"""Fine-tuning LLaMA 2 model on medical data."""

import argparse
import wandb
from contextlib import nullcontext
from tqdm.autonotebook import tqdm
from typing import Any

import torch
import torch.amp
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaTokenizer,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.optimizers import create_loraplus_optimizer

import bitsandbytes as bnb

import medical_llama2.opts as opts
import medical_llama2.utils as utils
from medical_llama2.medical_dataset import MedicalDataset
from medical_llama2.meters import AverageMeter


def train_model(args: argparse.Namespace) -> None:
    utils.set_seed(args.seed)

    # tokenizer
    tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

    if args.train_batch_size % args.world_size != 0:
        raise ValueError('train_batch_size must be divisible by world_size')
    if args.eval_batch_size % args.world_size != 0:
        raise ValueError('eval_batch_size must be divisible by world_size')
    per_device_train_batch_size = args.train_batch_size // args.world_size
    per_device_eval_batch_size = args.eval_batch_size // args.world_size
    effective_batch_size = per_device_train_batch_size * args.world_size * args.gradient_accum_step
    utils.master_print(
        f'Effective batch size: {effective_batch_size} '
        f'(micro_batch_size={per_device_train_batch_size}, '
        f'gradient_accum_step={args.gradient_accum_step}, '
        f'num_devices={args.world_size})'
    )

    # dataset
    raw_dataset: DatasetDict = load_dataset(
        'ruslanmv/ai-medical-chatbot',
        trust_remote_code=True,
    )  # pyright: ignore[reportAssignmentType]
    raw_dataset = raw_dataset['train'].train_test_split(
        test_size=args.test_size,
        shuffle=True,
        seed=args.seed,
    )
    old_dataset = raw_dataset
    raw_dataset = old_dataset['train'].train_test_split(
        test_size=args.validation_size,
        shuffle=True,
        seed=args.seed,
    )
    raw_dataset['validation'] = raw_dataset.pop('test')
    raw_dataset['test'] = old_dataset['test']

    # MedicalDataset
    train_dataset = MedicalDataset(dataset=raw_dataset['train'], tokenizer=tokenizer, seq_length=args.seq_length)
    validation_dataset = MedicalDataset(dataset=raw_dataset['validation'], tokenizer=tokenizer, seq_length=args.seq_length)
    test_dataset = MedicalDataset(dataset=raw_dataset['test'], tokenizer=tokenizer, seq_length=args.seq_length)
    data_collator = utils.CollatorWithPadding(tokenizer.pad_token_id, added_features=['input_ids', 'labels', 'attention_mask'])

    # data loaders
    train_data_loader, test_data_loader, validation_data_loader = utils.make_data_loaders(
        args=args,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        validation_dataset=validation_dataset,
        data_collator=data_collator,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
    )
    utils.master_print(
        f'Dataset: train_size={len(train_data_loader)}, '
        f'test_size={len(test_data_loader)}, '
        f'validation_size={len(validation_data_loader)}'
    )

    # training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    utils.master_print(f'Using device: {device}')

    # logging with wandb
    wandb_run = None
    if args.is_master and args.wandb_logging:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            id=args.wandb_resume_id,
            resume='must' if args.wandb_resume_id is not None else None,
        )

    # mixed precision training
    mp_dtype = utils.get_mp_dtype(args=args, device=device)
    autocast_context = torch.cuda.amp.autocast(enabled=(mp_dtype in (torch.float16, torch.bfloat16)), dtype=mp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(mp_dtype == torch.float16))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.bnb_load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,  # nf4
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,  # true
        bnb_4bit_compute_dtype=mp_dtype,  # bfloat16
        bnb_4bit_quant_storage=args.bnb_4bit_quant_storage,  # bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint,
        device_map=device,
        quantization_config=(bnb_config if device.type == 'cuda' else None),
        trust_remote_code=True,
        torch_dtype=args.bnb_4bit_quant_storage,
    )
    model.config.use_cache = False
    # setting config.pretraining_tp to a value different than 1 will activate the more accurate
    # but slower computation of the linear layers, which should better match the original logits
    model.config.pretraining_tp = 1;
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
    )
    model = prepare_model_for_kbit_training(model, args.use_gradient_checkpointing)
    model = get_peft_model(model, peft_config)
    if args.is_master:
        model.print_trainable_parameters()

    learning_rate = args.learning_rate
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=learning_rate,
    #     betas=args.betas,
    #     weight_decay=args.weight_decay,
    # )
    optimizer = create_loraplus_optimizer(
        model=model,
        optimizer_cls=bnb.optim.PagedAdam32bit,
        lr=learning_rate,
        loraplus_lr_ratio=16,
        betas=args.betas,
        weight_decay=args.weight_decay,
        percentile_clipping=5,
    )

    if args.decay_method == 'noam':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: utils.noam_decay(
                step, args.d_model, args.warmup_steps,
            ),
        )
    elif args.decay_method == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: utils.cosine_decay(
                step, learning_rate, args.min_lr,
                args.warmup_steps,
                args.decay_steps, factor=1/learning_rate,
            ),
        )
    else:
        raise ValueError(f'Unsupported scheduler decay method: {args.decay_method}')

    raw_model = model
    # convert the model to distributed data parallel
    if args.ddp_enabled:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    global_step = 0
    batch_loss = 0.0
    wandb_accum_logs: list[dict[str, Any]] = []
    running_loss = AverageMeter('running_loss', device=device)

    if args.ddp_enabled:
        train_progressbar = tqdm(
            range(args.train_steps),
            desc=f'GPU{args.rank} - Training model',
            disable=args.local_rank != 0,
            ncols=120,
        )
    else:
        train_progressbar = tqdm(
            range(args.train_steps),
            desc=f'Training model',
            ncols=120,
        )

    # set model in training mode
    model.train()
    optimizer.zero_grad()
    utils.master_print(
        f'Total training steps: {args.train_steps} '
        f'(roughly {args.train_steps / len(train_data_loader):0.2f} epoch(s))'
    )
    while global_step < args.train_steps:
        for batch_idx, batch in enumerate(train_data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if args.ddp_enabled:
                # we only sync gradients at the last step of gradient accumulation
                # we can use the below trick or model.no_sync context manager (see: https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py#L1404)
                model.require_backward_grad_sync = (batch_idx + 1) % args.gradient_accum_step == 0

            with autocast_context:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            if args.gradient_accum_step > 1:
                loss /= args.gradient_accum_step
            batch_loss += loss.detach()

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.gradient_accum_step == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # TODO: handle the case when wandb is disabled
                wandb_accum_logs.append({
                    f'learning_rate/group_{group_id}': group_lr
                    for group_id, group_lr in enumerate(lr_scheduler.get_last_lr())
                })
                wandb_accum_logs[-1].update({
                    'loss/batch_loss': batch_loss,
                    'step': global_step,
                })

                lr_scheduler.step()
                running_loss.update(batch_loss)

                if (global_step + 1) % args.valid_interval == 0:
                    if args.ddp_enabled:
                        running_loss.reduce(dst=args.master_rank)
                    valid_results = eval_model(
                        model,
                        device,
                        validation_data_loader,
                        args.valid_steps,
                        args,
                        autocast_context,
                    )
                    wandb_accum_logs[-1].update({
                        'loss/train': running_loss.average,
                        'loss/valid': valid_results['loss'],
                    })
                    running_loss.reset()

                if (
                    len(wandb_accum_logs) >= args.wandb_logging_interval or
                    (len(wandb_accum_logs) > 0 and batch_idx + 1 >= len(train_data_loader))
                ):
                    batch_loss_values = torch.tensor(
                        [loss['loss/batch_loss'] for loss in wandb_accum_logs],
                        dtype=torch.float32,
                        device=device,
                    )
                    if args.ddp_enabled:
                        dist.all_reduce(batch_loss_values, op=dist.ReduceOp.AVG)
                        batch_loss_values = batch_loss_values.tolist()
                    for idx in range(len(wandb_accum_logs)):
                        wandb_accum_logs[idx]['loss/batch_loss'] = batch_loss_values[idx]
                    if wandb_run is not None:
                        for log_idx in range(len(wandb_accum_logs)):
                            wandb_run.log(wandb_accum_logs[log_idx])
                    wandb_accum_logs = []
                    if args.ddp_enabled:
                        dist.barrier()

                if (global_step + 1) % args.save_interval == 0:
                    if args.is_master:
                        utils.save_model(
                            args=args,
                            model=raw_model,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            global_step=global_step + 1,
                            scaler=scaler,
                        )
                    if args.ddp_enabled:
                        dist.barrier()

                train_progressbar.set_postfix({
                    'loss': f'{batch_loss:0.3f}',
                })
                batch_loss = 0.0
                global_step += 1
                train_progressbar.update()
                if global_step >= args.train_steps:
                    break

def eval_model(
    model,
    device: torch.device,
    eval_data_loader,
    valid_steps: int,
    args: argparse.Namespace,
    autocast_context=None,
) -> dict[str, float]:
    evaluation_loss = AverageMeter('evaluation_loss', device=device)
    if autocast_context is None:
        autocast_context = nullcontext()

    if args.ddp_enabled:
        progress_bar = tqdm(
            range(valid_steps),
            total=valid_steps,
            desc=f'GPU{args.rank} - Evaluating model',
            disable=args.local_rank != 0,
            ncols=120,
        )
    else:
        progress_bar = tqdm(
            range(valid_steps),
            total=valid_steps,
            desc='Evaluating model',
            ncols=120,
        )

    # set model in evaluation mode
    is_training = model.training
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast_context:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            evaluation_loss.update(loss.detach())
            progress_bar.set_postfix({'loss': f'{loss:0.3f}'})
            progress_bar.update()
            if (batch_idx + 1) >= valid_steps:
                break

    # set model back to the original mode
    model.train(is_training)

    if args.ddp_enabled:
        evaluation_loss.reduce(dst=args.master_rank)

    return {
        'loss': evaluation_loss.average,
    }

def main():
    parser = argparse.ArgumentParser(
        description='Run fine-tuning LLaMA 2 model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    opts.add_run_finetune_opts(parser)
    args = parser.parse_args()

    utils.setup_ddp(args)

    train_model(args)

    utils.cleanup_ddp(args)


if __name__ == '__main__':
    main()
