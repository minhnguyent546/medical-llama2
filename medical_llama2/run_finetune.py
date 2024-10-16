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
from torch.utils.data import DataLoader

import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaModel,
    LlamaTokenizer,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

import medical_llama2.opts as opts
import medical_llama2.utils as utils
from medical_llama2.constants import SYSTEM_PROMPT
from medical_llama2.medical_conversation_dataset import MedicalConversationDataset
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
    raw_dataset: datasets.DatasetDict = datasets.load_dataset(
        'ruslanmv/ai-medical-chatbot',
        trust_remote_code=True,
    )  # pyright: ignore[reportAssignmentType]
    raw_dataset = raw_dataset['train'].shuffle(seed=args.seed).train_test_split(
        test_size=args.test_size,
        shuffle=True,
        seed=args.seed,
    )
    old_dataset = raw_dataset
    raw_dataset = old_dataset['train'].shuffle(seed=args.seed).train_test_split(
        test_size=args.validation_size,
        shuffle=True,
        seed=args.seed,
    )
    raw_dataset['validation'] = raw_dataset.pop('test')
    raw_dataset['test'] = old_dataset['test']

    # MedicalDataset
    train_dataset = MedicalConversationDataset(
        dataset=raw_dataset['train'], question_field=args.question_field,
        answer_field=args.answer_field, tokenizer=tokenizer,
        seq_length=args.seq_length, train_on_inputs=args.train_on_inputs,
    )
    validation_dataset = MedicalConversationDataset(
        dataset=raw_dataset['validation'], question_field=args.question_field,
        answer_field=args.answer_field, tokenizer=tokenizer,
        seq_length=args.seq_length, train_on_inputs=args.train_on_inputs,
    )
    test_dataset = MedicalConversationDataset(
        dataset=raw_dataset['test'], question_field=args.question_field,
        answer_field=args.answer_field, tokenizer=tokenizer,
        seq_length=args.seq_length, train_on_inputs=args.train_on_inputs,
    )
    data_collator = utils.CollatorWithPadding(
        tokenizer.pad_token_id,
        added_features=['input_ids', 'labels', 'attention_mask'],
    )

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
    device = torch.device(device, args.local_rank)
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
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=mp_dtype,
        bnb_4bit_quant_storage=args.bnb_4bit_quant_storage,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint,
        device_map=device,
        quantization_config=(bnb_config if device.type == 'cuda' else None),
        trust_remote_code=True,
        torch_dtype=args.bnb_4bit_quant_storage,
    )
    model.config.use_cache = args.use_cache
    # setting config.pretraining_tp to a value different than 1 will activate the more accurate
    # but slower computation of the linear layers, which should better match the original logits
    model.config.pretraining_tp = 1
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
    optimizer = utils.make_optimizer(model=model, args=args)
    loss_func = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

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

    unwrapped_model = model
    # convert the model to distributed data parallel
    if args.ddp_enabled:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    global_step = 0
    batch_loss = 0.0
    wandb_accum_logs: list[dict[str, Any]] = []
    running_loss = AverageMeter('running_loss', device=device)

    utils.master_print(
        f'Total training steps: {args.train_steps} '
        f'(roughly {args.train_steps / len(train_data_loader):0.2f} epoch(s))'
    )
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
            desc='Training model',
            ncols=120,
        )

    # set model in training mode
    model.train()
    optimizer.zero_grad()

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
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_func(outputs.logits.view(-1, tokenizer.vocab_size), labels.view(-1))

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
                        model=unwrapped_model,
                        device=device,
                        loss_func=loss_func,
                        tokenizer=tokenizer,
                        eval_data_loader=validation_data_loader,
                        validation_steps=args.valid_steps,
                        args=args,
                        autocast_context=autocast_context,
                    )
                    wandb_accum_logs[-1].update({
                        'loss/train': running_loss.average,
                        'loss/valid': valid_results['loss'],
                    })
                    running_loss.reset()

                if (global_step + 1) % args.generation_interval == 0:
                    eval_generation(
                        model=unwrapped_model,
                        device=device,
                        dataset=validation_dataset.dataset,
                        tokenizer=tokenizer,
                        generation_steps=args.generation_steps,
                        args=args,
                    )

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
                            model=unwrapped_model,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            global_step=global_step + 1,
                            scaler=scaler,
                            args=args,
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
    loss_func,
    tokenizer: LlamaTokenizer,
    eval_data_loader: DataLoader,
    validation_steps: int,
    args: argparse.Namespace,
    autocast_context=None,
) -> dict[str, float]:
    evaluation_loss = AverageMeter('evaluation_loss', device=device)
    if autocast_context is None:
        autocast_context = nullcontext()

    if args.ddp_enabled:
        progress_bar = tqdm(
            range(validation_steps),
            total=validation_steps,
            desc=f'GPU{args.rank} - Evaluating model',
            disable=args.local_rank != 0,
            ncols=120,
        )
    else:
        progress_bar = tqdm(
            range(validation_steps),
            total=validation_steps,
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
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_func(outputs.logits.view(-1, tokenizer.vocab_size), labels.view(-1))

            evaluation_loss.update(loss.detach())
            progress_bar.set_postfix({'loss': f'{loss:0.3f}'})
            progress_bar.update()
            if (batch_idx + 1) >= validation_steps:
                break

    # set model back to the original mode
    model.train(is_training)

    if args.ddp_enabled:
        evaluation_loss.reduce(dst=args.master_rank)

    return {
        'loss': evaluation_loss.average,
    }

def eval_generation(
    model,
    device: torch.device,
    dataset: datasets.Dataset,
    tokenizer: LlamaTokenizer,
    generation_steps: int,
    args: argparse.Namespace,
) -> None:
    generation_steps = min(generation_steps, len(dataset))

    if args.ddp_enabled:
        progress_bar = tqdm(
            range(generation_steps),
            total=generation_steps,
            desc=f'GPU{args.rank} - Evaluating generation',
            disable=args.local_rank != 0,
            ncols=120,
        )
    else:
        progress_bar = tqdm(
            range(generation_steps),
            total=generation_steps,
            desc='Evaluating generation',
            ncols=120,
        )

    is_training = model.training
    model.eval()
    for idx, item in enumerate(dataset):
        question = item[args.question_field]
        answer = item[args.answer_field]
        prompt = utils.generate_prompt(user_message=question, system_prompt=SYSTEM_PROMPT)
        model_inputs = tokenizer([prompt], return_tensors='pt').to(device)
        output = model.generate(
            **model_inputs,
            do_sample=args.do_sample,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            early_stopping=args.generation_early_stopping,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            num_return_sequences=args.num_return_sequences,
            repetition_penalty=args.repetition_penalty,
        )
        model_response = tokenizer.decode(output[0, len(model_inputs['input_ids'][0]):], skip_special_tokens=True)

        # TODO: calculate scores here (e.g. BERTScore)
        if args.is_master:
            progress_bar.write(f'>> QUESTION: {question}')
            progress_bar.write(f'>> ANSWER: {answer}')
            progress_bar.write(f'>> MODEL: {model_response}')

        progress_bar.update()
        if idx + 1 >= generation_steps:
            break

    model.train(is_training)

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
