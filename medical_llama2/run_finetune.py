"""Fine-tuning LLaMA 2 model on dialogue dataset (e.g. alpaca dataset, medical dataset, etc)"""

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

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    LlamaForCausalLM,
    LlamaModel,
    LlamaTokenizer,
)

from peft import (
    LoraConfig,
    TaskType,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

import medical_llama2.opts as opts
import medical_llama2.utils as utils
from medical_llama2.dialogue_dataset import DialogueDataset
from medical_llama2.meters import AverageMeter


def train_model(args: argparse.Namespace) -> None:
    utils.set_seed(args.seed)
    if not (args.do_train or args.do_test or args.do_test_generation):
        utils.master_print('Warning: please specify at least one of --do_train, --do_test, or --do_test_generation. Exiting...')
        return

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
    effective_batch_size = per_device_train_batch_size * args.world_size * args.gradient_accum_steps

    # dataset
    raw_dataset = utils.get_datasets(args)

    # MedicalDataset
    dialogue_dataset_common_kwargs = {
        'tokenizer': tokenizer, 'seq_length': args.max_seq_length,
        'input_field': args.input_field, 'output_field': args.output_field,
        'instruction_field': args.instruction_field, 'train_on_inputs': args.train_on_inputs,
        'prompt_template': args.prompt_template, 'dataset_num_procs': args.dataset_num_procs,
        'custom_instruction': args.custom_instruction,
    }
    train_dataset, validation_dataset, test_dataset = None, None, None
    if 'train' in raw_dataset:
        train_dataset = DialogueDataset(dataset=raw_dataset['train'], **dialogue_dataset_common_kwargs)  # pyright: ignore[reportArgumentType]
    if 'validation' in raw_dataset:
        validation_dataset = DialogueDataset(dataset=raw_dataset['validation'], **dialogue_dataset_common_kwargs)  # pyright: ignore[reportArgumentType]
    if 'test' in raw_dataset:
        test_dataset = DialogueDataset(dataset=raw_dataset['test'], **dialogue_dataset_common_kwargs)  # pyright: ignore[reportArgumentType]
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
        pad_to_multiple_of=8,
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

    # validation and test datasets for generation for this device (i.e. this node)
    per_device_validation_dataset_for_generation = None
    per_device_test_dataset_for_generation = None
    if validation_dataset is not None:
        per_device_validation_dataset_for_generation = validation_dataset.dataset.select(range(
            args.rank,
            len(validation_dataset),
            args.world_size,
        ))
    if test_dataset is not None:
        per_device_test_dataset_for_generation = test_dataset.dataset.select(range(
            args.rank,
            len(test_dataset),
            args.world_size,
        ))

    # training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device, args.local_rank)
    utils.master_print(f'Using device: {device}')

    # mixed precision training
    mp_dtype = None  # dtype for mixed-precision
    autocast_context = nullcontext()
    if args.mixed_precision is not None:
        mp_dtype = utils.get_mp_dtype(args.mixed_precision, device, verbose=args.is_master)
        autocast_context = torch.cuda.amp.autocast(enabled=(mp_dtype in (torch.float16, torch.bfloat16)), dtype=mp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(mp_dtype == torch.float16))

    is_load_in_kbit = args.load_in_8bit or args.load_in_4bit
    bnb_config = None
    if device.type == 'cuda' and is_load_in_kbit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            llm_int8_threshold=args.llm_int8_threshold,
            llm_int8_skip_modules=args.llm_int8_skip_modules,
            llm_int8_enable_fp32_cpu_offload=args.llm_int8_enable_fp32_cpu_offload,
            llm_int8_has_fp16_weight=args.llm_int8_has_fp16_weight,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_storage=args.bnb_4bit_quant_storage,
        )
    model = LlamaForCausalLM.from_pretrained(
        args.model_checkpoint,
        device_map=device,
        quantization_config=bnb_config,
        torch_dtype=args.model_torch_dtype,
        use_flash_attention_2=args.use_flash_attn_2,
        revision=args.model_checkpoint_revision,
    )
    model.config.use_cache = args.use_cache
    # setting config.pretraining_tp to a value different than 1 will activate the more accurate
    # but slower computation of the linear layers, which should better match the original logits
    model.config.pretraining_tp = 1
    if is_load_in_kbit:
        model = prepare_model_for_kbit_training(model, args.use_gradient_checkpointing)

    # getting peft model
    if args.peft_checkpoint is not None:
        utils.master_print(f'Getting peft model from {args.peft_checkpoint}')
        model = PeftModel.from_pretrained(
            model,
            args.peft_checkpoint,
            is_trainable=args.do_train,
            revision=args.peft_checkpoint_revision,
        )
        # override config in args with the config from the checkpoint
        peft_config = model.peft_config['default']
        keys_to_override = [
            'r',
            'lora_alpha',
            'lora_dropout',
            'target_modules',
            'modules_to_save',
            'bias',
            'task_type',
        ]
        for key in keys_to_override:
            if hasattr(peft_config, key):
                args_key = key
                if key != 'task_type' and not key.startswith('lora_'):
                    args_key = f'lora_{key}'
                setattr(args, args_key, getattr(peft_config, key))
    else:
        if args.task_type is None:
            raise ValueError('Please specify the task type for PEFT (via --task_type)')
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            modules_to_save=args.lora_modules_to_save,
            bias=args.lora_bias,
            task_type=args.task_type,
        )
        model = get_peft_model(model, peft_config)

    if args.is_master and args.do_train:
        model.print_trainable_parameters()

    learning_rate = args.learning_rate
    optimizer = utils.make_optimizer(model=model, args=args)

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
    # wrap the model with `DDP` to enable training with distributed data parallel
    if args.ddp_enabled and args.do_train:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

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

    # pushing to HuggingFace's hub
    push_steps = []
    tokenizer_pushed = False
    if args.is_master and args.push_to_hub and args.do_train and args.push_at_steps:
        push_steps = list(map(int, args.push_at_steps.split(',')))
        push_steps = sorted(push_steps)

    valid_steps = args.valid_steps
    test_steps = args.test_steps
    valid_generation_steps = args.valid_generation_steps
    test_generation_steps = args.test_generation_steps
    if valid_steps < 0 and validation_data_loader is not None:
        valid_steps = len(validation_data_loader)
    if test_steps < 0 and test_data_loader is not None:
        test_steps = len(test_data_loader)
    if valid_generation_steps < 0 and validation_dataset is not None:
        valid_generation_steps = len(validation_dataset.dataset)
    if test_generation_steps < 0 and test_dataset is not None:
        test_generation_steps = len(test_dataset.dataset)

    # divide generation steps across devices
    per_device_validation_generation_steps = utils.divide_across_device(
        valid_generation_steps,
        rank=args.rank, world_size=args.world_size, keep_rem=True,
    )
    per_device_test_generation_steps = utils.divide_across_device(
        test_generation_steps,
        rank=args.rank, world_size=args.world_size, keep_rem=True,
    )

    utils.master_print('******** General information ********')
    utils.master_print(
        f'  Dataset: train_size={len(train_data_loader) if train_data_loader else 0}, '
        f'test_size={len(test_data_loader) if test_data_loader else 0}, '
        f'validation_size={len(validation_data_loader) if validation_data_loader else 0}'
    )
    utils.master_print(
        f'  Effective batch size: {effective_batch_size} '
        f'(micro_batch_size={per_device_train_batch_size}, '
        f'gradient_accum_steps={args.gradient_accum_steps}, '
        f'num_devices={args.world_size})'
    )
    if train_data_loader is not None:
        utils.master_print(
            f'  Total training steps: {args.train_steps} '
            f'(roughly {args.train_steps * args.gradient_accum_steps / len(train_data_loader):0.2f} epoch(s))'
        )
    if args.valid_interval is not None:
        utils.master_print(
            f'  Validation interval: {args.valid_interval}, '
            f'validation steps: {valid_steps}'
        )
    if args.valid_generation_interval is not None:
        utils.master_print(
            f'  Validation generation interval: {args.valid_generation_interval}, '
            f'per device valid. generation steps: {per_device_validation_generation_steps}, '
            f'log interval: {args.generation_log_interval}'
        )
    if wandb_run is not None:
        utils.master_print(f'  Wandb logging interval: {args.wandb_logging_interval}')
    utils.master_print(f'  Push to hub: {args.push_to_hub}')
    if push_steps:
        utils.master_print(f'  Push at steps: {push_steps}')
    utils.master_print(f'  Do training: {args.do_train}')
    utils.master_print(f'  Do testing: {args.do_test}')
    utils.master_print(f'  Do testing generation: {args.do_test_generation}')

    if args.do_train:
        train_progressbar_desc = f'GPU{args.rank} - Training' if args.ddp_enabled else 'Training'
        train_progressbar = tqdm(
            range(args.train_steps),
            desc=train_progressbar_desc,
            disable=args.local_rank != 0,
            ncols=120,
        )

        # set model in training mode
        model.train()
        optimizer.zero_grad()

    wandb_accum_logs: list[dict[str, Any]] = []
    running_loss = AverageMeter('running_loss', device=device)
    global_step = 0

    # function definitions for training, testing model
    def do_train_single_epoch():
        assert train_data_loader is not None
        nonlocal global_step, wandb_accum_logs, train_progressbar, tokenizer_pushed

        total_num_samples = len(train_data_loader)
        last_iter_num_batches = total_num_samples % args.gradient_accum_steps
        if last_iter_num_batches == 0:
            last_iter_num_batches = args.gradient_accum_steps
        total_updates = (total_num_samples + args.gradient_accum_steps - 1) // args.gradient_accum_steps

        batch_loss = 0.0

        train_data_iterator = iter(train_data_loader)
        for update_step in range(total_updates):
            is_last_iteration = (global_step + 1 >= args.train_steps)

            num_batches = args.gradient_accum_steps if update_step < total_updates - 1 else last_iter_num_batches
            batches, num_items_in_batch = utils.get_batch_samples(train_data_iterator, num_batches)
            num_batches = len(batches)
            for batch_idx, batch in enumerate(batches):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                if args.ddp_enabled:
                    # only sync gradients at the last iteration of batches
                    # we can use the below trick or model.no_sync context manager (see: https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py#L1404)
                    model.require_backward_grad_sync = (batch_idx + 1 == num_batches)  # pyright: ignore[reportArgumentType]

                with autocast_context:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = utils.fixed_causal_lm_loss(
                        outputs.logits,
                        labels,
                        tokenizer.vocab_size,
                        num_items_in_batch=num_items_in_batch,
                    )

                scaler.scale(loss).backward()
                batch_loss += loss.detach()

            wandb_accum_logs.append({})
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                wandb_accum_logs[-1].update({
                    'grad_norm': grad_norm,
                    'step': global_step,
                })

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # TODO: handle the case when wandb is disabled
            wandb_accum_logs[-1].update({
                f'learning_rate/group_{group_id}': group_lr
                for group_id, group_lr in enumerate(lr_scheduler.get_last_lr())
            })
            wandb_accum_logs[-1].update({
                'loss/batch_loss': batch_loss,
                'step': global_step,
            })

            lr_scheduler.step()
            running_loss.update(batch_loss)  # pyright: ignore[reportArgumentType]

            if args.valid_interval is not None and (global_step + 1) % args.valid_interval == 0:
                assert validation_data_loader is not None
                if args.ddp_enabled:
                    running_loss.reduce(dst=args.master_rank)
                valid_results = utils.eval_model(
                    model=unwrapped_model,
                    device=device,
                    tokenizer=tokenizer,
                    eval_data_loader=validation_data_loader,
                    eval_steps=valid_steps,
                    args=args,
                    autocast_context=autocast_context,
                )
                wandb_accum_logs[-1].update({
                    'loss/train': running_loss.average,
                    'loss/valid': valid_results['loss'],
                })
                running_loss.reset()
                if args.ddp_enabled:
                    dist.barrier()

            if args.valid_generation_interval is not None and (global_step + 1) % args.valid_generation_interval == 0:
                assert per_device_validation_dataset_for_generation is not None
                gen_results = utils.eval_generation(
                    model=unwrapped_model,
                    device=device,
                    dataset=per_device_validation_dataset_for_generation,
                    tokenizer=tokenizer,
                    generation_steps=per_device_validation_generation_steps,
                    args=args,
                    generation_log_interval=args.generation_log_interval,
                )
                for bs_key in ('bert_score', 'bert_score_unscaled'):
                    if bs_key in gen_results:
                        wandb_accum_logs[-1].update({
                            f'{bs_key}/precision': gen_results[bs_key]['precision'],
                            f'{bs_key}/recall': gen_results[bs_key]['recall'],
                            f'{bs_key}/f1': gen_results[bs_key]['f1'],
                        })

                if args.ddp_enabled:
                    dist.barrier()

            if (
                len(wandb_accum_logs) >= args.wandb_logging_interval or
                (len(wandb_accum_logs) > 0 and is_last_iteration)
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
                        model=unwrapped_model,
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
            train_progressbar.update()
            global_step += 1

            if global_step in push_steps:
                trained_epoch = global_step * args.gradient_accum_steps / len(train_data_loader)
                unwrapped_model.push_to_hub(
                    args.repo_id,
                    commit_message=getattr(args, 'commit_message',
                    'Upload model') + f' (step {global_step}, epoch {trained_epoch:0.2f})',
                )
                if not tokenizer_pushed and args.push_tokenizer:
                    tokenizer.push_to_hub(args.repo_id)
                    tokenizer_pushed = True

            if is_last_iteration:
                break

    def do_test():
        assert test_data_loader is not None
        utils.master_print('******** Testing model ********')
        test_results = utils.eval_model(
            model=unwrapped_model,
            device=device,
            tokenizer=tokenizer,
            eval_data_loader=test_data_loader,
            eval_steps=test_steps,
            args=args,
            autocast_context=autocast_context,
        )
        utils.master_print(f'  Number of testing steps: {test_steps}')
        utils.master_print(f'  Test loss: {test_results["loss"]}')
        utils.master_print(f'  Test perplexity: {utils.get_perplexity(test_results["loss"])}')
        if wandb_run is not None:
            wandb_run.log({
                'test/loss': test_results['loss'],
                'test/perplexity': utils.get_perplexity(test_results["loss"]),
            })

    def do_test_generation():
        assert per_device_test_dataset_for_generation is not None
        utils.master_print('******** Testing generation ********')
        gen_results = utils.eval_generation(
            model=unwrapped_model,
            device=device,
            dataset=per_device_test_dataset_for_generation,
            tokenizer=tokenizer,
            generation_steps=per_device_test_generation_steps,
            args=args,
            generation_log_interval=args.generation_log_interval,
            is_test=True,
            log_file=args.test_generation_log_file,
        )

        utils.master_print(f'  Total generation steps: {test_generation_steps}')
        utils.master_print(f'  Per device generation steps: {per_device_test_generation_steps}')
        if args.test_generation_log_file is not None:
            utils.master_print(f'  Log file: {args.test_generation_log_file}')
        for bs_key in ('bert_score', 'bert_score_unscaled'):
            if bs_key in gen_results:
                utils.master_print(
                    f'{bs_key}: precision = {gen_results[bs_key]["precision"]:0.3f}, '
                    f'recall = {gen_results[bs_key]["recall"]:0.3f}, '
                    f'F1 = {gen_results[bs_key]["f1"]:0.3f} '
                    f'hash_code = {gen_results[bs_key]["hash_code"]}',
                )
                if wandb_run is not None:
                    wandb_run.log({
                        f'test/{bs_key}/precision': gen_results[bs_key]['precision'],
                        f'test/{bs_key}/recall': gen_results[bs_key]['recall'],
                        f'test/{bs_key}/f1': gen_results[bs_key]['f1'],
                    })

    # main stuff start here
    if args.do_train:
        while global_step < args.train_steps:
            do_train_single_epoch()

    if args.do_test:
        do_test()

    if args.do_test_generation:
        do_test_generation()

    if args.is_master and args.push_to_hub and not push_steps:
        unwrapped_model.push_to_hub(args.repo_id, commit_message=args.commit_message)
        if args.push_tokenizer:
            tokenizer.push_to_hub(args.repo_id)

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
