import argparse
import math
import os
from contextlib import nullcontext
from tqdm.autonotebook import tqdm
from typing import Any

import bert_score

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler

import datasets
from transformers import LlamaTokenizer

import bitsandbytes as bnb

from peft.optimizers import create_loraplus_optimizer

from medical_llama2.meters import AverageMeter
from medical_llama2.utils import (
    compute_bert_score,
    ensure_dir,
    ensure_num_saved_checkpoints,
    fixed_causal_lm_loss,
    gather_object,
    generate_alpaca_prompt,
    generate_llama2_prompt,
)


def noam_decay(step_num: int, d_model: int, warmup_steps: int, factor: float = 1.0) -> float:
    """As described in https://arxiv.org/pdf/1706.03762.pdf."""
    step_num = max(step_num, 1)
    return factor * d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))

def cosine_decay(
    step_num: int,
    lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_steps: int,
    factor: float = 1.0,
) -> float:
    """Cosine decay with warmup."""
    step_num = max(step_num, 1)
    decayed_lr = None
    if step_num <= warmup_steps:
        decayed_lr = lr * step_num / warmup_steps
    elif step_num > decay_steps:
        decayed_lr = min_lr
    else:
        decay_ratio = (step_num - warmup_steps) / (decay_steps - warmup_steps)
        assert 0 <= decay_ratio and decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        decayed_lr = min_lr + (lr - min_lr) * coeff
    return factor * decayed_lr

def get_optim_cls(optim_type: str):
    optim_type_to_cls = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'bnb_adam32bit': bnb.optim.Adam32bit,
        'bnb_adamw32bit': bnb.optim.AdamW32bit,
        'bnb_adam8bit': bnb.optim.Adam8bit,
        'bnb_adamw8bit': bnb.optim.AdamW8bit,
    }
    optim_type = optim_type.lower()
    if optim_type in optim_type_to_cls:
        return optim_type_to_cls[optim_type]
    raise ValueError(f'Unknown optim type: {optim_type}')

def make_optimizer(model, args):
    optim_cls = get_optim_cls(args.optim_type)
    if args.optim_type.startswith('bnb_'):
        optimizer = create_loraplus_optimizer(
            model=model,
            optimizer_cls=optim_cls,
            lr=args.learning_rate,
            loraplus_lr_ratio=args.loraplus_lr_ratio,
            percentile_clipping=args.bnb_optim_percentile_clipping,
            betas=args.betas,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim_cls(
            model.parameters(),
            lr=args.learning_rate,
            betas=args.betas,
            weight_decay=args.weight_decay,
        )
    return optimizer


class CollatorWithPadding:
    def __init__(self, padding_value: int, added_features: list[str], attention_mask_key: str = 'attention_mask') -> None:
        self.padding_value = padding_value
        self.added_features = added_features
        self.attention_mask_key = attention_mask_key

    def __call__(self, original_batch: list[dict[str, Any]]) -> dict[str, Any]:
        all_features = original_batch[0].keys()
        remain_features = [key for key in all_features if key not in self.added_features]

        feature_dict = {key: [item[key] for item in original_batch] for key in self.added_features}
        batch = {key: [item[key] for item in original_batch] for key in remain_features}

        # for attention mask: we will pad 0 (a.k.a false) instead of `padding_value`
        feature_dict = {
            key: pad_sequence(value, batch_first=True, padding_value=(self.padding_value if key != self.attention_mask_key else 0))
            for key, value in feature_dict.items()
        }
        batch.update(feature_dict)
        return batch

def save_model(args, model, optimizer=None, lr_scheduler=None, global_step=None, scaler=None):
    ensure_dir(args.checkpoints_dir)
    ensure_num_saved_checkpoints(
        checkpoints_dir=args.checkpoints_dir,
        file_or_dir_glob=r'medical_llama2-*',
        limit=args.saved_checkpoint_limit,
    )
    ck_save_path = os.path.join(args.checkpoints_dir, f'medical_llama2-{global_step}')
    model.save_pretrained(os.path.join(ck_save_path, 'peft_model'))
    if not args.save_model_only:
        checkpoint_dict = {}
        if optimizer is not None:
            checkpoint_dict['optimizer'] = optimizer.state_dict()
        if lr_scheduler is not None:
            checkpoint_dict['lr_scheduler'] = lr_scheduler.state_dict()
        if global_step is not None:
            checkpoint_dict['global_step'] = global_step
        if scaler is not None and scaler.is_enabled():
            checkpoint_dict['scaler'] = scaler.state_dict()

        if checkpoint_dict:
            torch.save(checkpoint_dict, os.path.join(ck_save_path, 'other_states.pt'))

def get_datasets(args):
    raw_dataset: datasets.DatasetDict = datasets.load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        data_files=args.dataset_data_files,
        num_proc=args.dataset_num_procs,
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
    return raw_dataset

def make_data_loaders(
    args,
    train_dataset,
    test_dataset,
    validation_dataset,
    data_collator,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int
):
    train_sampler, test_sampler, validation_sampler = None, None, None
    if args.ddp_enabled:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
            seed=args.seed,
            drop_last=True,
        )
        validation_sampler = DistributedSampler(
            validation_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
            seed=args.seed,
            drop_last=True,
        )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=True,
    )
    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
        sampler=validation_sampler,
        pin_memory=True,
    )
    return train_data_loader, test_data_loader, validation_data_loader

def get_mp_dtype(mixed_precision: str, device: torch.device, verbose: bool = True) -> torch.dtype:
    mp_dtype = torch.float32
    if device.type == 'cuda' and mixed_precision == 'float16':
        mp_dtype = torch.float16
        if verbose:
            print('Mixed precision training is enabled with float16')
    elif device.type == 'cuda' and mixed_precision == 'bfloat16':
        if torch.cuda.is_bf16_supported():
            mp_dtype = torch.bfloat16
            if verbose:
                print('Mixed precision training is enabled with bfloat16')
        else:
            mp_dtype = torch.float16
            if verbose:
                print('bfloat16 is not supported on your hardware, fallback to float16')

    return mp_dtype

def get_batch_samples(data_iterator, num_batches) -> tuple[list[Any], int | None]:
    batch_samples = []
    num_items_in_batch = None
    for _ in range(num_batches):
        try:
            batch_samples.append(next(data_iterator))
        except StopIteration:
            break
    if batch_samples and 'labels' in batch_samples[0]:
        num_items_in_batch = sum(
            torch.count_nonzero(batch_sample['labels'] != -100).item()
            for batch_sample in batch_samples
        )
    return batch_samples, num_items_in_batch

def eval_model(
    model,
    device: torch.device,
    tokenizer: LlamaTokenizer,
    eval_data_loader: DataLoader,
    eval_steps: int,
    args: argparse.Namespace,
    autocast_context=None,
) -> dict[str, float]:
    evaluation_loss = AverageMeter('evaluation_loss', device=device)
    if autocast_context is None:
        autocast_context = nullcontext()

    progress_bar_desc = f'GPU{args.rank} - Evaluating' if args.ddp_enabled else 'Evaluating'
    progress_bar = tqdm(
        range(eval_steps),
        total=eval_steps,
        desc=progress_bar_desc,
        disable=args.local_rank != 0,
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
                loss = fixed_causal_lm_loss(outputs.logits, labels, tokenizer.vocab_size)

            evaluation_loss.update(loss.detach())
            progress_bar.set_postfix({'loss': f'{loss:0.3f}'})
            progress_bar.update()
            if batch_idx + 1 >= eval_steps:
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
    generation_log_interval: int | None = None,
) -> dict[str, Any]:
    generation_steps = min(generation_steps, len(dataset))

    progress_bar_desc = f'GPU{args.rank} - Generating' if args.ddp_enabled else 'Generating'
    progress_bar = tqdm(
        range(generation_steps),
        total=generation_steps,
        desc=progress_bar_desc,
        disable=args.local_rank != 0,
    )

    bert_scorer = None
    bert_scorer_unscaled = None
    if args.is_master:
        bert_score_kwargs = {
            'model_type': 'roberta-large',
            'device': device,
            'lang': 'en',
            'use_fast_tokenizer': True,
        }
        if args.bert_score_type == 'unscaled' or args.bert_score_type == 'both':
            bert_scorer_unscaled = bert_score.BERTScorer(**bert_score_kwargs)
        if args.bert_score_type == 'scaled' or args.bert_score_type == 'both':
            bert_scorer = bert_score.BERTScorer(
                **bert_score_kwargs,
                rescale_with_baseline=True,
            )
    predictions: list[str] = []
    references: list[str] = []

    is_training = model.training
    model.eval()
    for idx, item in enumerate(dataset):
        input_data = item[args.input_field]
        output_data = item[args.output_field]
        if args.prompt_template == 'llama2':
            prompt = generate_llama2_prompt(
                user_message=item[args.input_field],
            )
        else:
            prompt = generate_alpaca_prompt(
                instruction=item[args.instruction_field],
                input=item[args.input_field],
                response='',
            )
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
        model_response = model_response.strip()

        if args.is_local_master:
            if generation_log_interval is not None and (idx + 1) % generation_log_interval == 0:
                if args.instruction_field in item:
                    progress_bar.write(f'>> INST: {item[args.instruction_field]}')
                progress_bar.write(f'>> INPUT: {input_data}')
                progress_bar.write(f'>> OUTPUT: {output_data}')
                progress_bar.write(f'>> MODEL: {model_response}')
            predictions.append(model_response)
            references.append(output_data)

        progress_bar.update()
        if idx + 1 >= generation_steps:
            break

    model.train(is_training)

    # gather responses across devices
    if args.ddp_enabled:
        predictions = gather_object(predictions, args)
        references = gather_object(references, args)
    outputs = {}
    if bert_scorer is not None:
        assert predictions is not None
        assert references is not None
        outputs['bert_score'] = compute_bert_score(bert_scorer, cands=predictions, refs=references)
    if bert_scorer_unscaled is not None:
        assert predictions is not None
        assert references is not None
        outputs['bert_score_unscaled'] = compute_bert_score(bert_scorer_unscaled, cands=predictions, refs=references)
    return outputs
