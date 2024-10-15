import math
import os
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler

from peft.optimizers import create_loraplus_optimizer

import bitsandbytes as bnb

from medical_llama2.utils import ensure_dir, ensure_num_saved_checkpoints


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

def save_model(model, optimizer, lr_scheduler, global_step, scaler, args):
    ensure_dir(args.checkpoints_dir)
    ensure_num_saved_checkpoints(
        checkpoints_dir=args.checkpoints_dir,
        file_or_dir_glob=r'medical_llama2-*',
        limit=args.saved_checkpoint_limit,
    )
    ck_save_path = os.path.join(args.checkpoints_dir, f'medical_llama2-{global_step}')
    model.save_pretrained(os.path.join(ck_save_path, 'hf_model'))
    if not args.save_model_only:
        checkpoint_dict = {
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'global_step': global_step + 1,
        }
        if scaler.is_enabled():
            checkpoint_dict['scaler'] = scaler.state_dict()
        torch.save(checkpoint_dict, os.path.join(ck_save_path, 'other_states.pt'))

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
            test_dataset,
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

def get_mp_dtype(args, device: torch.device) -> torch.dtype:
    mp_dtype = torch.float32
    if device.type == 'cuda' and args.mixed_precision == 'float16':
        mp_dtype = torch.float16
        if args.is_master:
            print('Mixed precision training is enabled with float16')
    elif device.type == 'cuda' and args.mixed_precision == 'bfloat16':
        if torch.cuda.is_bf16_supported():
            mp_dtype = torch.bfloat16
            if args.is_master:
                print('Mixed precision training is enabled with bfloat16')
        else:
            mp_dtype = torch.float16
            if args.is_master:
                print('bfloat16 is not supported on your hardware, fallback to float16')

    return mp_dtype
