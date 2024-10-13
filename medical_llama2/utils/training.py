import math
import os
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler

from medical_llama2.utils import ensure_num_saved_checkpoints, ensure_dir

if 'PJRT_DEVICE' in os.environ:
    import torch_xla as xla  # noqa: F401
    import torch_xla.amp.syncfree as syncfree  # provide modified version of optimizers to avoid the additional sync between device and host


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

def make_optimizer(
    model,
    device: torch.device,
    optim_type: str,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    use_syncfree_optim: bool = False,
) -> torch.optim.Optimizer:
    param_list = [param for param in model.parameters() if param.requires_grad]
    decay_params = [param for param in param_list if param.dim() >= 2]
    no_decay_params = [param for param in param_list if param.dim() < 2]
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    optim_type = optim_type.lower()
    use_fused_impl = device.type == 'cuda'
    if optim_type == 'adam':
        adam_optim = syncfree.Adam if use_syncfree_optim else torch.optim.Adam
        optimizer = adam_optim(param_groups, lr=lr, betas=betas, eps=eps, fused=use_fused_impl)
    elif optim_type == 'adamw':
        adamw_optim = syncfree.AdamW if use_syncfree_optim else torch.optim.AdamW
        optimizer = adamw_optim(param_groups, lr=lr, betas=betas, eps=eps, fused=use_fused_impl)
    else:
        raise ValueError(f'Unsupported optimizer type: {optim_type}. Possible values are: adam, adamw')

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

def save_model(args, model, optimizer, lr_scheduler, global_step, scaler):
    checkpoint_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'global_step': global_step + 1,
    }
    if scaler.is_enabled():
        checkpoint_dict['scaler'] = scaler.state_dict()
    ensure_dir(args.checkpoints_dir)
    ensure_num_saved_checkpoints(
        args.checkpoints_dir,
        'medical_llama2',
        args.saved_checkpoint_limit,
    )
    model_save_path = os.path.join(args.checkpoints_dir, f'medical_llama2-{global_step}.pt')
    torch.save(checkpoint_dict, model_save_path)

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
