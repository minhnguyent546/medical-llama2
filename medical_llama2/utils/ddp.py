import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def setup_ddp(args: argparse.Namespace) -> None:
    args.rank = int(os.environ.get('RANK', 0))
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.ddp_enabled = os.environ.get('RANK', -1) != -1
    args.master_rank = 0
    args.local_master_rank = 0
    args.is_master = (args.rank == args.master_rank)
    args.is_local_master = (args.local_rank == args.local_master_rank)
    os.environ['is_master'] = '1' if args.is_master else '0'
    os.environ['is_local_master'] = '1' if args.is_local_master else '0'
    if args.ddp_enabled:
        # set appropriate CUDA device
        torch.cuda.set_device(args.local_rank)
        # init process group
        dist.init_process_group(
            backend=getattr(args, 'ddp_backend', 'nccl'),
            timeout=timedelta(seconds=args.ddp_timeout),
        )  # nccl, gloo, etc

def cleanup_ddp(args: argparse.Namespace) -> None:
    if args.ddp_enabled:
        dist.destroy_process_group()

def gather_object(data_object: object, args: argparse.Namespace) -> list[object] | None:
    gathered_data = [None for _ in range(args.world_size)]
    dist.gather_object(
        data_object,
        gathered_data if args.is_master else None,
        dst=args.master_rank,
    )
    return gathered_data

def divide_across_device(value: int, rank: int, world_size: int, keep_rem: bool = True) -> int:
    rem = value % world_size
    ret = value // world_size
    if keep_rem:
        ret += int(rank < rem)
    return ret
