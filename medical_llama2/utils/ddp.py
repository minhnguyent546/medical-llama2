import argparse
import os

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
    os.environ['is_master'] = str(args.is_master)
    os.environ['is_local_master'] = str(args.is_local_master)
    if args.ddp_enabled:
        # set appropriate CUDA device
        torch.cuda.set_device(args.local_rank)
        # init process group
        dist.init_process_group(backend=getattr(args, 'ddp_backend', 'nccl'))  # nccl, gloo, etc

def cleanup_ddp(args: argparse.Namespace) -> None:
    if args.ddp_enabled:
        dist.destroy_process_group()
