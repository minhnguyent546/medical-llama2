import glob
import io
import math
import os
import random
import re
import shutil
import yaml
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from pickle import Pickler, Unpickler

from bert_score import BERTScorer

from medical_llama2.constants import (
    ALPACA_SYSTEM_PROMPT,
    ALPACA_SYSTEM_PROMPT_NO_INPUT,
    LLAMA_SYSTEM_PROMPT,
)
from medical_llama2.constants import SpecialToken


def set_seed(seed: int = 0x3f3f3f3f):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_yaml_config(config_path: str):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    return config

def master_print(*values: object, local_master: bool = True, **kwargs) -> None:
    is_master = bool(int(os.environ.get('is_master', 0)))
    is_local_master = bool(int(os.environ.get('is_local_master', 0)))
    should_print = (is_master or (local_master and is_local_master))
    if should_print:
        print(*values, **kwargs)

def chunks(data: list[Any] | str, chunk_size: int = 1_000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def is_xla_device(device: torch.device | None) -> bool:
    return device is not None and device.type == 'xla'

def get_checkpoint_timestamp(checkpoint_name: str) -> int:
    timestamp = re.findall(r'\d+', checkpoint_name)
    if not timestamp:
        raise RuntimeError(f'Unable to infer timestamp from checkpoint: {checkpoint_name}')
    return timestamp[0]

def ensure_num_saved_checkpoints(
    checkpoints_dir: str,
    file_or_dir_glob: str,
    limit: int,
) -> None:
    ck_basenames = glob.glob(file_or_dir_glob, root_dir=checkpoints_dir)
    ck_basenames = list(ck_basenames)
    if len(ck_basenames) <= limit:
        return

    ck_basenames = sorted(ck_basenames, key=get_checkpoint_timestamp)
    for ck_basename in ck_basenames[:-limit]:
        full_path = os.path.join(checkpoints_dir, ck_basename)
        if os.path.isfile(full_path):
            os.remove(full_path)
        else:
            shutil.rmtree(full_path)

def count_model_param(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def object_to_tensor(obj, device, group=None):
    """Modified from `torch/distributed/distributed_c10d.py`."""
    f = io.BytesIO()
    Pickler(f).dump(obj)
    byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # pyright: ignore[reportPrivateUsage]
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
    # Otherwise, it will casue 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage).to(device)
    local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
    return byte_tensor, local_size

def tensor_to_object(tensor, tensor_size, group=None):
    """Modified from `torch/distributed/distributed_c10d.py`."""
    tensor = tensor.cpu()
    buf = tensor.numpy().tobytes()[:tensor_size]
    return Unpickler(io.BytesIO(buf)).load()

def get_perplexity(loss: float) -> float:
    """Calculating perplexity, i.e. e^{loss}"""
    return math.exp(loss)

def generate_llama2_prompt(
    user_message: str,
    response: str | None = None,
) -> str:
    """
    For more information on Llama2 prompt format, see https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    """
    prompt = (
        f'{SpecialToken.SOS}{SpecialToken.START_INST} {SpecialToken.START_SYS}\n'
        f'{LLAMA_SYSTEM_PROMPT}\n'
        f'{SpecialToken.END_SYS}\n\n'
        f'{user_message} {SpecialToken.END_INST}'
    )
    if response is not None:
        prompt += f' {response}'
    return prompt.strip()

def generate_alpaca_prompt(
    instruction: str,
    input: str | None = None,
    response: str | None = None,
) -> str:
    prompt = ''
    if input is not None and input:
        prompt = (
            f'{ALPACA_SYSTEM_PROMPT}\n\n'
            f'### Instruction:\n'
            f'{instruction}\n\n'
            f'### Input:\n'
            f'{input}\n\n'
            f'### Response:\n'
            f'{response}'
        )
    else:
        prompt = (
            f'{ALPACA_SYSTEM_PROMPT_NO_INPUT}\n\n'
            f'### Instruction:\n'
            f'{instruction}\n\n'
            f'### Response:\n'
            f'{response}'
        )
    return prompt.strip()

def compute_bert_score(bert_scorer: BERTScorer, cands, refs) -> dict[str, float]:
    precision, recall, f1 = bert_scorer.score(
        cands=cands,
        refs=refs,
    )
    precision = precision.mean().item()
    recall = recall.mean().item()
    f1 = f1.mean().item()
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
