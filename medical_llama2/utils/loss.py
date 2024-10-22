"""
This util aims to fix the bug in LLM Training about gradient accumulation.
Relevant links:
    - https://github.com/unslothai/unsloth/releases/tag/October-2024
    - https://huggingface.co/blog/gradient_accumulation
    - https://www.reddit.com/r/MachineLearning/comments/1g8ymrn/r_gradient_accumulation_bug_fix_in_nightly/

The codes below are derived from https://github.com/huggingface/transformers/pull/34191/files.

If use you a newer version of Transformers, please ignore this.
"""

import torch
import torch.nn as nn


def fixed_cross_entropy(source, target, num_items_in_batch=None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def fixed_causal_lm_loss(
    logits, labels, vocab_size: int, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)

    num_items_in_batch = torch.count_nonzero(labels != -100)
    loss = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss
