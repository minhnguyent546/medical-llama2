"""
This util aims to fix the bug in LLM Training about gradient accumulation.
Relevant links:
    - https://github.com/unslothai/unsloth/releases/tag/October-2024
    - https://github.com/huggingface/trl/issues/2175
    - https://huggingface.co/blog/gradient_accumulation
    - https://www.reddit.com/r/LocalLLaMA/comments/1g4ego7/llm_training_bug_fixes_gradient_accumulation_was/
    - https://github.com/huggingface/transformers/pull/34198

The codes below are derived from https://github.com/huggingface/transformers/pull/34191/files.

If use you a newer version of Transformers, please ignore this.
"""

import torch.nn as nn


def fixed_causal_lm_loss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch=None,
    ignore_index: int = -100,
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

    if num_items_in_batch is not None:
        loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index, reduction='sum')
        loss = loss / num_items_in_batch
    else:
        loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index, reduction='mean')

    return loss
