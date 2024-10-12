import torch
import torch.nn.functional as Func
from torch import Tensor


def top_k_logits(logits: Tensor, top_k: int = 0) -> Tensor:
    if top_k <= 0:
        # no truncation
        return logits
    assert logits.dim() == 2
    top_k = min(top_k, logits.size(-1))
    topk_values = torch.topk(logits, k=top_k, dim=-1).values
    logits[logits < topk_values[:, [-1]]] = float('-inf')
    return logits

def top_p_logits(logits: Tensor, top_p: float = 1.0) -> Tensor:
    """Nucleus sampling (Nucleus decoding)"""
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cum_prob = torch.cumsum(Func.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cum_prob < top_p

    # shift one token to the right so that we have cum_prob >= top_p
    mask[:, 1:] = mask[:, :-1].clone()
    mask[:, 0] = True
    indices_to_keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        dim=-1,
        index=sorted_indices,
        src=mask,
    )
    logits[~indices_to_keep] = float('-inf')
    return logits
