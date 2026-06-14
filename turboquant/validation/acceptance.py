"""Speculative-decoding acceptance metrics — REAL on fake-quant hardware.

The headline number for the draft-model framing does NOT need Blackwell:
acceptance is a *distributional* quantity on logits, not a wall-clock timing.

Standard spec-decoding acceptance: a draft proposes token x ~ q (draft dist);
the target accepts with prob min(1, p(x)/q(x)). Expected per-token acceptance
  α = E_{x~q}[min(1, p(x)/q(x))] = Σ_x min(p(x), q(x)) = 1 − TV(p, q).
Greedy agreement = fraction of positions where argmax(draft) == argmax(target).

Both are measured by teacher-forcing draft and target over the SAME real text
(one forward each) and aggregating per-position — exact, hardware-agnostic.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def acceptance_stats(draft_logits: torch.Tensor, target_logits: torch.Tensor):
    """Per-position acceptance over flattened logits (N, V).

    Returns (alpha, greedy): alpha = mean Σ min(p, q) = 1 − TV; greedy = mean
    argmax agreement.
    """
    q = F.softmax(draft_logits.float(), dim=-1)
    p = F.softmax(target_logits.float(), dim=-1)
    alpha = torch.minimum(p, q).sum(dim=-1).mean().item()
    greedy = (draft_logits.argmax(-1) == target_logits.argmax(-1)).float().mean().item()
    return alpha, greedy


def expected_accepted_length(alpha: float, gamma: int) -> float:
    """Expected verified tokens per step for γ drafts under the iid block model.

    Each of γ drafted tokens is accepted iid w.p. α; the run stops at the first
    reject, then the target contributes one correction token. E[length] =
    (1 − α^(γ+1)) / (1 − α). At α=1 every draft passes → γ+1.
    """
    if alpha >= 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)

def speedup(alpha: float, gamma: int, cost_ratio: float) -> float:
    """Spec-decode speedup vs plain target decoding.

    cost_ratio c = (one draft step) / (one target step). One spec step runs γ
    draft steps + 1 target verify and yields E[length] tokens:
        speedup = E[length] / (γ·c + 1).
    """
    return expected_accepted_length(alpha, gamma) / (gamma * cost_ratio + 1.0)


@torch.no_grad()
def collect_acceptance(draft_model, target_model, input_ids: torch.Tensor,
                       chunk: int = 1, logits_fn=None):
    """Teacher-force both models over input_ids (B, T); return (alpha, greedy).

    Runs one forward of each model and compares next-token distributions at every
    position. ``chunk`` lets the caller split long sequences upstream; this just
    aggregates whatever batch it's given. GPU-side (real models); the math above
    is unit-tested on CPU.
    """
    if logits_fn is None:
        def logits_fn(m, x):
            return m(x).logits
    d = logits_fn(draft_model, input_ids)[:, :-1, :].reshape(-1, draft_model.config.vocab_size)
    t = logits_fn(target_model, input_ids)[:, :-1, :].reshape(-1, target_model.config.vocab_size)
    return acceptance_stats(d, t)
