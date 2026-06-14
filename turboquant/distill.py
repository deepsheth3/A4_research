"""Trained low-rank factors: distill the (free, already-deployed) correction.

The two-lever theorem says the additive low-rank channel is the ONLY place
injected information can help — the post-equalization residual is white, so
nothing analytic is left to exploit. SVD fills the factors to minimize
reconstruction MSE; but the deployment objective is the LOSS, not MSE. So we
freeze the 4-bit weight Wq and train ONLY the rank-r factors (L, R) by gradient
descent on KL(student ‖ FP16-teacher). This is Pareto-clean: the shipped model
is byte-identical (same Wq + same-shape factors) — only the factors are filled
better than SVD could.

For the draft-model use case, set the teacher to the *target* model you will
verify against: then the factors are trained to maximize acceptance directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankLinear(nn.Module):
    """Frozen NVFP4 weight Wq (buffer) + trainable rank-r factors L, R.

    forward: x @ Wqᵀ + (x @ Rᵀ) @ Lᵀ  (+ bias). Only L and R carry gradients;
    Wq and bias are frozen buffers (the deployed 4-bit path is unchanged).
    """

    def __init__(self, Wq: torch.Tensor, L: torch.Tensor, R: torch.Tensor,
                 bias: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("Wq", Wq)        # (out, in) dequantized, frozen
        self.L = nn.Parameter(L.contiguous())  # (out, r)
        self.R = nn.Parameter(R.contiguous())  # (r, in)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.Wq, self.bias)
        return y + F.linear(F.linear(x, self.R), self.L)   # (x Rᵀ) Lᵀ


def attach_lowrank(model: nn.Module, factors: dict[str, tuple],
                   dtype: torch.dtype = torch.float32) -> int:
    """Replace named nn.Linear modules with LowRankLinear (frozen Wq + trainable L,R).

    ``factors[name] = (Wq, L, R)``. Returns the number of layers swapped. After
    this, freeze the whole model and re-enable grad only on the L/R parameters
    (see ``trainable_factor_params``).
    """
    name_to_mod = dict(model.named_modules())
    n = 0
    for name, (Wq, L, R) in factors.items():
        mod = name_to_mod.get(name)
        if mod is None:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = name_to_mod[parent_name] if parent_name else model
        bias = getattr(mod, "bias", None)
        dev = mod.weight.device                       # place on the layer's device
        lr = LowRankLinear(Wq.to(device=dev, dtype=dtype), L.to(device=dev, dtype=dtype),
                           R.to(device=dev, dtype=dtype),
                           bias.detach().to(device=dev, dtype=dtype) if bias is not None else None)
        setattr(parent, child, lr)
        n += 1
    return n


def trainable_factor_params(model: nn.Module) -> list[nn.Parameter]:
    """Freeze everything; return only the LowRankLinear L/R params (grad-enabled)."""
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for m in model.modules():
        if isinstance(m, LowRankLinear):
            m.L.requires_grad_(True)
            m.R.requires_grad_(True)
            params += [m.L, m.R]
    return params


def kl_distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                    temperature: float = 1.0) -> torch.Tensor:
    """Temperature-scaled KL(teacher ‖ student) over the vocab (mean per token).

    Uses the standard distillation form: soft targets from the teacher, KL on
    log-softmax(student/T) vs softmax(teacher/T), scaled by T² so the gradient
    magnitude is temperature-independent.
    """
    t = temperature
    v = student_logits.size(-1)
    # flatten (B,T,V)->(B*T,V): kl_div(batchmean) divides by size(0); on a 3-D
    # tensor that's B only, inflating the loss ~T x and the effective lr with it.
    s = F.log_softmax(student_logits.float().reshape(-1, v) / t, dim=-1)
    p = F.softmax(teacher_logits.float().reshape(-1, v) / t, dim=-1)
    return F.kl_div(s, p, reduction="batchmean") * (t * t)


@torch.no_grad()
def _val_kl(student, teacher, val_batches, temperature, logits_fn):
    tot = 0.0
    for x in val_batches:
        tot += kl_distill_loss(logits_fn(student, x), logits_fn(teacher, x),
                               temperature).item()
    return tot / max(len(val_batches), 1)


def distill_factors(student: nn.Module, teacher: nn.Module,
                    batches: list[torch.Tensor], *, lr: float = 1e-3,
                    epochs: int = 1, temperature: float = 1.0,
                    logits_fn=None, log_every: int = 0, grad_clip: float = 1.0,
                    val_batches: list[torch.Tensor] | None = None,
                    eval_every: int = 0) -> list[float]:
    """Train the student's LowRankLinear factors to match teacher logits (KL).

    ``batches`` is a list of input_id tensors (B, T). ``logits_fn(model, x)``
    extracts logits (defaults to ``model(x).logits``). Teacher runs under no_grad.
    Only L/R params are updated.

    MONOTONE-SAFE: with ``val_batches`` + ``eval_every`` it snapshots the factors
    whenever held-out KL improves and restores the best set at the end — so a too-
    high lr (Adam wanders off a good warm-start) can never ship worse-than-start
    factors. ``grad_clip`` bounds the step. Returns the per-step train-loss history.
    """
    if logits_fn is None:
        def logits_fn(m, x):
            return m(x).logits

    params = trainable_factor_params(student)
    opt = torch.optim.Adam(params, lr=lr)
    teacher.eval()
    history: list[float] = []
    best_val, best_state = None, None
    if val_batches is not None:
        best_val = _val_kl(student, teacher, val_batches, temperature, logits_fn)
        best_state = [p.detach().clone() for p in params]
    step = 0
    for _ in range(epochs):
        for x in batches:
            with torch.no_grad():
                t_logits = logits_fn(teacher, x)
            s_logits = logits_fn(student, x)
            loss = kl_distill_loss(s_logits, t_logits, temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            history.append(loss.item())
            step += 1
            if log_every and step % log_every == 0:
                print(f"  distill step {step}: KL = {loss.item():.5f}", flush=True)
            if val_batches is not None and eval_every and step % eval_every == 0:
                v = _val_kl(student, teacher, val_batches, temperature, logits_fn)
                if v < best_val:
                    best_val = v
                    best_state = [p.detach().clone() for p in params]
    if best_state is not None:                       # restore the best factors
        with torch.no_grad():
            for p, b in zip(params, best_state):
                p.copy_(b)
    return history
