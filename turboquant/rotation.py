"""QuaRot-style data-oblivious rotation for activation quantization.

For a linear layer ``Y = X @ W``, any orthogonal ``R`` satisfies
``(X R)(Rᵀ W) = X W`` — so ``Rᵀ W`` can be folded into the weights **offline,
once**, and at runtime we quantize the rotated activation ``X R`` instead of
``X``. (The research doc's §3.2 claim that this needs per-layer un-rotation is
wrong; QuaRot/SpinQuant deploy exactly this.) In this fake-quant harness the
fold is simulated by ``unrotate(quant(rotate(x)))``, which is numerically
identical to quantizing ``X R`` against folded weights.

The rotation used here is seed-regenerated (zero storage, like QJL's G):
    R = P · S · H_blk
  - P: random channel permutation (spreads outlier channels across blocks)
  - S: random ±1 signs (de-correlates from any fixed pattern)
  - H_blk: normalized Walsh-Hadamard transform per ``blk`` channels, where
    ``blk`` is the largest power of two dividing the hidden dim (capped 256),
    so non-power-of-2 dims like 768 work.

Rotation Gaussianizes the per-element distribution: outlier energy is smeared
across the block, protecting the small-magnitude values that NVFP4's shared
block scale would otherwise crush. NOTE: this typically *raises* energy-weighted
NMSE while *lowering* downstream loss — NMSE is the wrong metric for it.
"""

from __future__ import annotations

import torch


def _wht(x: torch.Tensor) -> torch.Tensor:
    """Normalized in-place-style WHT over the last dim (power of 2)."""
    n = x.shape[-1]
    h = 1
    y = x.clone()
    while h < n:
        y = y.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2)
        h *= 2
    return y.reshape_as(x) / (n ** 0.5)


def _largest_pow2_block(dim: int, cap: int = 256) -> int:
    b = dim & (-dim)  # largest power of 2 dividing dim
    return min(b, cap)


def _perm_signs(dim: int, seed: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)  # CPU generator -> device-portable
    perm = torch.randperm(dim, generator=g).to(device)
    signs = (torch.randint(0, 2, (dim,), generator=g) * 2 - 1).float().to(device)
    return perm, signs


def rotate(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Apply R = P·S·H_blk to the last dim of ``x``."""
    dim = x.shape[-1]
    perm, signs = _perm_signs(dim, seed, x.device)
    y = x[..., perm] * signs
    blk = _largest_pow2_block(dim)
    return _wht(y.reshape(*x.shape[:-1], dim // blk, blk)).reshape_as(x)


def unrotate(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Apply Rᵀ (WHT is self-inverse; then undo signs and permutation)."""
    dim = x.shape[-1]
    perm, signs = _perm_signs(dim, seed, x.device)
    blk = _largest_pow2_block(dim)
    y = _wht(x.reshape(*x.shape[:-1], dim // blk, blk)).reshape_as(x) * signs
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(dim, device=x.device)
    return y[..., inv]
