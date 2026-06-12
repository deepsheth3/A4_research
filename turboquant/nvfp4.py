"""NVFP4 (E2M1 + MX4 microscaling) fake-quantization.

This is a *numerical simulation* of NVFP4, not a hardware kernel. FP4 tensor
cores exist only on Blackwell; on a Hopper H100 (and on a Mac) we round to the
E2M1 value grid in fp32 and matmul in fp16. That validates NVFP4's *accuracy*,
which is all the research's core claim needs — throughput needs real B200.

NVFP4 format:
  - E2M1: 1 sign, 2 exponent, 1 mantissa bit -> 16 representable values:
      {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}   (grid max = 6)
  - MX4 microscaling: one local scale per ``block`` elements (default 16), so a
    group's largest magnitude maps to the grid max. The scale is itself stored
    in fp8-e4m3 in real NVFP4 (the ~12.5% overhead); we optionally simulate that
    rounding so the measured accuracy matches hardware.
"""

from __future__ import annotations

import torch

# The 8 non-negative E2M1 magnitudes; the signed grid is built from these.
_E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
GRID_MAX = 6.0

# Full signed 16-value grid, e.g. [-6, -4, ..., -0.5, 0, 0.5, ..., 4, 6].
NVFP4_GRID = torch.tensor(
    sorted({v for m in _E2M1_POS for v in (m, -m)}),
    dtype=torch.float32,
)


def _round_to_grid(x: torch.Tensor) -> torch.Tensor:
    """Round each element of ``x`` to the nearest value in ``NVFP4_GRID``."""
    grid = NVFP4_GRID.to(device=x.device, dtype=x.dtype)
    idx = torch.bucketize(x, (grid[:-1] + grid[1:]) / 2)
    return grid[idx]


def round_e4m3(t: torch.Tensor) -> torch.Tensor:
    """Round to the fp8-e4m3fn grid in pure torch (any device, incl. MPS).

    E4M3fn: 3 mantissa bits, exponent in [-6, 8], max 448, subnormal step 2^-9.
    Matches ``t.to(torch.float8_e4m3fn)`` without needing fp8 device support
    (MPS lacks it, and a CPU round-trip per call stalls the MPS pipeline).
    """
    sign, a = t.sign(), t.abs().clamp(max=448.0)
    e = torch.floor(torch.log2(a.clamp_min(1e-45))).clamp(min=-6.0)
    step = torch.exp2(e - 3)  # 3-bit mantissa => 8 steps per octave
    return sign * (torch.round(a / step) * step).clamp(max=448.0)


def _quantize_scale_fp8(scale: torch.Tensor) -> torch.Tensor:
    """Round an MX4 block scale to fp8-e4m3, the real NVFP4 scale storage."""
    return round_e4m3(scale)


def _block_scale(x: torch.Tensor, block: int, quantize_scale: bool) -> torch.Tensor:
    """Per-block scale that maps each block's max magnitude to GRID_MAX.

    Returns a tensor broadcastable to ``x`` (last dim grouped into blocks).
    """
    *lead, n = x.shape
    if n % block != 0:
        raise ValueError(f"last dim {n} not divisible by block size {block}")
    xb = x.reshape(*lead, n // block, block)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    scale = (amax / GRID_MAX).clamp_min(1e-12)
    if quantize_scale:
        scale = _quantize_scale_fp8(scale)
    return scale.expand_as(xb).reshape(*lead, n)


# Clip-ratio candidates for optimal-clip scaling. gamma=1.0 is classic absmax
# (never clips); smaller gammas sacrifice the block max for finer resolution on
# the other 15 elements (ACIQ-style, applied per MX4 block, online).
_OPTCLIP_GAMMAS = (0.62, 0.75, 0.88, 1.0)


def nvfp4_quantize(
    x: torch.Tensor,
    block: int = 16,
    quantize_scale: bool = True,
    optclip: bool = False,
) -> torch.Tensor:
    """Round-trip NVFP4 fake-quant: returns the reconstructed fp tensor.

    Args:
        x: input activations, last dim divisible by ``block``.
        block: MX4 microscaling group size.
        quantize_scale: round the per-block scale to fp8-e4m3 (real NVFP4).
        optclip: per-block MSE-optimal scale search over ``_OPTCLIP_GAMMAS``
            instead of plain absmax. Output is still pure E2M1 + fp8 scale
            (hardware-native); costs len(gammas) quantization passes.

    Returns:
        ``x_hat`` of the same shape/dtype, each element snapped to the E2M1
        grid after per-block scaling.
    """
    if not optclip:
        scale = _block_scale(x, block, quantize_scale)
        return _round_to_grid(x / scale) * scale

    *lead, n = x.shape
    if n % block != 0:
        raise ValueError(f"last dim {n} not divisible by block size {block}")
    xb = x.reshape(-1, block)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    best_q = best_e = None
    for g in _OPTCLIP_GAMMAS:
        scale = (amax * g / GRID_MAX).clamp_min(1e-12)
        if quantize_scale:
            scale = _quantize_scale_fp8(scale)
        q = _round_to_grid(xb / scale) * scale
        e = ((xb - q) ** 2).sum(dim=-1, keepdim=True)
        if best_q is None:
            best_q, best_e = q, e
        else:
            better = e < best_e
            best_q = torch.where(better, q, best_q)
            best_e = torch.where(better, e, best_e)
    return best_q.reshape(*lead, n)


def nvfp4_quantize_zp(
    x: torch.Tensor,
    block: int = 16,
    quantize_scale: bool = True,
    optclip: bool = False,
) -> torch.Tensor:
    """NVFP4 with a per-block *optional* zero-point (best-of-N fake-quant).

    E2M1 is a float grid (dense near zero), so a blanket midpoint shift hurts
    blocks whose mass sits near zero while rescuing one-sided blocks. Each
    block therefore picks the MSE-better of {z=0 (symmetric), z=midpoint};
    measured: ~25% lower MSE than symmetric even on Gaussian data, 35x on
    one-sided. With ``optclip`` the search is joint over shift x clip-ratio
    (2 x 4 = 8 candidates), a strict superset of both single searches.
    z is stored fp8 (z=0 encodes the symmetric choice — no flag bit). Deploy
    cost: the matmul cross-term z · (block row-sums of W) is a precomputed
    (d/block, m) matrix times a tiny per-token vector — fused epilogue, main
    path stays pure E2M1 + fp8 scale.
    """
    *lead, n = x.shape
    if n % block != 0:
        raise ValueError(f"last dim {n} not divisible by block size {block}")
    xb = x.reshape(*lead, n // block, block)            # (..., nb, block)

    # All 2 x len(gammas) candidates are evaluated in one batched pass (a
    # candidate axis instead of a Python loop): same outputs and same per-block
    # MSE argmin as the sequential search, but ~8x fewer GPU kernel launches —
    # which dominates autoregressive (1-token) generation latency.
    gammas = torch.tensor(_OPTCLIP_GAMMAS if optclip else (1.0,),
                          device=x.device, dtype=x.dtype)   # (G,)
    zmid = round_e4m3((xb.amax(dim=-1, keepdim=True)
                       + xb.amin(dim=-1, keepdim=True)) / 2)  # (..., nb, 1)
    z_opts = torch.stack([torch.zeros_like(zmid), zmid], dim=-2)  # (..., nb, 2, 1)
    centered = xb.unsqueeze(-2) - z_opts                # (..., nb, 2, block)
    amax = centered.abs().amax(dim=-1, keepdim=True)    # (..., nb, 2, 1)

    centered = centered.unsqueeze(-2)                   # (..., nb, 2, 1, block)
    z_b = z_opts.unsqueeze(-2)                          # (..., nb, 2, 1, 1)
    scale = (amax.unsqueeze(-2) * gammas.view(-1, 1) / GRID_MAX).clamp_min(1e-12)
    if quantize_scale:                                  # (..., nb, 2, G, 1)
        scale = _quantize_scale_fp8(scale)
    q = _round_to_grid(centered / scale) * scale + z_b  # (..., nb, 2, G, block)
    e = ((xb[..., None, None, :] - q) ** 2).sum(dim=-1)  # (..., nb, 2, G)

    c = 2 * gammas.numel()
    q = q.reshape(*xb.shape[:-1], c, block)             # (..., nb, C, block)
    idx = e.reshape(*xb.shape[:-1], c).argmin(dim=-1)   # (..., nb)
    best_q = q.gather(-2, idx[..., None, None].expand(*idx.shape, 1, block)).squeeze(-2)
    return best_q.reshape(*lead, n)


def waware_comp(A: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Precompute the W-aware rounding feedback matrix (offline, from W only).

    ``A`` is the effective weight (d, m) with y = x @ A. The output-domain
    error metric is G = A Aᵀ; we take its (d/block) diagonal 16x16 blocks and
    derive the GPTQ-style feedback coefficients from the Cholesky factor of
    each block's inverse: U[k, j] = Hc[k, j] / Hc[k, k] for j > k, where
    Hc is the upper Cholesky factor of H⁻¹. Returns (d/block, block, block),
    strictly upper-triangular.
    """
    d = A.shape[0]
    if d % block != 0:
        raise ValueError(f"dim {d} not divisible by block size {block}")
    nb = d // block
    G = A @ A.T
    idx = torch.arange(nb, device=A.device)
    Gblk = G.reshape(nb, block, nb, block)[idx, :, idx, :]
    eps = 0.01 * Gblk.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    eye = torch.eye(block, device=A.device, dtype=A.dtype)
    H = Gblk + eps.view(-1, 1, 1) * eye
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    L = torch.linalg.cholesky(Hinv)  # lower factor; Hc = Lᵀ
    U = (L / L.diagonal(dim1=-2, dim2=-1).unsqueeze(-2)).transpose(-1, -2)
    return torch.triu(U, diagonal=1)


def nvfp4_quantize_waware(
    x: torch.Tensor,
    comp: torch.Tensor,
    block: int = 16,
    quantize_scale: bool = True,
) -> torch.Tensor:
    """W-aware (output-domain) NVFP4 rounding with per-block error feedback.

    Instead of rounding each element to the nearest grid point in the *input*
    domain, elements are rounded sequentially within each MX4 block and each
    rounding error is fed forward into the not-yet-rounded elements through
    ``comp`` (from :func:`waware_comp`), steering quantization noise into W's
    weak directions. Output format is unchanged (E2M1 + fp8 block scale);
    deploy cost is 16 fused multiply-add steps in the quantize epilogue and
    zero side bits.
    """
    *lead, n = x.shape
    if n % block != 0:
        raise ValueError(f"last dim {n} not divisible by block size {block}")
    nb = n // block
    xb = x.reshape(*lead, nb, block)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    scale = (amax / GRID_MAX).clamp_min(1e-12)
    if quantize_scale:
        scale = _quantize_scale_fp8(scale)
    s = scale.squeeze(-1)  # (..., nb)
    xw = xb.clone()
    q = torch.empty_like(xb)
    for k in range(block):
        qk = _round_to_grid(xw[..., k] / s) * s
        q[..., k] = qk
        if k + 1 < block:
            e = xw[..., k] - qk
            xw[..., k + 1:] = xw[..., k + 1:] - e.unsqueeze(-1) * comp[..., k, k + 1:]
    return q.reshape(*lead, n)
