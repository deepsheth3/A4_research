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

import math

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


def _largest_pow2_div(n: int, cap: int = 8192) -> int:
    """Largest power of two that divides ``n`` (capped). Hidden=4096 -> 4096;
    intermediate=14336 -> 2048. Used to size the Hadamard tile."""
    p = 1
    while (n % (p * 2) == 0) and (p * 2 <= cap):
        p *= 2
    return p


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Normalized fast Walsh-Hadamard transform over the last dim (power of 2).

    Returns ``H x / sqrt(n)`` with ``H`` the Walsh-Hadamard matrix. The transform
    is orthonormal and symmetric, hence an involution: applying it twice is the
    identity, so the same call inverts it."""
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"_fwht needs a power-of-2 last dim, got {n}")
    lead = x.shape[:-1]
    y = x.reshape(-1, n).clone()
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    return (y / math.sqrt(n)).reshape(*lead, n)


def nvfp4_quantize_ghad(
    x: torch.Tensor,
    block: int = 16,
    optclip: bool = True,
    had_size: int | None = None,
) -> torch.Tensor:
    """QuaRot-style rotated NVFP4: rotate by a (tiled) Walsh-Hadamard, quantize in
    the rotated frame, rotate back. For power-of-2 last dims this is a single
    global Hadamard; otherwise it is applied over tiles of the largest power-of-2
    divisor (a documented approximation of QuaRot's online Hadamard). Used as a
    rotation baseline. The rotate-back makes the weight matrix unchanged, so this
    is an activation-side fake-quant."""
    n = x.shape[-1]
    hs = had_size or _largest_pow2_div(n)
    if hs % block != 0:
        return nvfp4_quantize_zp(x, block=block, optclip=optclip)
    xt = x.reshape(*x.shape[:-1], n // hs, hs)
    xr = _fwht(xt)                                       # rotate
    q = nvfp4_quantize_zp(xr, block=block, optclip=optclip)
    xq = _fwht(q)                                        # involution -> rotate back
    return xq.reshape(*x.shape[:-1], n)


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
    imp: torch.Tensor | None = None,
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

    ``imp`` (optional, per-channel, shape broadcastable to the last dim) reweights
    the per-block candidate selection by *output-domain* importance instead of
    plain element MSE: each candidate's error is ``Σ imp_i·(q_i-x_i)²`` rather
    than ``Σ (q_i-x_i)²``. With imp_i = diag(W Wᵀ)_i (how much channel i drives
    the layer output, in equalized space) this is the diagonal of the true
    output error ‖(q-x)W‖² — a strictly better proxy for PPL than input MSE, at
    zero deploy cost (imp is precomputed offline from W). Default None = uniform.
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
    sq = (xb[..., None, None, :] - q) ** 2               # (..., nb, 2, G, block)
    if imp is not None:
        sq = sq * imp.reshape(n // block, 1, 1, block)   # output-domain weighting
    e = sq.sum(dim=-1)                                    # (..., nb, 2, G)

    c = 2 * gammas.numel()
    q = q.reshape(*xb.shape[:-1], c, block)             # (..., nb, C, block)
    idx = e.reshape(*xb.shape[:-1], c).argmin(dim=-1)   # (..., nb)
    best_q = q.gather(-2, idx[..., None, None].expand(*idx.shape, 1, block)).squeeze(-2)
    return best_q.reshape(*lead, n)


def _block_hwht(x: torch.Tensor, block: int) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform *within* each ``block`` (last dim).

    Block-diagonal: it never mixes across MX4 blocks, so it's distinct from the
    global rotation in ``rotation.py``. Self-inverse (H²=I), multiply-free.
    """
    from turboquant.rotation import _wht
    *lead, n = x.shape
    xb = x.reshape(*lead, n // block, block)
    return _wht(xb).reshape(*lead, n)


def nvfp4_quantize_hwht(
    x: torch.Tensor,
    block: int = 16,
    quantize_scale: bool = True,
    optclip: bool = False,
    hwht: str = "always",
) -> torch.Tensor:
    """NVFP4 with a per-block (16×16) Hadamard applied *before* E2M1 rounding.

    The within-block Hadamard spreads a block's outlier energy across its 16
    coefficients, lowering the block max so the shared MX4 scale resolves the
    other elements more finely — then it's inverted (H²=I, exact). Distinct from
    the global rotation we ruled out: it's a *local basis*, never mixing blocks.

    Deploy: weights fold ``H`` offline (store ``H·W`` blocks, zero cost);
    activations get a fused block-diagonal Hadamard prologue; ``H²=I`` keeps the
    FP4×FP4 GEMM exact. ``always`` = rotate every block (deployable, no flag).
    ``bestof`` = per-block min-MSE of {rotate, no-rotate} — the *upper bound*;
    deploying it needs a per-block-position pattern decided offline (a fixed
    choice per position, not a per-token flag), so it's a measurement aid here.

    NOTE: ``imp`` output-domain weighting isn't carried through — importance is a
    per-channel input-domain quantity that doesn't map trivially under H, and the
    gain it gave was ~0.01. Uniform-MSE block selection in the rotated domain.
    """
    h = _block_hwht(x, block)
    qh = nvfp4_quantize_zp(h, block=block, quantize_scale=quantize_scale, optclip=optclip)
    x_rot = _block_hwht(qh, block)                     # inverse = re-apply (H²=I)
    if hwht == "always":
        return x_rot
    if hwht != "bestof":
        raise ValueError(f"hwht must be 'always' or 'bestof', got {hwht!r}")
    q_plain = nvfp4_quantize_zp(x, block=block, quantize_scale=quantize_scale, optclip=optclip)
    *lead, n = x.shape
    nb = n // block
    er = ((x - x_rot).reshape(*lead, nb, block) ** 2).sum(-1, keepdim=True)
    eu = ((x - q_plain).reshape(*lead, nb, block) ** 2).sum(-1, keepdim=True)
    out = torch.where(er <= eu, x_rot.reshape(*lead, nb, block),
                      q_plain.reshape(*lead, nb, block))
    return out.reshape(*lead, n)


def nvfp4_quantize_hmask(
    x: torch.Tensor,
    mask: torch.Tensor,
    block: int = 16,
    quantize_scale: bool = True,
    optclip: bool = False,
) -> torch.Tensor:
    """Selective per-block-position Hadamard — the *deployable* best-of.

    ``mask`` is a fixed boolean pattern (shape ``(n//block,)``) decided offline
    from calibration win-rates: block positions where the within-block Hadamard
    usually beats plain quant are rotated, the rest are quantized normally. No
    per-token decision, no flag, one deterministic path — at deploy the masked
    positions fold ``HᵀW`` into the weights, unmasked positions keep ``W``.

    This avoids always-rotate's smooth-block penalty (those positions stay
    normal) while capturing the outlier-position win — a static approximation of
    the per-block best-of oracle, which itself isn't Pareto-safe (it needs a
    per-token branch / two weight paths).
    """
    from turboquant.rotation import _wht
    *lead, n = x.shape
    nb = n // block
    xb = x.reshape(*lead, nb, block)
    m = mask.view(*([1] * len(lead)), nb, 1).to(torch.bool)
    xin = torch.where(m, _wht(xb), xb)                       # rotate masked positions
    q = nvfp4_quantize_zp(xin.reshape(*lead, n), block=block,
                          quantize_scale=quantize_scale, optclip=optclip)
    qb = q.reshape(*lead, nb, block)
    return torch.where(m, _wht(qb), qb).reshape(*lead, n)    # inverse on masked (H²=I)


def nvfp4_quantize_perm(
    x: torch.Tensor,
    perm: torch.Tensor,
    inv: torch.Tensor,
    block: int = 16,
    quantize_scale: bool = True,
    optclip: bool = False,
) -> torch.Tensor:
    """NVFP4 with a fixed offline channel permutation before block grouping.

    ``perm`` reorders the last dim so each MX4 block groups channels of similar
    magnitude (decided offline from calibration amax — e.g. magnitude-sorted),
    shrinking each block's dynamic range → tighter shared scale. ``inv`` maps
    back. Distinct from rotation: it *regroups* channels (preserving MX4 locality
    and adding zero arithmetic), it doesn't mix them. Pareto-clean — ``perm``
    folds into the producing layer's weight columns and ``inv`` into this layer's
    rows, both offline; the main FP4 GEMM is unchanged.
    """
    q = nvfp4_quantize_zp(x[..., perm], block=block,
                          quantize_scale=quantize_scale, optclip=optclip)
    return q[..., inv]


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
