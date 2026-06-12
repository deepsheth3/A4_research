"""GPTQ-style weight quantization to NVFP4 (Hessian-aware error feedback).

Naive per-element rounding of weights to FP4 cost +0.58 PPL at 8B (raw W4A4
6.629 vs A4 6.050) because the ~10% per-element error compounds across layers.
GPTQ quantizes weight columns left-to-right and feeds each column's rounding
error into the not-yet-quantized columns through the inverse activation Hessian
H^-1 (H = XᵀX from calibration), steering error into directions the data rarely
excites. Output stays pure NVFP4 (E2M1 + per-16 fp8 block scale) — a one-time
OFFLINE pass, zero deploy cost, main path unchanged.

Reference: Frantar et al., "GPTQ" (2022). Blocked/lazy-batch variant.
"""

from __future__ import annotations

import torch

from turboquant.nvfp4 import GRID_MAX, _round_to_grid, round_e4m3


def _block_absmax_scale(W: torch.Tensor, block: int) -> torch.Tensor:
    """Per-(row, block) symmetric NVFP4 scale (fp8-e4m3) from the weight.

    Weights are zero-centered, so symmetric (no zero-point) — matching the
    hardware NVFP4 weight format. Returns (out, n_blocks)."""
    out, inn = W.shape
    nb = inn // block
    amax = W.reshape(out, nb, block).abs().amax(dim=-1)
    return round_e4m3((amax / GRID_MAX).clamp_min(1e-12))


def _quant_fp8_rows(A: torch.Tensor) -> torch.Tensor:
    """Per-row fp8-e4m3 quantization (for storing low-rank factors cheaply)."""
    s = A.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
    return round_e4m3(A / s) * s


@torch.no_grad()
def gptq_lowrank_factors(W: torch.Tensor, H: torch.Tensor, block: int = 16,
                         max_rank: int | None = None, percdamp: float = 0.01):
    """GPTQ-quantize W, then return the (untruncated) low-rank correction factors.

    Returns (Wq, Lfac, Rfac, S): the NVFP4 weight, factors Lfac (out, Q) and
    Rfac (Q, in) whose product is the rank-Q residual correction, and the
    singular values S (for cross-layer rank allocation). Truncating to the first
    r columns/rows gives the rank-r correction. Q = ``max_rank``.
    """
    W = W.float()
    H = 0.5 * (H.float() + H.float().t())
    out, inn = W.shape
    Wq = gptq_quantize_weight(W, H, block)
    E = W - Wq
    damp = percdamp * torch.diag(H).mean()
    L = torch.linalg.cholesky(H + damp * torch.eye(inn, device=W.device, dtype=W.dtype))
    B = E @ L
    Q = min(max_rank or min(out, inn), out, inn)
    U, S, V = torch.svd_lowrank(B, q=min(Q + 8, out, inn), niter=4)
    U, S, V = U[:, :Q], S[:Q], V[:, :Q]
    Rt = torch.linalg.solve_triangular(L.t(), V, upper=True)   # (in, Q)
    return Wq, U * S, Rt.t(), S                                 # Lfac, Rfac, S


@torch.no_grad()
def apply_lowrank(Wq: torch.Tensor, Lfac: torch.Tensor, Rfac: torch.Tensor, r: int,
                  block: int = 16, fp8_factors: bool = False,
                  fp4_factors: bool = False) -> torch.Tensor:
    """Q(W) + rank-r correction from precomputed factors (optionally quantized)."""
    from turboquant.nvfp4 import nvfp4_quantize
    if r <= 0:
        return Wq
    Lf, Rf = Lfac[:, :r].contiguous(), Rfac[:r, :].contiguous()
    if fp4_factors:
        Lf, Rf = nvfp4_quantize(Lf, block=block), nvfp4_quantize(Rf, block=block)
    elif fp8_factors:
        Lf, Rf = _quant_fp8_rows(Lf), _quant_fp8_rows(Rf)
    return Wq + Lf @ Rf


@torch.no_grad()
def lowrank_corrected_weight(W: torch.Tensor, H: torch.Tensor, block: int = 16,
                            rank_div: int = 16, use_gptq: bool = True,
                            percdamp: float = 0.01, fp8_factors: bool = False,
                            fp4_factors: bool = False) -> torch.Tensor:
    """Q(W) + an additive low-rank correction of the residual (LQER-style).

    Unlike scaling/redistribution (GPTQ/AWQ — redundant under microscaling),
    this *adds information back*: the quantization residual E = W - Q(W) gets an
    activation-aware rank-r approximation. Optimal in the output metric H: with
    H = C Cᵀ, the rank-r min of ‖(E-M)C‖_F is M = (top-r SVD of E·C)·C^-1. The
    correction rides a rank-r side matmul at deploy (r = in/rank_div ≈ our SVD's
    ~12% epilogue) — the weight-analog of the activation SVD side-channel that
    worked. Returns the dequantized effective weight Q(W) + M.
    """
    from turboquant.nvfp4 import nvfp4_quantize
    W = W.float()
    H = 0.5 * (H.float() + H.float().t())
    out, inn = W.shape
    Wq = gptq_quantize_weight(W, H, block) if use_gptq else nvfp4_quantize(W, block=block)
    E = W - Wq

    damp = percdamp * torch.diag(H).mean()
    L = torch.linalg.cholesky(H + damp * torch.eye(inn, device=W.device, dtype=W.dtype))
    B = E @ L                                   # error in the output (H) metric
    r = max(block, (inn // rank_div) // block * block)  # multiple of block (for fp4 factors)
    U, S, V = torch.svd_lowrank(B, q=min(r + 8, out, inn), niter=4)
    U, S, V = U[:, :r], S[:r], V[:, :r]         # B ≈ U diag(S) Vᵀ
    Rt = torch.linalg.solve_triangular(L.t(), V, upper=True)  # (in, r) = L^-ᵀ V
    Lf, Rf = U * S, Rt.t()                       # factors (out,r) and (r,in)
    if fp4_factors:                              # 4-bit factors: keeps "4-bit everywhere"
        Lf, Rf = nvfp4_quantize(Lf, block=block), nvfp4_quantize(Rf, block=block)
    elif fp8_factors:                            # cheap storage — keep memory axis honest
        Lf, Rf = _quant_fp8_rows(Lf), _quant_fp8_rows(Rf)
    return Wq + Lf @ Rf                          # Q(W) + correction


_AWQ_BETAS = (0.0, 0.25, 0.5, 0.75)


@torch.no_grad()
def awq_gptq_quantize_weight(W: torch.Tensor, H: torch.Tensor, block: int = 16) -> torch.Tensor:
    """AWQ-protected GPTQ: scale salient weight columns up before quantizing.

    Per-input-channel scale s_i = diag(H)_i^β protects channels the activations
    excite most (diag(H) = E[x_i²]). β is picked per layer (cheaply, via naive
    quant) to minimize the output-domain weight error trace(E H Eᵀ); GPTQ then
    runs once with the chosen scale. The inverse scale folds back: the deployed
    activation carries 1/s (composes with equalization at zero cost), so the
    stored weight Q(W·diag(s)) stays on the NVFP4 grid. β=0 (no scaling) is in
    the candidate set, so AWQ can never be selected worse than plain GPTQ.

    Returns the dequantized effective weight Q(W·diag(s))·diag(1/s).
    """
    from turboquant.nvfp4 import nvfp4_quantize
    W = W.float()
    H = H.float()
    d = torch.diag(H).clamp_min(1e-8)

    best_b, best_err = 0.0, None
    for b in _AWQ_BETAS:                    # cheap β search via fast naive quant
        s = (d ** b)
        s = s / s.mean()
        Weff = nvfp4_quantize(W * s[None, :], block=block) / s[None, :]
        E = W - Weff
        err = ((E @ H) * E).sum()           # trace(E H Eᵀ): output-domain error
        if best_err is None or err < best_err:
            best_b, best_err = b, err

    s = (d ** best_b)
    s = s / s.mean()
    Qs = gptq_quantize_weight(W * s[None, :], H / (s[:, None] * s[None, :]), block)
    return Qs / s[None, :]


@torch.no_grad()
def gptq_quantize_weight(W: torch.Tensor, H: torch.Tensor, block: int = 16,
                         blocksize: int = 128, percdamp: float = 0.01) -> torch.Tensor:
    """Return NVFP4-quantized ``W`` (out, in) using activation Hessian ``H`` (in, in).

    Static per-(row, block) absmax scale from the original weight; columns are
    quantized left-to-right with GPTQ error feedback through ``H^-1``. Output is
    E2M1 x fp8 block scale, identical format to naive NVFP4 — only the rounding
    is smarter.
    """
    W = W.clone().float()
    out, cols = W.shape
    H = H.clone().float()
    H = 0.5 * (H + H.t())              # symmetrize (numerical / accumulation drift)

    dead = torch.diag(H) == 0          # unexcited input channels
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    damp = percdamp * torch.diag(H).mean()
    diag = torch.arange(cols, device=W.device)
    H[diag, diag] += damp

    # H^-1 upper-Cholesky factor: row j gives the feedback weights for column j.
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)

    scale = _block_absmax_scale(W, block)              # (out, n_blocks), static
    Q = torch.zeros_like(W)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = i1 + j
            s = scale[:, col // block]                  # (out,)
            w = W1[:, j]
            q = _round_to_grid(w / s) * s               # symmetric NVFP4 round
            Q1[:, j] = q
            err = (w - q) / Hinv1[j, j]
            W1[:, j:] -= err.unsqueeze(1) * Hinv1[j, j:].unsqueeze(0)
            Err1[:, j] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]            # propagate to later blocks
    return Q
