"""Ceiling test for Output-Balanced Rounding (OBR) / computation-aware rounding.

The core question: nearest-rounding minimizes ||x-q||² (input error), but the layer
only cares about ||(x-q)W||² = e'Ge (output error), G = WᵀW. Does choosing rounding
in the G-metric beat greedy-nearest — and is the opportunity WITHIN a 16-block (OBR
can reach it) or CROSS-block (it can't)?

Three roundings, identical grid/scale machinery (GPTQ Babai = nearest-plane CVP):
  nearest   : G = I        (greedy per-element)
  block-G   : block-diag G (OBR's ceiling — best any within-16-block scheme can do)
  full-G    : full G        (absolute ceiling — cross-block coupling too)

Run: python -m turboquant.validation.rounding_ceiling
CAVEAT: synthetic. The within-block off-diagonal energy of G is the exact thing that
was ~0 on real activations (why `waware` was weak) — confirm any win on real data.
"""
from __future__ import annotations

import torch

from turboquant.gptq import gptq_quantize_weight


def _block_diag(G: torch.Tensor, block: int) -> torch.Tensor:
    d = G.shape[0]
    M = torch.zeros_like(G)
    for k in range(0, d, block):
        M[k:k + block, k:k + block] = G[k:k + block, k:k + block]
    return M


def _offblock_ratio(G: torch.Tensor, block: int) -> float:
    """Fraction of each 16-block's G energy that is OFF-diagonal (the coupling OBR
    needs). ~0 => channels in-block are output-uncorrelated => OBR can't cancel."""
    d = G.shape[0]
    on = off = 0.0
    for k in range(0, d, block):
        b = G[k:k + block, k:k + block]
        diag = torch.diag(b)
        on += (diag ** 2).sum().item()
        off += (b ** 2).sum().item() - (diag ** 2).sum().item()
    return off / (on + off + 1e-12)


@torch.no_grad()
def ceiling(d=512, n=2048, n_outlier=8, sig_rank=32, seed=0, within_block_corr=0.0):
    torch.manual_seed(seed)
    # W with decaying spectrum (realistic output metric)
    U, _ = torch.linalg.qr(torch.randn(d, d))
    sv = torch.logspace(0, -1.5, d)
    W = (U * sv) @ torch.linalg.qr(torch.randn(d, d))[0].T   # (out=d, in=d)

    # realistic activations: low-rank signal + outlier channels + noise
    Bm = torch.randn(n, sig_rank) @ torch.randn(sig_rank, d)
    x = Bm / Bm.std() + 0.3 * torch.randn(n, d)
    x[:, torch.randperm(d)[:n_outlier]] *= 12.0

    # optionally inject WITHIN-BLOCK channel correlation (to probe sensitivity:
    # real data had ~none, which is why waware failed; sweep this to see what OBR
    # would need). Mixes each 16-block's channels by a small random rotation.
    if within_block_corr > 0:
        xb = x.reshape(n, d // 16, 16)
        R = torch.randn(d // 16, 16, 16) * within_block_corr
        R = torch.matrix_exp(R - R.transpose(-1, -2))        # orthogonal mix
        x = torch.einsum('nbi,bij->nbj', xb, R).reshape(n, d)

    # equalize (mirror the real stack — eq is what washed out other ideas)
    s = x.abs().amax(0).clamp_min(1e-5) ** 0.5
    xe = x / s
    We = W * s[None, :]                                       # xe @ Weᵀ == x @ Wᵀ
    G = We.T @ We                                             # (d,d) output metric

    oe = lambda q: ((xe - q) @ We.T).pow(2).sum().item()
    eye = torch.eye(d)
    q_near = gptq_quantize_weight(xe, eye, block=16)          # G=I  -> nearest
    q_blk = gptq_quantize_weight(xe, _block_diag(G, 16), block=16)
    q_full = gptq_quantize_weight(xe, G, block=16)
    near, blk, full = oe(q_near), oe(q_blk), oe(q_full)

    # what our ACTUAL stack does: nearest base + additive SVD side-channel (k=d/16,
    # fp8 coeffs) — the DUAL of rounding (add error back vs shape it away). Both null
    # error in high-G directions. Does full-G rounding add anything OVER the SVD?
    def fp8(a):
        sc = a.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
        from turboquant.nvfp4 import round_e4m3
        return round_e4m3(a / sc) * sc
    k = d // 16
    _, _, Vg = torch.svd_lowrank(We, q=k + 8, niter=4)        # top-k input dirs (=top eigvecs of G)
    Vg = Vg[:, :k]
    svd = lambda q: oe(q + (fp8((xe - q) @ Vg) @ Vg.T))       # add SVD correction, then score
    near_svd, full_svd = svd(q_near), svd(q_full)

    print(f"--- within_block_corr={within_block_corr}, seed={seed} ---")
    print(f"within-block off-diagonal G energy: {100*_offblock_ratio(G,16):.1f}% "
          f"(near 0 => no within-block coupling => OBR can't help)")
    print(f"output error (lower=better, % = reduction vs nearest):")
    print(f"  nearest (greedy)        : {near:11.1f}")
    print(f"  block-G rounding (OBR)   : {blk:11.1f}   ({100*(near-blk)/near:+.1f}%)")
    print(f"  full-G rounding (ceiling): {full:11.1f}   ({100*(near-full)/near:+.1f}%)")
    print(f"  nearest + SVD (our stack): {near_svd:11.1f}   ({100*(near-near_svd)/near:+.1f}%)")
    print(f"  full-G rounding + SVD    : {full_svd:11.1f}   ({100*(near-full_svd)/near:+.1f}%)")
    gain = 100 * (near_svd - full_svd) / near_svd
    print(f"  => rounding adds {gain:+.1f}% OVER the SVD side-channel we already ship\n")


if __name__ == "__main__":
    for wbc in (0.0, 0.5, 1.0):     # none / mild / strong within-block correlation
        ceiling(within_block_corr=wbc)
