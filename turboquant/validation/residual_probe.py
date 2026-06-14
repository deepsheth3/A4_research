"""Local probe: would a STRUCTURED residual basis (residual-SVD) beat random QJL
on the leftover residual after the W-aware SVD correction? (A4, side-channel swap.)

Cheap synthetic test (CPU, no model) to decide whether a box run is warranted.
The whole question is whether the leftover residual's spectrum is concentrated
(low-rank -> residual-SVD wins) or white (-> QJL's many-1-bit directions win) at
MATCHED bit budget. Run: python -m turboquant.validation.residual_probe
"""
from __future__ import annotations

import torch

from turboquant._omnistack import RademacherQJL
from turboquant.nvfp4 import nvfp4_quantize_zp, round_e4m3


def _fp8(x, dim=-1):
    s = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-12) / 448.0
    return round_e4m3(x / s) * s


@torch.no_grad()
def probe(d=512, n=4096, n_outlier=8, sig_rank=32, seed=0):
    torch.manual_seed(seed)
    # realistic W: decaying spectrum (output metric G = W Wᵀ)
    U, _ = torch.linalg.qr(torch.randn(d, d))
    sv = torch.logspace(0, -2, d)                      # decaying singular values
    W = (U * sv) @ torch.linalg.qr(torch.randn(d, d))[0].T

    # realistic x: low-rank signal + a few outlier channels + noise
    Bm = torch.randn(n, sig_rank) @ torch.randn(sig_rank, d)
    x = Bm / Bm.std() + 0.3 * torch.randn(n, d)
    x[:, torch.randperm(d)[:n_outlier]] *= 12.0        # outlier channels

    # --- EQUALIZATION (mirror the real eqzp stack): fold s into the weight ---
    amax = x.abs().amax(0)
    s = amax.clamp_min(1e-5) ** 0.5                     # SmoothQuant alpha=0.5
    xe = x / s                                          # equalized activation
    We = (s[:, None] * W)                               # equalized weight: xe @ We == x @ W

    out_err = lambda e: ((e @ We) ** 2).sum().item()   # output-domain error (eq space)

    # --- A4 base + first W-aware SVD correction (k = d/16, fp8 coeffs) ---
    base = nvfp4_quantize_zp(xe, block=16, optclip=True)
    k = d // 16
    Uw, _, _ = torch.svd_lowrank(We, q=k + 8, niter=4)
    basis = Uw[:, :k]                                   # equalized-W's top-k input dirs
    coeff = _fp8((xe - base) @ basis)
    x_hat = base + coeff @ basis.T
    r = xe - x_hat                                      # leftover residual (eq space)
    W = We                                              # everything below in eq space
    x = xe
    base_oe = out_err(r)

    # --- spectrum concentration of the leftover residual (output metric) ---
    rW = r @ W                                          # residual in output space
    cov = rW.T @ rW / n
    evals = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)
    frac = lambda m: (evals[:m].sum() / evals.sum()).item()

    # --- QJL on r (sanity: should reduce INPUT residual energy ~31.8%) ---
    B_blk, Q = 128, 64                                  # QJL: 64 signs + 1 norm ~ 80 bits/128
    qjl = RademacherQJL(head_dim=B_blk, qjl_dim=Q)
    rb = r.reshape(n, d // B_blk, B_blk)
    sgn, nrm = qjl.encode(rb, head_idx=0)
    r_qjl = qjl.reconstruct(sgn, nrm, head_idx=0).reshape(n, d)
    qjl_in = 100 * (1 - ((r - r_qjl) ** 2).sum() / (r ** 2).sum()).item()
    qjl_oe = out_err(r - r_qjl)

    # --- residual-SVD (10 fp8 dirs, output metric) ---
    m = 10
    _, _, Vr = torch.svd_lowrank(rW, q=m + 8, niter=4)  # right singular vecs (d, q)
    Vr = Vr[:, :m]
    rc = _fp8(rW @ Vr)                                  # fp8 coeffs; deploy: (xe-x_hat)@(W@Vr)
    rsvd_oe = ((rW - rc @ Vr.T) ** 2).sum().item()      # output error after projection

    # --- CONTROL: just spend the same budget on MORE first-SVD rank (k -> k+m) ---
    basis2 = Uw[:, :k + m]
    coeff2 = _fp8((x - base) @ basis2)
    more_oe = out_err(x - (base + coeff2 @ basis2.T))

    print(f"leftover residual spectrum (output metric):")
    print(f"  top-10 {100*frac(10):4.1f}%  top-32 {100*frac(32):4.1f}%  "
          f"top-64 {100*frac(64):4.1f}%  (of {d} dims)")
    print(f"QJL sanity: input-domain residual energy removed = {qjl_in:.1f}% (expect ~31.8%)")
    print(f"output-domain error after side correction (lower=better):")
    print(f"  no side channel   : {base_oe:12.1f}")
    print(f"  QJL (64x1-bit)    : {qjl_oe:12.1f}   ({100*(base_oe-qjl_oe)/base_oe:+.1f}%)")
    print(f"  more W-SVD (+{m} fp8): {more_oe:12.1f}   ({100*(base_oe-more_oe)/base_oe:+.1f}%)")
    print(f"  residual-SVD ({m} fp8): {rsvd_oe:12.1f}   ({100*(base_oe-rsvd_oe)/base_oe:+.1f}%)")
    win = min([("QJL", qjl_oe), ("more-W-SVD", more_oe), ("residual-SVD", rsvd_oe)],
              key=lambda t: t[1])
    print(f"  => WINNER: {win[0]}  (matched ~80 bits / 128-block)")


if __name__ == "__main__":
    probe()
