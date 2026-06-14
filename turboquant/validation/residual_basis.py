"""Door 2: is a residual-derived SVD basis better than W's eigenvector basis?

Our current A4 side-channel projects the residual onto W's top input singular
vectors (= top eigenvecs of the output metric G). The "residual-Hessian SVD" idea:
instead use the basis that captures the most OUTPUT energy of the ACTUAL residual
(output-aware residual PCA). Same rank, same fp8 coeffs, same deploy cost — only
the basis differs. If it beats the G-basis on real data, A4 had headroom.

  G-basis  : project r onto top-k eigenvecs of Ge            (what we ship)
  res-basis: project r onto top-k output-PCA dirs of r       (Door 2)

output energy of e = e Ge eᵀ = ||e·L||² with Ge = L Lᵀ, so res-basis = PCA of r·L.

Run: python -m turboquant.validation.residual_basis --caps caps.pt
"""
from __future__ import annotations

import argparse
import torch

from turboquant.nvfp4 import nvfp4_quantize_zp, round_e4m3


def _fp8(a):
    s = a.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
    return round_e4m3(a / s) * s


@torch.no_grad()
def analyze_layer(x, G, amax, k_div=16, n_train=200):
    s = amax.clamp_min(1e-5) ** 0.5
    xe = (x / s).float()
    Ge = G.float() * s[:, None] * s[None, :]
    d = xe.shape[1]
    base = nvfp4_quantize_zp(xe, block=16, optclip=True)
    r = xe - base                                          # residual (input space)
    oe = lambda e: ((e @ Ge) * e).sum().item()
    k = max(8, d // k_div)

    # eigh of Ge (clamped PSD): gives both the G-basis and the output whitening
    evals, evecs = torch.linalg.eigh(Ge)
    evals = evals.clamp_min(0)
    H = evecs @ torch.diag(evals.sqrt()) @ evecs.T         # Ge^{1/2}: ||e·H||²=e Ge eᵀ

    n = xe.shape[0]
    r_tr, r_te = r[:n_train], r[n_train:]                  # honest held-out
    D0 = oe(r_te)

    # G-basis (current, data-independent): top-k eigenvecs of Ge
    Vg = evecs[:, -k:]
    cg = _fp8(r_te @ Vg)
    oe_g = oe(r_te - cg @ Vg.T)

    # residual output-PCA basis: fit on TRAIN (output-whitened), eval on TEST
    rp_tr = r_tr @ H
    _, _, Vr = torch.svd_lowrank(rp_tr, q=min(k + 8, *rp_tr.shape))
    Vr = Vr[:, :k]
    rp_te = r_te @ H
    cp = _fp8(rp_te @ Vr)
    oe_res = ((rp_te - cp @ Vr.T) ** 2).sum().item()       # output error, held-out

    return {"D0": D0, "G-basis": oe_g, "res-basis": oe_res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="caps.pt")
    args = ap.parse_args()
    blob = torch.load(args.caps, map_location="cpu")
    caps = blob["caps"]
    print(f"model {blob['model']}  ({len(caps)} layers)  — output distortion, lower=better\n")

    gg, rr = [], []
    for name, st in caps.items():
        r = analyze_layer(st["x"], st["G"], st["amax"])
        d0 = r["D0"]
        print(f"{name}")
        print(f"   G-basis (ours) {100*(d0-r['G-basis'])/d0:+5.1f}%   "
              f"res-basis {100*(d0-r['res-basis'])/d0:+5.1f}%")
        gg.append(r["G-basis"] / d0); rr.append(r["res-basis"] / d0)

    G = 100 * (1 - sum(gg) / len(gg)); R = 100 * (1 - sum(rr) / len(rr))
    print(f"\n=== AGGREGATE (residual energy removed, output metric) ===")
    print(f"  G-basis (what we ship): {G:+.1f}%")
    print(f"  res-basis (Door 2):     {R:+.1f}%")
    print(f"  Door 2 advantage: {R - G:+.1f} pp  ->",
          "WORTH IT — swap the basis" if R - G > 3 else
          "marginal — residual ≈ W structure (Door 2 ~ours)")


if __name__ == "__main__":
    main()
