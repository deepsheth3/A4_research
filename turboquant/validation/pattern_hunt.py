"""Pattern hunt: is the leftover quantization residual truly random, or is there a
FREE, foldable pattern hiding in it?

The decoder knows Q(x) and the channel index for free. If the residual r = x - Q(x)
is predictable from those, the correction folds into the weights — zero bits, zero
latency, keeps the rule. A non-uniform grid (E2M1) on real data can carry a
systematic, magnitude-dependent bias we never removed.

Honest test: fit per-channel predictors on TRAIN tokens, measure how much output
distortion they remove on HELD-OUT TEST tokens (so "pattern" = real, not memorized).
All predictors are foldable/free:
  bias       r_i ≈ b_i                          (fold into next layer's bias)
  affine     r_i ≈ a_i·Q(x)_i + b_i             (adjust per-channel dequant scale+bias)
  quadratic  r_i ≈ a_i·Q + c_i·Q² + b_i         (per-channel companding LUT)

Run: python -m turboquant.validation.pattern_hunt --caps caps.pt
"""
from __future__ import annotations

import argparse
import torch

from turboquant.nvfp4 import nvfp4_quantize_zp


@torch.no_grad()
def _fit_percol(feats, r_tr):
    """Per-channel least squares: solve for each column independently.
    feats: list of (n_tr, d) feature tensors; returns predictor fn on test feats."""
    # stack features along a new axis -> (n, d, p); solve d independent p-systems
    F = torch.stack(feats, -1)                      # (n_tr, d, p)
    y = r_tr.unsqueeze(-1)                          # (n_tr, d, 1)
    # normal equations per channel: (FᵀF) β = Fᵀy
    FtF = torch.einsum('ndp,ndq->dpq', F, F)
    Fty = torch.einsum('ndp,ndq->dpq', F, y).squeeze(-1)   # (d, p)
    ridge = 1e-3 * FtF.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True)
    FtF = FtF + ridge.unsqueeze(-1) * torch.eye(F.shape[-1])
    beta = torch.linalg.solve(FtF, Fty.unsqueeze(-1)).squeeze(-1)   # (d, p)
    return beta


@torch.no_grad()
def analyze_layer(x, G, amax, n_train=200):
    s = amax.clamp_min(1e-5) ** 0.5
    xe = (x / s).float()
    Ge = G.float() * s[:, None] * s[None, :]
    q = nvfp4_quantize_zp(xe, block=16, optclip=True)      # Q(x), decoder-known
    r = xe - q                                             # residual to hunt in
    n = xe.shape[0]
    tr, te = slice(0, n_train), slice(n_train, n)
    oe = lambda e: ((e @ Ge) * e).sum().item()
    D0 = oe(r[te])                                         # uncorrected residual (test)

    def predict(feat_fns):
        feats_tr = [f(q[tr]) for f in feat_fns]
        beta = _fit_percol(feats_tr, r[tr])               # (d, p)
        feats_te = torch.stack([f(q[te]) for f in feat_fns], -1)  # (n_te, d, p)
        rhat = (feats_te * beta).sum(-1)                  # (n_te, d)
        return oe(r[te] - rhat)

    one = lambda Q: torch.ones_like(Q)
    out = {
        "uncorrected": D0,
        "bias":      predict([one]),
        "affine":    predict([one, lambda Q: Q]),
        "quadratic": predict([one, lambda Q: Q, lambda Q: Q * Q]),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="caps.pt")
    args = ap.parse_args()
    blob = torch.load(args.caps, map_location="cpu")
    caps = blob["caps"]
    print(f"model {blob['model']}  ({len(caps)} layers)  — % = output distortion removed on HELD-OUT test\n")

    agg = {}
    for name, st in caps.items():
        r = analyze_layer(st["x"], st["G"], st["amax"])
        d0 = r["uncorrected"]
        print(f"{name}")
        for k, v in r.items():
            if k != "uncorrected":
                print(f"   {k:10s} removes {100*(d0-v)/d0:+5.1f}%")
                agg.setdefault(k, []).append((d0 - v) / d0)
        print()

    print("=== AGGREGATE (free, foldable correction — held-out test) ===")
    for k, v in agg.items():
        m = 100 * sum(v) / len(v)
        print(f"  {k:10s} {m:+5.1f}%  of residual is predictable-for-free")
    best = max(100 * sum(v) / len(v) for v in agg.values())
    print("\n  VERDICT:", "PATTERN FOUND — free foldable gain" if best > 3
          else "no free pattern — residual is genuinely random to these predictors")


if __name__ == "__main__":
    main()
