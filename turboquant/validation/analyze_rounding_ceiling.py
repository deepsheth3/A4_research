"""Offline rounding-ceiling analysis on REAL captured activations (free, no GPU).

Loads capture_activations.py output and, per layer, measures output-domain error
under nearest rounding vs output-metric (G) rounding at several block widths, with
and without the SVD side-channel. Answers:
  1. Does the synthetic +45%-over-SVD prize survive on REAL data?
  2. Is it reachable PARALLEL-friendly (wide-block) or only full serial?
  3. Rough end-to-end PPL estimate.

Run: python -m turboquant.validation.analyze_rounding_ceiling --caps caps.pt
"""
from __future__ import annotations

import argparse

import torch

from turboquant.gptq import gptq_quantize_weight
from turboquant.nvfp4 import round_e4m3

# our banked references (Llama-3.1-8B, full WikiText-2)
FP16, FP8, STACK = 5.918, 5.948, 6.050


def _block_diag(G, block):
    M = torch.zeros_like(G)
    for k in range(0, G.shape[0], block):
        M[k:k + block, k:k + block] = G[k:k + block, k:k + block]
    return M


def _fp8(a):
    sc = a.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
    return round_e4m3(a / sc) * sc


@torch.no_grad()
def analyze_layer(x, G, amax, widths, k_div=16):
    s = amax.clamp_min(1e-5) ** 0.5                     # eq scale
    xe = (x / s).float()
    Ge = G.float() * s[:, None] * s[None, :]            # output metric in eq space
    d = xe.shape[1]
    oe = lambda q: (((xe - q) @ Ge) * (xe - q)).sum().item()   # (x-q) Ge (x-q)ᵀ

    eye = torch.eye(d)
    qn = gptq_quantize_weight(xe, eye, block=16)        # nearest (identity metric)
    base = oe(qn)

    # SVD side-channel basis = top-k eigenvecs of Ge (our stack's correction)
    k = max(8, d // k_div)
    evals, evecs = torch.linalg.eigh(Ge)
    Vg = evecs[:, -k:]
    add_svd = lambda q: q + _fp8((xe - q) @ Vg) @ Vg.T

    out = {"nearest": base, "nearest+SVD": oe(add_svd(qn))}
    for B in widths:                                    # G-rounding at block width B
        Bq = gptq_quantize_weight(xe, _block_diag(Ge, B) if B < d else Ge, block=16)
        out[f"G{B}"] = oe(Bq)
        out[f"G{B}+SVD"] = oe(add_svd(Bq))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="caps.pt")
    ap.add_argument("--widths", type=int, nargs="+", default=[16, 128, 256])
    args = ap.parse_args()

    blob = torch.load(args.caps, map_location="cpu")
    caps = blob["caps"]
    print(f"model {blob['model']}  ({len(caps)} layers)\n")

    agg: dict = {}
    for name, st in caps.items():
        d = st["G"].shape[0]
        widths = [w for w in args.widths if w < d] + [d]   # include full-G (=d)
        r = analyze_layer(st["x"], st["G"], st["amax"], widths)
        b = r["nearest"]
        red = {kk: 100 * (b - v) / b for kk, v in r.items()}
        print(f"{name}  (d={d})")
        for kk in r:
            print(f"   {kk:14s} {red[kk]:+6.1f}%")
        for kk, v in r.items():
            agg.setdefault(kk, []).append(v / b)           # normalized error vs nearest
        print()

    print("=== AGGREGATE (mean normalized output error vs nearest=1.000) ===")
    means = {kk: sum(v) / len(v) for kk, v in agg.items()}
    for kk in sorted(means, key=lambda z: means[z]):
        print(f"  {kk:16s} {means[kk]:.3f}   ({100*(1-means[kk]):+.1f}% vs nearest)")

    # key comparison + rough PPL estimate
    stack = means.get("nearest+SVD")
    fullkey = max((k for k in means if k.startswith("G") and "+SVD" in k),
                  key=lambda z: int(z[1:].split("+")[0]))
    best = means[fullkey]
    if stack and best:
        f = (stack - best) / stack                          # extra error cut over our stack
        ppl_est = FP16 + (STACK - FP16) * (1 - f)            # linear-excess model
        print(f"\nfull-G+SVD ({fullkey}) cuts our-stack output error by {100*f:+.1f}%")
        print(f"rough PPL estimate: {ppl_est:.3f}   (now {STACK}, FP8 {FP8}, FP16 {FP16})")
        # parallel viability: how much of full-G does block-256 capture?
        for B in args.widths:
            kkey = f"G{B}+SVD"
            if kkey in means and stack > best:
                cap = (stack - means[kkey]) / (stack - best)
                print(f"  block-{B} captures {100*cap:.0f}% of the full-G gain "
                      f"(parallel across blocks => {'VIABLE' if cap>0.7 else 'partial'})")
        verdict = ("PRIZE REAL — chase parallel form" if ppl_est < FP8 - 0.005
                   else "below current but not FP8 — marginal" if ppl_est < STACK - 0.02
                   else "WASH — at the floor")
        print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
