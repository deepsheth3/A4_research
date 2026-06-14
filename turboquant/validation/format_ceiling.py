"""Format ceiling: how much could a BETTER 4-bit number format (Lloyd-Max scalar,
or trellis/VQ) beat E2M1 on REAL activations — in OUTPUT distortion?

Tests the hardware-codesign hypothesis. The theorem warns: equalization drives
blocks to the Gaussian-iid fixed point, where VQ/trellis gain over optimal scalar
is only the ~0.25-bit space-filling gain (memory gains are already eaten by eq +
additive low-rank). This measures whether that's true on real Llama-8B blocks.

  E2M1            : our fixed grid (scalar)
  Lloyd-Max (d=1) : OPTIMAL scalar  -> gap = cost of the fixed E2M1 grid (~4%)
  VQ d=2,3        : vector quant at matched 4 bit/elem -> the trellis/format ceiling
All at 4 bits/elem + the same per-16-block fp8 scale.

Run: python -m turboquant.validation.format_ceiling --caps caps.pt
"""
from __future__ import annotations

import argparse
import torch

from turboquant.nvfp4 import nvfp4_quantize_zp, round_e4m3, GRID_MAX


@torch.no_grad()
def _assign(X, C, chunk=20000):
    return torch.cat([torch.cdist(X[i:i+chunk], C).argmin(1)
                      for i in range(0, X.shape[0], chunk)])


@torch.no_grad()
def kmeans_codebook(X, n_clusters, iters=12, seed=0):
    """Lloyd's algorithm (vectorized update, chunked assign). Returns centroids C."""
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=g)[:n_clusters]].clone()
    for _ in range(iters):
        a = _assign(X, C)
        sums = torch.zeros_like(C).index_add_(0, a, X)
        cnt = torch.bincount(a, minlength=n_clusters)
        nz = cnt > 0
        C[nz] = sums[nz] / cnt[nz].unsqueeze(1).float()
    return C


@torch.no_grad()
def analyze_layer(x, G, amax, block=16):
    s = amax.clamp_min(1e-5) ** 0.5
    xe = (x / s).float()
    Ge = G.float() * s[:, None] * s[None, :]
    oe = lambda q: (((xe - q) @ Ge) * (xe - q)).sum().item()

    # per-16-block fp8 scale (the values the format quantizes are xe / s_b)
    n, d = xe.shape
    xb = xe.reshape(n, d // block, block)
    sb = round_e4m3(xb.abs().amax(-1, keepdim=True).clamp_min(1e-12) / GRID_MAX)
    scaled = (xb / sb)                               # what the 4-bit code sees
    flat = scaled.reshape(-1)                        # all scalar samples

    out = {"E2M1": oe(nvfp4_quantize_zp(xe, block=block, optclip=True))}

    # Lloyd-Max optimal scalar (16 levels) on the scaled samples
    train1 = flat[torch.randperm(flat.numel())[:200000], None]
    C1 = kmeans_codebook(train1, 16)
    qlm = C1[_assign(flat[:, None], C1)].reshape(scaled.shape) * sb
    out["LloydMax(d1)"] = oe(qlm.reshape(n, d))

    # VQ at matched 4 bit/elem: d-dim subvector -> 2^(4*dim) codewords
    for dim in (2, 4):
        K = 2 ** (4 * dim)
        sv = scaled.reshape(-1, dim)                 # subvectors
        if sv.shape[0] < 8 * K:                      # need enough data per codeword
            continue
        train = sv[torch.randperm(sv.shape[0])[:300000]]
        Cq = kmeans_codebook(train, K)
        qvq = Cq[_assign(sv, Cq)].reshape(scaled.shape) * sb
        out[f"VQ(d{dim})"] = oe(qvq.reshape(n, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="caps.pt")
    args = ap.parse_args()
    blob = torch.load(args.caps, map_location="cpu")
    caps = blob["caps"]
    print(f"model {blob['model']}  ({len(caps)} layers)\n")

    agg = {}
    for name, st in caps.items():
        r = analyze_layer(st["x"], st["G"], st["amax"])
        b = r["E2M1"]
        print(f"{name}")
        for k, v in r.items():
            print(f"   {k:14s} {v:11.1f}   ({100*(b-v)/b:+.1f}% vs E2M1)")
            agg.setdefault(k, []).append(v / b)
        print()

    print("=== AGGREGATE (mean output distortion, E2M1=1.000) ===")
    means = {k: sum(v) / len(v) for k, v in agg.items()}
    for k in sorted(means, key=lambda z: means[z]):
        print(f"  {k:14s} {means[k]:.3f}   ({100*(1-means[k]):+.1f}% vs E2M1)")
    if "LloydMax(d1)" in means:
        print(f"\n  grid cost (E2M1 vs optimal scalar): {100*(1-means['LloydMax(d1)']):.1f}%")
    vq = [k for k in means if k.startswith("VQ")]
    if vq:
        best = min(means[k] for k in vq)
        print(f"  format ceiling (best VQ vs E2M1): {100*(1-best):.1f}%")
        print("  => VERDICT:", "FORMAT HAS HEADROOM — build trellis/TCQ, box-test PPL"
              if best < 0.85 else "format ~floored too (eq+additive already ate the structure)")


if __name__ == "__main__":
    main()
