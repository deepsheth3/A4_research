"""Numerical error analysis for the TurboQuant activation codec (CPU, Mac-runnable).

Produces three things, all on synthetic outlier-laden activations (the documented
LLM pattern: persistent per-channel outliers):

  1. Component ablation     — NVFP4 / +PolarQuant / +QJL@full / +QJL@block, NMSE.
  2. QJL block & proj sweep — NMSE vs (qjl_block, qjl_dim), showing QJL only helps
                              when the k/block ratio is in the useful range.
  3. Outlier stress         — NMSE vs outlier magnitude, raw NVFP4 vs corrected.

Writes results/error_analysis.json and results/projection_sweep.png.

Run:  python -m turboquant.validation.error_analysis
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from turboquant import nvfp4_quantize
from turboquant.act_codec import TurboQuantActQuantizer
from turboquant.config import TurboQuantConfig

RESULTS = Path(__file__).resolve().parents[2] / "results"


def nmse(x: torch.Tensor, xh: torch.Tensor) -> float:
    return (((x - xh) ** 2).sum() / (x ** 2).sum()).item()


def make_activations(dim=4096, tokens=128, n_outlier=3, scale=60.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, dim, generator=g)
    chans = torch.randperm(dim, generator=g)[:n_outlier]
    x[:, chans] *= scale  # persistent per-channel outliers
    return x


def component_ablation(x):
    raw = nmse(x, nvfp4_quantize(x, block=16))
    polar = nmse(x, TurboQuantActQuantizer(
        TurboQuantConfig(use_polarquant=True, qjl_dim=0)).fake_quantize(x))
    qjl_full = nmse(x, TurboQuantActQuantizer(
        TurboQuantConfig(use_polarquant=False, qjl_block=x.shape[-1], qjl_dim=64)).fake_quantize(x))
    qjl_block = nmse(x, TurboQuantActQuantizer(
        TurboQuantConfig(use_polarquant=False, qjl_block=128, qjl_dim=64)).fake_quantize(x))
    return {
        "nvfp4_only": raw,
        "+polarquant": polar,
        "+qjl_full_vector(k=64)": qjl_full,
        "+qjl_per_block(128,k=64)": qjl_block,
    }


def qjl_sweep(x):
    rows = []
    for block in (16, 64, 128, 512):
        for k in (8, 16, 32, 64, 128):
            if k > block:
                continue
            cfg = TurboQuantConfig(qjl_block=block, qjl_dim=k, use_polarquant=False)
            e = nmse(x, TurboQuantActQuantizer(cfg).fake_quantize(x))
            rows.append({
                "qjl_block": block, "qjl_dim": k, "ratio": k / block,
                "theory_reduction": k * 2 / (math.pi * block),
                "bits_per_elem": 4.0 + k / block, "nmse": e,
            })
    return rows


def outlier_stress(dim=4096):
    rows = []
    for s in (1.0, 5.0, 20.0, 60.0, 150.0):
        x = make_activations(dim=dim, scale=s)
        raw = nmse(x, nvfp4_quantize(x, block=16))
        tq = nmse(x, TurboQuantActQuantizer(TurboQuantConfig()).fake_quantize(x))
        rows.append({"outlier_scale": s, "nvfp4": raw, "turboquant": tq})
    return rows


def _plot(sweep, stress, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # plotting is optional
        print(f"[plot skipped: {e}]")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for block in sorted({r["qjl_block"] for r in sweep}):
        pts = [(r["bits_per_elem"], r["nmse"]) for r in sweep if r["qjl_block"] == block]
        pts.sort()
        ax[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"block={block}")
    ax[0].set(xlabel="bits/element (4 + k/block)", ylabel="NMSE", title="QJL block & projection sweep")
    ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot([r["outlier_scale"] for r in stress], [r["nvfp4"] for r in stress], "o-", label="raw NVFP4")
    ax[1].plot([r["outlier_scale"] for r in stress], [r["turboquant"] for r in stress], "s-", label="TurboQuant")
    ax[1].set(xlabel="outlier magnitude (×)", ylabel="NMSE", title="Outlier stress")
    ax[1].legend(); ax[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    RESULTS.mkdir(exist_ok=True)
    x = make_activations()
    ablation = component_ablation(x)
    sweep = qjl_sweep(x)
    stress = outlier_stress()

    print("\n=== Component ablation (NMSE, lower is better) ===")
    for k, v in ablation.items():
        print(f"  {k:32s} {v:.4e}")
    print("\n=== Outlier stress (NMSE) ===")
    for r in stress:
        print(f"  scale {r['outlier_scale']:6.1f}x   nvfp4 {r['nvfp4']:.4e}   turboquant {r['turboquant']:.4e}")
    best = min(sweep, key=lambda r: r["nmse"])
    print(f"\nBest sweep config: block={best['qjl_block']} k={best['qjl_dim']} "
          f"(ratio {best['ratio']:.2f}, {best['bits_per_elem']:.2f} bits/elem) NMSE {best['nmse']:.4e}")

    out = {"ablation": ablation, "qjl_sweep": sweep, "outlier_stress": stress}
    (RESULTS / "error_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'error_analysis.json'}")
    _plot(sweep, stress, RESULTS / "projection_sweep.png")


if __name__ == "__main__":
    main()
