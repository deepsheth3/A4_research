r"""MoE phase (b): does token starvation actually HURT quantization, and can we fix it?

Phase (a) (`moe_expert_hessians.py`) proved starvation exists: on OLMoE the worst
expert saw 345 routed tokens against a 2048-dim Hessian, and effective rank degrades
monotonically with routing load. But that only showed the INPUT to our correction is
degraded. It did NOT show that this produces worse quantization, nor that anything
fixes it. That is this script.

Measured per expert, bucketed by routing load:
  err = ||(W - Wq) @ C||_F / ||W @ C||_F      where H = C Cᵀ

i.e. reconstruction error in the OUTPUT metric (weighted by the Hessian), not raw
weight-space MSE — because output error is what actually moves PPL.

Methods compared per expert:
  rtn             plain NVFP4 rounding                    (no Hessian used at all)
  ours            eq-free additive low-rank correction    (uses that expert's own H)
  ours_damp       + load-adaptive damping                 (mitigation 1)
  ours_shrink     + H shrunk toward the layer-mean H      (mitigation 2)

The prediction under test: `ours` beats `rtn` for well-fed experts but degrades (or
even LOSES to rtn) for starved ones, because their H is noise — and the mitigations
recover that. If `ours` never degrades, starvation is a non-issue for our method and
that itself is a clean, publishable answer.

SCOPE: gate_up_proj only. down_proj consumes the INTERMEDIATE activation (a different
tensor, different dim) which our Experts-module hook does not capture — quantizing it
against the hidden-dim Hessian would be silently wrong, so it is excluded rather than
faked.

Run:  python moe_phase_b.py --model allenai/OLMoE-1B-7B-0924
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from turboquant.gptq import lowrank_corrected_weight
from turboquant.nvfp4 import nvfp4_quantize
from turboquant.validation.moe_expert_hessians import (
    collect_expert_hessians, find_expert_modules,
)

RESULTS = Path(__file__).resolve().parent / "results"


def metric_factor(H, eps=1e-6):
    """A factor C with C Cᵀ ≈ H, robust to H being singular.

    Starved experts' Hessians are NOT positive-definite — with 345 routed tokens
    against 2048 dimensions, H is rank-deficient by construction and plain Cholesky
    fails outright (observed: "leading minor of order 1659 is not positive-definite").
    That IS the phenomenon under study, so the scorer must survive it rather than
    crash. Eigendecomposition with clamped eigenvalues always succeeds and gives the
    same C Cᵀ = H when H is well-conditioned.
    """
    d = H.shape[0]
    Hs = 0.5 * (H + H.t())
    ridge = eps * torch.diag(Hs).mean().clamp_min(1e-12)
    Hs = Hs + ridge * torch.eye(d, device=H.device, dtype=H.dtype)
    try:
        return torch.linalg.cholesky(Hs)
    except Exception:
        ev, Q = torch.linalg.eigh(Hs.double())
        ev = ev.clamp_min(ridge.double())          # kill the negative/zero directions
        return (Q * ev.sqrt()).to(H.dtype)


def output_metric_err(W, Wq, H, eps=1e-6):
    """||(W-Wq) C||_F / ||W C||_F with H ≈ C Cᵀ — error as the OUTPUT sees it."""
    C = metric_factor(H, eps)
    num = ((W - Wq).to(C) @ C).norm().item()
    den = (W.to(C) @ C).norm().item()
    return num / max(den, 1e-12)


def quantize_expert(W, H, method, tokens, dim, layer_mean_H, block=16, rank_div=16):
    """One expert's gate_up_proj under one method. Returns the dequantized weight."""
    if method == "rtn":
        return nvfp4_quantize(W, block=block)

    damp, Hu = 0.01, H
    if method == "ours_damp":
        # Fewer samples than dimensions => trust cross-channel structure less.
        # ratio<1 means rank-deficient by construction; damp up to 10x there.
        ratio = tokens / max(dim, 1)
        damp = 0.01 * max(1.0, min(10.0, 1.0 / max(ratio, 0.1)))
    elif method == "ours_shrink":
        # Shrink toward the layer's token-weighted mean H. Weight on the prior grows
        # as the expert's own sample shrinks (standard shrinkage estimator).
        lam = 1.0 / (1.0 + tokens / max(dim, 1))      # tokens>>dim -> lam->0
        Hu = (1 - lam) * H + lam * layer_mean_H

    return lowrank_corrected_weight(W, Hu, block=block, rank_div=rank_div,
                                    use_gptq=False, percdamp=damp, fp8_factors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--n-windows", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3, help="MoE layers to analyze")
    ap.add_argument("--rank-div", type=int, default=16)
    ap.add_argument("--methods", nargs="+",
                    default=["rtn", "ours", "ours_damp", "ours_shrink"])
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} model={args.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(device).eval()

    cfg = model.config
    E = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts")
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    calib = tok("\n\n".join(train["text"][:4000]), return_tensors="pt"
                ).input_ids[:, : args.n_windows * args.max_len]

    counts, hess = collect_expert_hessians(model, calib, args.max_len, device,
                                           args.n_windows, E)

    mods = dict(find_expert_modules(model))
    report = {"model": args.model, "num_experts": E,
              "calib_tokens": args.n_windows * args.max_len,
              "note": "gate_up_proj only; down_proj uses the intermediate activation "
                      "(different Hessian) and is excluded",
              "layers": {}}

    for nm in sorted(hess.keys())[: args.layers]:
        m = mods[nm]
        Wall = m.gate_up_proj.data                    # (E, 2*inter, hidden)
        c, Hs = counts[nm], hess[nm]
        dim = Hs.shape[-1]
        # token-weighted layer-mean Hessian (the shrinkage prior)
        tw = c.float().clamp_min(1).to(Hs.device)
        layer_mean_H = (Hs * tw[:, None, None]).sum(0) / tw.sum()

        print(f"\n  {nm}  ({E} experts, dim {dim})", flush=True)
        rows = []
        t0 = time.time()
        for e in range(E):
            W = Wall[e].float()
            H = Hs[e].float()
            tokens = int(c[e].item())
            row = {"expert": e, "tokens": tokens, "starved": bool(tokens < dim)}
            for meth in args.methods:
                Wq = quantize_expert(W, H, meth, tokens, dim, layer_mean_H,
                                     rank_div=args.rank_div)
                # Score every method against the expert's TRUE Hessian, even when the
                # method used a modified one — otherwise a method could "win" just by
                # being graded on its own altered metric.
                row[meth] = output_metric_err(W, Wq, H)
            rows.append(row)
        report["layers"][nm] = rows
        print(f"    quantized {E} experts x {len(args.methods)} methods "
              f"in {time.time()-t0:.0f}s", flush=True)

        # --- verdict for this layer: starved vs well-fed buckets ---
        rows_sorted = sorted(rows, key=lambda r: r["tokens"])
        q = max(1, len(rows_sorted) // 4)
        lo, hi = rows_sorted[:q], rows_sorted[-q:]
        print(f"    {'method':14s}{'starved-Q err':>15}{'well-fed-Q err':>16}{'gap':>9}")
        for meth in args.methods:
            a = sum(r[meth] for r in lo) / len(lo)
            b = sum(r[meth] for r in hi) / len(hi)
            print(f"    {meth:14s}{a:>15.4f}{b:>16.4f}{a-b:>9.4f}")
        # does our correction ever LOSE to plain rounding on starved experts?
        if "ours" in args.methods and "rtn" in args.methods:
            worse = [r for r in rows if r["ours"] > r["rtn"]]
            print(f"    experts where 'ours' is WORSE than plain rtn: "
                  f"{len(worse)}/{E}"
                  + (f"  (tokens: {[r['tokens'] for r in worse][:6]}...)" if worse else ""))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"moe_phase_b_{args.model.replace('/', '_')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
