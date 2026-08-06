r"""Our full method on a REAL MoE, measured end-to-end for perplexity.

Everything in this project — including the two-lever theorem — was derived and
verified on DENSE models. MoE adds structure the theorem does not model: routing,
expert specialization, and load imbalance severe enough that some experts see 6x
fewer calibration tokens than their Hessian has dimensions
(`moe_expert_hessians.py`). Whether our levers survive that is untested.

This quantizes an actual MoE's expert weights and measures WikiText PPL, comparing:
  fp16          reference
  rtn           plain NVFP4 rounding, no calibration at all
  ours          equalization-free additive low-rank (per-expert H)
  ours_shrink   + per-expert H shrunk toward the layer-mean H (the starvation fix)
  rot_gptq      rotation-aware GPTQ (the competing method) -- does it fare better
                or worse than ours under routing?

Only the MoE EXPERT weights are quantized (gate_up_proj), so the delta isolates the
expert path — attention and shared layers stay FP16. That is deliberate: it makes the
comparison about the thing MoE actually changes, rather than diluting it across a
model that is mostly not experts.

Run:  python moe_e2e.py --model allenai/OLMoE-1B-7B-0924
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from turboquant.gptq import lowrank_corrected_weight, rotated_gptq_weight
from turboquant.nvfp4 import nvfp4_quantize
from turboquant.validation.hf_perplexity import perplexity
from turboquant.validation.moe_expert_hessians import (
    collect_expert_hessians, find_expert_modules,
)

RESULTS = Path(__file__).resolve().parent / "results"


def quantize_moe(model, counts, hess, method, rank_div=16, block=16):
    """Quantize every expert's gate_up_proj in place. Returns (n_experts, seconds)."""
    mods = dict(find_expert_modules(model))
    n, t0 = 0, time.time()
    for nm, m in mods.items():
        if nm not in hess:
            continue
        W_all = m.gate_up_proj.data
        c, Hs = counts[nm], hess[nm]
        dim = Hs.shape[-1]
        tw = c.float().clamp_min(1).to(Hs.device)
        layer_mean_H = (Hs * tw[:, None, None]).sum(0) / tw.sum()
        new = torch.empty_like(W_all)
        for e in range(W_all.shape[0]):
            W = W_all[e].float()
            H = Hs[e].float()
            tokens = int(c[e].item())
            if method == "rtn":
                Wq = nvfp4_quantize(W, block=block)
            elif method == "rot_gptq":
                Wq = rotated_gptq_weight(W, H, block=block, use_gptq=True)
            else:
                Hu = H
                if method == "ours_shrink":
                    lam = 1.0 / (1.0 + tokens / max(dim, 1))
                    Hu = (1 - lam) * H + lam * layer_mean_H
                Wq = lowrank_corrected_weight(W, Hu, block=block, rank_div=rank_div,
                                              use_gptq=False, fp8_factors=True)
            new[e] = Wq.to(W_all.dtype)
            n += 1
        m.gate_up_proj.data = new
    return n, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--n-windows", type=int, default=32)
    ap.add_argument("--rank-div", type=int, default=16)
    ap.add_argument("--methods", nargs="+",
                    default=["rtn", "ours", "ours_shrink", "rot_gptq"])
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} model={args.model} rank=in/{args.rank_div}", flush=True)

    def load():
        try:
            return AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
    if args.limit:
        ids = ids[:, : args.limit]
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    calib = tok("\n\n".join(train["text"][:4000]), return_tensors="pt"
                ).input_ids[:, : args.n_windows * args.max_len]
    L = args.max_len
    print(f"eval tokens: {ids.size(1)}", flush=True)

    # Hessians collected ONCE on the clean model, shared by every method.
    base = load().to(device).eval()
    E = getattr(base.config, "num_experts", None) or base.config.num_local_experts
    counts, hess = collect_expert_hessians(base, calib, L, device, args.n_windows, E)
    counts = {k: v.cpu() for k, v in counts.items()}
    hess = {k: v.cpu() for k, v in hess.items()}
    fp16 = perplexity(base, ids, L, L, device)
    print(f"  {'fp16':12s} PPL = {fp16:8.4f}", flush=True)
    del base
    torch.cuda.empty_cache()

    results = {"fp16": {"ppl": fp16}}
    for method in args.methods:
        model = load().to(device).eval()
        h = {k: v.to(device) for k, v in hess.items()}
        c = {k: v.to(device) for k, v in counts.items()}
        n, tq = quantize_moe(model, c, h, method, rank_div=args.rank_div)
        ppl = perplexity(model, ids, L, L, device)
        results[method] = {"ppl": ppl, "experts": n, "quant_s": round(tq, 1)}
        print(f"  {method:12s} PPL = {ppl:8.4f}   ({n} experts, {tq:.0f}s)", flush=True)
        del model, h, c
        torch.cuda.empty_cache()

    got = {k: v["ppl"] for k, v in results.items() if k != "fp16"}
    best = min(got, key=got.get)
    summary = {"best": best, "best_ppl": round(got[best], 4),
               "fp16": round(fp16, 4)}
    if "ours" in got and "rot_gptq" in got:
        summary["ours_minus_rot_gptq"] = round(got["ours"] - got["rot_gptq"], 4)
        summary["ours_beats_rot_gptq_on_moe"] = bool(got["ours"] < got["rot_gptq"])
    if "ours" in got and "ours_shrink" in got:
        summary["shrinkage_gain"] = round(got["ours"] - got["ours_shrink"], 4)
    if "ours" in got and "rtn" in got:
        summary["ours_vs_rtn"] = round(got["rtn"] - got["ours"], 4)
    print("\n  " + json.dumps(summary, indent=2).replace("\n", "\n  "), flush=True)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"moe_e2e_{args.model.replace('/', '_')}.json"
    out.write_text(json.dumps({"model": args.model, "rank_div": args.rank_div,
                               "results": results, "summary": summary}, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
