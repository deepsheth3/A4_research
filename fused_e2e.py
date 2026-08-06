r"""End-to-end: run a REAL model with the fused Triton kernels in the forward path.

Everything before this measured the kernels in isolation. This wires them into an
actual TinyLlama forward pass as module hooks and answers the two questions that
actually decide whether the fusion is real:

  1. CORRECTNESS AT SCALE — does the fused path give the same PPL as the PyTorch
     reference? Kernel-level relerr of 1e-7 is necessary but not sufficient; errors
     can compound across 22 layers x hundreds of tokens.
  2. DOES IT ACTUALLY HELP — wall-clock forward-pass time, fused vs unfused. A 14x
     win on an isolated op means little if that op is 2% of the forward pass. This
     is where Amdahl gets a vote.

Configuration measured: W4A8 (our best Pareto point from T3) — NVFP4 weights with
equalization + additive low-rank correction, FP8 activations. The fused kernel
replaces the per-layer (equalize -> fp8-quantize) activation hook.

Run:  python fused_e2e.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from triton_fusion_suite import fused_eq_fp8, ref_eq_fp8, HAS_TRITON
from turboquant.validation.hf_perplexity import (
    _is_linear, calibrate_eq_scales, collect_weight_hessians, fp8_quantize,
    perplexity, quantize_weights_gptq,
)

RESULTS = Path(__file__).resolve().parent / "results"


def install_act_hooks(model, eq_scales: dict, fused: bool):
    """Install the W4A8 activation path: equalize + fp8-quantize, per linear layer.

    fused=True  -> one Triton kernel, one HBM round trip.
    fused=False -> the PyTorch reference: divide, absmax-reduce, quantize, multiply.
    """
    handles = []
    for i, m in enumerate(mod for mod in model.modules() if _is_linear(mod)):
        s = eq_scales.get(i)
        if s is None:
            continue

        def mk(scale):
            def hook(module, args):
                x = args[0]
                if x.shape[-1] != scale.numel():
                    return None
                flat = x.reshape(-1, x.shape[-1])
                sc = scale.to(device=flat.device, dtype=torch.float32)
                if fused:
                    y = fused_eq_fp8(flat.float().contiguous(), sc)
                else:
                    y = ref_eq_fp8(flat.float(), sc)
                return (y.reshape(x.shape).to(x.dtype), *args[1:])
            return hook
        handles.append(m.register_forward_pre_hook(mk(s)))
    return handles


def timed_forward(model, ids, max_len, device, n_windows=8):
    """Wall-clock of n_windows full forward passes (what a user actually waits on)."""
    wins = [ids[:, i * max_len:(i + 1) * max_len].to(device)
            for i in range(min(n_windows, ids.size(1) // max_len))]
    with torch.no_grad():
        for w in wins[:2]:                      # warmup
            model(w)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for w in wins:
            model(w)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / len(wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--calib-windows", type=int, default=32)
    ap.add_argument("--w4-rank-div", type=int, default=16)
    args = ap.parse_args()

    if not (torch.cuda.is_available() and HAS_TRITON):
        print("needs CUDA + triton"); return

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = "cuda", torch.bfloat16
    print(f"device={device} model={args.model}", flush=True)

    def load():
        try:
            return AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[:, : args.limit]
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    calib = tok("\n\n".join(train["text"][:4000]), return_tensors="pt"
                ).input_ids[:, : args.calib_windows * args.max_len]
    L = args.max_len
    print(f"eval tokens: {ids.size(1)}", flush=True)

    results = {}

    # ---- FP16 reference (no quantization at all) ----
    model = load().to(device).eval()
    t_fp16 = timed_forward(model, ids, L, device)
    ppl_fp16 = perplexity(model, ids, L, L, device)
    results["fp16"] = {"ppl": ppl_fp16, "fwd_ms": t_fp16 * 1e3}
    print(f"  fp16          PPL {ppl_fp16:8.4f}   fwd {t_fp16*1e3:7.1f} ms", flush=True)

    # ---- W4A8: quantize weights once (shared by both activation paths) ----
    hess = collect_weight_hessians(model, calib, L, device, n_seq=args.calib_windows)
    eq = calibrate_eq_scales(model, calib, L, device)
    quantize_weights_gptq(model, hess, 16, lowrank=True, rank_div=args.w4_rank_div,
                          fp8_factors=True, use_gptq=False)   # OUR method: RTN + lowrank

    for tag, fused in (("w4a8_unfused", False), ("w4a8_fused", True)):
        h = install_act_hooks(model, eq, fused=fused)
        t = timed_forward(model, ids, L, device)
        p = perplexity(model, ids, L, L, device)
        for x in h:
            x.remove()
        results[tag] = {"ppl": p, "fwd_ms": t * 1e3}
        print(f"  {tag:13s} PPL {p:8.4f}   fwd {t*1e3:7.1f} ms", flush=True)

    # ---- verdict ----
    u, f = results["w4a8_unfused"], results["w4a8_fused"]
    dppl = abs(f["ppl"] - u["ppl"])
    summary = {
        "ppl_delta_fused_vs_unfused": round(dppl, 6),
        "ppl_identical": bool(dppl < 1e-3),
        "fwd_speedup": round(u["fwd_ms"] / f["fwd_ms"], 4),
        "fwd_saved_ms": round(u["fwd_ms"] - f["fwd_ms"], 2),
        "w4a8_vs_fp16_ppl": round(f["ppl"] - results["fp16"]["ppl"], 4),
        "fused_overhead_vs_fp16_fwd": round(f["fwd_ms"] / results["fp16"]["fwd_ms"], 3),
    }
    print("\n  " + json.dumps(summary, indent=2).replace("\n", "\n  "), flush=True)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"fused_e2e_{args.model.replace('/', '_')}.json"
    out.write_text(json.dumps({"model": args.model, "results": results,
                               "summary": summary}, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
