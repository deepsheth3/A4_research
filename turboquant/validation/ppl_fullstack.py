"""PPL of the full deployed stack (W4 eq+lowrank / A4 eq+SVD+QJL / KV4) — the
combination `accept_fullstack.py` measures acceptance on but never PPL'd.

Two questions this answers, neither previously measured:

1. **Does KV4 compose?** `kv4_correction.py` measured the KV4 leg *in isolation*
   (everything else FP16): +0.119 PPL raw, ~45% of that recovered by a rank-64
   additive residual correction. Whether that still holds once W4+A4 are also on
   is untested — this reports full-stack with KV4 off / raw / +lowrank64.

2. **Is GPTQ's coordinated rounding inert here?** The two-lever theorem says
   equalization + additive low-rank are the levers and coordinated rounding is
   redundant post-equalization. `--ablate-gptq` runs the no-KV4 config both ways
   (GPTQ base vs plain RTN base, same low-rank correction on top). A tie is the
   direct evidence that our result comes from eq+additive, not from GPTQ.

Run (local/CPU-MPS is fine for TinyLlama):
  python -m turboquant.validation.ppl_fullstack \
     --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --limit 200000 --ablate-gptq
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from turboquant.config import TurboQuantConfig
from turboquant.validation.hf_perplexity import (
    calibrate_eq_scales, collect_weight_hessians, quantize_weights_gptq,
    install_hooks, perplexity,
)
from turboquant.validation.kv4_correction import calib_residual_basis, install_kv4

RESULTS = Path(__file__).resolve().parents[2] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="eval token cap (0 = full test split)")
    ap.add_argument("--calib-windows", type=int, default=32)
    ap.add_argument("--w4-rank-div", type=int, default=16)
    ap.add_argument("--qjl-block", type=int, default=128)
    ap.add_argument("--qjl-dim", type=int, default=64)
    ap.add_argument("--kv-rank", type=int, default=64, help="rank of the KV residual correction")
    ap.add_argument("--ablate-gptq", action="store_true",
                    help="also run the no-KV4 config with plain RTN base (two-lever ablation)")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32   # MPS lacks some bf16 linalg ops
    else:
        device, dtype = "cpu", torch.float32
    print(f"device={device} dtype={dtype} model={args.model} rank=d/{args.w4_rank_div}", flush=True)

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
    L = args.max_len
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    calib = tok("\n\n".join(train["text"][:4000]),
                return_tensors="pt").input_ids[:, : args.calib_windows * L]
    print(f"eval tokens: {ids.size(1)}", flush=True)

    cfg = TurboQuantConfig(qjl_block=args.qjl_block, qjl_dim=args.qjl_dim)
    results: dict = {}

    def run(tag: str, *, quantize: bool, kv: str, use_gptq: bool = True) -> float:
        """Build a fresh model, apply the requested stack, return PPL.

        kv: "off" | "raw" | "lowrank" — the KV-cache leg.
        """
        model = load().to(device).eval()
        handles = []
        if quantize:
            hess = collect_weight_hessians(model, calib, L, device, n_seq=args.calib_windows)
            eq = calibrate_eq_scales(model, calib, L, device)
            quantize_weights_gptq(model, hess, cfg.mx_block, lowrank=True,
                                  rank_div=args.w4_rank_div, fp8_factors=True,
                                  use_gptq=use_gptq)
            handles += install_hooks(model, "nvfp4_eqzp_svd_qjl", cfg, eq)
        if kv == "raw":
            handles += install_kv4(model, cfg.mx_block)
        elif kv == "lowrank":
            basis = calib_residual_basis(model, calib, L, device, cfg.mx_block,
                                         [args.kv_rank], args.calib_windows)
            handles += install_kv4(model, cfg.mx_block, basis=basis, rank=args.kv_rank)
        t0 = time.time()
        ppl = perplexity(model, ids, L, L, device)
        for h in handles:
            h.remove()
        results[tag] = {"perplexity": ppl, "seconds": round(time.time() - t0, 1)}
        print(f"  {tag:24s} PPL = {ppl:8.4f}   ({results[tag]['seconds']}s)", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return ppl

    fp16 = run("fp16", quantize=False, kv="off")
    stack = run("stack", quantize=True, kv="off")
    kv_raw = run("stack+kv4", quantize=True, kv="raw")
    kv_lr = run(f"stack+kv4+lr{args.kv_rank}", quantize=True, kv="lowrank")
    if args.ablate_gptq:
        run("stack(RTN base)", quantize=True, kv="off", use_gptq=False)

    # The deltas that answer the two questions.
    kv_cost = kv_raw - stack
    kv_recovered = (kv_raw - kv_lr) / kv_cost if abs(kv_cost) > 1e-9 else float("nan")
    summary = {
        "stack_vs_fp16": round(stack - fp16, 4),
        "kv4_cost_on_top_of_stack": round(kv_cost, 4),
        "kv4_lowrank_recovered_frac": round(kv_recovered, 4),
    }
    if args.ablate_gptq:
        summary["gptq_vs_rtn_base"] = round(
            results["stack(RTN base)"]["perplexity"] - stack, 4)
    print("\n  " + json.dumps(summary, indent=2).replace("\n", "\n  "), flush=True)

    RESULTS.mkdir(exist_ok=True)
    out = {"model": args.model, "max_len": L, "rank_div": args.w4_rank_div,
           "kv_rank": args.kv_rank, "results": results, "summary": summary}
    path = RESULTS / f"ppl_fullstack_{args.model.replace('/', '_')}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
