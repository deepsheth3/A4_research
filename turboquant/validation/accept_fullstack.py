"""Acceptance of the full deployed draft (W4 GPTQ+lowrank / A4 eq+SVD+QJL / KV4) vs
the FP16 target — the number that sets the spec-decode throughput/latency win.

Output is FP16-quality by construction (target verifies); the codec gap shows up
only as acceptance alpha = 1 - TV(draft, target). Higher alpha -> more accepted draft
tokens per verify -> more speedup. We measure alpha on the SAME stack we PPL'd, then
project the spec-decode speedup with acceptance.speedup().

Run (Blackwell/H100):
  python -m turboquant.validation.accept_fullstack \
     --model unsloth/Meta-Llama-3.1-8B --w4-rank-div 6 --gamma 4 --cost-ratio 0.3
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
    install_hooks, install_kv_hooks,
)
from turboquant.validation.acceptance import (
    acceptance_stats, expected_accepted_length, speedup,
)

RESULTS = Path(__file__).resolve().parents[2] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--n-windows", type=int, default=8, help="eval windows for alpha")
    ap.add_argument("--w4-rank-div", type=int, default=6)
    ap.add_argument("--qjl-block", type=int, default=128)
    ap.add_argument("--qjl-dim", type=int, default=64)
    ap.add_argument("--gamma", type=int, default=4, help="draft tokens per verify step")
    ap.add_argument("--cost-ratio", type=float, default=0.3,
                    help="draft-step / target-step cost (FP4 draft vs FP16 target)")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = "cuda", torch.bfloat16
    print(f"device={device} dtype={dtype} model={args.model} rank=d/{args.w4_rank_div}")

    def load():
        try:
            return AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
    L = args.max_len
    windows = [ids[:, i * L:(i + 1) * L].to(device) for i in range(ids.size(1) // L)][: args.n_windows]
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    calib = tok("\n\n".join(train["text"][:4000]),
                return_tensors="pt").input_ids[:, : 32 * L]

    teacher = load().to(device).eval()                            # FP16 target
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = load().to(device).eval()                            # FP4 draft (full stack)
    hess = collect_weight_hessians(student, calib, L, device, n_seq=32)
    eq = calibrate_eq_scales(student, calib, L, device)
    quantize_weights_gptq(student, hess, 16, lowrank=True,
                          rank_div=args.w4_rank_div, fp8_factors=True)
    cfg = TurboQuantConfig(qjl_block=args.qjl_block, qjl_dim=args.qjl_dim)
    handles = install_hooks(student, "nvfp4_eqzp_svd_qjl", cfg, eq) + install_kv_hooks(student, 16)

    V = teacher.config.vocab_size
    alphas, greedys = [], []
    t0 = time.time()
    with torch.no_grad():
        for x in windows:
            t = teacher(x).logits[:, :-1, :].reshape(-1, V)
            d = student(x).logits[:, :-1, :].reshape(-1, V)
            a, g = acceptance_stats(d, t)
            alphas.append(a); greedys.append(g)
    for h in handles:
        h.remove()
    alpha = sum(alphas) / len(alphas)
    greedy = sum(greedys) / len(greedys)

    eal = expected_accepted_length(alpha, args.gamma)
    spd = speedup(alpha, args.gamma, args.cost_ratio)
    out = {"model": args.model, "rank_div": args.w4_rank_div,
           "alpha_1_minus_TV": round(alpha, 4), "greedy_agreement": round(greedy, 4),
           "gamma": args.gamma, "cost_ratio": args.cost_ratio,
           "expected_accepted_len": round(eal, 3),
           "projected_speedup": round(spd, 3), "seconds": round(time.time() - t0, 1)}
    print(f"\n=== full-stack draft acceptance (W4 d/{args.w4_rank_div}+lowrank, A4, KV4) ===")
    print(f"  alpha (1-TV)         = {alpha:.4f}")
    print(f"  greedy agreement     = {greedy:.4f}")
    print(f"  E[accepted len] (g={args.gamma}) = {eal:.3f} tokens/verify")
    print(f"  projected speedup    = {spd:.3f}x  (cost_ratio={args.cost_ratio})")
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"accept_fullstack_{args.model.replace('/', '_')}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
