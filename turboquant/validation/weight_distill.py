"""Path 1: do TRAINED weight low-rank factors beat the ANALYTICAL (LQER/SVD) fill?

Weights are the dominant leg of the W4A4KV4->FP8 gap (+0.48). The repo fills the
rank-r weight correction analytically (SVD of the residual in the H-metric). The bet:
freeze the FP4 weight Wq, warm-start L,R from the analytical factors, and TRAIN them
against the FP16 teacher (KL) — optimizing the deploy objective (acceptance/loss),
not reconstruction MSE. Same Wq, same rank, same bytes; only the factors differ.

Isolates the weight leg (no A4/KV4 quant) so the comparison is clean:
  analytical factors  vs  trained factors  ->  PPL and acceptance (1-TV vs FP16).
If trained wins, that's higher accuracy/acceptance-per-byte than the standard fill
(the novel result). If it ties, Path 1 is closed cleanly.

Run (Blackwell/H100):
  python -m turboquant.validation.weight_distill --model unsloth/Meta-Llama-3.1-8B \
     --rank-div 6 --steps 300 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from turboquant.gptq import gptq_lowrank_factors
from turboquant.distill import LowRankLinear, attach_lowrank, distill_factors
from turboquant.validation.hf_perplexity import _is_linear, collect_weight_hessians, perplexity
from turboquant.validation.acceptance import acceptance_stats, expected_accepted_length, speedup

RESULTS = Path(__file__).resolve().parents[2] / "results"


@torch.no_grad()
def _acceptance(student, teacher, windows):
    V = teacher.config.vocab_size
    a = g = 0.0
    for x in windows:
        t = teacher(x).logits[:, :-1, :].reshape(-1, V)
        d = student(x).logits[:, :-1, :].reshape(-1, V)
        ai, gi = acceptance_stats(d, t)
        a += ai; g += gi
    n = len(windows)
    return a / n, g / n


def _snapshot(student):
    return [(m.L.detach().clone(), m.R.detach().clone())
            for m in student.modules() if isinstance(m, LowRankLinear)]


def _restore(student, snap):
    for m, (L, R) in zip((m for m in student.modules() if isinstance(m, LowRankLinear)), snap):
        m.L.data.copy_(L); m.R.data.copy_(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--rank-div", type=int, default=6)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--n-accept", type=int, default=8)
    ap.add_argument("--ppl-limit", type=int, default=30000)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--cost-ratio", type=float, default=0.3)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = "cuda", torch.bfloat16
    print(f"device={device} dtype={dtype} model={args.model} rank=d/{args.rank_div}")

    def load():
        try:
            return AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    L = args.max_len
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    test_ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
    accept_w = [test_ids[:, i * L:(i + 1) * L].to(device) for i in range(test_ids.size(1) // L)][: args.n_accept]
    ppl_ids = test_ids[:, : args.ppl_limit]
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    tr_ids = tok("\n\n".join(train["text"][:4000]), return_tensors="pt").input_ids
    tr_w = [tr_ids[:, i * L:(i + 1) * L].to(device) for i in range(tr_ids.size(1) // L)]
    train_b, val_b = tr_w[args.n_val:args.n_val + args.n_train], tr_w[:args.n_val]
    calib = tr_ids[:, : 32 * L]
    print(f"  train {len(train_b)}  val {len(val_b)}  accept {len(accept_w)} windows")

    student = load().to(device).eval()                # build factors first (teacher loaded later)
    hess = collect_weight_hessians(student, calib, L, device, n_seq=32)
    hess = {i: h.cpu() for i, h in hess.items()}      # park Hessians on CPU (in*in each is huge)
    torch.cuda.empty_cache()

    # analytical factors per layer (Wq frozen + LQER/SVD L,R), truncated to d/rank_div
    print("  building analytical low-rank factors...", end=" ", flush=True)
    t0 = time.time()
    linears = [(name, m) for name, m in student.named_modules() if _is_linear(m)]
    factors = {}
    for i, (name, m) in enumerate(linears):
        W = m.weight.data
        if i not in hess or W.shape[-1] % 16:
            continue
        r = max(16, (W.shape[1] // args.rank_div) // 16 * 16)
        Wq, Lf, Rf, _ = gptq_lowrank_factors(W.float(), hess[i].to(W.device), block=16, max_rank=r)
        factors[name] = (Wq.to(dtype), Lf.to(dtype), Rf.to(dtype))
        m.weight.data = W.new_zeros(1)               # free original (LowRankLinear replaces it)
        del Wq, Lf, Rf
    hess.clear()
    torch.cuda.empty_cache()
    attach_lowrank(student, factors, dtype=dtype)
    factors.clear()
    torch.cuda.empty_cache()
    teacher = load().to(device).eval()               # load teacher now that build is done
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"done ({len(linears)} linears, {time.time() - t0:.1f}s)", flush=True)

    out = {"model": args.model, "rank_div": args.rank_div, "gamma": args.gamma,
           "cost_ratio": args.cost_ratio}

    def measure(tag):
        a, g = _acceptance(student, teacher, accept_w)
        ppl = perplexity(student, ppl_ids, L, L, device)
        spd = speedup(a, args.gamma, args.cost_ratio)
        out[tag] = {"alpha": round(a, 4), "greedy": round(g, 4), "ppl": round(ppl, 4),
                    "speedup": round(spd, 3)}
        print(f"  [{tag}] alpha={a:.4f} greedy={g:.4f} ppl={ppl:.4f} speedup={spd:.3f}x", flush=True)

    # --- analytical baseline ---
    measure("analytical")
    analytical = _snapshot(student)

    # --- train the factors against the FP16 teacher (monotone-safe) ---
    print("  distilling factors (KL vs FP16 teacher)...", flush=True)
    distill_factors(student, teacher, train_b, lr=args.lr, epochs=max(1, args.steps // len(train_b)),
                    grad_clip=1.0, val_batches=val_b, eval_every=10, log_every=20)
    measure("trained")

    da = out["trained"]["alpha"] - out["analytical"]["alpha"]
    dp = out["analytical"]["ppl"] - out["trained"]["ppl"]
    out["delta_alpha_trained_minus_analytical"] = round(da, 4)
    out["delta_ppl_analytical_minus_trained"] = round(dp, 4)
    verdict = ("BEAT — trained > analytical (novel: better acceptance-per-byte)"
               if da > 0.002 else "TIE/NULL — trained ~= analytical (Path 1 closed)")
    out["verdict"] = verdict
    print(f"\n  delta alpha (trained-analytical) = {da:+.4f}")
    print(f"  delta ppl   (analytical-trained) = {dp:+.4f}")
    print(f"  VERDICT: {verdict}")

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"weight_distill_{args.model.replace('/', '_')}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
