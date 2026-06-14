"""The "secret sauce" run: train the free low-rank factors (KL-distill vs the
FP16 teacher), fold them into W4 weights, then evaluate the full W4A4KV4 student's
PPL and its speculative-decoding ACCEPTANCE against the teacher.

Loss-optimal factors instead of SVD/MSE-optimal — the one lever the two-lever
theorem leaves open. Pareto-clean: the deployed model is byte-identical (same Wq,
same-shape fp8 factors), only filled better. Reuses the validated hf_perplexity
pieces (eq scales, Hessians, A4/KV4 hooks, perplexity). One H100, Llama-8B, ~$5.

  python -m turboquant.validation.distill_accept --model unsloth/Meta-Llama-3.1-8B \
      --rank-div 6 --kv4 --epochs 1 --lr 2e-3
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from turboquant.config import TurboQuantConfig
from turboquant.gptq import gptq_lowrank_factors, _quant_fp8_rows
from turboquant.distill import (
    LowRankLinear, attach_lowrank, distill_factors,
)
from turboquant.validation.acceptance import (
    acceptance_stats, expected_accepted_length, speedup,
)
from turboquant.validation import hf_perplexity as H


def _device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def _load(model_name, dtype):
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)


def _make_factors(model, hessians, rank_div, block):
    """Per-layer (Wq, L, R) with SVD warm-start (non-zero -> no saddle).

    MUST index hessians by the same enumeration collect_weight_hessians uses:
    position among ALL _is_linear modules (named_modules traverses in the same
    order as modules), else factors silently pair with the wrong Hessian.
    """
    factors = {}
    for i, (name, m) in enumerate(
            (n, mod) for n, mod in model.named_modules() if H._is_linear(mod)):
        if i not in hessians:
            continue
        W = m.weight.data.float()
        inn = W.shape[1]
        if inn % block != 0:
            continue
        r = max(block, (inn // rank_div) // block * block)
        Wq, L, R, _ = gptq_lowrank_factors(W, hessians[i].to(W.device), block, max_rank=r)
        # store on CPU — attach_lowrank moves them to GPU per-layer; keeping every
        # layer's (Wq,L,R) resident is a second full-model copy and OOMs the 8B.
        factors[name] = (Wq.cpu(), L[:, :r].cpu(), R[:r, :].cpu())
        del Wq, L, R
    return factors


@torch.no_grad()
def _fold_back(model, block):
    """Replace each trained LowRankLinear with a plain nn.Linear whose weight is
    Wq + fp8(L) @ fp8(R) — Pareto-clean (fp8 factors), standard module so the
    existing A4/KV4 eval hooks work unchanged."""
    name_to_mod = dict(model.named_modules())
    folded = 0
    for name, mod in list(name_to_mod.items()):
        if not isinstance(mod, LowRankLinear):
            continue
        L = _quant_fp8_rows(mod.L.data.float())
        R = _quant_fp8_rows(mod.R.data.float())
        W = (mod.Wq.float() + L @ R)
        lin = torch.nn.Linear(W.shape[1], W.shape[0],
                              bias=mod.bias is not None).to(W.device)
        lin.weight.data = W.to(mod.Wq.dtype)
        if mod.bias is not None:
            lin.bias.data = mod.bias.to(mod.Wq.dtype)
        parent_name, _, child = name.rpartition(".")
        parent = name_to_mod[parent_name] if parent_name else model
        setattr(parent, child, lin)
        folded += 1
    return folded


@torch.no_grad()
def collect_acceptance(student, teacher, ids, max_len, device, max_tokens):
    """Windowed acceptance (alpha = 1-TV, greedy) over the first max_tokens, so the
    huge (T x vocab) logits never materialize at once."""
    asum = gsum = ntok = 0.0
    pos = 0
    while pos < ids.size(1) - 1 and ntok < max_tokens:
        end = min(pos + max_len, ids.size(1))
        win = ids[:, pos:end].to(device)
        s = student(win).logits[:, :-1].reshape(-1, student.config.vocab_size)
        t = teacher(win).logits[:, :-1].reshape(-1, teacher.config.vocab_size)
        a, g = acceptance_stats(s, t)
        n = s.size(0)
        asum += a * n; gsum += g * n; ntok += n
        pos = end
    return asum / max(ntok, 1), gsum / max(ntok, 1), int(ntok)


def _eval_student(load_fn, dtype, device, ids, factors, distill_cfg, cfg,
                  eq_scales, max_len, stride, kv4, accept_tokens, teacher, label):
    """Fold/eval a student. If distill_cfg is given, train factors first.

    ``load_fn()`` returns a fresh model on the device (decoupled from any HF name
    so the pipeline is testable offline)."""
    student = load_fn()
    attach_lowrank(student, factors, dtype=dtype)
    if distill_cfg is not None:
        # distill in bf16 (full fp32 range -> no fp16 overflow/NaN from the large
        # SVD singular values) + gradient checkpointing (fits the backward graph).
        student = student.to(torch.bfloat16)
        student.config.use_cache = False
        try:
            student.gradient_checkpointing_enable()
        except Exception:
            pass
        student.train()
        batches = distill_cfg["batches"]
        hist = distill_factors(student, teacher, batches,
                               lr=distill_cfg["lr"], epochs=distill_cfg["epochs"],
                               temperature=distill_cfg["temperature"],
                               log_every=distill_cfg["log_every"],
                               val_batches=distill_cfg.get("val_batches"),
                               eval_every=distill_cfg.get("eval_every", 0))
        student.eval()
        try:
            student.gradient_checkpointing_disable()
        except Exception:
            pass
        student.config.use_cache = True
        student = student.to(dtype)
        print(f"  [{label}] distill KL {hist[0]:.4f} -> {min(hist):.4f} "
              f"(best; {len(hist)} steps)", flush=True)
    n = _fold_back(student, cfg.mx_block)
    handles = H.install_hooks(student, "nvfp4_eqzp_svd_qjl", cfg, eq_scales)
    if kv4:
        handles += H.install_kv_hooks(student, cfg.mx_block)
    ppl = H.perplexity(student, ids, max_len, stride, device)
    a, g, nt = collect_acceptance(student, teacher, ids, max_len, device, accept_tokens)
    for h in handles:
        h.remove()
    del student
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"ppl": round(ppl, 4), "alpha": round(a, 4), "greedy": round(g, 4),
            "accept_tokens": nt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B")
    ap.add_argument("--rank-div", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--calib-seqs", type=int, default=64,
                    help="number of calib windows for distillation")
    ap.add_argument("--accept-tokens", type=int, default=8192)
    ap.add_argument("--kv4", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gamma", type=int, default=4, help="draft length for speedup")
    ap.add_argument("--cost-ratio", type=float, default=0.1,
                    help="draft/target per-step cost for the speedup formula")
    ap.add_argument("--skip-svd", action="store_true",
                    help="skip the SVD baseline eval (already known) -> only distilled")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    device, dtype = _device_dtype()
    print(f"device={device} dtype={dtype} model={args.model}", flush=True)
    cfg = TurboQuantConfig(qjl_block=128, qjl_dim=64, use_polarquant=False)

    tok = AutoTokenizer.from_pretrained(args.model)
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
    if args.limit:
        ids = ids[:, : args.limit]
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ctext = "\n\n".join(train["text"][:6000])
    calib = tok(ctext, return_tensors="pt").input_ids[:, : (args.calib_seqs + 32) * args.max_len]
    print(f"test tokens: {ids.size(1)}", flush=True)

    # offline calibration (eq scales + weight Hessians) on a quantized copy
    qmodel = _load(args.model, dtype).to(device).eval()
    eq_scales = H.calibrate_eq_scales(qmodel, calib, args.max_len, device)
    hessians = H.collect_weight_hessians(qmodel, calib, args.max_len, device, n_seq=32)
    hessians = {k: v.cpu() for k, v in hessians.items()}   # free ~30GB GPU (8B)
    if device == "cuda":
        torch.cuda.empty_cache()
    print("  computing SVD warm-start factors...", flush=True)
    factors = _make_factors(qmodel, hessians, args.rank_div, cfg.mx_block)
    del qmodel
    if device == "cuda":
        torch.cuda.empty_cache()

    # the FP16 teacher (the target the draft verifies against)
    teacher = _load(args.model, dtype).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # distillation calibration batches (token-id windows); hold out the last 4
    # windows for keep-best validation (monotone-safe distillation)
    cl = calib[:, : args.calib_seqs * args.max_len].reshape(-1, args.max_len)
    n_val = min(4, max(1, cl.size(0) // 8))
    train_w = [cl[j:j + 1].to(device) for j in range(cl.size(0) - n_val)]
    val_w = [cl[j:j + 1].to(device) for j in range(cl.size(0) - n_val, cl.size(0))]

    load_fn = lambda: _load(args.model, dtype).to(device).eval()
    results = {}
    t0 = time.time()
    # baseline: SVD factors, no training (= the current W4A4KV4)
    if not args.skip_svd:
        results["svd"] = _eval_student(
            load_fn, dtype, device, ids, factors, None, cfg, eq_scales,
            args.max_len, args.stride, args.kv4, args.accept_tokens, teacher, "svd")
        print(f"  SVD baseline: {results['svd']}", flush=True)

    # trained factors (the secret sauce); keep-best on held-out val
    dcfg = {"batches": train_w, "lr": args.lr, "epochs": args.epochs,
            "temperature": args.temperature, "log_every": 50,
            "val_batches": val_w, "eval_every": max(1, len(train_w) // 2)}
    results["distilled"] = _eval_student(
        load_fn, dtype, device, ids, factors, dcfg, cfg, eq_scales,
        args.max_len, args.stride, args.kv4, args.accept_tokens, teacher, "distilled")
    print(f"  Distilled:    {results['distilled']}", flush=True)

    # acceptance -> expected accepted length + speedup (iid block model)
    for k in [k for k in ("svd", "distilled") if k in results]:
        a = results[k]["alpha"]
        results[k]["exp_len"] = round(expected_accepted_length(a, args.gamma), 3)
        results[k]["speedup"] = round(speedup(a, args.gamma, args.cost_ratio), 3)

    results["_meta"] = {"seconds": round(time.time() - t0, 1),
                        "rank_div": args.rank_div, "epochs": args.epochs,
                        "lr": args.lr, "kv4": args.kv4, "gamma": args.gamma}
    H.RESULTS.mkdir(exist_ok=True)
    out = H.RESULTS / f"distill_accept_{args.model.replace('/', '_')}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n==== RESULT ====\n{json.dumps(results, indent=2)}\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
