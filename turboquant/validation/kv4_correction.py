"""KV4 correction lever test: does correcting the KV cache beat today's raw-rounded KV4?

Today `install_kv_hooks` does plain NVFP4 (optclip+zp) on K/V with ZERO correction,
while activations get equalization + SVD/QJL side-channels. The gap decomposition flagged
KV4 as "the untapped lever". This script tests the additive low-rank residual lever (the
one the two-lever theorem leaves open) ON the KV projections:

  out_corrected = Q(out) + (R @ Vr) @ Vr.T ,   R = out - Q(out)

Vr = top-r right singular vectors of the calibration residual (precomputed per k/v_proj).
Runtime cost = one (d x r) projection per token = a small side channel.

Quick, TinyLlama, zero extra disk (model already cached). Reports FP16 / KV4-raw /
KV4+lowrank(r) PPL so we see if the lever is real BEFORE spending 8B disk+time on it.

  python -m turboquant.validation.kv4_correction --ranks 16 32 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from turboquant.nvfp4 import nvfp4_quantize_zp
from turboquant.validation.hf_perplexity import _is_linear, perplexity

RESULTS = Path(__file__).resolve().parents[2] / "results"


def _kv_modules(model):
    return [(n, m) for n, m in model.named_modules()
            if n.endswith(("k_proj", "v_proj")) and _is_linear(m)]


@torch.no_grad()
def calib_residual_basis(model, calib_ids, max_len, device, block, ranks, n_windows):
    """Per k/v_proj: accumulate residual covariance R^T R over calibration, return top-r
    right vectors for the largest r in `ranks` (smaller r = slicing this)."""
    mods = _kv_modules(model)
    cov = {n: None for n, _ in mods}                       # (d, d) per module

    handles = []
    for name, m in mods:
        def mk(nm):
            def hook(module, inp, output):
                if not torch.is_tensor(output) or output.shape[-1] % block:
                    return None
                o = output.float()
                q = nvfp4_quantize_zp(o, block=block, optclip=True)
                r = (o - q).reshape(-1, o.shape[-1])       # (T, d)
                c = r.t() @ r                              # (d, d)
                cov[nm] = c if cov[nm] is None else cov[nm] + c
                return None
            return hook
        handles.append(m.register_forward_hook(mk(name)))

    L = max_len
    wins = [calib_ids[:, i * L:(i + 1) * L].to(device)
            for i in range(calib_ids.size(1) // L)][:n_windows]
    for x in wins:
        model(x)
    for h in handles:
        h.remove()

    rmax = max(ranks)
    basis = {}
    for n, c in cov.items():
        evals, evecs = torch.linalg.eigh(c.double())       # ascending
        basis[n] = evecs[:, -rmax:].flip(1).to(device)     # (d, rmax), desc
    return basis


def install_kv4(model, block, basis=None, rank=0):
    """KV4 hooks. basis=None -> raw KV4 (current). Else add low-rank residual correction."""
    handles = []
    for name, m in _kv_modules(model):
        Vr = None if basis is None else basis[name][:, :rank]   # (d, r)
        def mk(V):
            def hook(module, inp, output):
                if not torch.is_tensor(output) or output.shape[-1] % block:
                    return None
                o = output.float()
                q = nvfp4_quantize_zp(o, block=block, optclip=True)
                if V is not None:
                    r = o - q
                    q = q + (r @ V) @ V.t()                  # additive low-rank residual
                return q.to(output.dtype)
            return hook
        handles.append(m.register_forward_hook(mk(Vr)))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--calib-windows", type=int, default=32)
    ap.add_argument("--ppl-limit", type=int, default=40000)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = "cuda", torch.float16
    tok = AutoTokenizer.from_pretrained(args.model)
    L = args.max_len
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ppl_ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[:, :args.ppl_limit]
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    calib_ids = tok("\n\n".join(train["text"]), return_tensors="pt").input_ids

    def load():
        try:
            return AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    model = load().to(device).eval()
    out = {"model": args.model, "block": args.block}

    fp16 = round(perplexity(model, ppl_ids, L, L, device), 4)
    out["fp16"] = fp16
    print(f"  FP16            ppl = {fp16}", flush=True)

    h = install_kv4(model, args.block)                     # raw KV4
    raw = round(perplexity(model, ppl_ids, L, L, device), 4)
    for x in h:
        x.remove()
    out["kv4_raw"] = raw
    print(f"  KV4 raw         ppl = {raw}  (+{raw - fp16:.4f} vs FP16)", flush=True)

    basis = calib_residual_basis(model, calib_ids, L, device, args.block,
                                 args.ranks, args.calib_windows)
    out["kv4_lowrank"] = {}
    for r in args.ranks:
        h = install_kv4(model, args.block, basis=basis, rank=r)
        ppl = round(perplexity(model, ppl_ids, L, L, device), 4)
        for x in h:
            x.remove()
        out["kv4_lowrank"][str(r)] = ppl
        rec = raw - ppl
        print(f"  KV4 + lowrank r={r:<3d} ppl = {ppl}  (recovers {rec:+.4f} of the KV4 leg)",
              flush=True)

    best_r = min(out["kv4_lowrank"], key=lambda k: out["kv4_lowrank"][k])
    best = out["kv4_lowrank"][best_r]
    out["verdict"] = ("LEVER REAL — correction helps KV4" if (raw - best) > 0.01
                      else "no help — KV4 is already near its floor")
    print(f"\n  best: r={best_r} ppl={best}  raw={raw} fp16={fp16}")
    print(f"  VERDICT: {out['verdict']}")
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"kv4_correction_{args.model.split('/')[-1]}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
