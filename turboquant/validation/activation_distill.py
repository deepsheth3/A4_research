"""Part 2: train the activation low-rank correction against LOSS, not MSE.

The two-lever law says the additive low-rank channel is the only place injected
information can help. Today we fill its basis with SVD (W's top input singular
vectors) — optimal for reconstruction MSE. But the deployment objective is the
LOSS. So: make the per-layer correction basis a trainable parameter, warm-start it
from the SVD basis (no zero-init saddle), freeze everything else, and train ONLY
the bases by KL to the FP16 teacher. Same rank, same bytes, same fp8 coeffs — the
shipped model is byte-identical, the basis is just filled better than SVD can.

THE BET: training against loss beats MSE-optimal SVD iff the post-eq residual is
non-Gaussian enough that output-importance reweighting helps. If trained only
MATCHES SVD, that is the null (and the same null the weight-path distill hit).

Monotone-safe: starts from the SVD-init held-out KL and only keeps improvements,
so a bad lr can never ship worse-than-SVD bases. The kill-line: one run; if
held-out KL never drops below the SVD baseline, stop — that is the answer.

Smoke (CPU/Mac, gpt2):
    python -m turboquant.validation.activation_distill --model gpt2 --steps 20
Run (H100, one shot):
    python -m turboquant.validation.activation_distill \
        --model unsloth/Meta-Llama-3.1-8B --w4 --steps 300 --eval-ppl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from turboquant.nvfp4 import nvfp4_quantize_zp, round_e4m3
from turboquant.distill import kl_distill_loss
from turboquant.validation.hf_perplexity import (
    _is_linear, _precompute_aux, perplexity, quantize_weights_nvfp4, install_kv_hooks,
)
from turboquant.validation.runtime_lambda_accept import calibrate_eq_full
from turboquant.act_codec import TurboQuantActQuantizer
from turboquant.config import TurboQuantConfig

RESULTS = Path(__file__).resolve().parents[2] / "results"


def _correct(x, s, basis, mx_block=16):
    """eq -> NVFP4 base (frozen) -> trainable-basis low-rank correction (fp8 STE).

    Differentiable in ``basis`` only: ``base`` is detached, and the fp8 rounding of
    the coefficients uses a straight-through estimator so the deployed fp8 path is
    simulated while gradients still reach the basis.
    """
    xe = x / s
    base = nvfp4_quantize_zp(xe, block=mx_block, optclip=True).detach()
    coeff = (xe - base) @ basis                              # (..., k)
    cs = coeff.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
    coeff_q = round_e4m3(coeff / cs) * cs
    coeff = coeff + (coeff_q - coeff).detach()               # straight-through fp8
    xh = base + coeff @ basis.T
    return xh * s


@torch.no_grad()
def _correct_full(x, s, basis, codec, layer, mx_block=16):
    """Deployed-stack correction for EVAL: eq -> base -> basis -> per-block QJL."""
    xe = x / s
    base = nvfp4_quantize_zp(xe, block=mx_block, optclip=True)
    coeff = (xe - base) @ basis
    cs = coeff.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
    coeff = round_e4m3(coeff / cs) * cs
    xh = base + coeff @ basis.T
    if codec.cfg.qjl_dim > 0:
        xh = xh + codec.qjl_correct(xe - xh, layer)
    return xh * s


def _install_full(model, bases, eq, codec, mx_block=16):
    """Full-stack (basis + QJL) eval hooks; caller adds KV4 separately."""
    handles = []
    for i, m in enumerate(mod for mod in model.modules() if _is_linear(mod)):
        if i not in bases:
            continue
        def mk(idx):
            def hook(module, args):
                x = args[0]
                if x.shape[-1] % mx_block:
                    return None
                xq = _correct_full(x.float(), eq[idx], bases[idx], codec, idx, mx_block).to(x.dtype)
                return (xq, *args[1:])
            return hook
        handles.append(m.register_forward_pre_hook(mk(i)))
    return handles


def _install(model, bases, eq, mx_block=16):
    """Grad-enabled forward-pre-hooks applying _correct with the trainable bases."""
    handles = []
    for i, m in enumerate(mod for mod in model.modules() if _is_linear(mod)):
        if i not in bases:
            continue
        def mk(idx):
            def hook(module, args):
                x = args[0]
                if x.shape[-1] % mx_block:
                    return None
                xq = _correct(x.float(), eq[idx], bases[idx], mx_block).to(x.dtype)
                return (xq, *args[1:])
            return hook
        handles.append(m.register_forward_pre_hook(mk(i)))
    return handles


@torch.no_grad()
def _val_kl(student, teacher, batches, sh, th):
    tot = 0.0
    for x in batches:
        tot += kl_distill_loss(student(x).logits, teacher(x).logits).item()
    return tot / max(len(batches), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--w4", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--n-train", type=int, default=24, help="train windows")
    ap.add_argument("--n-val", type=int, default=8, help="held-out windows")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-ppl", action="store_true", help="final W4A4 PPL: SVD-init vs trained")
    ap.add_argument("--full-stack", action="store_true", help="eval on full stack: +QJL +KV4")
    ap.add_argument("--cosine", action="store_true", help="cosine lr decay for a stable climb")
    ap.add_argument("--qjl-dim", type=int, default=64)
    ap.add_argument("--eval-len", type=int, default=0, help="PPL eval context (0=max_len)")
    ap.add_argument("--limit", type=int, default=20000)
    args = ap.parse_args()
    args.eval_len = args.eval_len or args.max_len

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float32     # fp32: training stability (fp16 NaNs)
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32
    print(f"device={device} dtype={dtype} model={args.model} w4={args.w4} steps={args.steps}")

    def load():
        try:
            return AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    train_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(train_ds["text"][:3000]), return_tensors="pt").input_ids
    L = args.max_len
    windows = [ids[:, i * L:(i + 1) * L].to(device)
               for i in range((ids.size(1)) // L)][: args.n_train + args.n_val]
    train_b, val_b = windows[args.n_val:], windows[:args.n_val]
    print(f"  train windows {len(train_b)}  val windows {len(val_b)}")

    teacher = load().to(device).eval()                       # FP16, frozen
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = load().to(device).eval()
    if args.w4:
        quantize_weights_nvfp4(student, 16)
    for p in student.parameters():
        p.requires_grad_(False)

    # eq scales (frozen) + SVD-init bases (in eq space) -> trainable params
    frozen, _, _ = calibrate_eq_full(student, ids[:, : 4 * L].to(device), L, device)
    aux = _precompute_aux(student, 16, device, need_basis=True, need_comp=False,
                          eq_scales=frozen)
    eq = {i: frozen[i].to(device) for i in frozen}
    bases = {i: nn.Parameter(aux[i]["basis"].clone().float())
             for i in frozen if i in aux and "basis" in aux[i]}
    svd_init = {i: bases[i].detach().clone() for i in bases}  # for the baseline + PPL
    print(f"  trainable bases: {len(bases)} layers")

    handles = _install(student, bases, eq)
    params = list(bases.values())
    opt = torch.optim.Adam(params, lr=args.lr)

    # baseline = SVD-init held-out KL (monotone-safe floor)
    best = _val_kl(student, teacher, val_b, None, None)
    best_state = {i: p.detach().clone() for i, p in bases.items()}
    print(f"  SVD-init held-out KL = {best:.5f}")

    step = 0
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
             if args.cosine else None)
    t0 = time.time()
    while step < args.steps:
        for x in train_b:
            if step >= args.steps:
                break
            loss = kl_distill_loss(student(x).logits, teacher(x).logits)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            if sched:
                sched.step()
            step += 1
            if step % args.eval_every == 0:
                v = _val_kl(student, teacher, val_b, None, None)
                tag = ""
                if v < best:
                    best, tag = v, "  <- best (kept)"
                    best_state = {i: p.detach().clone() for i, p in bases.items()}
                print(f"  step {step:4d}  train KL {loss.item():.5f}  val KL {v:.5f}{tag}",
                      flush=True)
    for i, p in bases.items():                                # restore best
        p.data.copy_(best_state[i])

    out = {"model": args.model, "w4": args.w4, "steps": args.steps,
           "best_val_kl": round(best, 5), "seconds": round(time.time() - t0, 1)}
    # explicit SVD-init vs trained held-out KL (restore each, measure)
    for i, p in bases.items():
        p.data.copy_(svd_init[i])
    out["svd_init_val_kl"] = round(_val_kl(student, teacher, val_b, None, None), 5)
    for i, p in bases.items():
        p.data.copy_(best_state[i])
    out["trained_val_kl"] = round(_val_kl(student, teacher, val_b, None, None), 5)
    out["kl_improvement"] = round(out["svd_init_val_kl"] - out["trained_val_kl"], 5)
    print(f"\nHELD-OUT KL: SVD-init {out['svd_init_val_kl']:.5f} -> "
          f"trained {out['trained_val_kl']:.5f}  (improvement {out['kl_improvement']:+.5f})")
    verdict = ("LAND — trained beats SVD" if out["kl_improvement"] > 1e-4
               else "NULL — trained ~= SVD (stop, this is the answer)")
    print(f"VERDICT: {verdict}")
    out["verdict"] = verdict

    if args.eval_ppl:
        for h in handles:
            h.remove()
        test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        tids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[:, : args.limit]
        codec = TurboQuantActQuantizer(
            TurboQuantConfig(qjl_dim=args.qjl_dim if args.full_stack else 0))
        stack = "W4A4KV4+QJL" if args.full_stack else "W4A4"
        for name, src in (("svd", svd_init), ("trained", best_state)):
            for i, p in bases.items():
                p.data.copy_(src[i])
            if args.full_stack:
                hs = _install_full(student, bases, eq, codec) + install_kv_hooks(student, 16)
            else:
                hs = _install(student, bases, eq)
            out[f"ppl_{name}"] = round(perplexity(student, tids, args.eval_len, args.eval_len, device), 4)
            for h in hs:
                h.remove()
            print(f"  {stack} PPL ({name}-basis) = {out[f'ppl_{name}']:.4f}")
        out["eval_stack"] = stack
        out["ppl_improvement"] = round(out["ppl_svd"] - out["ppl_trained"], 4)
        print(f"  PPL improvement (svd - trained) = {out['ppl_improvement']:+.4f}  [{stack}]")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"activation_distill_{args.model.replace('/', '_')}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
