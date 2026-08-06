r"""Head-to-head: rotation-aware GPTQ vs equalization + additive low-rank (ours).

Until now our claim that "rotation + GPTQ is the weaker design" rested on the
two-lever theorem plus older activation-side measurements. This measures the
competing method DIRECTLY, on the same model, same calibration, same eval, in one
table — including the combination (rotation AND our correction) to check whether
rotation adds anything ON TOP of our levers.

Methods compared (all W4, NVFP4 block 16, same Hessians):
  rtn           plain round-to-nearest                     (floor)
  gptq          GPTQ coordinated rounding                  (their leg 1)
  rot_rtn       rotation only, RTN in the rotated basis    (their leg 2)
  rot_gptq      ROTATION-AWARE GPTQ                        (their full method)
  ours          equalization + additive low-rank           (our method)
  ours_rot      rotation + our method                      (does rotation ADD anything?)

Why rotation is free at inference (so this is a fair fight, not a strawman): for
orthogonal Q, (XQ)(QᵀW) = XW, so Q folds into the neighbouring layer offline.
Rotation costs nothing at runtime — unlike our low-rank correction, which needs a
side matmul. If rotation won on accuracy it would be the better engineering choice.

NOTE ON CONDITIONING: QᵀHQ has the SAME eigenvalues as H, so rotation cannot improve
Hessian conditioning — that theoretical leg of the rotation story is dead on arrival
(verified numerically: cond 29.4 -> 29.4). Rotation can only redistribute energy
across the basis (incoherence), which is exactly what equalization already does.

Run:  python rotation_headtohead.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from turboquant.config import TurboQuantConfig
from turboquant.gptq import (gptq_quantize_weight, lowrank_corrected_weight,
                             rotated_gptq_weight)
from turboquant.validation.hf_perplexity import (
    _is_linear, calibrate_eq_scales, collect_weight_hessians, install_hooks,
    perplexity,
)

RESULTS = Path(__file__).resolve().parent / "results"


def quantize_all(model, hessians, method, block=16, rank_div=16, seed=0):
    """Apply one weight-quantization method to every linear layer."""
    from turboquant.nvfp4 import nvfp4_quantize
    n = 0
    t0 = time.time()
    for i, m in enumerate(mod for mod in model.modules() if _is_linear(mod)):
        W = m.weight.data
        if W.shape[-1] % block != 0 or i not in hessians:
            continue
        Wf, H = W.float(), hessians[i].to(W.device).float()
        if method == "rtn":
            Wq = nvfp4_quantize(Wf, block=block)
        elif method == "gptq":
            Wq = gptq_quantize_weight(Wf, H, block)
        elif method == "rot_rtn":
            Wq = rotated_gptq_weight(Wf, H, block, seed=seed, use_gptq=False)
        elif method == "rot_gptq":
            Wq = rotated_gptq_weight(Wf, H, block, seed=seed, use_gptq=True)
        elif method == "ours":
            Wq = lowrank_corrected_weight(Wf, H, block=block, rank_div=rank_div,
                                          use_gptq=False, fp8_factors=True)
        elif method == "ours_rot":
            # our correction computed in the rotated basis, then rotated back
            from turboquant.rotation import rotate, unrotate
            Wr = rotate(Wf, seed=seed)
            Hr = rotate(rotate(H, seed=seed).t(), seed=seed).t()
            Wq = unrotate(lowrank_corrected_weight(Wr, 0.5 * (Hr + Hr.t()), block=block,
                                                   rank_div=rank_div, use_gptq=False,
                                                   fp8_factors=True), seed=seed)
        else:
            raise ValueError(method)
        m.weight.data = Wq.to(W.dtype)
        n += 1
    return n, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=200000)
    ap.add_argument("--calib-windows", type=int, default=32)
    ap.add_argument("--rank-div", type=int, default=16)
    ap.add_argument("--acts", default="fp8", choices=["fp8", "none"],
                    help="activation path: fp8 (the W4A8 config) or none (W4A16, isolates weights)")
    ap.add_argument("--methods", nargs="+",
                    default=["rtn", "gptq", "rot_rtn", "rot_gptq", "ours", "ours_rot"])
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} model={args.model} acts={args.acts} rank=in/{args.rank_div}",
          flush=True)

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
    calib = tok("\n\n".join(train["text"][:4000]), return_tensors="pt"
                ).input_ids[:, : args.calib_windows * L]
    print(f"eval tokens: {ids.size(1)}", flush=True)

    # Hessians + eq scales computed ONCE on the clean model, shared by every method
    # (so no method gets a calibration advantage).
    base = load().to(device).eval()
    hess = collect_weight_hessians(base, calib, L, device, n_seq=args.calib_windows)
    eq = calibrate_eq_scales(base, calib, L, device)
    fp16 = perplexity(base, ids, L, L, device)
    print(f"  {'fp16':12s} PPL = {fp16:8.4f}", flush=True)
    del base
    torch.cuda.empty_cache()

    cfg = TurboQuantConfig(mx_block=16)
    results = {"fp16": {"ppl": fp16}}
    for method in args.methods:
        model = load().to(device).eval()
        n, tq = quantize_all(model, hess, method, rank_div=args.rank_div)
        handles = []
        if args.acts == "fp8":
            # same activation path for every method: equalize + fp8 (the W4A8 config)
            handles = install_hooks(model, "w4a8", cfg, eq)
        t0 = time.time()
        ppl = perplexity(model, ids, L, L, device)
        results[method] = {"ppl": ppl, "quant_s": round(tq, 1),
                           "eval_s": round(time.time() - t0, 1), "layers": n}
        print(f"  {method:12s} PPL = {ppl:8.4f}   (quant {tq:5.1f}s, {n} layers)",
              flush=True)
        for h in handles:
            h.remove()
        del model
        torch.cuda.empty_cache()

    # verdict
    got = {k: v["ppl"] for k, v in results.items() if k != "fp16"}
    best = min(got, key=got.get)
    summary = {"best_method": best, "best_ppl": round(got[best], 4)}
    if "ours" in got and "rot_gptq" in got:
        summary["ours_minus_rot_gptq"] = round(got["ours"] - got["rot_gptq"], 4)
        summary["ours_beats_rot_gptq"] = bool(got["ours"] < got["rot_gptq"])
    if "ours" in got and "ours_rot" in got:
        summary["rotation_adds_to_ours"] = round(got["ours"] - got["ours_rot"], 4)
    if "gptq" in got and "rot_gptq" in got:
        summary["rotation_adds_to_gptq"] = round(got["gptq"] - got["rot_gptq"], 4)
    print("\n  " + json.dumps(summary, indent=2).replace("\n", "\n  "), flush=True)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"rotation_headtohead_{args.model.replace('/', '_')}_{args.acts}.json"
    out.write_text(json.dumps({"model": args.model, "acts": args.acts,
                               "rank_div": args.rank_div, "results": results,
                               "summary": summary}, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
