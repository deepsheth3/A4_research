"""Equal-byte baseline comparison: PPL and draft acceptance for every config.

Runs the same model/eval through several activation-quantization modes and reports
both perplexity and speculative-decoding acceptance against the FP16 target, so the
paper's central claim (rotations add little after equalization; additive correction
gives the gain) can be read off one table. Modes:

  fp16, fp8, nvfp4_raw            -- references
  nvfp4_ghad                      -- QuaRot-style global Hadamard rotation, no eq
  nvfp4_eqzp                      -- equalization only
  nvfp4_eqzp_ghad                 -- rotation AFTER equalization (the key test)
  nvfp4_eqzp_svd_qjl              -- our full activation codec

External baselines (stock ModelOpt NVFP4 checkpoint) are loaded separately; this
driver covers the in-harness, equal-semantics comparison. One cheap H100.

  python -m turboquant.validation.baseline_compare --model unsloth/Meta-Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from turboquant.config import TurboQuantConfig
from turboquant.validation import hf_perplexity as H
from turboquant.validation.acceptance import (
    acceptance_stats, expected_accepted_length, speedup,
)

MODES = ["fp16", "fp8", "nvfp4_raw", "nvfp4_ghad", "nvfp4_eqzp",
         "nvfp4_eqzp_ghad", "nvfp4_eqzp_svd_qjl"]


def _device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def _load(name, dtype):
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)


@torch.no_grad()
def _acceptance(student, teacher, ids, max_len, device, max_tokens):
    asum = gsum = ntok = 0.0
    pos = 0
    V = student.config.vocab_size
    while pos < ids.size(1) - 1 and ntok < max_tokens:
        end = min(pos + max_len, ids.size(1))
        win = ids[:, pos:end].to(device)
        s = student(win).logits[:, :-1].reshape(-1, V)
        t = teacher(win).logits[:, :-1].reshape(-1, V)
        a, g = acceptance_stats(s, t)
        n = s.size(0)
        asum += a * n; gsum += g * n; ntok += n
        pos = end
    return asum / max(ntok, 1), gsum / max(ntok, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B")
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--accept-tokens", type=int, default=8192)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--cost-ratio", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
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
    print(f"test tokens: {ids.size(1)}", flush=True)

    # equalization scales (shared by all eq modes), calibrated once
    need_eq = any(m.startswith("nvfp4_eq") for m in args.modes)
    eq_scales = None
    if need_eq:
        train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        calib = tok("\n\n".join(train["text"][:2000]),
                    return_tensors="pt").input_ids[:, : 4 * args.max_len]
        m0 = _load(args.model, dtype).to(device).eval()
        eq_scales = H.calibrate_eq_scales(m0, calib, args.max_len, device)
        del m0
        if device == "cuda":
            torch.cuda.empty_cache()

    teacher = _load(args.model, dtype).to(device).eval()

    results = {}
    for mode in args.modes:
        model = _load(args.model, dtype).to(device).eval()
        handles = []
        if mode != "fp16":
            handles = H.install_hooks(model, mode, cfg, eq_scales)
        t0 = time.time()
        ppl = H.perplexity(model, ids, args.max_len, args.stride, device)
        if mode == "fp16":
            alpha, greedy = 1.0, 1.0
        else:
            alpha, greedy = _acceptance(model, teacher, ids, args.max_len, device,
                                        args.accept_tokens)
        row = {"ppl": round(ppl, 4), "alpha": round(alpha, 4),
               "greedy": round(greedy, 4),
               "exp_emitted": round(1 + sum(alpha ** k for k in range(1, args.gamma + 1)), 3),
               "speedup": round(speedup(alpha, args.gamma, args.cost_ratio), 3),
               "sec": round(time.time() - t0, 1)}
        results[mode] = row
        print(f"  {mode:22s} PPL={row['ppl']:8.4f}  alpha={row['alpha']:.4f}  "
              f"greedy={row['greedy']:.4f}  speedup={row['speedup']:.2f}", flush=True)
        for h in handles:
            h.remove()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    H.RESULTS.mkdir(exist_ok=True)
    out = H.RESULTS / f"baseline_compare_{args.model.replace('/', '_')}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
