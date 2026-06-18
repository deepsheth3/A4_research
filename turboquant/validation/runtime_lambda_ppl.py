"""Experiment C, end-to-end: does runtime-adaptive lambda HOLD perplexity?

The Door C claim (see DOOR_C.md): warm-started runtime-adaptive equalization
matches frozen calibrated lambda IN-DISTRIBUTION (by construction) while holding up
OUT-OF-DISTRIBUTION, where the frozen scales — calibrated on WikiText — drift.

This scores frozen-lambda vs adaptive-lambda perplexity on two corpora:
  in-dist : WikiText-2 test (the calibration domain)
  ood     : code (clearly off the calibration distribution)

Expected: indist delta ~= 0 (warm start makes token 0 identical); ood frozen drifts
up while adaptive holds — that gap is the contribution. Everything except lambda is
held fixed: eq -> zp+optclip -> W-SVD side-channel (anchored at warm-start) -> QJL.
The same per-position causal-decaying-max scale as runtime_lambda_accept.py.

A4-only by default (isolates the lambda lever); --w4 / --kv4 compose the rest of the
4-bit-everything stack to confirm the claim holds for the deployed configuration.

Run (H100):
    python -m turboquant.validation.runtime_lambda_ppl \
        --model unsloth/Meta-Llama-3.1-8B --w4 --kv4 --decay 0.98
    # point --ood-file at a real code/math corpus for the headline run
Smoke (Mac, tiny):
    python -m turboquant.validation.runtime_lambda_ppl --model gpt2 --limit 2000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from turboquant.config import TurboQuantConfig
from turboquant.act_codec import TurboQuantActQuantizer
from turboquant.validation.hf_perplexity import (
    _is_linear, _precompute_aux, perplexity, quantize_weights_nvfp4, install_kv_hooks,
)
from turboquant.validation.runtime_lambda_accept import calibrate_eq_full, _make_hook

RESULTS = Path(__file__).resolve().parents[2] / "results"

# Default OOD corpus (code) — clearly off the WikiText calibration distribution.
# For the headline H100 run pass --ood-file pointing at a larger code/math corpus.
_OOD_CODE = '''\
import math
from dataclasses import dataclass


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self):
        return math.sqrt(self.dot(self))

    def normalized(self):
        n = self.norm()
        return Vec3(self.x / n, self.y / n, self.z / n)


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)


def binary_search(sorted_arr, target):
    lo, hi = 0, len(sorted_arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_arr[mid] == target:
            return mid
        if sorted_arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.store = {}
        self.order = []

    def get(self, key):
        if key not in self.store:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.store[key]

    def put(self, key, value):
        if key in self.store:
            self.order.remove(key)
        elif len(self.store) >= self.capacity:
            oldest = self.order.pop(0)
            del self.store[oldest]
        self.store[key] = value
        self.order.append(key)


def matmul(a, b):
    n, k, m = len(a), len(b), len(b[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for p in range(k):
            aip = a[i][p]
            for j in range(m):
                out[i][j] += aip * b[p][j]
    return out
'''


@torch.no_grad()
def _build_cache(model, frozen, warm, alpha, device):
    """Per-layer {frozen, warm, alpha, basis}; basis = W-SVD anchored at frozen eq."""
    aux = _precompute_aux(model, 16, device, need_basis=True, need_comp=False,
                          eq_scales=frozen)
    cache = {}
    for i in frozen:
        if i in aux and "basis" in aux[i]:
            cache[i] = {"frozen": frozen[i].to(device), "warm": warm[i].to(device),
                        "alpha": alpha[i], "basis": aux[i]["basis"]}
    return cache


@torch.no_grad()
def _ppl_with_lambda(model, ids, max_len, stride, device, cache, codec, decay,
                     adaptive, kv4):
    handles = []
    for i, m in enumerate(mod for mod in model.modules() if _is_linear(mod)):
        if i in cache:
            handles.append(m.register_forward_pre_hook(
                _make_hook(i, codec, 16, cache[i], adaptive, decay)))
    if kv4:
        handles += install_kv_hooks(model, 16)
    try:
        return perplexity(model, ids, max_len, stride, device)
    finally:
        for h in handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="cap in-dist test tokens (smoke)")
    ap.add_argument("--ood-file", default=None, help="path to OOD corpus; default embedded code")
    ap.add_argument("--decay", type=float, default=0.98)
    ap.add_argument("--qjl-block", type=int, default=128)
    ap.add_argument("--qjl-dim", type=int, default=64)
    ap.add_argument("--w4", action="store_true", help="also NVFP4-quantize weights")
    ap.add_argument("--kv4", action="store_true", help="also NVFP4-quantize KV cache")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32
    print(f"device={device} dtype={dtype} model={args.model} "
          f"decay={args.decay} w4={args.w4} kv4={args.kv4}")

    tok = AutoTokenizer.from_pretrained(args.model)
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    indist = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
    if args.limit:
        indist = indist[:, : args.limit]
    ood_text = Path(args.ood_file).read_text() if args.ood_file else _OOD_CODE
    ood = tok(ood_text, return_tensors="pt").input_ids
    print(f"in-dist tokens: {indist.size(1)}   ood tokens: {ood.size(1)}")

    def load_model():
        try:
            return AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    model = load_model().to(device).eval()

    # eq calibration on the in-dist (WikiText) train split — frozen scale, warm
    # amax, per-layer alpha (the only calibration-dependent step).
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    calib_ids = tok("\n\n".join(train["text"][:2000]),
                    return_tensors="pt").input_ids[:, : 4 * args.max_len]
    frozen, warm, alpha = calibrate_eq_full(model, calib_ids, args.max_len, device)

    corpora = {"indist_wikitext": indist, "ood_code": ood}
    out = {"model": args.model, "config": vars(args), "fp16": {}, "frozen": {}, "adaptive": {}}

    # fp16 references (no hooks) before any weight mutation.
    for name, ids in corpora.items():
        out["fp16"][name] = round(perplexity(model, ids, args.max_len, args.stride, device), 4)
        print(f"  fp16  {name:18s} PPL = {out['fp16'][name]:.4f}")

    if args.w4:
        quantize_weights_nvfp4(model, 16)        # basis below is then built on quantized W

    cache = _build_cache(model, frozen, warm, alpha, device)
    print(f"  active layers (eq+basis): {len(cache)}")
    codec = TurboQuantActQuantizer(TurboQuantConfig(qjl_block=args.qjl_block,
                                                    qjl_dim=args.qjl_dim))

    for regime, adaptive in (("frozen", False), ("adaptive", True)):
        for name, ids in corpora.items():
            t0 = time.time()
            ppl = _ppl_with_lambda(model, ids, args.max_len, args.stride, device,
                                   cache, codec, args.decay, adaptive, args.kv4)
            out[regime][name] = round(ppl, 4)
            print(f"  {regime:8s} {name:18s} PPL = {ppl:.4f}   ({time.time() - t0:.1f}s)")

    # the claim: indist delta ~0, ood frozen drift > adaptive drift
    for name in corpora:
        f16 = out["fp16"][name]
        df = out["frozen"][name] - f16
        da = out["adaptive"][name] - f16
        print(f"\n{name}: fp16 {f16:.4f} | frozen +{df:.4f} | adaptive +{da:.4f} "
              f"| adaptive-vs-frozen {out['adaptive'][name] - out['frozen'][name]:+.4f}")
    out["claim"] = {
        "indist_delta_adaptive_minus_frozen":
            round(out["adaptive"]["indist_wikitext"] - out["frozen"]["indist_wikitext"], 4),
        "ood_delta_adaptive_minus_frozen":
            round(out["adaptive"]["ood_code"] - out["frozen"]["ood_code"], 4),
    }
    print(f"\nCLAIM CHECK: in-dist should be ~0 "
          f"({out['claim']['indist_delta_adaptive_minus_frozen']:+.4f}); "
          f"ood should be negative = adaptive holds "
          f"({out['claim']['ood_delta_adaptive_minus_frozen']:+.4f})")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"runtime_lambda_ppl_{args.model.replace('/', '_')}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
