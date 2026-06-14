"""Real FP4-vs-FP8-vs-bf16 decode throughput on Blackwell, via vLLM.

This is the ONE thing our fake-quant research code can't give: actual tokens/sec
on real FP4 tensor cores. It benchmarks NVIDIA's *stock* NVFP4 path (a ModelOpt
checkpoint through vLLM) — NOT our method — to measure the FP4-vs-FP8 hardware
speedup, i.e. the `cost_ratio` we had to assume (0.1) in the 2.86x projection.
Combined with our REAL acceptance (alpha=0.889), it grounds the speedup number.

Caveat: this is the BASE format's ceiling. Our method adds a ~<12% rank-r + QJL
epilogue, so our deployed draft lands ~10-15% under these numbers.

Run on a B200 (needs a vLLM image / `pip install vllm`):
  python -m turboquant.validation.vllm_fp4_bench --model <ckpt> [--quantization fp8] \
      --batch 1 --in-len 128 --out-len 256 --iters 3
Decode-dominated (out_len >> in_len) so the number reflects per-token decode speed.
"""

from __future__ import annotations

import argparse
import json
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF id or path (bf16 base, or a ModelOpt FP8/FP4 checkpoint)")
    ap.add_argument("--quantization", default=None,
                    help="vLLM quantization arg (e.g. fp8, modelopt, modelopt_fp4); "
                         "omit to let vLLM auto-detect from a quantized checkpoint")
    ap.add_argument("--label", default=None, help="row label for the results table")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--in-len", type=int, default=128)
    ap.add_argument("--out-len", type=int, default=256)
    ap.add_argument("--iters", type=int, default=3, help="timed iters (after 1 warmup)")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--out", default="results/vllm_bench.jsonl")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(model=args.model, max_model_len=args.max_model_len,
                      enforce_eager=False, gpu_memory_utilization=0.9)
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)

    # fixed-length decode: ignore EOS so every request emits exactly out_len tokens
    sp = SamplingParams(temperature=0.0, max_tokens=args.out_len, ignore_eos=True)
    # deterministic synthetic prompt of in_len tokens (token id 100 repeated)
    prompt_ids = [100] * args.in_len
    prompts = [{"prompt_token_ids": prompt_ids} for _ in range(args.batch)]

    def run_once():
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        return dt, gen

    run_once()                                   # warmup (compile/caches)
    times, toks = [], 0
    for _ in range(args.iters):
        dt, gen = run_once()
        times.append(dt)
        toks = gen
    avg = sum(times) / len(times)
    tok_per_s = toks / avg                        # output tokens/sec (decode-dominated)

    label = args.label or (args.quantization or "base")
    row = {"label": label, "model": args.model, "quantization": args.quantization,
           "batch": args.batch, "in_len": args.in_len, "out_len": args.out_len,
           "avg_sec": round(avg, 4), "out_tokens": toks,
           "tokens_per_sec": round(tok_per_s, 1),
           "per_request_tok_s": round(tok_per_s / args.batch, 1)}
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\n==== {label} ====\n{json.dumps(row, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
