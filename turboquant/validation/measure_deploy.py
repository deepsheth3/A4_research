"""Measure real FP4-tensor-core throughput on our exported NVFP4 checkpoint.

Dual backend because the one-shot box is sm_120 (RTX Pro 6000): TRT-LLM's
trtllm-gen FP4 attention has no precompiled cubins for sm_120, so it may fail or
fall back there, whereas vLLM's NVFP4 path is verified on sm_120. Try trtllm,
fall back to vllm. NVFP4 (and KV precision if exported) is auto-detected from the
checkpoint config — no flags needed to activate FP4.

Accuracy is NOT measured here — it's the deployed-QAT PPL from
`export_nvfp4 --eval-ppl`, which doesn't depend on the serving runtime.

    python -m turboquant.validation.measure_deploy --model nvfp4_ckpt --backend vllm

Verified APIs: tensorrt_llm.LLM / tensorrt_llm.llmapi.KvCacheConfig; vllm.LLM.
Both share llm.generate(prompts, SamplingParams(max_tokens=...)).
"""
from __future__ import annotations

import argparse
import time


def _load(backend: str, model: str, max_batch: int):
    """Return (llm, SamplingParams-class). Raises if the backend can't load."""
    if backend == "trtllm":
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig
        llm = LLM(model=model, max_batch_size=max_batch,
                  kv_cache_config=KvCacheConfig(dtype="auto"))
        return llm, SamplingParams
    if backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=model, max_num_seqs=max_batch, enforce_eager=False)
        return llm, SamplingParams
    raise ValueError(f"unknown backend {backend!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="exported NVFP4 checkpoint dir")
    ap.add_argument("--backend", default="auto", choices=["auto", "trtllm", "vllm"],
                    help="'auto' tries trtllm then falls back to vllm (safe on sm_120)")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 64, 128, 256])
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    order = ["trtllm", "vllm"] if args.backend == "auto" else [args.backend]
    llm = SP = used = None
    for b in order:
        try:
            print(f"loading via {b} ...", flush=True)
            llm, SP = _load(b, args.model, max(args.batches))
            used = b
            break
        except Exception as ex:
            print(f"  {b} load failed: {repr(ex)[:120]}", flush=True)
    if llm is None:
        raise SystemExit("ABORT: no serving backend could load the checkpoint.")
    print(f"backend in use: {used}", flush=True)

    # Sanity: coherent generation proves the NVFP4 checkpoint loaded correctly.
    sanity = llm.generate(["The capital of France is"], SP(max_tokens=16, temperature=0))
    print("SANITY:", repr(sanity[0].outputs[0].text), flush=True)

    prompt = "The quick brown fox " * (args.prompt_len // 4)
    sp = SP(max_tokens=args.gen_tokens, temperature=0)

    print(f"\nbackend={used}\n{'batch':>6}{'tok/s':>12}{'ms/req':>10}")
    for B in args.batches:
        prompts = [prompt] * B
        llm.generate(prompts, sp)                       # warmup / graph capture
        best = min(_time_once(llm, prompts, sp) for _ in range(args.iters))
        print(f"{B:>6}{B * args.gen_tokens / best:>12.0f}{best / B * 1e3:>10.1f}", flush=True)


def _time_once(llm, prompts, sp) -> float:
    t = time.perf_counter()
    llm.generate(prompts, sp)
    return time.perf_counter() - t


if __name__ == "__main__":
    main()
