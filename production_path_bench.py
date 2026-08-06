r"""The SHIPPABLE configuration: production FP4 kernel + our correction beside it.

`triton_lowrank_fusion_bench.py` replaced the whole GEMM with hand-written Triton to
prove the fusion mechanism. That is not shippable: it loses to cuBLAS at 0.62-0.72x
for large M, and a real deployment would never swap out a tuned FP4 kernel.

This measures what you would ACTUALLY deploy:

    main GEMM   : vLLM/FlashInfer `cutlass_scaled_fp4_mm`  (untouched, production)
    correction  : coeff = X @ Rf.T   then   Y += coeff @ Lf.T   (small, separate)

At rank r = K/16 the correction is ~2*r/N of the main GEMM's FLOPs — single-digit
percent. The question is whether that theoretical overhead survives contact with
real kernel-launch costs and HBM traffic.

Reported:
  fp4_only      the production FP4 GEMM alone (what you ship WITHOUT our method)
  fp4_plus_corr the same GEMM + our correction (what you ship WITH it)
  bf16          dense reference, for the FP4 speed context

The number that matters is `corr_overhead` = fp4_plus_corr / fp4_only. If it stays
near the FLOP ratio, our accuracy win is essentially free on top of production FP4.

Run:  python production_path_bench.py            (needs vllm + Blackwell)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parent / "results"
FP8_MAX, FP4_MAX = 448.0, 6.0


def bench(fn, it=50, warm=15):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it


def gscale(x):
    return (FP8_MAX * FP4_MAX / x.abs().amax().clamp_min(1e-8)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank-div", type=int, default=16)
    args = ap.parse_args()

    try:
        from vllm import _custom_ops as ops
    except ImportError:
        print("vllm not available — this benchmark needs the production FP4 kernel.")
        print("(setup_box.sh installs it best-effort; it failed on the last sm_120 box.)")
        return

    if not torch.cuda.is_available():
        print("needs CUDA"); return
    cap = torch.cuda.get_device_capability()
    print(f"GPU {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}", flush=True)
    try:
        print(f"supports_fp4={ops.cutlass_scaled_mm_supports_fp4(cap[0]*10+cap[1])}",
              flush=True)
    except Exception as e:
        print(f"fp4 support probe failed: {e!r}")

    shapes = [
        (8192, 8192), (8192, 28672), (28672, 8192),   # Llama-70B attn / MLP / down
        (2048, 2048), (2048, 5632),                   # TinyLlama-scale
    ]
    rows = []
    print(f"\nProduction FP4 GEMM + our correction (r = K/{args.rank_div})")
    print(f"{'M':>6}{'K':>7}{'N':>7}{'R':>6}{'bf16_ms':>9}{'fp4_ms':>9}"
          f"{'fp4+corr':>10}{'overhead':>10}{'vs_bf16':>9}")
    for (K, N) in shapes:
        R = max(8, K // args.rank_div)
        for M in (1, 16, 512, 4096):
            A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
            B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
            RfT = torch.randn(K, R, device="cuda", dtype=torch.bfloat16) * 0.1
            LfT = torch.randn(R, N, device="cuda", dtype=torch.bfloat16) * 0.1
            try:
                tb = bench(lambda: torch.matmul(A, B))
                gsa, gsb = gscale(A), gscale(B)
                Bt = B.t().contiguous()
                a4, asf = ops.scaled_fp4_quant(A, gsa)
                b4, bsf = ops.scaled_fp4_quant(Bt, gsb)
                alpha = (1.0 / (gsa * gsb)).float()

                def fp4_only():
                    return ops.cutlass_scaled_fp4_mm(a4, b4, asf, bsf, alpha,
                                                     torch.bfloat16)

                def fp4_plus_corr():
                    y = ops.cutlass_scaled_fp4_mm(a4, b4, asf, bsf, alpha,
                                                  torch.bfloat16)
                    return y + (A @ RfT) @ LfT      # the correction, beside the GEMM

                t4 = bench(fp4_only)
                tc = bench(fp4_plus_corr)
            except Exception as e:
                print(f"{M:>6}{K:>7}{N:>7}{R:>6}  ERR {repr(e)[:45]}")
                continue
            row = {"M": M, "K": K, "N": N, "R": R, "bf16_ms": tb * 1e3,
                   "fp4_ms": t4 * 1e3, "fp4_corr_ms": tc * 1e3,
                   "corr_overhead": tc / t4, "fp4_vs_bf16": tb / t4}
            rows.append(row)
            print(f"{M:>6}{K:>7}{N:>7}{R:>6}{tb*1e3:>9.3f}{t4*1e3:>9.3f}"
                  f"{tc*1e3:>10.3f}{tc/t4:>9.2f}x{tb/t4:>8.2f}x")

    if rows:
        big = [r for r in rows if r["K"] >= 8192 and r["M"] >= 512]
        small = [r for r in rows if r["M"] <= 16]
        summary = {
            "mean_corr_overhead_all": round(
                sum(r["corr_overhead"] for r in rows) / len(rows), 4),
            "mean_corr_overhead_large_gemm": round(
                sum(r["corr_overhead"] for r in big) / len(big), 4) if big else None,
            "mean_corr_overhead_decode_M<=16": round(
                sum(r["corr_overhead"] for r in small) / len(small), 4) if small else None,
        }
        print("\n  " + json.dumps(summary, indent=2).replace("\n", "\n  "))
        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / "production_path_bench.json"
        out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
