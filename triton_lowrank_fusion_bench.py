r"""Fuse our low-rank weight correction into the main GEMM's epilogue — one kernel
launch instead of two.

WHY THIS EXISTS
---------------
Our method (equalization + additive low-rank correction) is the one PTQ approach that
survives NVFP4's per-16 microscaling, but unlike rotation or GPTQ it does NOT fold to
zero runtime cost: rotation folds into an offline weight transform, GPTQ folds into
the rounded weight, while ours needs a genuine extra matmul at inference:

    Y = X @ Wq.T  +  (X @ Rf.T) @ Lf.T
        \_________/   \_________________/
         NVFP4 GEMM    rank-r correction (the side-channel)

At rank r = in/16 the correction is only ~6% extra FLOPs, but as a SEPARATE kernel
launch that small matmul pays full launch overhead + a round trip of the (M,N) output
through HBM. The hypothesis this script tests: if the main GEMM is memory-bandwidth-
bound (which is exactly the regime NVFP4 weights create), fusing the correction into
its epilogue hides the extra FLOPs in existing slack -> near-free correction.

This mirrors the arithmetic-intensity crossover already measured in
`fp4_gemm_bench_flashinfer.py` (FP4 3.2-3.8x faster than BF16 at 70B-layer shapes,
SLOWER at TinyLlama shapes): fusion should pay off where the GEMM is big/bandwidth-
bound and matter least where it is launch-latency-dominated.

SCOPE: standalone proof-of-concept microbenchmark. Not wired into export_nvfp4.py,
vLLM, or any deploy path. Proves or kills the "near-free fusion" hypothesis, nothing
more.

Run:
  python triton_lowrank_fusion_bench.py            # GPU: correctness + timing
  python triton_lowrank_fusion_bench.py --check    # CPU/any: correctness only
"""
from __future__ import annotations

import argparse
import time

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# The fused kernel: C = A @ B + (A @ Rf.T) @ Lf.T, in one launch.
#
# The correction rides in registers: each block loads the full (r, N) slice of the
# correction's right factor once, accumulates the rank-r term alongside the main
# K-loop, and writes a single fused result. No second (M,N) HBM round trip.
# ---------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def _fused_lowrank_gemm(
        A, B, Coeff, LfT, C,
        M, N, K, R,
        sam, sak, sbk, sbn, scfm, scfr, slr, sln, scm, scn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, BLOCK_R: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # --- main GEMM: sum over K ---
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(A + offs_m[:, None] * sam + offs_k[None, :] * sak,
                        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            b = tl.load(B + offs_k[:, None] * sbk + offs_n[None, :] * sbn,
                        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            # ieee: fp32 inputs otherwise go through TF32 tensor cores (10-bit mantissa),
            # which silently costs ~4 decimal digits and fails the correctness gate.
            acc += tl.dot(a, b, input_precision="ieee")

        # --- fused epilogue: add the rank-r correction, SAME launch ---
        # `Coeff` = (A @ Rf.T) was computed ONCE outside, as its own small GEMM.
        # Here we only stream the small (M,R) x (R,N) term — one pass over a narrow
        # matrix, not a re-read of A. (v1 recomputed A@Rf.T inside this loop, which
        # re-streamed the whole of A once per rank block: 4-100x SLOWER. See
        # results/box_threads/*/t4_fusion_fixed.log for that negative result.)
        for r0 in range(0, R, BLOCK_R):
            offs_r = r0 + tl.arange(0, BLOCK_R)
            cf = tl.load(Coeff + offs_m[:, None] * scfm + offs_r[None, :] * scfr,
                         mask=(offs_m[:, None] < M) & (offs_r[None, :] < R), other=0.0)
            lf = tl.load(LfT + offs_r[:, None] * slr + offs_n[None, :] * sln,
                         mask=(offs_r[:, None] < R) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(cf, lf, input_precision="ieee")

        tl.store(C + offs_m[:, None] * scm + offs_n[None, :] * scn, acc,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_lowrank_matmul(A, B, RfT, LfT, block_m=64, block_n=64, block_k=32, block_r=16):
    """A @ B + (A @ RfT) @ LfT with the correction fused into the GEMM epilogue.

    Two steps, but only ONE (M,N)-sized kernel: the (M,R) coefficient projection is a
    thin GEMM (~R/N of the main one's work), then the main GEMM absorbs the
    correction in its epilogue. The naive path instead materializes a second full
    (M,N) result and adds it — an extra HBM round trip of the large output.
    """
    M, K = A.shape
    _, N = B.shape
    R = RfT.shape[1]
    coeff = torch.matmul(A, RfT)                     # (M,R) — thin, done once
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fused_lowrank_gemm[grid](
        A, B, coeff, LfT, C, M, N, K, R,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        coeff.stride(0), coeff.stride(1), LfT.stride(0), LfT.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_R=block_r,
    )
    return C


def naive_lowrank_matmul(A, B, RfT, LfT):
    """Two launches (what we do today): main GEMM, then a separate correction matmul."""
    return torch.matmul(A, B) + torch.matmul(torch.matmul(A, RfT), LfT)


def triton_plain_matmul(A, B, block_m=64, block_n=64, block_k=32):
    """Same Triton kernel with R=0 — a GEMM carrying NO correction.

    This is the honest baseline for the fusion question. Comparing the fused kernel
    to cuBLAS conflates two effects (is fusion good? is my Triton as fast as
    cuBLAS?); comparing it to this isolates the one we care about: what does adding
    the correction COST, given the same kernel quality.
    """
    M, K = A.shape
    _, N = B.shape
    empty_c = torch.empty((M, 0), device=A.device, dtype=A.dtype)
    empty_l = torch.empty((0, N), device=A.device, dtype=A.dtype)
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fused_lowrank_gemm[grid](
        A, B, empty_c, empty_l, C, M, N, K, 0,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        1, 1, 1, 1, C.stride(0), C.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_R=16,
    )
    return C


def bench(fn, it=50, warm=15):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it


def correctness(device="cpu"):
    """Fused output must match the naive two-launch reference. Must pass before any
    timing number is trusted — a fast kernel that is wrong is worse than no kernel."""
    torch.manual_seed(0)
    M, K, N, R = 128, 256, 64, 16
    A = torch.randn(M, K, device=device) * 0.1
    B = torch.randn(K, N, device=device) * 0.1
    RfT = torch.randn(K, R, device=device) * 0.1
    LfT = torch.randn(R, N, device=device) * 0.1

    ref = naive_lowrank_matmul(A, B, RfT, LfT)
    if device == "cuda" and HAS_TRITON:
        got = fused_lowrank_matmul(A, B, RfT, LfT)
        err = (got - ref).abs().max().item() / ref.abs().max().item()
        ok = err < 1e-3
        print(f"  fused vs naive: max relerr {err:.2e}  {'PASS' if ok else 'FAIL'}")
        assert ok, f"fused kernel disagrees with reference (relerr {err:.2e})"
    else:
        # No GPU: at least verify the reference decomposition itself is what we claim,
        # i.e. Wq + Lf@Rf applied to X equals correcting the weight offline.
        Wq, Lf, Rf = B.T, LfT.T, RfT.T           # (N,K), (N,R), (R,K)
        offline = torch.matmul(A, (Wq + Lf @ Rf).T)
        err = (offline - ref).abs().max().item() / ref.abs().max().item()
        print(f"  side-channel == offline-corrected weight: max relerr {err:.2e} "
              f"{'PASS' if err < 1e-4 else 'FAIL'}")
        assert err < 1e-4, "low-rank side-channel is not equivalent to the merged weight"
    print("  correctness OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="correctness only, no timing")
    ap.add_argument("--rank-div", type=int, default=16, help="r = K // rank_div")
    args = ap.parse_args()

    has_cuda = torch.cuda.is_available()
    print(f"triton={HAS_TRITON} cuda={has_cuda}")
    correctness("cuda" if (has_cuda and HAS_TRITON) else "cpu")
    if args.check or not has_cuda:
        if not has_cuda:
            print("\nNo CUDA — correctness only. Timing requires the GPU box.")
        return
    if not HAS_TRITON:
        print("\nTriton not installed — cannot benchmark the fused path.")
        return

    # Same shape ladder as fp4_gemm_bench_flashinfer.py, so the fusion result is
    # directly comparable to the NVFP4-vs-BF16 crossover measured there.
    shapes = [
        (8192, 8192), (8192, 28672), (28672, 8192),   # Llama-70B attn / MLP / down
        (2048, 2048), (2048, 5632), (5632, 2048),     # TinyLlama attn / MLP / down
    ]
    print(f"\nFused low-rank epilogue (r = K/{args.rank_div})")
    print("  cublas   = torch: main GEMM + separate correction (2 launches, today's path)")
    print("  tri_bare = same Triton kernel, NO correction (isolates fusion's true cost)")
    print("  tri_fuse = Triton kernel WITH the correction fused into the epilogue")
    print(f"{'M':>6}{'K':>7}{'N':>7}{'R':>5}{'cublas':>9}{'tri_bare':>10}"
          f"{'tri_fuse':>10}{'fuse_cost':>10}{'vs_cublas':>10}")
    for (K, N) in shapes:
        R = max(8, K // args.rank_div)
        for M in (16, 512, 4096):
            A = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
            B = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
            RfT = torch.randn(K, R, device="cuda", dtype=torch.float16) * 0.1
            LfT = torch.randn(R, N, device="cuda", dtype=torch.float16) * 0.1
            try:
                tn = bench(lambda: naive_lowrank_matmul(A, B, RfT, LfT))
                tb = bench(lambda: triton_plain_matmul(A, B))
                tf = bench(lambda: fused_lowrank_matmul(A, B, RfT, LfT))
            except Exception as e:
                print(f"{M:>6}{K:>7}{N:>7}{R:>5}  ERR {repr(e)[:50]}")
                continue
            # fuse_cost = what the correction actually costs, kernel quality held fixed.
            print(f"{M:>6}{K:>7}{N:>7}{R:>5}{tn*1e3:>9.3f}{tb*1e3:>10.3f}"
                  f"{tf*1e3:>10.3f}{tf/tb:>9.2f}x{tn/tf:>9.2f}x")


if __name__ == "__main__":
    main()
