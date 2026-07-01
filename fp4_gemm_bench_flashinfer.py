import torch, time
from vllm import _custom_ops as ops
torch.manual_seed(0)

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

print("Real NVFP4 GEMM (cutlass_scaled_fp4_mm) vs BF16 — RTX PRO 6000 sm_120")
print(f"supports_fp4={ops.cutlass_scaled_mm_supports_fp4(120)}")
print(f"{'M':>6}{'K':>7}{'N':>7}{'bf16_TF':>9}{'fp4_TF':>9}{'speedup':>9}{'relerr':>9}")
shapes = [(8192, 8192), (8192, 28672), (28672, 8192)]  # Llama-70B attn / MLP / down
for (K, N) in shapes:
    for M in (16, 512, 4096, 8192):
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
        B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
        tb = bench(lambda: torch.matmul(A, B))
        try:
            gsa, gsb = gscale(A), gscale(B)
            Bt = B.t().contiguous()               # (N,K), quantized along K
            a4, asf = ops.scaled_fp4_quant(A, gsa)
            b4, bsf = ops.scaled_fp4_quant(Bt, gsb)
            alpha = (1.0 / (gsa * gsb)).float()
            f = lambda: ops.cutlass_scaled_fp4_mm(a4, b4, asf, bsf, alpha, torch.bfloat16)
            out = f()
            ref = torch.matmul(A, B)
            relerr = (out.float() - ref.float()).norm().item() / ref.float().norm().item()
            t4 = bench(f)
        except Exception as e:
            print(f"{M:>6}{K:>7}{N:>7}  fp4 ERR {repr(e)[:60]}")
            continue
        fl = 2 * M * K * N
        print(f"{M:>6}{K:>7}{N:>7}{fl/tb/1e12:>9.0f}{fl/t4/1e12:>9.0f}{tb/t4:>8.2f}x{relerr:>9.3f}")
