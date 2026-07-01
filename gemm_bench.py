"""FP4 vs FP8 vs BF16 GEMM throughput on Blackwell — PRODUCTION batches (64/128/256),
no batch-1 (single-stream is a benchmark fiction; prod is concurrent).

Goal: a REAL NVFP4 CUTLASS tensor-core number, not torchao's emulation fallback.
Self-validates: if FP4 TFLOPS < FP8, it's emulation/broken -> printed as SUSPECT.

When the box is back:  python gemm_bench.py
If FP4 shows SUSPECT, run probe_fp4() output and switch to the path that dispatches
to the SM12x CUTLASS kernel (try NVFP4Tensor block-16 before MXTensor block-32).
"""
import sys
import torch

dev = "cuda"
def tf(M, K, N, t): return 2 * M * K * N / t / 1e12
def bench(fn, iters=50, warm=15):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3


def preflight():
    """Fail loud BEFORE benchmarking if this box can't do real FP4 tensor cores.
    FP4 needs Blackwell (sm_100 B200 / sm_120 RTX Pro 6000); older SMs emulate."""
    if not torch.cuda.is_available():
        sys.exit("ABORT: no CUDA device — FP4 tensor cores need a Blackwell GPU.")
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    try:
        import torchao; tao = torchao.__version__
    except Exception as ex:
        tao = f"MISSING ({repr(ex)[:40]})"
    print(f"device: {name}  sm_{cap[0]}{cap[1]}  torch {torch.__version__}  torchao {tao}")
    if cap[0] < 10:
        print(f"WARNING: sm_{cap[0]}{cap[1]} predates Blackwell — FP4 will emulate, not use tensor cores.")


def fp4_rel_err(a, b):
    """Relative error of the FP4 GEMM vs the bf16 reference. Guards both failure modes:
    ~0 error => not actually quantizing (fp16 passthrough); ~1 => broken kernel.
    Genuine FP4 on N(0,1) operands lands in a sane mid band."""
    ref = torch.matmul(a, b.t()).float()
    ax, bx = fp4_pack(a, b)
    got = fp4_mm(ax, bx).float()
    return ((got - ref).norm() / ref.norm()).item()


def probe_fp4():
    """Report every FP4 mm entry point torchao exposes, so we pick the CUTLASS one."""
    import importlib
    for mod, names in [
        ("torchao.prototype.mx_formats.nvfp4_tensor", ["NVFP4Tensor"]),
        ("torchao.prototype.mx_formats.mx_tensor", ["MXTensor"]),
        ("torchao.prototype.mx_formats", ["MXGemmKernelChoice", "NVFP4Tensor"]),
        ("torchao.prototype.mx_formats.config", ["MXGemmKernelChoice"]),
    ]:
        try:
            m = importlib.import_module(mod)
            print("  ok:", mod, "->", [n for n in names if hasattr(m, n)])
        except Exception as ex:
            print("  no:", mod, repr(ex)[:70])


# --- FP4 path: prefer NVFP4 (block-16) CUTLASS, fall back to mxfp4 ---
fp4_pack = fp4_mm = None
try:
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    def fp4_pack(a, b):
        return NVFP4Tensor.to_nvfp4(a), NVFP4Tensor.to_nvfp4(b)
    def fp4_mm(ax, bx): return torch.mm(ax, bx.t())
    print("FP4 path: NVFP4Tensor (block-16, target CUTLASS)")
except Exception as ex:
    print("FP4 path: NVFP4Tensor unavailable ->", repr(ex)[:90])
    try:
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
        from torchao.prototype.mx_formats import MXGemmKernelChoice
        def fp4_pack(a, b):
            mk = MXGemmKernelChoice.CUTLASS
            return (MXTensor.to_mx(a, torch.float4_e2m1fn_x2, 32, gemm_kernel_choice=mk),
                    MXTensor.to_mx(b, torch.float4_e2m1fn_x2, 32, gemm_kernel_choice=mk))
        def fp4_mm(ax, bx): return torch.mm(ax, bx.t())
        print("FP4 path: MXTensor block-32 CUTLASS")
    except Exception as ex2:
        print("FP4 path: none ->", repr(ex2)[:90]); probe_fp4()

preflight()
if fp4_mm is None:
    sys.exit("ABORT: no FP4 GEMM path — install torchao with NVFP4/MX CUTLASS support.")

# Correctness gate: one representative shape. FP4 error on N(0,1) operands must sit in
# a plausible band — near-zero means fp16 passthrough, near-one means a broken kernel.
a0 = torch.randn(256, 4096, device=dev, dtype=torch.bfloat16)
b0 = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
err = fp4_rel_err(a0, b0)
print(f"FP4 correctness: rel_err={err:.4f}  (expect ~0.02-0.25 for real FP4)")
if err < 5e-3:
    sys.exit(f"ABORT: rel_err {err:.4g} ~ 0 => FP4 not quantizing (fp16 passthrough), numbers meaningless.")
if err > 0.5:
    sys.exit(f"ABORT: rel_err {err:.4g} too high => FP4 kernel broken.")

print(f"\n{'shape':18}{'M':>5}{'BF16':>8}{'FP8':>8}{'FP4':>8}{'FP4/FP8':>9}{'flag':>10}")
rows = [("attn 4096x4096", 4096, 4096), ("mlp 4096x14336", 4096, 14336)]
suspect = False
for name, K, N in rows:
    for M in (64, 128, 256):
        a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        b = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        tb = bench(lambda: torch.matmul(a, b.t()))
        af, bf = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
        sa = sb = torch.tensor(1.0, device=dev)
        t8 = bench(lambda: torch._scaled_mm(af, bf.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
        t4 = None
        if fp4_mm is not None:
            try:
                ax, bx = fp4_pack(a, b)
                t4 = bench(lambda: fp4_mm(ax, bx))
            except Exception as ex:
                if M == 64: print("  FP4 mm err:", repr(ex)[:80])
        v8, v4 = tf(M, K, N, t8), (tf(M, K, N, t4) if t4 else 0)
        r = f"{v4/v8:>8.2f}x" if t4 else f"{'-':>9}"
        is_suspect = bool(t4) and v4 < v8
        suspect = suspect or is_suspect
        flag = "" if not t4 else ("SUSPECT" if is_suspect else "real")
        s4 = f"{v4:>8.0f}" if t4 else f"{'n/a':>8}"
        print(f"{name:18}{M:>5}{tf(M,K,N,tb):>8.0f}{v8:>8.0f}{s4}{r}{flag:>10}")

print("\nSUSPECT = FP4 slower than FP8 => emulation kernel, not real tensor cores.")
if suspect:
    sys.exit("ABORT: FP4 emulating on at least one shape — fix the kernel dispatch before trusting numbers.")
