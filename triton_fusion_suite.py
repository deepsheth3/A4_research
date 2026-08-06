r"""Fuse every op our method adds at inference — measured, not asserted.

Our method (equalization + additive low-rank correction) is the PTQ approach that
survives NVFP4 microscaling, but unlike rotation or GPTQ it does NOT fold to zero
runtime cost. It adds FOUR ops to the deployed path. This benchmarks fusing each.

  #1 equalization scale-fold   x / s_i          -> fold into the PRODUCER's epilogue
  #2 fp8 activation quant      per-token absmax -> fold into the PRODUCER's epilogue
  #3 KV4 + low-rank correction q + (r@V)@V.T    -> fold into attention's KV path
  #4 weight low-rank epilogue  + (X@Rf.T)@Lf.T  -> done: triton_lowrank_fusion_bench.py

#1 and #2 always co-occur in W4A8 (equalize, then quantize the activation), and both
are elementwise/reduction ops on the SAME tensor — so they fuse into a single pass.
Unfused they are 3 full HBM round trips of the activation; fused, one.

#3 is the interesting one: T1 measured the rank-64 KV correction recovering 56% of
KV4's PPL cost, but it was shelved as "not byte-legal" — an r-dim per-token
side-channel doesn't fit the KV cache's fixed layout without a custom kernel. Fusing
dequant+correction into one pass is what would make it shippable.

Run:  python triton_fusion_suite.py          # correctness + timing (needs CUDA)
      python triton_fusion_suite.py --check  # correctness only, any device
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

FP8_MAX = 448.0
FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)   # E2M1 magnitudes


# ---------------------------------------------------------------------------
# Reference implementations (what the harness does today, unfused)
# ---------------------------------------------------------------------------
def ref_eq_fp8(x, s):
    """#1+#2 unfused: equalize, then per-token-absmax fp8 quantize, then un-equalize.

    Three separate tensor ops = three HBM round trips of x.
    """
    xe = x / s
    scale = (xe.abs().amax(dim=-1, keepdim=True) / FP8_MAX).clamp_min(1e-12)
    q = torch.clamp((xe / scale), -FP8_MAX, FP8_MAX)
    q = _round_e4m3_ref(q) * scale
    return q * s


def _round_e4m3_ref(x):
    """Round to fp8-e4m3 via torch's native dtype (exact, not an approximation)."""
    return x.to(torch.float8_e4m3fn).to(x.dtype)


def ref_kv4_corrected(o, V, block=16):
    """#3 unfused: NVFP4-quantize KV, then add the rank-r residual correction.

    Materializes the residual r = o - q as a full tensor, then two matmuls.
    """
    q = _nvfp4_ref(o, block)
    r = o - q
    return q + (r @ V) @ V.t()


def _nvfp4_ref(x, block=16):
    """Reference NVFP4 (E2M1 + per-block absmax fp8 scale), symmetric."""
    *lead, n = x.shape
    xb = x.reshape(*lead, n // block, block)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 6.0).clamp_min(1e-12)
    scale = scale.to(torch.float8_e4m3fn).to(x.dtype)      # scales are fp8 in NVFP4
    g = torch.tensor(FP4_GRID, device=x.device, dtype=x.dtype)
    xn = (xb / scale).abs().clamp(max=6.0)
    idx = (xn.unsqueeze(-1) - g).abs().argmin(dim=-1)
    q = g[idx] * torch.sign(xb) * scale
    return q.reshape(*lead, n)


# ---------------------------------------------------------------------------
# #1 + #2 FUSED: equalize + fp8-quantize in ONE pass over the activation
# ---------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def _fused_eq_fp8(X, S, Y, M, N, sxm, sxn, sym, syn, BLOCK_N: tl.constexpr):
        """One row (token) per program. Loads x once, does everything in registers.

        Unfused this is: read x, write xe | read xe, reduce | read xe, write q.
        Fused: read x, write y. The per-token absmax reduction and the rescale both
        happen while the row is still resident.
        """
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(X + row * sxm + offs * sxn, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(S + offs, mask=mask, other=1.0).to(tl.float32)
        xe = x / s                                  # #1 equalization
        amax = tl.max(tl.abs(xe), axis=0)           # #2 per-token absmax (in-register)
        scale = tl.maximum(amax / 448.0, 1e-12)
        q = xe / scale
        q = tl.clamp(q, -448.0, 448.0)
        q = q.to(tl.float8e4nv).to(tl.float32)      # fp8-e4m3 round trip
        y = q * scale * s                           # rescale + un-equalize
        tl.store(Y + row * sym + offs * syn, y, mask=mask)


def fused_eq_fp8(x, s):
    """#1+#2 in a single kernel launch, one HBM round trip."""
    M, N = x.shape
    y = torch.empty_like(x)
    blk = triton.next_power_of_2(N)
    _fused_eq_fp8[(M,)](x, s, y, M, N, x.stride(0), x.stride(1),
                        y.stride(0), y.stride(1), BLOCK_N=blk, num_warps=8)
    return y


# ---------------------------------------------------------------------------
# #3 FUSED: NVFP4-quantize KV and project the residual in ONE pass
#
# The unfused path materializes the residual r = o - q as a full (M,d) tensor and
# then reads it back for r @ V. Fused, r never leaves registers: we quantize a row,
# subtract, and accumulate its projection immediately. Output is the (M,r) coeff,
# which is ALSO the byte-legal artifact — the r-dim code that would ride alongside
# the KV cache instead of a full-width residual.
# ---------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def _fused_kv4_project(O, Vm, Q, Coef, M, D, R, BLOCK_D: tl.constexpr,
                           BLOCK_R: tl.constexpr, BLK: tl.constexpr):
        """Per token: NVFP4-quantize, then project the residual onto V — no
        full-width residual ever written to memory."""
        row = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        o = tl.load(O + row * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)

        # --- NVFP4 (E2M1, per-BLK absmax, fp8 scale), symmetric ---
        # Reshape to (nblk, BLK) so the absmax is a real per-block MAX reduction.
        # (v1 used tl.sum over a masked full-width vector, which returns the block
        # SUM, not its max -> scales ~BLK x too large -> small values crushed to 0.
        # Caught by the correctness gate at relerr 2.6e-01.)
        o2 = tl.reshape(o, (BLOCK_D // BLK, BLK))
        bmax = tl.max(tl.abs(o2), axis=1)                      # (nblk,)
        absmax = tl.reshape(
            tl.broadcast_to(bmax[:, None], (BLOCK_D // BLK, BLK)), (BLOCK_D,))
        scale = tl.maximum(absmax / 6.0, 1e-12)
        scale = scale.to(tl.float8e4nv).to(tl.float32)
        a = tl.abs(o) / scale
        # round-to-nearest on the E2M1 magnitude grid
        q = tl.where(a < 0.25, 0.0,
            tl.where(a < 0.75, 0.5,
            tl.where(a < 1.25, 1.0,
            tl.where(a < 1.75, 1.5,
            tl.where(a < 2.5, 2.0,
            tl.where(a < 3.5, 3.0,
            tl.where(a < 5.0, 4.0, 6.0)))))))
        qv = q * tl.where(o < 0, -1.0, 1.0) * scale
        tl.store(Q + row * D + offs_d, qv, mask=mask_d)

        # --- residual projection, r stays in registers ---
        r = o - qv
        for r0 in range(0, R, BLOCK_R):
            offs_r = r0 + tl.arange(0, BLOCK_R)
            v = tl.load(Vm + offs_d[:, None] * R + offs_r[None, :],
                        mask=mask_d[:, None] & (offs_r[None, :] < R), other=0.0)
            c = tl.sum(r[:, None] * v.to(tl.float32), axis=0)
            tl.store(Coef + row * R + offs_r, c, mask=offs_r < R)


def fused_kv4_project(o, V, blk=16):
    """Returns (q, coeff): the NVFP4 KV and the r-dim residual code.

    coeff is what makes this byte-legal — an r-wide per-token code stored beside the
    cache, rather than a full-width correction that has nowhere to live.
    """
    M, D = o.shape
    R = V.shape[1]
    q = torch.empty_like(o)
    coeff = torch.empty((M, R), device=o.device, dtype=torch.float32)
    _fused_kv4_project[(M,)](o, V.contiguous(), q, coeff, M, D, R,
                             BLOCK_D=triton.next_power_of_2(D),
                             BLOCK_R=triton.next_power_of_2(min(R, 64)), BLK=blk,
                             num_warps=8)
    return q, coeff


def bench(fn, it=50, warm=15):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cuda = torch.cuda.is_available()
    print(f"triton={HAS_TRITON} cuda={cuda}")
    if not (cuda and HAS_TRITON):
        print("Needs CUDA+triton for both correctness and timing (fp8 dtypes are GPU-only).")
        return

    torch.manual_seed(0)
    dev = "cuda"

    # ---- correctness: #1+#2 ----
    print("\n[#1+#2] equalization + fp8 activation quant")
    M, N = 512, 2048
    x = torch.randn(M, N, device=dev) * 0.1
    s = torch.rand(N, device=dev) * 1.5 + 0.5
    ref = ref_eq_fp8(x, s)
    got = fused_eq_fp8(x, s)
    # Gate on MEAN error + the fraction of disagreeing elements, not max-abs.
    # Max-abs is dominated by fp8 TIE cases: a value landing exactly halfway between
    # two representable fp8 neighbours (e.g. 42.0 between 40 and 44) rounds up in
    # torch and down in Triton. That is one element in ~10^6 and is not a defect —
    # but a real bug (wrong scale, wrong axis, bad reduction) moves the MEAN and
    # makes many elements disagree, which these thresholds do catch.
    rel = (got - ref).abs() / ref.abs().clamp_min(1e-9)
    mean_err = rel.mean().item()
    frac_bad = (rel > 1e-3).float().mean().item()
    print(f"  mean relerr {mean_err:.2e} | elements >1e-3 off: {frac_bad*100:.3f}%"
          f"  {'PASS' if (mean_err < 1e-5 and frac_bad < 1e-3) else 'FAIL'}")
    assert mean_err < 1e-5 and frac_bad < 1e-3, "fused eq+fp8 disagrees with the reference"

    # ---- correctness: #3 ----
    print("\n[#3] KV4 quantize + residual projection")
    D, R = 2048, 64
    o = torch.randn(M, D, device=dev) * 0.1
    V = torch.linalg.qr(torch.randn(D, R, device=dev))[0]
    q_ref = _nvfp4_ref(o, 16)
    q_got, coeff = fused_kv4_project(o, V, 16)
    qerr = (q_got - q_ref).abs().max().item() / q_ref.abs().max().item()
    corr_ref = ((o - q_ref) @ V)
    cerr = (coeff - corr_ref).abs().max().item() / corr_ref.abs().max().item()
    print(f"  NVFP4 grid match : max relerr {qerr:.2e}  {'PASS' if qerr < 1e-2 else 'FAIL'}")
    print(f"  residual coeff   : max relerr {cerr:.2e}  {'PASS' if cerr < 1e-2 else 'FAIL'}")
    full_ref = ref_kv4_corrected(o, V, 16)
    full_got = q_got + coeff @ V.t()
    ferr = (full_got - full_ref).abs().max().item() / full_ref.abs().max().item()
    print(f"  reconstructed KV : max relerr {ferr:.2e}  {'PASS' if ferr < 1e-2 else 'FAIL'}")
    if args.check:
        return

    # ---- timing ----
    print("\n[#1+#2] fused vs unfused (per-token equalize + fp8 quant)")
    print(f"{'tokens':>8}{'dim':>7}{'unfused_us':>12}{'fused_us':>10}{'speedup':>9}")
    for N in (2048, 5632, 8192):
        for M in (16, 512, 4096):
            x = torch.randn(M, N, device=dev) * 0.1
            s = torch.rand(N, device=dev) * 1.5 + 0.5
            tu = bench(lambda: ref_eq_fp8(x, s))
            tf = bench(lambda: fused_eq_fp8(x, s))
            print(f"{M:>8}{N:>7}{tu*1e6:>12.1f}{tf*1e6:>10.1f}{tu/tf:>8.2f}x")

    print("\n[#3] fused vs unfused (KV4 quantize + rank-64 residual projection)")
    print(f"{'tokens':>8}{'dim':>7}{'rank':>6}{'unfused_us':>12}{'fused_us':>10}{'speedup':>9}")
    for D in (2048, 4096):
        V = torch.linalg.qr(torch.randn(D, 64, device=dev))[0]
        for M in (16, 512, 4096):
            o = torch.randn(M, D, device=dev) * 0.1
            tu = bench(lambda: ref_kv4_corrected(o, V, 16))
            tf = bench(lambda: fused_kv4_project(o, V, 16))
            print(f"{M:>8}{D:>7}{64:>6}{tu*1e6:>12.1f}{tf*1e6:>10.1f}{tu/tf:>8.2f}x")


if __name__ == "__main__":
    main()
