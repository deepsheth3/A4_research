import torch

from turboquant.act_codec import TurboQuantActQuantizer
from turboquant.config import TurboQuantConfig
from turboquant.nvfp4 import nvfp4_quantize, round_e4m3


def test_e4m3_matches_native_cast_in_range():
    torch.manual_seed(0)
    t = torch.cat([torch.rand(5000) * 448, torch.rand(5000) * 0.02,
                   -torch.rand(5000) * 448, torch.zeros(3)])
    ref = t.to(torch.float8_e4m3fn).to(torch.float32)
    assert torch.equal(round_e4m3(t), ref)


def test_e4m3_saturates_instead_of_nan():
    out = round_e4m3(torch.tensor([1000.0, -1000.0]))
    assert torch.equal(out, torch.tensor([448.0, -448.0]))


def test_optclip_never_worse_than_absmax():
    # gamma=1.0 (absmax) is in the candidate set, so per-block MSE can only drop.
    torch.manual_seed(0)
    x = torch.randn(64, 512)
    x[:, 7] *= 50.0
    e_abs = ((x - nvfp4_quantize(x)) ** 2).sum()
    e_opt = ((x - nvfp4_quantize(x, optclip=True)) ** 2).sum()
    assert e_opt <= e_abs + 1e-6


def test_optclip_output_is_grid_times_fp8_scale():
    # Hardware-nativity: every output must still be (E2M1 value) x (fp8 scale).
    from turboquant.nvfp4 import NVFP4_GRID
    torch.manual_seed(1)
    x = torch.randn(4, 64)
    q = nvfp4_quantize(x, optclip=True).reshape(-1, 16)
    for blk in q:
        nz = blk[blk != 0].abs()
        if nz.numel() == 0:
            continue
        # ratio of any two nonzero magnitudes must equal a ratio of grid values
        ratios = nz / nz.min()
        grid = NVFP4_GRID[NVFP4_GRID > 0]
        valid = (grid.unsqueeze(0) / grid.unsqueeze(1)).flatten()
        assert all(torch.isclose(r, valid, rtol=1e-4).any() for r in ratios)


def test_zeropoint_beats_symmetric_on_skewed_input():
    # Post-SiLU-style skewed activations: shifting each block to its midpoint
    # must recover grid resolution lost to the unused negative half.
    from turboquant.nvfp4 import nvfp4_quantize_zp
    torch.manual_seed(0)
    x = torch.nn.functional.silu(torch.randn(256, 512) * 2)
    e_sym = ((x - nvfp4_quantize(x)) ** 2).sum()
    e_zp = ((x - nvfp4_quantize_zp(x)) ** 2).sum()
    assert e_zp < e_sym


def test_channel_equalization_beats_raw_on_hot_channel():
    # A persistently hot channel eats its block's scale every token; dividing
    # it out (folded into W offline) must lower quantization error.
    torch.manual_seed(0)
    x = torch.randn(256, 512)
    x[:, 7] *= 50.0  # persistent outlier channel
    s = x.abs().amax(dim=0).clamp_min(1e-5) ** 0.5
    e_raw = ((x - nvfp4_quantize(x)) ** 2).sum()
    e_eq = ((x - nvfp4_quantize(x / s) * s) ** 2).sum()
    assert e_eq < e_raw


def test_joint_optclip_zp_dominates_both_single_searches():
    # The 8-candidate joint search is a strict superset of zp-only (gamma=1)
    # and optclip-only (z=0), so its per-block MSE can never be worse.
    from turboquant.nvfp4 import nvfp4_quantize_zp
    torch.manual_seed(0)
    x = torch.cat([torch.randn(128, 256),
                   torch.randn(128, 256).abs() + 2.0], dim=0)

    def err(q):
        return ((x - q) ** 2).sum()

    e_joint = err(nvfp4_quantize_zp(x, optclip=True))
    assert e_joint <= err(nvfp4_quantize_zp(x)) + 1e-6
    assert e_joint <= err(nvfp4_quantize(x, optclip=True)) + 1e-6


def test_gptq_weight_beats_naive_on_correlated_hessian():
    # GPTQ error feedback through the activation Hessian must lower the
    # output-domain weight error ‖(W-Wq)Xᵀ‖² vs naive nearest NVFP4 rounding,
    # at the same E2M1+fp8 format and zero side bits.
    from turboquant.gptq import gptq_quantize_weight
    from turboquant.nvfp4 import nvfp4_quantize
    torch.manual_seed(0)
    out_d, in_d, T = 128, 256, 1024
    X = torch.randn(T, 48) @ torch.randn(48, in_d)   # correlated activations
    W = torch.randn(out_d, in_d)
    H = X.t() @ X / T

    def oerr(Wq):
        return (((W - Wq) @ X.t()) ** 2).sum()

    e_naive = oerr(nvfp4_quantize(W, block=16))
    e_gptq = oerr(gptq_quantize_weight(W, H, block=16))
    assert e_gptq < e_naive


def test_gptq_output_is_nvfp4_grid():
    # GPTQ output must still be (E2M1 value) x (fp8 block scale), per row/block.
    from turboquant.gptq import gptq_quantize_weight
    from turboquant.nvfp4 import NVFP4_GRID
    torch.manual_seed(2)
    W = torch.randn(8, 64)
    X = torch.randn(512, 64)
    H = X.t() @ X / 512                       # realistic positive-definite Hessian
    Wq = gptq_quantize_weight(W, H, block=16)
    grid = NVFP4_GRID[NVFP4_GRID > 0]
    for blk in Wq.reshape(-1, 16):
        nz = blk[blk != 0].abs()
        if nz.numel() == 0:
            continue
        ratios = nz / nz.min()
        valid = (grid.unsqueeze(0) / grid.unsqueeze(1)).flatten()
        assert all(torch.isclose(r, valid, rtol=1e-4).any() for r in ratios)


def test_output_domain_importance_lowers_output_error():
    # Weighting the per-block scale search by output-domain importance
    # (imp_i = ||W[i,:]||²) must lower ‖(Q(x)-x)W‖² vs plain element-MSE, since
    # it picks scales that protect the channels that actually drive the output.
    from turboquant.nvfp4 import nvfp4_quantize_zp
    torch.manual_seed(0)
    d, m, T = 256, 512, 256
    W = torch.randn(d, m)
    W[:8] *= 9.0   # a few input channels drive the output far more than the rest
    x = torch.randn(T, d) * 2
    imp = (W ** 2).sum(dim=1)  # diag(W Wᵀ): per-input-channel output importance

    def oerr(xh):
        return (((x - xh) @ W) ** 2).sum()

    assert oerr(nvfp4_quantize_zp(x, optclip=True, imp=imp)) < \
        oerr(nvfp4_quantize_zp(x, optclip=True))


def test_importance_none_matches_original():
    # imp=None must be byte-identical to the original uniform selection.
    from turboquant.nvfp4 import nvfp4_quantize_zp
    torch.manual_seed(1)
    x = torch.randn(64, 512) * 2
    assert torch.equal(nvfp4_quantize_zp(x, optclip=True),
                       nvfp4_quantize_zp(x, optclip=True, imp=None))


def test_qjl_correct_estimates_residual():
    torch.manual_seed(0)
    codec = TurboQuantActQuantizer(TurboQuantConfig(qjl_block=128, qjl_dim=64))
    r = torch.randn(64, 512)
    rec = codec.qjl_correct(r)
    assert ((r - rec) ** 2).sum() < (r ** 2).sum()


def test_eq_alpha_sweep_never_worse_than_fixed_half():
    # alpha=0.5 is in the candidate set, so the chosen scale's measured error
    # can only be <= the fixed-0.5 error on the same sample.
    from turboquant.nvfp4 import nvfp4_quantize_zp
    from turboquant.validation.hf_perplexity import _pick_eq_scale
    torch.manual_seed(0)
    x = torch.randn(128, 256)
    x[:, 3] *= 80.0
    amax = x.abs().amax(dim=0)

    def err(s):
        return ((x - nvfp4_quantize_zp(x / s, optclip=True) * s) ** 2).sum()

    assert err(_pick_eq_scale(x, amax, 16)) <= err(amax.clamp_min(1e-5) ** 0.5) + 1e-6


def test_waware_rounding_beats_nearest_in_output_domain():
    # Error feedback through G = AAᵀ must lower ||(x - x̂)A||² vs nearest
    # rounding, at zero side bits. Structured A gives within-block correlation.
    from turboquant.nvfp4 import nvfp4_quantize_waware, waware_comp
    torch.manual_seed(0)
    d, m, T = 256, 512, 256
    A = torch.randn(d, 32) @ torch.randn(32, m)
    x = torch.randn(T, d)
    comp = waware_comp(A)

    def oerr(xh):
        return (((x - xh) @ A) ** 2).sum()

    e_nearest = oerr(nvfp4_quantize(x))
    e_waware = oerr(nvfp4_quantize_waware(x, comp))
    assert e_waware < e_nearest


def test_waware_rounding_harmless_on_uncorrelated_weight():
    # With near-diagonal G the feedback is ~0 and waware must not be much
    # worse than nearest rounding (it can only differ via tiny eps terms).
    from turboquant.nvfp4 import nvfp4_quantize_waware, waware_comp
    torch.manual_seed(1)
    d, m, T = 128, 4096, 128
    A = torch.randn(d, m)  # i.i.d. -> G ~ m*I, off-diagonals ~ sqrt(m)
    x = torch.randn(T, d)
    comp = waware_comp(A)

    def oerr(xh):
        return (((x - xh) @ A) ** 2).sum()

    assert oerr(nvfp4_quantize_waware(x, comp)) <= 1.05 * oerr(nvfp4_quantize(x))


def test_svd_side_channel_beats_qjl_on_structured_input():
    # Low-rank-structured activations + structured W: SVD-aligned correction must
    # capture more *output-domain* error than random-direction QJL at equal bits.
    torch.manual_seed(0)
    d, m, T, k = 256, 512, 128, 16
    W = torch.randn(d, 32) @ torch.randn(32, m)            # structured weight
    x = torch.randn(T, 24) @ torch.randn(24, d) + 0.1 * torch.randn(T, d)
    basis = torch.linalg.svd(W, full_matrices=False).U[:, :k]

    q = TurboQuantActQuantizer(TurboQuantConfig(use_optclip=True, qjl_dim=0))
    qjl = TurboQuantActQuantizer(TurboQuantConfig(use_optclip=True, qjl_block=128, qjl_dim=64))

    def oerr(xh):
        return (((x - xh) @ W) ** 2).sum() / ((x @ W) ** 2).sum()

    assert oerr(q.fake_quantize_svd(x, basis)) < oerr(qjl.fake_quantize(x))

def test_hwht_transform_is_exact_inverse():
    # H²=I: applying the block Hadamard twice (no quant) returns the input.
    from turboquant.nvfp4 import _block_hwht
    torch.manual_seed(0)
    x = torch.randn(8, 256)
    assert torch.allclose(_block_hwht(_block_hwht(x, 16), 16), x, atol=1e-5)


def test_hwht_helps_outlier_blocks():
    # One large value per 16-block: the within-block Hadamard spreads it, so the
    # shared MX4 scale resolves the rest finer -> lower per-block MSE than plain.
    from turboquant.nvfp4 import nvfp4_quantize_hwht, nvfp4_quantize_zp
    torch.manual_seed(0)
    x = 0.1 * torch.randn(64, 512)
    x.reshape(64, 32, 16)[:, :, 0] += 6.0  # one outlier per block
    e_plain = ((x - nvfp4_quantize_zp(x, optclip=True)) ** 2).sum()
    e_hwht = ((x - nvfp4_quantize_hwht(x, optclip=True, hwht="always")) ** 2).sum()
    assert e_hwht < e_plain


def test_hwht_bestof_never_worse_than_plain():
    # best-of {rotate, no-rotate} per block includes the no-rotate option, so it
    # can't lose to plain zp on any distribution.
    from turboquant.nvfp4 import nvfp4_quantize_hwht, nvfp4_quantize_zp
    torch.manual_seed(1)
    x = torch.randn(64, 512)
    e_plain = ((x - nvfp4_quantize_zp(x, optclip=True)) ** 2).sum()
    e_best = ((x - nvfp4_quantize_hwht(x, optclip=True, hwht="bestof")) ** 2).sum()
    assert e_best <= e_plain + 1e-6

def test_hmask_all_false_equals_plain_and_all_true_equals_always():
    from turboquant.nvfp4 import nvfp4_quantize_hmask, nvfp4_quantize_hwht, nvfp4_quantize_zp
    torch.manual_seed(0)
    x = torch.randn(16, 256)
    nb = 256 // 16
    none = torch.zeros(nb, dtype=torch.bool)
    allm = torch.ones(nb, dtype=torch.bool)
    assert torch.allclose(nvfp4_quantize_hmask(x, none, optclip=True),
                          nvfp4_quantize_zp(x, optclip=True), atol=1e-5)
    assert torch.allclose(nvfp4_quantize_hmask(x, allm, optclip=True),
                          nvfp4_quantize_hwht(x, optclip=True, hwht="always"), atol=1e-5)


def test_hmask_selective_beats_always_on_mixed_blocks():
    # half the block positions are outlier-heavy (rotation helps), half smooth
    # (rotation hurts). A mask rotating only the outlier positions should beat
    # both always-rotate and plain on the mixed tensor.
    from turboquant.nvfp4 import nvfp4_quantize_hmask, nvfp4_quantize_hwht, nvfp4_quantize_zp
    torch.manual_seed(0)
    nb = 16
    x = 0.1 * torch.randn(128, nb * 16)
    xb = x.reshape(128, nb, 16)
    xb[:, : nb // 2, 0] += 6.0              # outliers in first half of positions
    x = xb.reshape(128, nb * 16)
    mask = torch.zeros(nb, dtype=torch.bool); mask[: nb // 2] = True
    e_plain = ((x - nvfp4_quantize_zp(x, optclip=True)) ** 2).sum()
    e_always = ((x - nvfp4_quantize_hwht(x, optclip=True, hwht="always")) ** 2).sum()
    e_mask = ((x - nvfp4_quantize_hmask(x, mask, optclip=True)) ** 2).sum()
    assert e_mask < e_plain and e_mask < e_always

def test_perm_identity_equals_plain():
    from turboquant.nvfp4 import nvfp4_quantize_perm, nvfp4_quantize_zp
    torch.manual_seed(0)
    x = torch.randn(8, 256)
    ident = torch.arange(256)
    assert torch.allclose(nvfp4_quantize_perm(x, ident, ident, optclip=True),
                          nvfp4_quantize_zp(x, optclip=True), atol=1e-5)


def test_perm_isolates_outlier_channels():
    # The real mechanism: a few *consistent* outlier channels scattered across
    # blocks pollute many block scales (each forces a coarse scale, crushing the
    # other 15). Grouping them (highest-amax-first) confines the damage to one
    # block, leaving the rest tight -> lower total error. (Magnitude-sort only
    # helps under this outlier-channel regime, NOT when large channels are
    # abundant — see the channel-permutation finding in the README.)
    from turboquant.nvfp4 import nvfp4_quantize_perm, nvfp4_quantize_zp
    torch.manual_seed(0)
    d = 256
    x = torch.randn(128, d)
    x[:, [3, 40, 77, 140, 201]] *= 30.0
    amax = x.abs().amax(0)
    perm = torch.argsort(amax, descending=True)
    inv = torch.empty_like(perm); inv[perm] = torch.arange(d)
    e_plain = ((x - nvfp4_quantize_zp(x, optclip=True)) ** 2).sum()
    e_perm = ((x - nvfp4_quantize_perm(x, perm, inv, optclip=True)) ** 2).sum()
    assert e_perm < e_plain
