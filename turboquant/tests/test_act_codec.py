import torch

from turboquant import nvfp4_quantize
from turboquant.act_codec import TurboQuantActQuantizer
from turboquant.config import TurboQuantConfig


def nmse(x, xh):
    return ((x - xh) ** 2).sum().item() / (x ** 2).sum().item()


def outlier_activations(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(64, 4096)
    x[:, [11, 880, 2000]] *= 60.0  # persistent per-channel outliers (the LLM pattern)
    return x


def test_round_trip_shape_and_dtype():
    q = TurboQuantActQuantizer()
    x = torch.randn(4, 256)
    out = q.fake_quantize(x)
    assert out.shape == x.shape and out.dtype == x.dtype


def test_per_block_qjl_reduces_error():
    # The corrected default (per-128-block QJL) must beat plain NVFP4.
    x = outlier_activations()
    raw = nvfp4_quantize(x, block=16)
    tq = TurboQuantActQuantizer(TurboQuantConfig()).fake_quantize(x)
    assert nmse(x, tq) < nmse(x, raw)


def test_whole_vector_qjl_is_negligible():
    # QJL@4096 with k=64 has ratio ~0.02 -> essentially no correction.
    x = outlier_activations()
    raw = nvfp4_quantize(x, block=16)
    full = TurboQuantActQuantizer(
        TurboQuantConfig(qjl_block=4096, use_polarquant=False)
    ).fake_quantize(x)
    assert abs(nmse(x, full) - nmse(x, raw)) / nmse(x, raw) < 0.05


def test_polarquant_does_not_help():
    # Adding PolarQuant should not reduce error vs the same config without it.
    x = outlier_activations()
    base = TurboQuantConfig(use_polarquant=False, qjl_block=128)
    withp = TurboQuantConfig(use_polarquant=True, qjl_block=128)
    e_base = nmse(x, TurboQuantActQuantizer(base).fake_quantize(x))
    e_with = nmse(x, TurboQuantActQuantizer(withp).fake_quantize(x))
    assert e_with >= e_base - 1e-6


def test_qjl_disabled_equals_nvfp4():
    x = outlier_activations()
    raw = nvfp4_quantize(x, block=16)
    off = TurboQuantActQuantizer(TurboQuantConfig(qjl_dim=0)).fake_quantize(x)
    assert torch.allclose(off, raw, atol=1e-5)


def test_non_divisible_dim_falls_back():
    q = TurboQuantActQuantizer(TurboQuantConfig(qjl_block=128))
    x = torch.randn(2, 4080)  # not divisible by 128
    out = q.fake_quantize(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
