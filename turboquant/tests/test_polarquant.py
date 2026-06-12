import torch

from turboquant.nvfp4 import nvfp4_quantize
from turboquant.polarquant import polar_decompose, polar_reconstruct


def test_decompose_reconstruct_identity():
    x = torch.randn(16, 256) * 3.0
    unit, mag = polar_decompose(x)
    assert torch.allclose(unit.norm(dim=-1), torch.ones(16), atol=1e-5)
    assert torch.allclose(polar_reconstruct(unit, mag), x, atol=1e-5)


def test_handles_zero_vector():
    x = torch.zeros(2, 8)
    unit, mag = polar_decompose(x)
    assert torch.isfinite(unit).all()
    assert torch.allclose(polar_reconstruct(unit, mag), x)


def test_polarquant_is_noop_over_mx4():
    # Key finding: with fp32 block scales, PolarQuant before NVFP4 is identical to
    # raw NVFP4 — MX4 microscaling already absorbs the per-vector magnitude.
    x = torch.randn(32, 512)
    x[:, [3, 200]] *= 40.0
    raw = nvfp4_quantize(x, block=16, quantize_scale=False)
    unit, mag = polar_decompose(x)
    polar = polar_reconstruct(nvfp4_quantize(unit, block=16, quantize_scale=False), mag)
    assert (raw - polar).abs().max() < 1e-3
