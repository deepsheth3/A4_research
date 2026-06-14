"""CPU tests for the global Walsh-Hadamard rotation baseline (QuaRot-style)."""

import torch

from turboquant.nvfp4 import _fwht, _largest_pow2_div, nvfp4_quantize_ghad


def test_fwht_orthonormal_involution():
    x = torch.randn(4, 64)
    y = _fwht(x)
    # orthonormal: norm preserved
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)
    # involution: applying twice is identity
    assert torch.allclose(_fwht(y), x, atol=1e-4)


def test_fwht_matches_dense_hadamard():
    # H_2 Kronecker construction vs the fast transform (n=8)
    H = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    Hn = H
    for _ in range(2):  # 2 -> 4 -> 8
        Hn = torch.kron(Hn, H)
    x = torch.randn(3, 8)
    ref = (x @ Hn.t()) / (8 ** 0.5)
    assert torch.allclose(_fwht(x), ref, atol=1e-4)


def test_largest_pow2_div():
    assert _largest_pow2_div(4096) == 4096
    assert _largest_pow2_div(14336) == 2048   # 2^11 * 7
    assert _largest_pow2_div(48) == 16


def test_ghad_quantize_runs_and_is_reasonable():
    x = torch.randn(2, 64)
    xq = nvfp4_quantize_ghad(x, block=16, optclip=True)
    assert xq.shape == x.shape
    # rotated quantization should track the input (cosine close to 1)
    cos = torch.nn.functional.cosine_similarity(x.flatten(), xq.flatten(), dim=0)
    assert cos > 0.9


def test_ghad_handles_non_power_of_2_tiles():
    # 48 = 16 * 3 -> tile size 16, applied over 3 tiles
    x = torch.randn(2, 48)
    xq = nvfp4_quantize_ghad(x, block=16, optclip=True)
    assert xq.shape == x.shape and torch.isfinite(xq).all()
