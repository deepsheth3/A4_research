import torch

from turboquant.nvfp4 import NVFP4_GRID, GRID_MAX, nvfp4_quantize, _round_to_grid


def test_grid_is_e2m1():
    # E2M1 has 16 codes but +0/-0 coincide -> 15 distinct values, symmetric.
    vals = NVFP4_GRID.tolist()
    assert vals == sorted(vals)
    assert max(vals) == GRID_MAX and min(vals) == -GRID_MAX
    assert set(vals) == {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0}


def test_round_to_grid_snaps_to_grid_points():
    # Exact grid points are fixed points; midpoints round to a neighbour.
    g = NVFP4_GRID
    assert torch.equal(_round_to_grid(g), g)
    assert _round_to_grid(torch.tensor([5.0])).item() in (4.0, 6.0)
    assert _round_to_grid(torch.tensor([0.2])).item() in (0.0, 0.5)


def test_block_max_is_exactly_representable():
    # With fp32 scale, each block's max magnitude maps to GRID_MAX*scale = itself.
    x = torch.randn(4, 64)
    xh = nvfp4_quantize(x, block=16, quantize_scale=False)
    xb, xhb = x.reshape(-1, 16), xh.reshape(-1, 16)
    amax_idx = xb.abs().argmax(dim=-1)
    rows = torch.arange(xb.shape[0])
    assert torch.allclose(xhb[rows, amax_idx], xb[rows, amax_idx], atol=1e-5)


def test_round_trip_is_scale_invariant():
    # NVFP4 relative error is invariant to a global per-vector constant (MX4).
    x = torch.randn(8, 128)
    e1 = (x - nvfp4_quantize(x, quantize_scale=False)).pow(2).sum()
    e2 = (10 * x - nvfp4_quantize(10 * x, quantize_scale=False)).pow(2).sum()
    assert torch.allclose(e2, 100 * e1, rtol=1e-4)


def test_block_isolates_outlier():
    # A planted outlier only damages its own 16-block, not the rest.
    x = torch.randn(1, 64)
    x[0, 5] = 50.0
    xh = nvfp4_quantize(x, block=16, quantize_scale=False)
    clean_err = (x[0, 16:] - xh[0, 16:]).abs().max()
    assert clean_err < 0.2  # blocks 1..3 untouched by the block-0 outlier


def test_requires_divisible_last_dim():
    import pytest
    with pytest.raises(ValueError):
        nvfp4_quantize(torch.randn(2, 10), block=16)
