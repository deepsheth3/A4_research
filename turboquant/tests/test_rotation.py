import torch

from turboquant.act_codec import TurboQuantActQuantizer
from turboquant.config import TurboQuantConfig
from turboquant.rotation import rotate, unrotate


def test_rotation_is_orthogonal_round_trip():
    # R must be exactly orthogonal: unrotate(rotate(x)) == x. This is what makes
    # the offline weight-fold (X R)(Rᵀ W) = X W valid.
    for dim in (128, 768, 4096):  # incl. non-power-of-2 (gpt2's 768)
        x = torch.randn(8, dim)
        assert torch.allclose(unrotate(rotate(x, seed=3), seed=3), x, atol=1e-5)


def test_rotation_preserves_norm():
    x = torch.randn(16, 768)
    assert torch.allclose(rotate(x).norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_rotation_spreads_outliers():
    # A single outlier channel should be smeared across its block: post-rotation
    # kurtosis (outlier-ness) must drop sharply.
    x = torch.randn(64, 4096)
    x[:, 11] *= 100.0
    def kurt(t):
        t = (t - t.mean()) / t.std()
        return (t ** 4).mean().item()
    assert kurt(rotate(x)) < kurt(x) / 10


def test_seed_changes_rotation():
    x = torch.randn(4, 256)
    assert not torch.allclose(rotate(x, seed=0), rotate(x, seed=1))


def test_codec_round_trip_with_hadamard():
    cfg = TurboQuantConfig(use_hadamard=True)
    q = TurboQuantActQuantizer(cfg)
    x = torch.randn(4, 768)
    out = q.fake_quantize(x, layer_idx=5)
    assert out.shape == x.shape and torch.isfinite(out).all()
