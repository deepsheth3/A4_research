"""Parity between our fake-quant (nvfp4.py) and ModelOpt's NVFP4 definition.

Why this matters for productionization: our QAT trains weights to be robust to
*our* fake-quantizer. At deploy we export via ModelOpt (`mtq.quantize` +
`export_hf_checkpoint`), which re-quantizes those weights with *its* NVFP4. If the
two grids disagree, the exported checkpoint won't reproduce our measured PPL and
we'd only find out on a rented Blackwell box. This test pins the assumption on CPU.

ModelOpt NVFP4 (verified from modelopt_recipes/configs/numerics/nvfp4.yaml):
    num_bits: e2m1
    block_sizes: {-1: 16, type: dynamic, scale_bits: e4m3}
i.e. E2M1 values, block of 16 along the last axis, per-block scale stored in
fp8-e4m3, amax-based ("dynamic"). Optionally a per-tensor fp32 *global* scale sits
on top (the "two-level" scale) so the fp8 block scales use their range well; that
is the only structural difference from our single-level `nvfp4_quantize`.
"""
from __future__ import annotations

import torch

from turboquant.nvfp4 import (
    NVFP4_GRID,
    GRID_MAX,
    round_e4m3,
    _round_to_grid,
    nvfp4_quantize,
)

# The exact E2M1 value set ModelOpt targets (num_bits: e2m1).
_E2M1_SPEC = sorted(
    {v for m in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0) for v in (m, -m)}
)


def test_grid_is_exactly_e2m1():
    """Our grid is the E2M1 value set, max magnitude 6 — matches ModelOpt.

    E2M1 has 16 *codes* but +0 and -0 map to the same real value, so the distinct
    reconstruction grid is 15 values. That's what both formats round to.
    """
    assert GRID_MAX == 6.0
    assert NVFP4_GRID.tolist() == _E2M1_SPEC
    assert len(NVFP4_GRID) == 15


def _modelopt_nvfp4_two_level(x: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Reference two-level NVFP4 exactly as ModelOpt reconstructs it.

    Per-tensor fp32 global scale maps the largest block amax so block scales land
    in fp8-e4m3 range; each block scale is then stored in fp8. Effective per-block
    scale = (fp8 block scale) * (fp32 global). This is the deploy-time grid.
    """
    *lead, n = x.shape
    xb = x.reshape(*lead, n // block, block)
    block_amax = xb.abs().amax(dim=-1, keepdim=True)          # (..., nb, 1)
    tensor_amax = x.abs().amax().clamp_min(1e-12)
    s_global = tensor_amax / (448.0 * GRID_MAX)               # fp32 per-tensor
    block_scale = round_e4m3((block_amax / GRID_MAX) / s_global)  # fp8-e4m3 code
    eff = (block_scale * s_global).clamp_min(1e-12)           # effective scale
    q = _round_to_grid(xb / eff) * eff
    return q.reshape(*lead, n)


def _rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b) ** 2).mean().item() / (b ** 2).mean().item()


def test_export_preserves_our_accuracy():
    """Our single-level fake-quant is a *conservative* proxy for ModelOpt's deploy
    grid: it is never materially better, and at normal weight scales it's within a
    few percent. So exporting via ModelOpt cannot silently *lose* the accuracy we
    measured — at worst the deploy grid is slightly better (its per-tensor global
    scale rescues small-magnitude blocks out of fp8-e4m3 subnormals)."""
    torch.manual_seed(0)
    for scale in (0.02, 0.1, 1.0):                            # realistic weight stds
        w = torch.randn(64, 4096) * scale
        d_ours = _rel_mse(nvfp4_quantize(w, block=16, quantize_scale=True), w)
        d_ref = _rel_mse(_modelopt_nvfp4_two_level(w, block=16), w)
        # Direction: two-level (deploy) is at least as good as ours everywhere.
        assert d_ours >= d_ref * 0.99, (scale, d_ours, d_ref)
        # Tight parity once block scales clear the fp8 subnormal band (std >= 0.1).
        if scale >= 0.1:
            assert abs(d_ours - d_ref) / d_ref < 0.05, (scale, d_ours, d_ref)


def test_boundary_ignore_recipe():
    """The ModelOpt-style recipe keeps first+last decoder block and lm_head in BF16,
    and quantizes the interior — mirroring what actually deploys."""
    import torch.nn as nn
    from turboquant.validation.qat_nvfp4 import (
        replace_linears, boundary_ignore, QATLinear, _layer_indices,
    )

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(32, 32, bias=False)
            self.mlp = nn.Module()
            self.mlp.up_proj = nn.Linear(32, 64, bias=False)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block() for _ in range(4)])
            self.lm_head = nn.Linear(32, 100, bias=False)

    m = Model()
    assert _layer_indices(m) == [0, 1, 2, 3]
    pats = boundary_ignore(m)
    assert pats == ["lm_head", ".layers.0.", ".layers.3."]

    n = replace_linears(m, block=16, ignore=pats)
    bf16 = {nm for nm, mod in m.named_modules() if isinstance(mod, nn.Linear)}
    quant = {nm for nm, mod in m.named_modules() if isinstance(mod, QATLinear)}
    assert n == 4  # 2 interior blocks x 2 linears
    assert "lm_head" in bf16
    assert all(".layers.1." in x or ".layers.2." in x for x in quant)
    assert not any(".layers.1." in x or ".layers.2." in x for x in bf16)


def test_export_build_quant_cfg():
    """The exporter composes ModelOpt's QuantizeConfig correctly: KV merged, boundary
    blocks disabled *last* (so they win), and the base preset never mutated."""
    from turboquant.validation.export_nvfp4 import build_quant_cfg, layer_indices_from_names

    base = {"algorithm": "max", "quant_cfg": [
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": "e2m1"}},
        {"quantizer_name": "*lm_head*", "enable": False},
    ]}
    kv = {"quant_cfg": [{"quantizer_name": "*k_bmm_quantizer", "cfg": {"num_bits": "e2m1"}}]}
    out = build_quant_cfg(base, boundary_layers=[0, 31], kv_cfg=kv)
    names = [e.get("quantizer_name") for e in out["quant_cfg"]]

    assert len(base["quant_cfg"]) == 2                      # base untouched
    assert "*k_bmm_quantizer" in names                      # KV merged
    for i in (0, 31):
        entry = {"quantizer_name": f"*layers.{i}.*", "enable": False}
        assert entry in out["quant_cfg"]
        assert out["quant_cfg"].index(entry) > names.index("*lm_head*")  # appended last
    assert layer_indices_from_names(["x.layers.0.a", "x.layers.5.b", "n"]) == [0, 5]


def test_quant_error_is_in_expected_nvfp4_band():
    """Sanity: NVFP4 relative MSE on Gaussian weights sits in the known ~1e-3 band
    (a regression guard, and confirms we're not accidentally near-lossless/broken)."""
    torch.manual_seed(1)
    w = torch.randn(128, 4096) * 0.1
    d = _rel_mse(nvfp4_quantize(w, block=16, quantize_scale=True), w)
    assert 1e-4 < d < 1e-2, d
