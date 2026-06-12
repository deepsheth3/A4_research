"""TurboQuant activation codec: (PolarQuant) + NVFP4 + per-block 1-bit QJL.

Pipeline for an activation tensor ``x`` (last dim = the vector):
  1. PolarQuant (optional):  unit, magnitude = x/||x||, ||x||
  2. NVFP4:                   q = round_to_E2M1(unit, MX4 block scale)
  3. QJL (per sub-block):     encode residual (unit - q) to 1-bit signs + norm,
                              applied over ``qjl_block``-sized chunks so QJL's
                              k/dim ratio lands in the useful range.

Two stages are flag-gated so the ablation in ``validation/error_analysis.py`` can
show, on data, that PolarQuant is redundant with MX4 and that whole-vector QJL is
negligible. The QJL primitive itself is reused verbatim from OmniStack-RS.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from turboquant._omnistack import RademacherQJL
from turboquant.config import DEFAULT_CONFIG, TurboQuantConfig
from turboquant.nvfp4 import nvfp4_quantize, nvfp4_quantize_waware, round_e4m3
from turboquant.polarquant import polar_decompose, polar_reconstruct
from turboquant.rotation import rotate, unrotate


class PackedActivation(NamedTuple):
    """Quantized representation of one batch of activation vectors."""

    base_q: torch.Tensor          # (..., D) NVFP4-quantized (unit) direction
    magnitude: torch.Tensor | None  # (..., 1) L2 magnitude, or None if no PolarQuant
    signs: torch.Tensor | None    # (..., nblk, qjl_dim) bool, or None if QJL off
    norms: torch.Tensor | None    # (..., nblk, 1), or None
    block: int                    # effective QJL sub-block size used


class TurboQuantActQuantizer:
    """Online, calibration-free 4-bit activation quantizer."""

    def __init__(self, cfg: TurboQuantConfig = DEFAULT_CONFIG) -> None:
        self.cfg = cfg

    def _effective_block(self, dim: int) -> int:
        """QJL sub-block that divides ``dim``; fall back to the whole vector."""
        b = self.cfg.qjl_block
        return b if dim % b == 0 else dim

    def quantize(self, x: torch.Tensor, layer_idx: int = 0) -> PackedActivation:
        if self.cfg.use_hadamard:
            x = rotate(x, seed=layer_idx)
        if self.cfg.use_polarquant:
            unit, magnitude = polar_decompose(x)
        else:
            unit, magnitude = x, None

        base_q = nvfp4_quantize(
            unit, block=self.cfg.mx_block,
            quantize_scale=self.cfg.quantize_scale_fp8,
            optclip=self.cfg.use_optclip,
        )

        signs = norms = None
        block = self._effective_block(x.shape[-1])
        if self.cfg.qjl_dim > 0:
            residual = (unit - base_q).reshape(*x.shape[:-1], -1, block)
            qjl = RademacherQJL(head_dim=block, qjl_dim=self.cfg.qjl_dim)
            signs, norms = qjl.encode(residual, head_idx=layer_idx)

        return PackedActivation(base_q=base_q, magnitude=magnitude,
                                signs=signs, norms=norms, block=block)

    def dequantize(self, packed: PackedActivation, layer_idx: int = 0) -> torch.Tensor:
        unit_hat = packed.base_q
        if packed.signs is not None:
            qjl = RademacherQJL(head_dim=packed.block, qjl_dim=self.cfg.qjl_dim)
            rec = qjl.reconstruct(packed.signs, packed.norms, head_idx=layer_idx)
            unit_hat = unit_hat + rec.reshape_as(unit_hat)
        if packed.magnitude is not None:
            unit_hat = polar_reconstruct(unit_hat, packed.magnitude)
        if self.cfg.use_hadamard:
            unit_hat = unrotate(unit_hat, seed=layer_idx)
        return unit_hat

    def fake_quantize(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """Encode then decode — the quantization error a hook injects."""
        return self.dequantize(self.quantize(x, layer_idx), layer_idx).to(x.dtype)

    def fake_quantize_svd(self, x: torch.Tensor, basis: torch.Tensor,
                          layer_idx: int = 0) -> torch.Tensor:
        """NVFP4 + W-aware SVD-aligned residual side-channel (replaces QJL).

        ``basis`` is the layer weight's top-k *input-side* singular vectors
        (d, k), precomputed offline from W — no calibration data. The residual
        is projected onto it and the k coefficients stored in fp8. In deployment
        the correction rides a tiny side matmul ``c @ (basisᵀ W)`` (precomputed),
        so the main path stays pure FP4 tensor cores. At fp8 coeffs, k = d/16
        costs the same side-bits as QJL's 0.5 bits/element.

        If ``cfg.qjl_dim > 0``, per-block QJL is stacked on the residual that
        survives the SVD correction (doubles side info to ~1 bit/element).
        """
        base_q = nvfp4_quantize(
            x, block=self.cfg.mx_block,
            quantize_scale=self.cfg.quantize_scale_fp8,
            optclip=self.cfg.use_optclip,
        )
        coeff = (x - base_q) @ basis                       # (..., k) side info
        cs = coeff.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
        coeff = round_e4m3(coeff / cs) * cs                # fp8-quantized coeffs
        x_hat = base_q + coeff @ basis.T
        if self.cfg.qjl_dim > 0:
            block = self._effective_block(x.shape[-1])
            residual = (x - x_hat).reshape(*x.shape[:-1], -1, block)
            qjl = RademacherQJL(head_dim=block, qjl_dim=self.cfg.qjl_dim)
            signs, norms = qjl.encode(residual, head_idx=layer_idx)
            rec = qjl.reconstruct(signs, norms, head_idx=layer_idx)
            x_hat = x_hat + rec.reshape_as(x_hat)
        return x_hat.to(x.dtype)

    def qjl_correct(self, residual: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """Per-block QJL estimate of ``residual`` (encode + reconstruct)."""
        block = self._effective_block(residual.shape[-1])
        r = residual.reshape(*residual.shape[:-1], -1, block)
        qjl = RademacherQJL(head_dim=block, qjl_dim=self.cfg.qjl_dim)
        signs, norms = qjl.encode(r, head_idx=layer_idx)
        return qjl.reconstruct(signs, norms, head_idx=layer_idx).reshape_as(residual)

    def fake_quantize_war(self, x: torch.Tensor, comp: torch.Tensor,
                          basis: torch.Tensor | None = None,
                          layer_idx: int = 0) -> torch.Tensor:
        """W-aware-rounded NVFP4, optionally + SVD side-channel.

        ``comp`` is the per-layer feedback matrix from ``waware_comp`` (offline,
        from W only). Rounding error is steered into W's weak directions at the
        moment of quantization (zero side bits); ``basis`` then corrects what
        leaks into W's top subspace, as in :meth:`fake_quantize_svd`.
        """
        base_q = nvfp4_quantize_waware(
            x, comp, block=self.cfg.mx_block,
            quantize_scale=self.cfg.quantize_scale_fp8,
        )
        x_hat = base_q
        if basis is not None:
            coeff = (x - base_q) @ basis
            cs = coeff.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
            coeff = round_e4m3(coeff / cs) * cs
            x_hat = base_q + coeff @ basis.T
        return x_hat.to(x.dtype)
