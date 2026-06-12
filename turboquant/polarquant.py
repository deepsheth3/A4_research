"""PolarQuant: separate a vector's magnitude from its direction.

For each activation vector ``x`` (the last dim is the vector), the L2 magnitude
is a per-vector scalar and the direction is the unit vector ``x / ||x||``. This
moves outliers — large values in a few dimensions — into the scalar magnitude,
leaving a bounded direction whose components lie in [-1, 1]. NVFP4's limited
dynamic range then quantizes the direction accurately, which is the whole point:
the within-group outlier problem that breaks raw NVFP4 is gone.
"""

from __future__ import annotations

import torch

_EPS = 1e-12


def polar_decompose(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``x`` into (unit direction, magnitude) along the last dim.

    Returns:
        unit: ``x / ||x||``, same shape as ``x``.
        magnitude: ``||x||`` per vector, shape ``(..., 1)``.
    """
    magnitude = x.norm(dim=-1, keepdim=True)
    unit = x / magnitude.clamp_min(_EPS)
    return unit, magnitude


def polar_reconstruct(unit_hat: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """Rebuild the activation from a (possibly quantized) direction + magnitude."""
    return unit_hat * magnitude
