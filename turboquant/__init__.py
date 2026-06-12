"""TurboQuant: near-lossless 4-bit activation quantization.

PolarQuant (magnitude/direction split) + NVFP4 (E2M1 fake-quant) + 1-bit QJL
residual correction, reusing OmniStack-RS's validated QJL primitive.
"""

from turboquant.act_codec import TurboQuantActQuantizer
from turboquant.config import DEFAULT_CONFIG, TurboQuantConfig
from turboquant.nvfp4 import NVFP4_GRID, nvfp4_quantize
from turboquant.polarquant import polar_decompose, polar_reconstruct

__all__ = [
    "TurboQuantActQuantizer",
    "TurboQuantConfig",
    "DEFAULT_CONFIG",
    "NVFP4_GRID",
    "nvfp4_quantize",
    "polar_decompose",
    "polar_reconstruct",
]
