"""Single source of truth for TurboQuant activation-codec hyperparameters.

The defaults encode the *corrected* design found empirically (see README and the
annotated research docs): PolarQuant is redundant with MX4 microscaling and is
off by default, and QJL is applied per sub-block (so its k/dim ratio is in the
useful range), not over the full hidden vector.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TurboQuantConfig:
    """Configuration for the (PolarQuant) + NVFP4 + per-block QJL activation codec.

    Attributes:
        mx_block: MX4 microscaling group size — one local scale per this many
            elements (NVFP4 uses 16).
        qjl_block: Sub-block size over which the 1-bit QJL residual is applied.
            QJL's MSE reduction is ~ qjl_dim*2/(pi*qjl_block), so this must be
            small relative to qjl_dim to matter (128 with qjl_dim=64 -> ratio
            0.5, ~31.8%). If the activation's last dim isn't divisible by it, the
            codec falls back to the full vector for that tensor.
        qjl_dim: Number of 1-bit QJL sign projections per block. 0 disables QJL.
        use_polarquant: Apply PolarQuant magnitude/direction split before NVFP4.
            Off by default — empirically a no-op over MX4 and slightly harmful
            with fp8 scales. Kept as a flag so the ablation can demonstrate this.
        use_hadamard: Apply a QuaRot-style seed-regenerated rotation
            (perm + signs + block-WHT) before quantization and invert after.
            In deployment the inverse folds into the weights offline (free);
            here it is simulated. Spreads outlier energy so NVFP4's block scale
            stops crushing small values. Off by default pending model-level
            validation (it raises NMSE while typically lowering loss).
        use_optclip: Per-block MSE-optimal scale search (ACIQ-style) instead of
            absmax. Pure hardware-native (still E2M1 + fp8 scale), no side info,
            ~3-7% lower output error measured on real gpt2 activations.
        quantize_scale_fp8: Round each MX4 block scale to fp8-e4m3 (real NVFP4
            storage) vs keep fp32 (idealized).
    """

    mx_block: int = 16
    qjl_block: int = 128
    qjl_dim: int = 64
    use_polarquant: bool = False
    use_hadamard: bool = False
    use_optclip: bool = False
    quantize_scale_fp8: bool = True

    def __post_init__(self) -> None:
        if self.mx_block <= 0:
            raise ValueError("mx_block must be positive")
        if self.qjl_block <= 0:
            raise ValueError("qjl_block must be positive")
        if self.qjl_dim < 0:
            raise ValueError("qjl_dim must be non-negative")


DEFAULT_CONFIG = TurboQuantConfig()
