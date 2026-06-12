"""Single import choke-point for the reused OmniStack-RS primitives.

We reuse OmniStack's already-validated 1-bit QJL residual codec rather than
reimplementing it. OmniStack lives in a sibling git repo, so we put it on
``sys.path`` here. The location is overridable via the ``OMNISTACK_PATH`` env
var; it defaults to ``../Omnistack_RS`` relative to this repo.

Only the pure-PyTorch ``omnistack_rs.quantization`` subpackage is imported —
NOT ``omnistack_rs.kernels.*``, which pulls in Triton (CUDA-only, unimportable
on a Mac).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parents[2] / "Omnistack_RS"
_OMNISTACK_PATH = Path(os.environ.get("OMNISTACK_PATH", _DEFAULT))

if not (_OMNISTACK_PATH / "omnistack_rs").is_dir():
    raise ImportError(
        f"OmniStack-RS not found at {_OMNISTACK_PATH}. Clone it as a sibling of "
        f"this repo, or set OMNISTACK_PATH to its location."
    )

if str(_OMNISTACK_PATH) not in sys.path:
    sys.path.insert(0, str(_OMNISTACK_PATH))

from omnistack_rs.quantization import RademacherQJL  # noqa: E402

__all__ = ["RademacherQJL"]
