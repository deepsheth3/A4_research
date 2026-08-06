#!/usr/bin/env bash
# Bootstrap a TRULY BARE vast.ai box (vastai/base-image: CUDA + python, NO torch).
#
# Difference from setup_box.sh: that one assumes vast.ai's *PyTorch* image, where a
# Blackwell torch build is preinstalled in /venv/main and must be PINNED so nothing
# replaces it. Here there is no torch at all, so we INSTALL it first (matched to the
# box's CUDA), then pin that version for everything downstream.
#
# Usage:  bash setup_box_bare.sh            # auto-detect CUDA, pick wheel index
#         TORCH_INDEX=cu128 bash setup_box_bare.sh   # force a specific wheel build
set -u

echo "== 0. environment probe =="
echo "nvidia-smi:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || {
  echo "FATAL: no nvidia-smi — this box has no usable GPU"; exit 1; }

# CUDA runtime version the DRIVER supports (not the toolkit's).
CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "driver: $CUDA_VER"
nvcc --version 2>/dev/null | tail -2 || echo "(no nvcc — fine, we use prebuilt wheels)"

# Some vast base images ship a venv, some don't. Use one if present, else make one.
if [ -d /venv/main ]; then
  echo "found /venv/main — activating"
  source /venv/main/bin/activate
else
  echo "no /venv/main — creating /venv/box"
  python3 -m venv /venv/box 2>/dev/null || { apt-get update -qq && apt-get install -y -qq python3-venv && python3 -m venv /venv/box; }
  source /venv/box/bin/activate
fi
python -V

# uv is dramatically faster on a fresh box; install it if missing.
if ! command -v uv >/dev/null 2>&1; then
  pip install --no-input -q uv 2>/dev/null || true
fi
if command -v uv >/dev/null 2>&1; then PIPI="uv pip install"; else PIPI="pip install --no-input"; fi
echo "installer: $PIPI"

echo ""
echo "== 1. PyTorch (Blackwell sm_120 needs a recent CUDA wheel) =="
if python -c "import torch" 2>/dev/null; then
  echo "torch already present: $(python -c 'import torch;print(torch.__version__)')"
else
  # sm_120 (Blackwell) support landed in the cu128+ wheels. Default to cu128; allow
  # override via TORCH_INDEX for newer boxes (e.g. cu130 on CUDA 13 images).
  IDX="${TORCH_INDEX:-cu128}"
  echo "installing torch from the $IDX wheel index..."
  $PIPI torch --index-url "https://download.pytorch.org/whl/$IDX" || {
    echo "WARN: $IDX wheels failed; retrying with the default PyPI index"
    $PIPI torch || { echo "FATAL: could not install torch"; exit 1; }
  }
fi

python - <<'PY' || { echo "FATAL: torch cannot see the GPU"; exit 1; }
import torch
assert torch.cuda.is_available(), "torch installed but CUDA unavailable"
cap = torch.cuda.get_device_capability()
print(f"torch {torch.__version__} | {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}")
if cap[0] < 10:
    print("WARNING: not Blackwell — FP4 tensor-core paths will EMULATE, timings meaningless")
PY

# Pin whatever torch we ended up with so no downstream dep swaps it out.
TORCH_BASE=$(python -c "import torch;print(torch.__version__.split('+')[0])")
echo "torch==$TORCH_BASE" > /tmp/box_constraints.txt
echo "pinned torch==$TORCH_BASE"
if command -v uv >/dev/null 2>&1; then PIP="uv pip install -c /tmp/box_constraints.txt"
else PIP="pip install --no-input -c /tmp/box_constraints.txt"; fi

echo ""
echo "== 2. CORE deps (accuracy headline) — MANDATORY =="
$PIP "transformers>=4.44" "datasets>=2.20" accelerate sentencepiece pytest \
  || { echo "FATAL: core deps failed"; exit 1; }

echo ""
echo "== 3. BEST-EFFORT deps (failure is OK — the thread that needs it is skipped) =="
# triton: Thread 4's fused kernel. Usually ships WITH torch on linux; install only if absent.
python -c "import triton" 2>/dev/null && echo "triton already present (bundled with torch)" \
  || $PIP triton || echo "WARN: triton failed -> Thread 4 fusion timing will be skipped"
# modelopt: only needed for the export/deploy path, not for any of the 4 threads.
$PIP "nvidia-modelopt[torch]" || echo "WARN: modelopt failed -> export path unavailable (threads 1-4 unaffected)"

echo ""
echo "== 4. pre-download models + data (so the run needs no network) =="
python - <<'PY' || { echo "FATAL: model/dataset download failed"; exit 1; }
from huggingface_hub import snapshot_download
from datasets import load_dataset
snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
print("cached TinyLlama")
try:
    snapshot_download('allenai/OLMoE-1B-7B-0924')
    print("cached OLMoE (thread 2)")
except Exception as e:
    print(f"WARN: OLMoE download failed -> thread 2 will be skipped ({e!r:.100})")
for s in ('test', 'train'):
    load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=s)
print("cached wikitext")
PY

echo ""
echo "== 5. verify torch intact after installs =="
python -c "import torch;assert torch.cuda.is_available(),'an install broke torch CUDA!'" \
  || { echo "FATAL: an install broke torch"; exit 1; }

echo ""
echo "SETUP OK. Next: bash run_the_box_threads.sh"
