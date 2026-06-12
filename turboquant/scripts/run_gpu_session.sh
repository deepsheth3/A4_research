#!/usr/bin/env bash
# One-shot TurboQuant activation-quantization run for a SINGLE ephemeral H100.
# No persistent storage assumed: clone, install, run, copy ./results off the box.
#
# Usage:
#   bash turboquant/scripts/run_gpu_session.sh [MODEL]
#   MODEL defaults to a small open model; pass e.g. meta-llama/Llama-3.1-8B
#   (set HF_TOKEN first for gated repos: export HF_TOKEN=hf_...).
set -euo pipefail

MODEL="${1:-mistralai/Mistral-7B-v0.3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # NVFP4_Research root
cd "$HERE"

# --- OmniStack-RS (reused QJL primitive) must be a sibling, or set OMNISTACK_PATH ---
if [[ ! -d "${OMNISTACK_PATH:-../Omnistack_RS}/omnistack_rs" ]]; then
  echo "Cloning OmniStack-RS as a sibling repo..."
  git clone https://github.com/deepsheth3/Omnistack-RS ../Omnistack_RS
fi

echo "=== Installing deps ==="
pip install -q -r requirements.txt

PYTHON=$(command -v python3 || command -v python)

echo "=== Sanity: unit tests (CPU) ==="
$PYTHON -m pytest turboquant/tests -q

echo "=== Experiment B: WikiText-2 perplexity (6 modes) on $MODEL ==="
# Loads the model once per mode; full WikiText-2 test set. One H100, no sharding.
# turboquant_svd uses W-aware SVD residual; SVD basis computed on CPU (offline step).
$PYTHON -m turboquant.validation.hf_perplexity \
  --model "$MODEL" \
  --modes fp16 fp8 nvfp4_raw turboquant turboquant_opt turboquant_svd

echo
echo "=== DONE. Copy results off the box before it terminates: ==="
ls -la results/
echo "  scp -r <box>:$HERE/results ."
