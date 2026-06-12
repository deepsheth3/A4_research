#!/usr/bin/env bash
# GSM8K accuracy run: fp16 vs nvfp4_eqzp_svd_qjl (TurboQuant best mode).
# One ephemeral H100; no persistent storage assumed.
#
# Usage:
#   bash turboquant/scripts/run_gsm8k_session.sh [MODEL]
#   MODEL defaults to unsloth/Meta-Llama-3.1-8B (ungated mirror).
set -euo pipefail

MODEL="${1:-unsloth/Meta-Llama-3.1-8B}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"

if [[ ! -d "${OMNISTACK_PATH:-../Omnistack_RS}/omnistack_rs" ]]; then
  echo "Cloning OmniStack-RS..."
  git clone https://github.com/deepsheth3/Omnistack-RS ../Omnistack_RS
fi

echo "=== Installing deps ==="
pip install -q -r requirements.txt

PYTHON=$(command -v python3 || command -v python)

echo "=== GSM8K: fp16 vs nvfp4_eqzp_svd_qjl on $MODEL ==="
$PYTHON -m turboquant.validation.gsm8k_eval \
  --model "$MODEL" \
  --modes fp16 nvfp4_eqzp_svd_qjl

echo
echo "=== DONE. Copy results off the box: ==="
ls -la results/
echo "  rsync -avz <box>:$HERE/results/ ./results/"
