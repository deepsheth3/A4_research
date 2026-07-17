#!/usr/bin/env bash
# Bootstrap a BARE Blackwell box (B200) for the FP8-vs-NVFP4 throughput sweep on
# Llama-3.1-8B. Separate from setup_box.sh (which targets the TinyLlama QAT run) --
# this run's model, deps, and deliverable are different, so it gets its own script
# rather than overloading the existing one.
#
# Copy the repo onto the box, cd into it, then: bash setup_box_throughput.sh
set -u

source /venv/main/bin/activate 2>/dev/null || { echo "FATAL: no /venv/main venv"; exit 1; }

echo "== 0. base check (torch in /venv/main, Blackwell) =="
python -c "import torch;assert torch.cuda.is_available();cap=torch.cuda.get_device_capability();print('torch',torch.__version__,'|',torch.cuda.get_device_name(),'sm_%d%d'%cap);assert cap[0]>=10,'not Blackwell — FP4 will emulate'" \
  || { echo "FATAL: torch/CUDA not usable on this box"; exit 1; }

TORCH_BASE=$(python -c "import torch;print(torch.__version__.split('+')[0])")
echo "torch==$TORCH_BASE" > /tmp/box_constraints.txt
echo "pinning torch==$TORCH_BASE"
if command -v uv >/dev/null 2>&1; then PIP="uv pip install -c /tmp/box_constraints.txt"
else PIP="pip install --no-input -c /tmp/box_constraints.txt"; fi
echo "using: $PIP"

echo "== 1. CORE deps (export path) — MANDATORY =="
$PIP "transformers>=4.44" "datasets>=2.20" accelerate sentencepiece pytest \
  || { echo "FATAL: core deps failed"; exit 1; }
$PIP "nvidia-modelopt[torch]" \
  || { echo "FATAL: modelopt failed — no export path"; exit 1; }

echo "== 2. vLLM — MANDATORY this run (throughput IS the deliverable here) =="
$PIP vllm \
  || { echo "FATAL: vllm failed to install — this run has nothing to measure without it"; exit 1; }

echo "== 3. pre-download Llama-3.1-8B (so the benchmark needs no network) =="
python -c "
from huggingface_hub import snapshot_download
snapshot_download('unsloth/Meta-Llama-3.1-8B')
print('cached Llama-3.1-8B')
" || { echo "FATAL: model download failed"; exit 1; }

echo "== 4. verify torch intact + repo preflight =="
python -c "import torch;assert torch.cuda.is_available(),'torch CUDA broke after installs!'" \
  || { echo "FATAL: an install broke torch — uninstall the last dep and re-run"; exit 1; }
python -m pytest turboquant/tests/test_modelopt_parity.py -q || { echo "FATAL: repo preflight failed"; exit 1; }

echo ""
echo "SETUP OK. Next: bash run_throughput_bench.sh"
