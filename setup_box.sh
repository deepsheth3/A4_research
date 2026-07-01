#!/usr/bin/env bash
# Bootstrap a BARE Blackwell box (only CUDA + PyTorch present) for the one-shot run.
# Copy the repo onto the box, cd into it, then: bash setup_box.sh
#
# Safety model: the box's PyTorch is a special Blackwell (sm_120) build. We PIN it so
# NO dependency can replace it. Core deps (the accuracy headline) are mandatory; the
# throughput deps (torchao/vllm) are best-effort — if they can't accept the pinned
# torch they fail to install and are skipped, leaving torch and the headline intact.
set -u

# vast.ai PyTorch image: torch is preinstalled in the /venv/main venv, NOT system python3.
source /venv/main/bin/activate 2>/dev/null || { echo "FATAL: no /venv/main venv"; exit 1; }

echo "== 0. base check (torch in /venv/main) =="
python -c "import torch;assert torch.cuda.is_available();cap=torch.cuda.get_device_capability();print('torch',torch.__version__,'|',torch.cuda.get_device_name(),'sm_%d%d'%cap);assert cap[0]>=10,'not Blackwell — FP4 will emulate'" \
  || { echo "FATAL: torch/CUDA not usable on this box"; exit 1; }

# Pin torch (base version, ignoring the +cuXXX local tag) so no dep can replace the
# Blackwell build. Prefer uv (fast, per the vast guide); fall back to pip.
TORCH_BASE=$(python -c "import torch;print(torch.__version__.split('+')[0])")
echo "torch==$TORCH_BASE" > /tmp/box_constraints.txt
echo "pinning torch==$TORCH_BASE"
if command -v uv >/dev/null 2>&1; then PIP="uv pip install -c /tmp/box_constraints.txt"
else PIP="pip install --no-input -c /tmp/box_constraints.txt"; fi
echo "using: $PIP"

echo "== 1. CORE deps (accuracy headline) — MANDATORY =="
$PIP "transformers>=4.44" "datasets>=2.20" accelerate sentencepiece pytest \
  || { echo "FATAL: core deps failed"; exit 1; }
$PIP "nvidia-modelopt[torch]" \
  || { echo "FATAL: modelopt failed — no export path"; exit 1; }

echo "== 2. THROUGHPUT deps — best-effort (failure is OK, headline unaffected) =="
$PIP torchao || echo "WARN: torchao failed -> gemm_bench will be skipped"
$PIP vllm     || echo "WARN: vllm failed -> throughput will be skipped. For sm_120 vllm see github.com/elsung/blackwell-llm-toolkit"

echo "== 3. pre-download model + wikitext (so the run needs no network) =="
python -c "
from huggingface_hub import snapshot_download
snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
from datasets import load_dataset
for s in ('test','train'): load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split=s)
print('cached model + wikitext')
" || { echo "FATAL: model/dataset download failed"; exit 1; }

echo "== 4. verify torch intact + repo preflight =="
python -c "import torch;assert torch.cuda.is_available(),'torch CUDA broke after installs!'" \
  || { echo "FATAL: an install broke torch — uninstall the last best-effort dep and re-run"; exit 1; }
python -m pytest turboquant/tests/test_modelopt_parity.py -q || { echo "FATAL: repo preflight failed"; exit 1; }

echo ""
echo "SETUP OK. Next: bash run_the_box.sh"
