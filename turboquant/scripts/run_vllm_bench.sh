#!/usr/bin/env bash
# Real Blackwell decode throughput: bf16 vs FP8 vs NVFP4, Llama-3.1-8B, via vLLM.
# Goal = the FP4-vs-FP8 hardware speedup (the cost_ratio we assumed). ~20 min on a B200.
#
# REQUIRES a Blackwell GPU (SM100, e.g. B200) + vLLM >= 0.15 (use a vLLM-ready
# image; do NOT pip-install from scratch in a 20-min window). vLLM auto-detects
# the NVFP4/FP8 format from each ModelOpt checkpoint's config.json (no flag needed).
set -u
cd "$(dirname "$0")/../.." || exit 1
export HF_HOME="${HF_HOME:-/workspace/.hf_home}" HF_HUB_DISABLE_XET=1

BF16="${BF16:-unsloth/Meta-Llama-3.1-8B-Instruct}"
FP8="${FP8:-nvidia/Llama-3.1-8B-Instruct-FP8}"     # NVIDIA ModelOpt FP8
FP4="${FP4:-nvidia/Llama-3.1-8B-Instruct-NVFP4}"   # NVIDIA ModelOpt NVFP4 (verified ID)
B="${1:-1}"                                         # batch (1 = draft single-stream)
OUT=results/vllm_bench.jsonl; mkdir -p results; : > "$OUT"

echo "=== nvidia-smi ==="; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "=== gpu compute capability (need sm_100 Blackwell for FP4) ==="
python3 -c "import torch; print(torch.cuda.get_device_capability())" 2>/dev/null || true

run() { echo "-------------------- $1 --------------------"
        python3 -m turboquant.validation.vllm_fp4_bench --label "$1" \
          --batch "$B" --in-len 128 --out-len 256 --iters 3 "${@:2}" \
          2>&1 | grep -vE "^(INFO|WARNING|DEBUG)" | tail -25; }

run bf16 --model "$BF16"
run fp8  --model "$FP8"
run fp4  --model "$FP4"

echo "==================== SUMMARY (tokens/sec) ===================="
python3 - "$OUT" << 'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
d = {r["label"]: r["tokens_per_sec"] for r in rows}
for r in rows:
    print(f'  {r["label"]:6s}  {r["tokens_per_sec"]:>9.1f} tok/s')
if "fp8" in d and "fp4" in d and d["fp8"]:
    print(f'\n  FP4 vs FP8 speedup = {d["fp4"]/d["fp8"]:.2f}x  '
          f'(this is the cost_ratio^-1 our 2.86x projection assumed)')
if "bf16" in d and "fp4" in d and d["bf16"]:
    print(f'  FP4 vs bf16 speedup = {d["fp4"]/d["bf16"]:.2f}x')
PY
echo "DONE. copy results/vllm_bench.jsonl off the box."
