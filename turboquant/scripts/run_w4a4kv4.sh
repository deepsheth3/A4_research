#!/usr/bin/env bash
# The "4-bit everything" composition: A4 -> W4A4 -> W4A4KV4 on one box.
# Decomposes the cost of each leg (same-box, no cross-box drift) + fp16/fp8 anchors.
#
# Usage (box, after sync + pip install):
#   bash turboquant/scripts/run_w4a4kv4.sh [MODEL]
set -u
export OMNISTACK_PATH="${OMNISTACK_PATH:-$HOME/proj/Omnistack_RS}"
cd "$(dirname "$0")/../.." || exit 1

M="${1:-unsloth/Meta-Llama-3.1-8B}"
W4="--w4-gptq --w4-lowrank --w4-rank-div 8 --w4-lowrank-fp8"
LOG=results/w4a4kv4.log
mkdir -p results; : > "$LOG"
echo "model=$M" | tee -a "$LOG"

echo "==================== A4 (+ fp16/fp8 anchors) ====================" | tee -a "$LOG"
python3 -m turboquant.validation.hf_perplexity --model "$M" \
    --modes fp16 fp8 nvfp4_eqzp_svd_qjl 2>&1 \
  | grep -E "PPL =|test tokens|Traceback|Error|out of memory" | tee -a "$LOG"

echo "==================== W4A4 ====================" | tee -a "$LOG"
python3 -m turboquant.validation.hf_perplexity --model "$M" \
    --modes nvfp4_eqzp_svd_qjl $W4 2>&1 \
  | grep -E "PPL =|Traceback|Error|out of memory" | tee -a "$LOG"

echo "==================== W4A4KV4 (4-bit everything) ====================" | tee -a "$LOG"
python3 -m turboquant.validation.hf_perplexity --model "$M" \
    --modes nvfp4_eqzp_svd_qjl $W4 --kv4 2>&1 \
  | grep -E "PPL =|Traceback|Error|out of memory" | tee -a "$LOG"

echo "==================== SUMMARY ====================" | tee -a "$LOG"
echo "A4 / W4A4 / W4A4KV4 PPLs above; compare to fp8 (the bar)." | tee -a "$LOG"
echo "DONE. copy results/w4a4kv4.log + results/perplexity_*.json off the box." | tee -a "$LOG"
