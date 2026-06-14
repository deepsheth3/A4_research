#!/usr/bin/env bash
# W4 study: find the Pareto-optimal weight quantization (the unoptimized leg).
# Isolates the two levers vs the current best (uniform in/8 = 6.294):
#   - RANK effect:       uniform in/8  ->  uniform in/6   (more rank, still < FP8 bytes)
#   - ALLOCATION effect: uniform in/8  ->  Fisher-alloc in/8 (same budget, smarter spread)
# All configs are Pareto-clean (< FP8's 1.0 byte/elem). Then compose W4A4KV4 on the winner.
#
# Usage (box): bash turboquant/scripts/run_w4_study.sh [MODEL]
set -u
export OMNISTACK_PATH="${OMNISTACK_PATH:-$HOME/proj/Omnistack_RS}"
cd "$(dirname "$0")/../.." || exit 1
M="${1:-unsloth/Meta-Llama-3.1-8B}"
A4="nvfp4_eqzp_svd_qjl"
LOG=results/w4_study.log; mkdir -p results; : > "$LOG"
echo "model=$M  (A4 control ~6.050, current best W4A4 uniform-in/8 ~6.294, fp8 5.948)" | tee -a "$LOG"

run() { echo "==================== $1 ====================" | tee -a "$LOG"
        shift; python3 -m turboquant.validation.hf_perplexity --model "$M" "$@" 2>&1 \
          | grep -E "PPL =|test tokens|Hessian|Fisher|ranks|Traceback|Error|out of memory" | tee -a "$LOG"; }

run "anchors: fp16 / fp8 / A4"            --modes fp16 fp8 $A4
run "W4A4 uniform in/8 (control)"         --modes $A4 --w4-gptq --w4-lowrank --w4-rank-div 8 --w4-lowrank-fp8
run "W4A4 uniform in/6 (more rank)"       --modes $A4 --w4-gptq --w4-lowrank --w4-rank-div 6 --w4-lowrank-fp8
run "W4A4 Fisher-alloc in/8 (new alloc)"  --modes $A4 --w4-gptq --w4-rank-alloc --w4-rank-div 8 --w4-lowrank-fp8

echo "==================== SUMMARY ====================" | tee -a "$LOG"
grep -E "====|PPL =" "$LOG" | tee -a "$LOG".summary
echo "DONE. Best W4 config -> use it in run_w4a4kv4.sh. Copy results/ off." | tee -a "$LOG"
