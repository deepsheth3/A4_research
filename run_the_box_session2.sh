#!/usr/bin/env bash
# Session 2: close the gaps session 1 left open, so the story is shippable.
#
#   A. MoE phase (b)   — does starvation HURT, and does shrinkage fix it?
#                        (session 1 proved starvation exists, not that it matters)
#   B. MoE end-to-end  — our full method on a real MoE, PPL measured, vs rot_gptq
#                        (the two-lever theorem was derived on DENSE models)
#   C. 8B confirmation — rotation head-to-head + W4A8 at Llama-3.1-8B scale
#                        (every accuracy number so far is TinyLlama-1.1B)
#   D. Production path — real cutlass FP4 GEMM + correction beside it
#                        (the shippable config, not Triton-replaces-cuBLAS)
#
# Ordered cheapest-first so a truncated run still yields the most valuable results.
# Fail-soft: one experiment crashing does not stop the rest.
#
# Usage:  bash run_the_box_session2.sh
#         EXPS="A B" bash run_the_box_session2.sh      # subset
#         MODEL_8B=meta-llama/Llama-3.1-8B bash ...     # gated-repo alternative

set -u
source /venv/main/bin/activate 2>/dev/null || source /venv/box/bin/activate 2>/dev/null \
  || { echo "FATAL: no venv"; exit 1; }

MOE_MODEL="${MOE_MODEL:-allenai/OLMoE-1B-7B-0924}"
MODEL_8B="${MODEL_8B:-unsloth/Meta-Llama-3.1-8B}"
LIMIT="${LIMIT:-200000}"
LIMIT_8B="${LIMIT_8B:-100000}"
RANK_DIV="${RANK_DIV:-16}"
EXPS="${EXPS:-A B C D}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN="results/box_session2/${TS}"
mkdir -p "$RUN"
SUMMARY="${RUN}/SUMMARY.txt"
export PYTHONUNBUFFERED=1

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }
step() { log ""; log "===== $* ====="; }
has()  { echo " $EXPS " | grep -q " $1 "; }

log "run: $RUN | moe: $MOE_MODEL | 8b: $MODEL_8B | exps: $EXPS"

step "preflight"
python -c "
import torch
cap = torch.cuda.get_device_capability()
print('GPU', torch.cuda.get_device_name(), 'sm_%d%d' % cap,
      '| %.0f GB' % (torch.cuda.get_device_properties(0).total_memory/1e9))
assert cap[0] >= 10, 'NOT Blackwell — FP4 timings meaningless'
" 2>&1 | tee -a "$SUMMARY" || log "WARN: preflight failed"

# -- A: MoE phase (b) — the biggest hole in the story ---------------------------
if has A; then
step "A  MoE phase (b): does starvation hurt, does shrinkage fix it"
python moe_phase_b.py --model "$MOE_MODEL" --rank-div "$RANK_DIV" \
    > "${RUN}/a_moe_phase_b.log" 2>&1 \
  && log "A: OK" || log "A: FAILED (see a_moe_phase_b.log)"
grep -aE "starved-Q|rtn |ours|WORSE|wrote " "${RUN}/a_moe_phase_b.log" 2>/dev/null \
  | tail -30 | tee -a "$SUMMARY"
fi

# -- B: MoE end-to-end PPL ------------------------------------------------------
if has B; then
step "B  MoE end-to-end: our method vs rot_gptq on a real MoE"
python moe_e2e.py --model "$MOE_MODEL" --limit "$LIMIT" --rank-div "$RANK_DIV" \
    > "${RUN}/b_moe_e2e.log" 2>&1 \
  && log "B: OK" || log "B: FAILED (see b_moe_e2e.log)"
grep -aE "PPL = |best|ours_|shrinkage|wrote " "${RUN}/b_moe_e2e.log" 2>/dev/null \
  | tee -a "$SUMMARY"
fi

# -- C: 8B confirmation ---------------------------------------------------------
if has C; then
step "C1  8B rotation head-to-head"
python rotation_headtohead.py --model "$MODEL_8B" --limit "$LIMIT_8B" \
    --acts fp8 --rank-div "$RANK_DIV" \
    > "${RUN}/c1_8b_headtohead.log" 2>&1 \
  && log "C1: OK" || log "C1: FAILED (see c1_8b_headtohead.log)"
grep -aE "PPL = |best_|ours_|rotation_|wrote " "${RUN}/c1_8b_headtohead.log" 2>/dev/null \
  | tee -a "$SUMMARY"

step "C2  8B W4A8"
python -m turboquant.validation.hf_perplexity --model "$MODEL_8B" \
    --limit "$LIMIT_8B" --max-len 1024 \
    --modes fp16 fp8 nvfp4_raw w4a8 --w4 --w4-gptq --w4-rtn --w4-lowrank \
    --w4-lowrank-fp8 --w4-rank-div "$RANK_DIV" \
    > "${RUN}/c2_8b_w4a8.log" 2>&1 \
  && log "C2: OK" || log "C2: FAILED (see c2_8b_w4a8.log)"
grep -aE "PPL = |wrote " "${RUN}/c2_8b_w4a8.log" 2>/dev/null | tee -a "$SUMMARY"
fi

# -- D: production path ---------------------------------------------------------
if has D; then
step "D  production path: real cutlass FP4 GEMM + correction beside it"
python production_path_bench.py --rank-div "$RANK_DIV" \
    > "${RUN}/d_production_path.log" 2>&1 \
  && log "D: OK" || log "D: FAILED/SKIPPED (needs vllm; see d_production_path.log)"
grep -aE "supports_fp4|overhead|mean_corr|^ *[0-9]+ +[0-9]+|wrote " \
  "${RUN}/d_production_path.log" 2>/dev/null | tail -30 | tee -a "$SUMMARY"
fi

step "DONE"
log "artifacts: $RUN"
log "RETRIEVE BEFORE TEARDOWN:"
log "  rsync -avz --include='*/' --include='*.json' --include='*.log' --exclude='*' \\"
log "        -e 'ssh -p PORT' root@HOST:/workspace/NVFP4_Research/results/ ./results/"
echo; echo "===== SUMMARY ($RUN) ====="; cat "$SUMMARY"
