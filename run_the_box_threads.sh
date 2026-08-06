#!/usr/bin/env bash
# Four-thread research run (see .claude/plans/ four-thread plan).
#
# The method under test everywhere is OURS: equalization + additive low-rank
# correction. NOT rotation, NOT GPTQ's coordinated rounding — thread 1 measures the
# GPTQ ablation precisely to show that leg is inert.
#
#   T3. W4A8        — NVFP4 weights (eq+lowrank) + FP8 activations. Cheapest, first.
#   T1. KV4+stack   — full stack PPL w/ KV4 off/raw/+lowrank, + the GPTQ ablation.
#   T2. MoE         — per-expert Hessian conditioning vs token count (OLMoE).
#   T4. fusion      — fused low-rank epilogue vs two kernel launches (timing).
#
# Everything below was written and correctness-checked on CPU/Mac first; the GPU is
# needed because the low-rank path uses linalg_qr/solve_triangular (absent on MPS)
# and because fusion timing is meaningless off real FP4 hardware.
#
# Fail-soft: a thread that crashes logs and is skipped; the rest still run.
# Usage:  bash run_the_box_threads.sh          (env overrides below)
#         THREADS="3 4" bash run_the_box_threads.sh    # run a subset

set -u
source /venv/main/bin/activate 2>/dev/null || source /venv/box/bin/activate 2>/dev/null \
  || { echo "FATAL: no venv (/venv/main or /venv/box) — run setup first"; exit 1; }

MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MOE_MODEL="${MOE_MODEL:-allenai/OLMoE-1B-7B-0924}"
LIMIT="${LIMIT:-200000}"          # eval tokens; 0 = full wikitext-2 test split
MAXLEN="${MAXLEN:-1024}"
RANK_DIV="${RANK_DIV:-16}"        # low-rank correction rank = in/RANK_DIV
THREADS="${THREADS:-3 1 2 4}"     # order = cheapest/most-certain first

TS="$(date +%Y%m%d_%H%M%S)"
RUN="results/box_threads/${TS}"
mkdir -p "$RUN"
SUMMARY="${RUN}/SUMMARY.txt"
export PYTHONUNBUFFERED=1

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }
step() { log ""; log "===== $* ====="; }
has()  { echo " $THREADS " | grep -q " $1 "; }

log "run dir: $RUN"
log "model: $MODEL | moe: $MOE_MODEL | limit: $LIMIT | rank: in/$RANK_DIV"
log "threads: $THREADS"

# -- preflight ----------------------------------------------------------------
step "preflight"
python -c "
import torch
cap = torch.cuda.get_device_capability()
print('GPU', torch.cuda.get_device_name(), 'sm_%d%d' % cap)
assert cap[0] >= 10, 'NOT Blackwell — FP4 timings would be EMULATED and meaningless'
" 2>&1 | tee -a "$SUMMARY" || log "WARN: preflight GPU check failed (T4 timings suspect)"
python -c "import transformers, datasets; print('deps OK')" 2>&1 | tee -a "$SUMMARY" \
  || { log "FATAL: core deps missing — run setup_box.sh first"; exit 1; }

# -- T3: W4A8 (cheapest, most certain) ----------------------------------------
if has 3; then
step "T3  W4A8: NVFP4 weights (eq + additive low-rank) + FP8 activations"
python -m turboquant.validation.hf_perplexity --model "$MODEL" \
    --limit "$LIMIT" --max-len "$MAXLEN" \
    --modes fp16 fp8 nvfp4_raw w4a8 turboquant \
    --w4 --w4-gptq --w4-rtn --w4-lowrank --w4-lowrank-fp8 \
    --w4-rank-div "$RANK_DIV" \
    > "${RUN}/t3_w4a8.log" 2>&1 \
  && log "T3: OK" || log "T3: FAILED (see t3_w4a8.log)"
grep -aE "PPL =|vs fp8|wrote " "${RUN}/t3_w4a8.log" 2>/dev/null | tee -a "$SUMMARY"
fi

# -- T1: KV4 + full stack, with the two-lever (GPTQ-inert) ablation ------------
if has 1; then
step "T1  full stack PPL + KV4 legs + GPTQ ablation"
python -m turboquant.validation.ppl_fullstack --model "$MODEL" \
    --limit "$LIMIT" --max-len "$MAXLEN" --w4-rank-div "$RANK_DIV" --ablate-gptq \
    > "${RUN}/t1_fullstack.log" 2>&1 \
  && log "T1: OK" || log "T1: FAILED (see t1_fullstack.log)"
grep -aE "PPL =|stack_vs_fp16|kv4_|gptq_vs_rtn|wrote " "${RUN}/t1_fullstack.log" 2>/dev/null | tee -a "$SUMMARY"
fi

# -- T2: MoE token-starved experts (new code — most likely to need a fix) ------
if has 2; then
step "T2  MoE per-expert Hessian conditioning vs token count ($MOE_MODEL)"
if [ -f turboquant/validation/moe_expert_hessians.py ]; then
  python -m turboquant.validation.moe_expert_hessians --model "$MOE_MODEL" \
      --max-len "$MAXLEN" \
      > "${RUN}/t2_moe.log" 2>&1 \
    && log "T2: OK" || log "T2: FAILED (see t2_moe.log) — new code, check the traceback"
  grep -aE "experts|bucket|cond|rank|tokens|wrote " "${RUN}/t2_moe.log" 2>/dev/null | tail -25 | tee -a "$SUMMARY"
else
  log "T2: SKIPPED — moe_expert_hessians.py not present in this checkout"
fi
fi

# -- T4: fused low-rank epilogue timing ---------------------------------------
if has 4; then
step "T4  fused low-rank epilogue vs two kernel launches"
python triton_lowrank_fusion_bench.py --rank-div "$RANK_DIV" \
    > "${RUN}/t4_fusion.log" 2>&1 \
  && log "T4: OK" || log "T4: FAILED (see t4_fusion.log)"
grep -aE "correctness|PASS|FAIL|speedup|^ *[0-9]+ +[0-9]+" "${RUN}/t4_fusion.log" 2>/dev/null | tee -a "$SUMMARY"
fi

# -- wrap ---------------------------------------------------------------------
step "DONE"
log "artifacts: $RUN"
log "RETRIEVE BEFORE TEARDOWN:  results/  (JSONs + all four logs)"
log "  rsync -avz --include='*/' --include='*.json' --include='*.log' --exclude='*' \\"
log "        root@<box>:/workspace/NVFP4_Research/results/ ./results/"
echo; echo "===== SUMMARY ($RUN) ====="; cat "$SUMMARY"
