#!/usr/bin/env bash
# ONE-SHOT box run (RTX Pro 6000 / sm_120). There is no second box — this must
# complete unattended and leave every artifact + log on disk.
#
# Pipeline (QAT is the deliverable; no PTQ anywhere):
#   0. preflight   — GPU/SM check + CPU test suite (repo intact, grid parity holds)
#   1. gemm sanity — real FP4 tensor-core GEMM (self-aborts if emulating)
#   2. QAT         — train TinyLlama to survive NVFP4, save BF16 QAT'd HF model
#   3. export+ppl  — ModelOpt NVFP4 export; PPL of the deployed grid (headline)
#   4. measure     — real FP4 throughput/latency in TRT-LLM on the exported ckpt
#
# Fail-soft: each step logs to results/box_run/<ts>/ and never aborts the script;
# a step that needs a missing upstream artifact is skipped and noted. Read SUMMARY
# at the end. Usage:  bash run_the_box.sh   (env overrides below)

set -u
source /venv/main/bin/activate 2>/dev/null || { echo "FATAL: no /venv/main venv"; exit 1; }
MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
STEPS="${STEPS:-3000}"          # heavy KL-QAT = the 70%-gap-closed recipe
LR="${LR:-3e-5}"
NTRAIN="${NTRAIN:-1500}"
EXTRA_QAT="${EXTRA_QAT:-}"      # e.g. "--kv4" for W4A4KV4, "--ignore-boundary" for the recipe variant
EXTRA_EXPORT="${EXTRA_EXPORT:-}"  # mirror --kv4 / --ignore-boundary here if used above

TS="$(date +%Y%m%d_%H%M%S)"
RUN="results/box_run/${TS}"
mkdir -p "$RUN"
QAT_DIR="${RUN}/qat_ckpt"
NVFP4_DIR="${RUN}/nvfp4_ckpt"
SUMMARY="${RUN}/SUMMARY.txt"
export PYTHONUNBUFFERED=1

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }
step() { log "STEP $1"; }

log "run dir: $RUN | model: $MODEL | steps: $STEPS | extra: '$EXTRA_QAT'"

# 0. preflight -----------------------------------------------------------------
step "0 preflight"
python -c "import modelopt, transformers, datasets" 2>/dev/null \
  || { log "FATAL: core deps missing — run 'bash setup_box.sh' first"; exit 1; }
python -c "import torch;cap=torch.cuda.get_device_capability();print('GPU',torch.cuda.get_device_name(),'sm_%d%d'%cap);assert cap[0]>=10,'NOT Blackwell — FP4 will emulate'" 2>&1 | tee -a "$SUMMARY"
python -m pytest turboquant/tests/test_modelopt_parity.py -q 2>&1 | tail -3 | tee -a "$SUMMARY"

# 1. gemm sanity (non-blocking) ------------------------------------------------
step "1 gemm_bench (real FP4 GEMM)"
python gemm_bench.py > "${RUN}/gemm_bench.log" 2>&1 \
  && log "gemm_bench: OK -> ${RUN}/gemm_bench.log" \
  || log "gemm_bench: FAILED/SUSPECT (see log) — throughput numbers below still worth capturing"
grep -E "device:|correctness|SUSPECT|real" "${RUN}/gemm_bench.log" 2>/dev/null | tee -a "$SUMMARY"

# 2. QAT (the deliverable) -----------------------------------------------------
step "2 QAT $MODEL ($STEPS steps)"
python -m turboquant.validation.qat_nvfp4 --model "$MODEL" \
    --objective kl --steps "$STEPS" --lr "$LR" --n-train "$NTRAIN" \
    --save-dir "$QAT_DIR" $EXTRA_QAT 2>&1 | tee "${RUN}/qat.log"
grep -E "FP16|PTQ|QAT|ACCEPTANCE|gap closed" "${RUN}/qat.log" 2>/dev/null | tee -a "$SUMMARY"

# 3. export + deployed-QAT PPL -------------------------------------------------
step "3 export NVFP4 + PPL"
if [ -f "${QAT_DIR}/config.json" ]; then
  python -m turboquant.validation.export_nvfp4 --model "$QAT_DIR" --out "$NVFP4_DIR" \
      --eval-ppl $EXTRA_EXPORT 2>&1 | tee "${RUN}/export.log"
  grep -E "DEPLOYED QAT|WikiText PPL|exporting|done" "${RUN}/export.log" 2>/dev/null | tee -a "$SUMMARY"
else
  log "export: SKIPPED — no QAT checkpoint at $QAT_DIR (step 2 failed)"
fi

# 4. measure throughput on real FP4 (trtllm->vllm auto-fallback; sm_120-safe) --
step "4 FP4 throughput (auto backend)"
if [ -f "${NVFP4_DIR}/config.json" ] || ls "${NVFP4_DIR}"/*.safetensors >/dev/null 2>&1; then
  python -m turboquant.validation.measure_deploy --model "$NVFP4_DIR" \
      --backend auto --batches 1 64 128 256 2>&1 | tee "${RUN}/measure.log"
  grep -E "backend in use|SANITY|batch|tok/s" "${RUN}/measure.log" 2>/dev/null | tee -a "$SUMMARY"
else
  log "measure: SKIPPED — no exported checkpoint (step 3 failed)"
fi

log "DONE. Artifacts in $RUN — SAVE/UPLOAD ${NVFP4_DIR} before tearing the box down."
echo; echo "===== SUMMARY ($RUN) ====="; cat "$SUMMARY"
