#!/usr/bin/env bash
# ONE-SHOT B200 run: reproduce + extend the FP8-vs-NVFP4 decode throughput table
# (paper Table 6) on Llama-3.1-8B, adding batch 128 to the existing 1/32/64 rows,
# all measured in one consistent session on the box actually rented for this.
#
# Pipeline:
#   0. preflight   — GPU/SM check + CPU parity test
#   1. export      — stock ModelOpt NVFP4 checkpoint of Llama-3.1-8B (PTQ, no QAT,
#                     no kv4, no boundary tricks -- "NVIDIA's stock NVFP4 path",
#                     matching what produced the original table)
#   2. sweep       — vllm_fp4_bench.py at batch {1,32,64,128} x {fp8, nvfp4}
#   3. table       — parse the run's jsonl into the paper's Table 6 format
#
# Fail-soft per config: one batch/format failing (e.g. OOM at 128) is logged and
# skipped, not fatal -- every dollar of rental buys whatever rows DID complete.
set -u
source /venv/main/bin/activate 2>/dev/null || { echo "FATAL: no /venv/main venv"; exit 1; }

MODEL="${MODEL:-unsloth/Meta-Llama-3.1-8B}"
BATCHES="${BATCHES:-1 32 64 128}"
IN_LEN="${IN_LEN:-128}"
OUT_LEN="${OUT_LEN:-256}"
ITERS="${ITERS:-3}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN="results/box_run/${TS}"
mkdir -p "$RUN"
NVFP4_DIR="${RUN}/nvfp4_ckpt_stock"
BENCH_JSONL="${RUN}/vllm_bench.jsonl"
SUMMARY="${RUN}/SUMMARY.txt"
export PYTHONUNBUFFERED=1

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }
step() { log "STEP $1"; }

log "run dir: $RUN | model: $MODEL | batches: $BATCHES | in/out len: $IN_LEN/$OUT_LEN"

# 0. preflight -------------------------------------------------------------
step "0 preflight"
python -c "import modelopt, transformers, vllm" 2>/dev/null \
  || { log "FATAL: core deps missing — run 'bash setup_box_throughput.sh' first"; exit 1; }
python -c "import torch;cap=torch.cuda.get_device_capability();print('GPU',torch.cuda.get_device_name(),'sm_%d%d'%cap);assert cap[0]>=10,'NOT Blackwell — FP4 will emulate'" 2>&1 | tee -a "$SUMMARY"
python -m pytest turboquant/tests/test_modelopt_parity.py -q 2>&1 | tail -3 | tee -a "$SUMMARY"

# 1. export stock NVFP4 checkpoint ------------------------------------------
step "1 export stock NVFP4 checkpoint of $MODEL"
python -m turboquant.validation.export_nvfp4 --model "$MODEL" --out "$NVFP4_DIR" \
    2>&1 | tee "${RUN}/export.log"
grep -E "exporting|done" "${RUN}/export.log" 2>/dev/null | tee -a "$SUMMARY"
if [ ! -d "$NVFP4_DIR" ]; then
  log "FATAL: export produced no checkpoint at $NVFP4_DIR — aborting sweep"
  exit 1
fi

# 2. sweep: FP8 (dynamic, base model) and NVFP4 (exported ckpt), per batch --
step "2 throughput sweep"
for B in $BATCHES; do
  log "  batch=$B quantization=fp8"
  python -m turboquant.validation.vllm_fp4_bench --model "$MODEL" --quantization fp8 \
      --label "fp8_b${B}" --batch "$B" --in-len "$IN_LEN" --out-len "$OUT_LEN" \
      --iters "$ITERS" --out "$BENCH_JSONL" \
      >> "${RUN}/sweep.log" 2>&1 || log "    FAILED (see ${RUN}/sweep.log) — continuing"

  log "  batch=$B quantization=nvfp4 (stock, exported)"
  python -m turboquant.validation.vllm_fp4_bench --model "$NVFP4_DIR" \
      --label "nvfp4_b${B}" --batch "$B" --in-len "$IN_LEN" --out-len "$OUT_LEN" \
      --iters "$ITERS" --out "$BENCH_JSONL" \
      >> "${RUN}/sweep.log" 2>&1 || log "    FAILED (see ${RUN}/sweep.log) — continuing"
done

# 3. assemble the paper's Table 6 format -------------------------------------
step "3 results table"
python3 <<PYEOF | tee -a "$SUMMARY"
import json, collections
rows = collections.defaultdict(dict)
try:
    with open("$BENCH_JSONL") as f:
        for line in f:
            r = json.loads(line)
            fmt = "nvfp4" if r["label"].startswith("nvfp4") else "fp8"
            rows[r["batch"]][fmt] = r["tokens_per_sec"]
except FileNotFoundError:
    print("no results file — sweep produced nothing")
    raise SystemExit(0)

print(f"\n{'Batch':>6}{'FP8':>12}{'NVFP4':>12}{'NVFP4/FP8':>12}")
for b in sorted(rows):
    fp8 = rows[b].get("fp8")
    nvfp4 = rows[b].get("nvfp4")
    ratio = f"{nvfp4/fp8:.2f}" if fp8 and nvfp4 else "n/a"
    print(f"{b:>6}{fp8 if fp8 else 'n/a':>12}{nvfp4 if nvfp4 else 'n/a':>12}{ratio:>12}")
PYEOF

log "DONE. Raw rows: $BENCH_JSONL | NVFP4 ckpt: $NVFP4_DIR"
log "SAVE/UPLOAD $RUN before tearing the box down -- there is no second box."
echo; echo "===== SUMMARY ($RUN) ====="; cat "$SUMMARY"
