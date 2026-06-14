#!/usr/bin/env bash
# Cross-model floor-travels test (one box session, ~1 hour).
#
# For each model: run the 4-mode A4 perplexity (fp16 / fp8 / nvfp4_raw / our stack)
# AND capture activations for the free offline analyses (pattern-hunt, format-ceiling,
# rounding-ceiling). Confirms whether the FP4 floor — and the two-lever theorem —
# travel across model sizes and families, killing the single-model bias.
#
# Usage (on the box, after sync + pip install):
#   bash turboquant/scripts/run_crossmodel.sh
# Then copy results/ off before killing the box.
#
# Resilient: a failed model (gated/OOM) is logged and skipped, not fatal.

set -u
export OMNISTACK_PATH="${OMNISTACK_PATH:-$HOME/proj/Omnistack_RS}"
cd "$(dirname "$0")/../.." || exit 1            # -> repo root (NVFP4_Research)

# Ungated mirrors where possible; edit freely (set HF_TOKEN if a repo is gated).
MODELS=(
  "unsloth/Llama-3.2-1B"
  "unsloth/Llama-3.2-3B"
  "Qwen/Qwen2.5-7B"
  "unsloth/mistral-7b-v0.3"
)
MODES="fp16 fp8 nvfp4_raw nvfp4_eqzp_svd_qjl"
mkdir -p results
SUMMARY="results/crossmodel_summary.txt"
: > "$SUMMARY"

t0=$(date +%s)
for M in "${MODELS[@]}"; do
  tag=$(echo "$M" | tr '/' '_')
  echo "==================== $M ====================" | tee -a "$SUMMARY"

  echo "--- PPL: $MODES ---"
  if python3 -m turboquant.validation.hf_perplexity --model "$M" --modes $MODES 2>&1 \
       | grep -E "PPL =|test tokens|Traceback|Error|out of memory" | tee -a "$SUMMARY"; then :; fi
  [ "${PIPESTATUS[0]:-1}" -ne 0 ] && echo "  !! PPL FAILED for $M" | tee -a "$SUMMARY"

  echo "--- capture activations ---"
  python3 -m turboquant.validation.capture_activations --model "$M" \
      --every 16 --max-layers 10 --out "results/caps_${tag}.pt" 2>&1 \
      | grep -E "wrote|capturing|Traceback|Error|out of memory" | tee -a "$SUMMARY" \
      || echo "  !! CAPTURE FAILED for $M" | tee -a "$SUMMARY"
  echo "" | tee -a "$SUMMARY"
done

echo "ALL DONE in $(( ($(date +%s)-t0)/60 )) min." | tee -a "$SUMMARY"
echo "Copy results off:  scp -P <port> -r root@<ip>:~/proj/NVFP4_Research/results ." | tee -a "$SUMMARY"
echo "Then locally, per model:" | tee -a "$SUMMARY"
echo "  python3 -m turboquant.validation.pattern_hunt    --caps results/caps_<tag>.pt" | tee -a "$SUMMARY"
echo "  python3 -m turboquant.validation.format_ceiling  --caps results/caps_<tag>.pt" | tee -a "$SUMMARY"
echo "  python3 -m turboquant.validation.analyze_rounding_ceiling --caps results/caps_<tag>.pt" | tee -a "$SUMMARY"
