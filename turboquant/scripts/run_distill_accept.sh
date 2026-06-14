#!/usr/bin/env bash
# The secret-sauce session: trained low-rank factors (KL-distill vs FP16 teacher)
# -> W4A4KV4 PPL + speculative-decoding ACCEPTANCE, with a QuaRot baseline.
# One H100, Llama-8B, ~$5. Code is pre-debugged offline (saddle-safe, keep-best).
set -u
export OMNISTACK_PATH="${OMNISTACK_PATH:-$HOME/proj/Omnistack_RS}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.." || exit 1
M="${1:-unsloth/Meta-Llama-3.1-8B}"
mkdir -p results; LOG=results/distill_accept.log; : > "$LOG"

echo "==================== TRAINED FACTORS: W4A4KV4 + acceptance ====================" | tee -a "$LOG"
# in/6 weights (the W4 study winner) + KV4; SVD baseline vs distilled, both eval'd.
python3 -m turboquant.validation.distill_accept --model "$M" \
    --rank-div 6 --kv4 --epochs 1 --lr 2e-3 --calib-seqs 64 --accept-tokens 8192 \
    2>&1 | tee -a "$LOG"

echo "==================== ROTATION ABLATION (theorem check) ====================" | tee -a "$LOG"
# nvfp4_hwht = per-block WHT rotation (NOT global QuaRot). Theorem predicts
# rotation is neutral-to-harmful post-equalization (Gaussian fixed point). This
# is the cheap on-harness evidence; a proper global QuaRot/SpinQuant baseline is
# a separate task (needs a global-rotation mode), TODO before submission.
python3 -m turboquant.validation.hf_perplexity --model "$M" \
    --modes fp16 fp8 nvfp4_raw nvfp4_eqzp_svd_qjl nvfp4_hwht 2>&1 \
  | grep -E "PPL =|test tokens" | tee -a "$LOG"

echo "DONE. copy results/distill_accept* + results/perplexity_*.json off the box." | tee -a "$LOG"
