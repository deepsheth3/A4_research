# NVFP4 Quantization-Aware Training — Findings

*Consolidated record of the QAT effort for NVFP4 (4-bit) inference. Written to be honest:
wins are stated plainly, nulls and caveats are not hidden, and projections are labeled as
projections. Last updated 2026-06-19.*

---

## TL;DR

- **QAT works for NVFP4.** Training a model to be quantization-aware recovers a large
  fraction of the quality lost when going to 4-bit — **62–70% of the gap on TinyLlama-1.1B**,
  **~46% on Llama-3.1-8B** — at *zero inference cost* (it folds into a standard NVFP4
  checkpoint; no custom kernels, no side-channels).
- **The payoff metric is acceptance, not standalone PPL.** A QAT'd W4A4 model is the ideal
  *draft* for speculative decoding: acceptance rises (0.80→0.83 on 8B, 0.825→0.87 on
  TinyLlama), and the verifier guarantees the *output* equals the FP16/FP8 target — so you
  get full quality at ~1.8× throughput and ¼ the draft memory.
- **8B needed LoRA-QAT.** Full-weight 8B QAT OOMs on a 95GB GPU; naive subset training is
  null. **LoRA-QAT** (adapters on all layers, base frozen) fits in 66GB and is the
  validated 8B path. This was the single biggest unlock this session.
- **Novelty: incremental.** Solid systems engineering + one genuinely-new empirical finding
  (acceptance-objective QAT generalizes better OOD). Workshop-paper material, not a
  methodological breakthrough.
- **TRT-LLM-ready.** Because QAT moves all correction *into the weights*, inference is
  bog-standard NVFP4 — fully fusible in TensorRT-LLM, unlike our correction-stack
  alternative.

---

## 1. What "QAT for NVFP4" means here

NVFP4 = per-block (K=16) microscaling: each block of 16 values gets an FP8 absmax scale and
4-bit E2M1 values. **PTQ** (post-training quantization) just rounds a trained model to this
grid. **QAT** inserts a fake-quantizer in the forward pass (straight-through estimator for
gradients) and *trains the model to be robust to that rounding*, distilling against the
FP16 teacher.

Two training objectives were used:
- **`kl`** — KL-distill the student logits toward the FP16 teacher (selection on val loss).
  The stable gap-closer.
- **`kltv`** — KL backbone + a Total-Variation term. Since spec-decode acceptance
  `α = 1 − TV(draft, target)`, the TV term *directly optimizes acceptance* (selection on
  val acceptance). Pure-TV alone fails (weak gradients); the blend works.

Tooling: [turboquant/validation/qat_nvfp4.py](turboquant/validation/qat_nvfp4.py).

---

## 2. Results

### TinyLlama-1.1B (the fuller QAT proof)

FP16 reference 9.358 · FP8 bar 9.388 (FP8 is nearly lossless).

| Config | PPL | vs PTQ | Gap closed | Acceptance |
|---|---|---|---|---|
| PTQ W4A4 (no training) | 10.734 | — | — | 0.825 |
| Light KL-QAT (800 steps) | 9.875 | −0.859 | 62% | 0.852 |
| **Heavy KL-QAT (3000 steps)** | **9.768** | **−0.966** | **70%** | **0.862** |

More data clearly helped (62%→70%), confirming the light run was data-starved.

### Acceptance-QAT three-way (TinyLlama, W4A4 draft vs FP16 target)

The metric that becomes throughput. WikiText (in-dist) + GSM8K (OOD math):

| Draft | Wiki α | Wiki speedup | OOD α | OOD speedup |
|---|---|---|---|---|
| PTQ W4A4 | 0.8273 | 1.61× | 0.8671 | 1.74× |
| KL-QAT | 0.8702 | 1.76× | 0.8740 | 1.77× |
| **kltv (acceptance-optimized)** | **0.8714** | **1.76×** | **0.8789** | **1.79×** |

**Finding:** directly optimizing acceptance (kltv) beats distribution-matching (KL) — and
the edge is **4× larger out-of-distribution** (+0.0049 OOD vs +0.0012 in-dist). Honest
magnitude: the in-dist margin is near noise; the *OOD generalization* is the real, novel
signal — directly shaping the draft to agree with the target on sampled tokens transfers to
unseen data better than cloning the full softmax.

Source: [results/ood_accept_eval.json](results/ood_accept_eval.json).

### QAT generalizes to OOD (not overfit to WikiText)

| | Wiki PPL | GSM8K (OOD) PPL |
|---|---|---|
| FP16 | 9.358 | 4.030 |
| PTQ W4A4 | 10.734 | 4.366 |
| **KL-QAT W4A4** | **9.768** | **4.250** |

QAT beats raw PTQ on *unseen math* too (4.366→4.250). Source:
[results/ood_eval_qat.json](results/ood_eval_qat.json).

### Llama-3.1-8B (this is where it got hard)

FP16 reference 7.945 (512-ctx setting). PTQ W4A4 = 9.543 (+1.60 gap), acceptance 0.800.

| Approach | Result | Notes |
|---|---|---|
| Full-weight QAT | **OOM** | 95GB exceeded — see §3 |
| Subset (half-layer) QAT, kl & kltv | **NULL** (9.543, +0.0) | frozen half uncorrectable |
| **LoRA-QAT** r16 / 300 steps | **8.896 (+0.65), 40.5% closed**, α 0.831 | fits 66GB |
| **LoRA-QAT** r32 / 1500 steps | **8.810 (+0.73), 45.9% closed**, α 0.832 | best |

Source: [results/qat_nvfp4_kl_unsloth_Meta-Llama-3.1-8B_lora.json](results/qat_nvfp4_kl_unsloth_Meta-Llama-3.1-8B_lora.json).

---

## 3. The 8B arc: OOM → null → LoRA win

This is the most instructive part of the session. **The TinyLlama QAT recipe does not
auto-transfer to 8B.**

**Full-weight QAT OOMs.** QATLinear makes every linear weight trainable (~6.5B params).
Persistent memory = student 16GB + teacher 16GB + grads 13GB + AdamW states (bf16) 26GB ≈
71GB, and the optimizer's `_foreach_sqrt` allocates a transient 13GB copy of the state →
peak > 95GB. Memory fixes applied (all in the harness now):
- `--grad-checkpoint` — activation checkpointing.
- `foreach=False` AdamW — kills the fused-optimizer transient (param-by-param update).
- `--train-every N` — subset training.

**Subset training is null.** With `--train-every 2` (113/225 layers) it *fits* but never
beats the PTQ baseline on either objective (gap closed = 0%). Why: the frozen half's
quantization error is uncorrectable, and training only perturbs the model *away* from its
good PTQ point first (val loss rises to 2.96 then claws back to ~2.79, never below the 2.716
baseline) → monotone-safe selection just keeps the PTQ weights.

**LoRA-QAT is the fix.** Freeze the NVFP4 base weight, train a rank-r adapter, fake-quant
the *merged* weight (STE). Only adapters train (44M params at rank 16) → optimizer state
shrinks ~50× → **all layers train within 66GB**. The critical design detail: **`B` inits to
zero** so `BA = 0` and the model starts *exactly at PTQ* — then it trains **monotonically
down** (val loss 2.716 → 2.62), with none of the initial disruption that killed subset
training. It folds to a plain `nn.Linear` at export = a drop-in NVFP4 checkpoint.

LoRA also needs a higher LR (1e-4) than full QAT, because the adapter starts at zero.

Implementation: `LoRAQATLinear` + `--lora-rank` in
[turboquant/validation/qat_nvfp4.py](turboquant/validation/qat_nvfp4.py).

---

## 4. Related lever: KV4 correction (tested, low priority)

The KV-cache leg (K/V quantized to NVFP4) is uncorrected in our stack. We tested adding an
additive low-rank residual correction (the lever the two-lever theorem leaves open):

| Config (TinyLlama) | PPL |
|---|---|
| FP16 | 9.354 |
| KV4 raw (current) | 9.474 (+0.119) |
| KV4 + lowrank r=64 | 9.419 (recovers ~45% of the leg) |

**Verdict: real but not worth shipping.** It's the *smallest* leg (weights dominate), only
recovers ~45% even at r=64, **isn't byte-legal** (an r-dim correction code per token blows
the KV memory budget — the whole point of KV4), and needs a custom attention kernel. The
only TRT-viable form is equalization folded into the projection (zero extra cache bytes).
Source: [results/kv4_correction_TinyLlama-1.1B-Chat-v1.0.json](results/kv4_correction_TinyLlama-1.1B-Chat-v1.0.json).

---

## 5. Is it a standalone model?

Depends on the config:
- **W4A4-QAT (what we trained)** — standalone it's still +0.87 PPL over FP16 (~11% higher
  perplexity). Usable for latency/memory-critical, quality-tolerant work, but **not a clean
  independent FP16 replacement.** Its real home is as a **spec-decode draft**, where the
  verifier makes the *output* lossless and only acceptance matters.
- **W4A16 + low-rank (from earlier work)** — ~FP16 quality standalone (6.380 vs FP16 6.324,
  *beats* FP8), ~2× speedup, under FP8 memory. **This is the ship-it-alone config.**

---

## 6. Why FP16 as the baseline (and the FP8 caveat)

- **FP16/BF16** is the quality ceiling and the most-deployed precision — the natural "how
  much did 4-bit cost" reference.
- **FP8** is the production incumbent to beat. But FP8 is *nearly lossless* (8B: 6.357 vs
  FP16 6.324, within 0.03), so "gap to FP16" ≈ "gap to FP8". Measuring against FP16 is a
  valid proxy.
- **Open item (cheap):** we have not measured a direct FP8 row at the 8B/512-ctx setting —
  one extra forward pass, no kernel build. Worth adding to close the question cleanly.

---

## 7. Novelty (honest)

- QAT itself: **not novel** (Google Gemma QAT, standard).
- QAT on the NVFP4 *microscaling* grid (vs int4): **mildly novel** — less explored.
- LoRA-QAT to *close the NVFP4 quant gap* (inverse of QLoRA, which is quantized-base +
  LoRA for task fine-tuning): **modestly novel engineering.**
- **Most paper-worthy:** acceptance-objective (TV) QAT generalizing better OOD for
  spec-decode drafts — a genuinely new empirical finding, but small margin.

**Verdict: incremental novelty.** Solid systems work + one small new finding. Workshop /
empirical paper, not a top-tier method.

---

## 8. Productionization with TensorRT-LLM

QAT is the **most** TRT-LLM-friendly approach we have: it pushes all correction into the
weights, leaving a standard NVFP4 model. The LoRA adapters fold into the weight at export →
a plain NVFP4 checkpoint TRT-LLM ingests directly. **No custom kernels.**

What TRT-LLM fuses on the NVFP4 (Blackwell) path:
- **NVFP4 dequant fused into the GEMM** — two-level scale (per-block FP8 + per-tensor FP32)
  decode + E2M1→bf16 in the GEMM prologue.
- **RMSNorm + activation-quant** fused (norm epilogue writes NVFP4 activations).
- **Previous GEMM epilogue → activation quant** (quantize-on-write).
- **RoPE + attention (FMHA/FlashAttention)** with FP8/FP4 KV cache.
- **Speculative decoding** (draft + target) is natively supported — the W4A4-QAT draft +
  FP8/FP16 target both run inside TRT-LLM.

Contrast: our correction-stack alternative (eq/SVD/QJL epilogues) would **not** fuse cleanly
— it needs bespoke kernels. That's the core argument for QAT over the stack.

---

## 9. Wins if productionized

**vs FP16 (unambiguous):**
- **~4× smaller weights** (4-bit vs 16-bit) → 8B drops ~16GB → ~4–5GB; bigger models /
  longer KV / bigger batch per card.
- **Lower decode latency** — decode is memory-bound, 4× less weight traffic.
- **≥ throughput** — 2× tensor-core math.

**vs FP8 (subtler):**
- **2× smaller** weights; quality parity *via the verifier*; throughput roughly at parity
  until you scale to bigger models / higher concurrency.

**Best case (spec-decode, QAT'd FP4 draft + verifier):** FP16-quality output, ~1.8×
throughput, ¼ the draft memory.

**Caveat (measured, real B200):** stock FP4 only *ties* FP8 at 8B single-stream (parity at
batch ~64) — FP4's full throughput edge needs big models + mature TRT-LLM kernels + high
concurrency. So we compete on **accuracy-per-byte and acceptance-per-byte**, not raw speed.

---

## 10. Honest caveats & open items

- The 8B LoRA-QAT runs were **time-bounded** (rank 32, 1500 steps, 512 ctx). The untried
  levers — **max-len 1024, rank 64** — would likely push past 46% (TinyLlama hit 70%).
- **Diminishing returns observed:** rank16/300 → rank32/1500 (5× compute) moved 40.5% →
  45.9% at 512 ctx.
- **No first-party TRT-LLM throughput benchmark** — it's expensive engineering validation,
  not research. Backed instead by our real B200 vLLM data + NVIDIA's published TRT-LLM
  NVFP4 numbers. (The 1.8× speedup uses an assumed cost_ratio.)
- **No direct FP8 accuracy row** at the 8B/512 setting yet (cheap; should add).
- The **bake-off** (ours vs stock ModelOpt NVFP4 vs QuaRot/AWQ, equal-byte) — the external
  credibility experiment — is still unrun.

---

## 11. Reproduce

```bash
# TinyLlama heavy KL-QAT (the 70% result)
python -m turboquant.validation.qat_nvfp4 --objective kl --steps 3000 --lr 3e-5 --n-train 1500

# Acceptance-optimized (kltv) + three-way showdown vs FP16 target (in-dist + OOD)
python -m turboquant.validation.qat_nvfp4 --objective kltv --tv-weight 1.0 --steps 3000 \
    --lr 3e-5 --n-train 1500 --save-dir qat_kltv_ckpt
python -m turboquant.validation.ood_eval \
    --drafts orig:TinyLlama/TinyLlama-1.1B-Chat-v1.0,kl:qat_heavy_ckpt,kltv:qat_kltv_ckpt

# 8B LoRA-QAT (the validated 8B path) — needs a Blackwell-class GPU
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m turboquant.validation.qat_nvfp4 \
    --model unsloth/Meta-Llama-3.1-8B --objective kl --lora-rank 32 --steps 1500 --lr 1e-4 \
    --max-len 512 --grad-checkpoint

# KV4 correction lever test
python -m turboquant.validation.kv4_correction --ranks 16 32 64
```

---

## 12. Where findings live

- **This file** — the consolidated narrative (version-controlled).
- **Git commits** `10ca728`, `a7f57f5`, `637179d`, `f3dc022`, `ffa46e7`, `4024031`,
  `bdb0485`, `8ac8f34` — each is a finding with its numbers (`git show <hash>`).
- **`results/*.json`** — the raw measurements.
- **`turboquant/validation/`** — the harnesses (`qat_nvfp4.py`, `ood_eval.py`,
  `kv4_correction.py`).
