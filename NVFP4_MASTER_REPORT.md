# NVFP4 4-bit Quantization for LLMs — Master Research Report

*A complete, honest record of the NVFP4 (W4A4 + KV4) research effort: the theory, every
technique tried, every result, every failure, every parity (wash), the spec-decode framing,
the QAT work, and the TensorRT-LLM deployment story. Written to be learned from and
presented. Last updated 2026-06-19.*

> **How to read this:** §0 is the elevator pitch. §1–2 are the intellectual core (the
> theorem). §3–8 are the empirical codec results. §9 is the "everything that washed out"
> ledger. §10–11 are spec-decode + QAT. §12–13 are throughput + TensorRT-LLM. §14–17 are
> honesty, novelty, the hypothesis ledger, and an interview cheat-sheet.

---

## 0. Executive summary

**Goal.** Make NVFP4 (4-bit) inference as production-viable as FP8 — *match FP8 accuracy
while beating it on memory*, with a **Pareto-clean** codec (every correction either folds
into weights offline or runs as a fused epilogue; the main GEMM path stays pure FP4; no
axis regresses vs FP8).

**The references that matter (Llama-3.1-8B):**
- **FP16** = quality ceiling (most-deployed precision).
- **FP8** = the production incumbent to beat. *FP8 is nearly lossless* (within ~0.03 PPL of
  FP16), so "gap to FP16" ≈ "gap to FP8".

**Three headline contributions:**

1. **A distortion theory of microscaled FP4 (the "two-lever theorem").** Under per-block
   microscaling with equalization, the *only* distortion-reducing levers are (i)
   **equalization** and (v) **additive low-rank correction**. Rotation, permutation,
   per-block Hadamard, channel re-grouping, output-weighted scale selection, and
   coordinated rounding are all **provably redundant** — and we confirmed each empirically
   on real Llama-8B. This turns "we tried 11 tricks" into "the framework predicted all 11."

2. **A near-FP8 calibration-free codec.** `eq + SVD side-channel + QJL` closes **~68% of the
   FP4→FP8 activation gap** (A4 = 6.050 vs FP8 5.948 at 1024 ctx), and the law **travels
   across 5 models / 3 families** (Llama, Qwen, Mistral; up to 80% on Qwen). Adding GPTQ +
   additive low-rank **weights** recovers ~58% of the weight cliff. The deployable
   **W4A16+low-rank** config is *FP16-quality and beats FP8*, under FP8 memory.

3. **Quantization-Aware Training (QAT) as the step off the post-training floor**, and the
   **acceptance** reframing for speculative decoding. QAT recovers 62–70% of the W4A4 gap on
   TinyLlama and generalizes OOD; **LoRA-QAT** is the validated path to 8B (full-weight
   OOMs). Optimizing **acceptance** (not perplexity) gives a draft that generalizes better
   out-of-distribution.

**The honest verdict.** The *post-training* codec is at its practical floor on every leg —
that itself is a clean, theorem-backed result. The remaining quality lives in **training
(QAT)** and is monetized via **spec-decode acceptance**, where the verifier guarantees
FP16-quality output at 4-bit economics. We compete on **accuracy-per-byte and
acceptance-per-byte**, *not* raw single-stream speed (measured: stock FP4 ≈ FP8 at 8B).

**Real-hardware validation (§13.5).** The full QAT→ModelOpt-export→deploy path ran
end-to-end on an RTX PRO 6000 (sm_120): QAT reproduced 9.766 PPL (70% gap closed), produced
a real loadable NVFP4 checkpoint (deployed 9.94 PPL), and real FP4 GEMM measured **3.7×
faster than BF16 at 70B-layer shapes** (but slower at TinyLlama shapes — the speed win is a
large-GEMM property).

---

## 1. Setup & notation

A linear layer computes `y = Wx`. The accuracy-relevant distortion of a quantizer `Q` is the
**output** error:

```
D(Q) = E‖W(x − Q(x))‖² = E[ eᵀ G e ],   e = x − Q(x),   G = WᵀW ⪰ 0.
```

`G` is the **output metric**: `G_ii = ‖W_:i‖²` is how much input channel *i* drives the
output; `G_ij` is channel coupling.

**NVFP4 microscaling quantizer.** Partition the `d` channels into blocks of size `K=16`.
Fix the E2M1 grid `𝒢` (4-bit, symmetric, dense near zero, `max 𝒢 = g`). Each block gets one
**absmax** scale stored in **fp8** (the "MX4" overhead): `s_b = (1/g)·max_{i∈B_b}|x_i|`,
`Q(x)_i = s_b · Π_𝒢(x_i/s_b)`. This is what Blackwell hardware implements.

**Bits.** "4-bit" = the *main GEMM path* (E2M1 weights × E2M1 activations, fp16 accum, fp8
block scales). Side-channels (SVD/QJL/low-rank factors) may be fp8 — they ride a thin
epilogue, not the main path.

---

## 2. The Two-Lever Theorem (the intellectual centerpiece)

*Full proof in [THEORY.md](THEORY.md). This is the part to present.*

**Assumptions (idealizations, stated honestly):**
- **(A1) High-resolution white error:** `E[e_i e_j | x] = δ_ij · κ · s_b²` (standard
  high-res / subtractive-dither model; exact under dither, approximate for E2M1 at K=16).
- **(A2) Absmax scaling** (what the hardware uses).

Under (A1), the distortion collapses to a single clean expression:

```
D = (κ/g²) · Σ_b ( max_{i∈B_b} x_i² ) · tr(G_bb).        (★)
```

> **In words:** distortion = the **per-block peak magnitude**, weighted by the block's
> **output energy** `tr(G_bb)`, summed over blocks. Every "trick" is just an action on (★).

**The design space** (every quant trick is one of these): (i) diagonal preconditioning
(equalization), (ii) orthogonal/permutation transforms (rotation, Hadamard, channel
permute), (iii) scale-objective tweaks (output-weighted scale selection), (iv) coordinated
rounding (GPTQ-style), (v) additive low-rank residual correction.

**Theorem.** `D` is minimized by **(i) equalization + (v) additive low-rank correction**;
classes (ii)–(iv) are **redundant**.

**The four lemmas (why):**
1. **Equalization is the optimal diagonal preconditioner** — it acts per-channel *before*
   blocking, directly lowering the block-peak term in (★). Special among multiplicative
   tricks.
2. **Rotation Gaussianizes within a block, and the Gaussian is its fixed point.** For a
   Gaussian block, peak/avg is neutral (4.55→4.55 at K=16); rotation helps super-Gaussian
   (outliers: 14.5→3.5) and *hurts* sub-Gaussian (flat: 1.2→4.5). **Equalization already
   drives blocks to the Gaussian fixed point**, so rotation/permutation are neutral-to-harmful
   *post-equalization*. (This is the crux; numerically verified, matches the per-block
   Hadamard data exactly.)
3. **Additive low-rank is the rate-distortion-optimal side-channel** (reverse water-filling
   / KLT of the residual covariance). Our SVD side-channel *is* this; QJL is its 1-bit
   relaxation. No input transform can produce a residual-information term.
4. **Coordinated rounding is the dual of additive correction** (shape-away vs add-back, same
   low-G subspace). Within-block it's inert (G_bb is diagonal post-eq); cross-block it lives
   in the additive subspace but is serial-only — explaining the "rounding ceiling" result.

**What it predicts (all confirmed on real Llama-8B):** PolarQuant, QuaRot global rotation,
per-block Hadamard (×3 forms), channel permutation, output-weighted scale selection, GPTQ &
AWQ scaling, rank allocation, and coordinated rounding → all redundant. Equalization +
additive → the two levers.

**One honest refinement** (`baseline_compare.py`, 8B): after equalization, rotation gives a
*small residual* gain (6.155→6.136, ~0.02), consistent with SpinQuant — but additive
correction gives **~5× more** (6.155→6.050, ~0.10) and the best acceptance. So the
defensible claim is **"additive correction dominates; rotation gives a small residual
benefit,"** not "rotation is strictly redundant." (This run saved us from an overclaim.)

**Why weights are different:** equalization drives *activations* to the white-Gaussian fixed
point, which is why every PTQ trick closes for **A4**. **Weights** are static and structured
(not white after eq), so the same doors stay *live* for **W4** — which is exactly where GPTQ
and additive low-rank pay off, and where QAT/training is the open frontier.

---

## 3. Activation quantization (A4) — the full arc

Reference (Llama-3.1-8B, full WikiText-2, 1024 ctx): **FP16 5.918**, **FP8 5.948** (+0.030).

| Mode | PPL | vs FP8 | Gap to FP8 closed |
|---|---|---|---|
| raw NVFP4 | 6.263 | +0.315 | 0% |
| + channel equalization + best-of-{0,mid} zero-point | 6.168 | +0.220 | 30% |
| + W-aware SVD side-channel (top d/16 singular vecs, fp8 coeffs) | 6.065 | +0.117 | 63% |
| **+ QJL residual (`nvfp4_eqzp_svd_qjl`)** | **6.050** | **+0.102** | **68%** |

**A4 reaches near-FP8 (+0.10 PPL) — the hard half, essentially solved, calibration-free,
hardware-native main path.**

**Key A4 sub-findings:**
- **PolarQuant redundant** with MX4 (per-16 block scale already normalizes per group; with
  fp8 scales PolarQuant *hurts*). Dropped.
- **QJL must be per-sub-block** (per-128, not per-4096): reduction = `qjl_dim·2/(π·block)`;
  full-vector QJL is ~1% (useless), per-128-block is ~22% NMSE reduction.
- **Zero-point:** naive per-block zero-point *hurts* E2M1 (float grid is dense near zero;
  shifting moves mass to the coarse mid-grid — which is why hardware NVFP4 has no zp). But
  **best-of-two {z=0, z=mid}** wins ~25% MSE. `optclip` extends this to a joint
  shift×clip-ratio search (8 candidates), vectorized into one batched op.
- **W-aware SVD beats random-projection QJL** decisively (project the residual onto W's top
  d/16 input singular vectors). For composed modes, the SVD basis must come from the
  **equalized** weight `diag(s)·A`.

**A4 floor proven (rounding-ceiling analysis, real captured 8B activations):** within-block
coordinated rounding is **inert** (G16 flat); the real ceiling needs *full serial 4096-channel*
rounding (3–50× latency, not Pareto-viable) and **still misses FP8**. So the additive
side-channel already captures **all parallel-deployable structure**. A4 = 6.050 is the floor.

**Cross-model (the law travels — 5 models, 3 families):** our stack lands near-FP8 on every
model. gap-to-FP8 / closed: 1B +0.359/47%, 3B +0.107/65%, 8B +0.102/67%, **Qwen-7B
+0.056/80%**, Mistral-7B +0.058/47%. Best result is a *different family* → kills
single-model bias. The mechanism (white residual → floored) also travels.

---

## 4. Weight quantization (W4) — the full arc

Refs (8B, 1024 ctx): FP8 5.948, A4 6.050, **naive W4A4 6.629 (+0.579 cliff over A4)**.

| Method | W4A4 PPL | Cliff recovered |
|---|---|---|
| naive NVFP4 rounding | 6.629 | 0% |
| GPTQ (Hessian error feedback) | 6.552 | 13% |
| AWQ + GPTQ (salient-channel scaling) | 6.581 | *worse than GPTQ* |
| **GPTQ + additive low-rank (LQER-style), rank d/8, fp8 factors** | **6.294** | **58%** |
| GPTQ + low-rank rank d/6 | 6.258 | (the W4 winner; 0.83 B/elem) |

**The breakthrough = additive low-rank weight correction**, mirroring the activation story:
**scaling/redistribution methods (GPTQ alone, AWQ) are redundant under microscaling; only the
additive mechanism breaks the wall** (it *adds information back*: corrects E = W−Q(W) in the
activation-Hessian metric via a rank-r side matmul). Monotonic with rank, diminishing
returns. Byte-legal: d/6 fp8 factors = 0.83 B/elem < FP8's 1.0.

**W4 sub-findings:**
- **fp4/NVFP4 factors are worse than fp8 at equal bytes** (E2M1 too coarse for factors).
- **Joint W+A Hessian gives no signal** (both quantizers individually strong → Q(x)≈x →
  W–A interaction negligible).
- **Per-layer rank allocation HURTS** (water-fill by singular-energy 6.363 > uniform 6.294;
  Fisher-alloc 6.314 > uniform too). **Rank is the lever; allocation is not.** (Per-layer
  recon error ≠ end-to-end importance.)

---

## 5. KV-cache quantization (KV4)

KV4 = fake-quantize K/V to NVFP4 (the "4-bit everything" composition). Today's
`install_kv_hooks` does **raw nearest-round NVFP4 on K/V with zero correction** (no eq, no
SVD, no QJL) — the one uncorrected leg.

**KV4 correction lever test (TinyLlama, this session):** adding an additive low-rank residual
to K/V *does* help — FP16 9.354 / KV4-raw 9.474 (+0.119) / +lowrank r=64 9.419 (recovers
~45% of the leg, monotone with rank). **But it's not worth shipping:** it's the *smallest*
leg, only ~45% recoverable, **not byte-legal** (an r-dim correction code per token blows the
KV-cache memory budget — the whole point of KV4), and needs a custom attention kernel. The
only TRT-viable form is **equalization folded into the projection** (zero extra cache bytes).

---

## 6. The full composition: W4A4KV4 + honest gap decomposition

**4-bit-everything (Llama-3.1-8B, in/6 weights + A4 + KV4) = 6.363 PPL** (1024 ctx; FP16
5.919, FP8 5.948). Decomposition: A4 6.051 → W4A4 6.258 (+0.21) → W4A4KV4 6.363 (KV4 +0.105,
cheapest leg). vs FP8 +0.415. Every axis under FP8's 1 byte → Pareto-clean.

**Honest 2048-context (the realistic deploy config):** FP16 6.324, FP8 6.357, full W4A4KV4
7.047 (+0.69 vs FP8). **Leg split:** activations +0.136, **weights +0.478 (DOMINANT)**, KV4
+0.109. Adding low-rank weight correction (d/6, fp8) → **6.762** (gap to FP8 +0.405).
Plateaus ~+0.4 short of FP8; pushing rank to parity bytes only buys +0.049 → not worth it.

**The clean positive:** **W4A16 + low-rank (weights-only 4-bit) is FP16-quality (6.380,
*beats* FP8 6.357), α=0.935, ~2× speedup, under FP8 memory** — the deployable standalone
sweet spot. Full W4A4KV4 (lossy) is only worth it *with* the spec-decode verifier on top.

---

## 7. The redundancy law — the complete "wash" ledger (negative results)

Every basis/regroup/scale-objective trick collapses to a dead heat with the
`eqzp_svd_qjl` control (8B, ~6.05). **This is a feature: the theorem predicts each one.**

| Trick | Result vs control | Verdict |
|---|---|---|
| PolarQuant | == raw (fp8 scales: hurts) | redundant |
| QuaRot global rotation | NMSE ×3, gpt2 PPL 44.7→94.2 | harmful under MX4 |
| Per-block Hadamard (always) | 6.050 vs 6.057 | wash (within noise) |
| Per-block Hadamard (best-of, isolation) | +0.057 | works as a *choice* only |
| Per-block Hadamard fixed-mask (deployable) | 6.052 | wash (eq already did its job) |
| Channel permutation | 6.051 | wash (redundant with eq) |
| Output-weighted scale selection (`oda`) | 6.052 | wash (SVD mops up the residual) |
| GPTQ / AWQ weight *scaling* | 6.55 / 6.58 | redundant (vs additive low-rank) |
| Per-layer rank allocation (energy / Fisher) | 6.36 / 6.31 | worse than uniform |
| Coordinated rounding within-block (OBR) | ~0% (G16 flat) | inert |
| Residual-derived SVD basis (vs W-eigenvector) | −18.8pp | worse (residual is white) |
| Trained *weight* factors vs analytical LQER | identical (α 0.935, 6.380) | null (no room) |

**Consolidated law (shown 3 independent ways):** *under per-16 microscaling with
equalization, channel-level redistribution has nothing left to do; only additive
side-channels move PPL.* Mirrors the PolarQuant/rotation findings — now airtight.

---

## 8. The post-training floor (a clean negative result)

The PTQ codec is at its **practical floor on every leg**, six ways:
- A4 floored (rounding ceiling, format ceiling, residual-basis, error-diffusion).
- Trained *activation* correction: signal at short context (+0.10–0.14 PPL) but **deflates
  to +0.032 at 2048 ctx** and training is fragile (diverges ~step 40); the post-training
  activation residual is **nearly irreducible** (it's white).
- Trained *weight* factors: **null** (analytical LQER fill already within 0.056 of FP16).

**This is a publishable mechanism, not a dead end:** *additive side-channels subsume
basis/regroup/scale-objective tricks under microscaling; the residual gain needs full serial
rounding (not Pareto-viable) and still misses FP8.* The frontier is **mapped, not guessed.**
The remaining quality requires **breaking a premise → training (QAT)**.

---

## 9. Spec-decode & the acceptance reframing

The key economic insight: a 4-bit model doesn't have to be a *standalone* FP16 replacement.
Used as a **speculative-decoding draft** with an FP8/FP16 **verifier**, the *output quality
equals the target* — the 4-bit codec's PPL gap surfaces only as an **acceptance tax**, never
a quality loss.

**Acceptance** `α = 1 − TV(draft, target) = Σ min(p,q)` — distributional, so it's **real on
a fake-quant H100, no Blackwell needed.** Spec-decode speedup ≈ E[accepted]/(γ·cost_ratio+1).

**Headline (8B, full deployed W4A4KV4 draft vs FP16 target):** α = **0.8955** (~90%),
E[accepted] = 4.06 @ γ=4, **projected 1.85× speedup**. All three production constraints met:
quality FP16-identical (verifier), memory under FP8, throughput ~1.85×.

This reframes the whole project: **we don't need to close the codec gap — we need high
acceptance per byte.** That is what QAT then optimizes directly.

---

## 10. Quantization-Aware Training (QAT) — the step off the floor

PTQ is at its floor; QAT *trains the model to be quantization-aware* (fake-quant in the
forward, straight-through gradient, distill against FP16 teacher). Two objectives:
`kl` (match teacher) and `kltv` (KL backbone + a TV term that **directly optimizes
acceptance**, since α = 1 − TV).

### TinyLlama-1.1B (the fuller proof)
Refs: FP16 9.358, FP8 9.388. PTQ W4A4 10.734, α 0.825.

| Config | PPL | Gap closed | Acceptance |
|---|---|---|---|
| Light KL-QAT (800 steps) | 9.875 | 62% | 0.852 |
| **Heavy KL-QAT (3000 steps)** | **9.768** | **70%** | **0.862** |

QAT **generalizes OOD** (GSM8K math, unseen): PTQ-W4A4 4.366 → KL-QAT 4.250 — beats raw PTQ
on unseen math too. **Not overfit to WikiText.**

### Acceptance-QAT three-way (the novel finding)
W4A4 draft vs FP16 target, WikiText (in-dist) + GSM8K (OOD):

| Draft | Wiki α | OOD α |
|---|---|---|
| PTQ W4A4 | 0.8273 | 0.8671 |
| KL-QAT | 0.8702 | 0.8740 |
| **kltv (acceptance-optimized)** | **0.8714** | **0.8789** |

**Directly optimizing acceptance beats distribution-matching — and the edge is 4× larger
out-of-distribution** (+0.0049 OOD vs +0.0012 in-dist). Shaping the draft to agree with the
target on *sampled* tokens transfers to unseen data better than cloning the full softmax.
(Honest: in-dist margin is near noise; the OOD generalization is the real signal. Pure-TV
alone fails — it needs the KL backbone for gradients.)

### Llama-3.1-8B — the OOM → null → LoRA arc (the instructive part)
Refs (512 ctx): FP16 7.945, PTQ W4A4 9.543 (+1.60), α 0.800.

| Approach | Result | Why |
|---|---|---|
| Full-weight QAT | **OOM (95GB)** | grads + Adam states for 6.5B params + teacher + fake-quant temporaries; the `_foreach_sqrt` optimizer step alone spikes +13GB |
| Subset (half-layer) QAT | **NULL (+0.0)** | frozen half's quant error is uncorrectable; training perturbs *away* from PTQ then claws back, never beating baseline → selection keeps PTQ |
| **LoRA-QAT r16 / 300 steps** | **8.896, 40.5% closed**, α 0.831 | fits **66GB** |
| **LoRA-QAT r32 / 1500 steps** | **8.810, 45.9% closed**, α 0.832 | best |

**LoRA-QAT is the validated 8B path.** Freeze the NVFP4 base weight, train a rank-r adapter,
fake-quant the *merged* weight (STE); only adapters train (~44M params) → optimizer state
shrinks ~50× → all layers fit. **Critical design detail: `B` inits to zero** → the model
starts *exactly at PTQ* and trains **monotonically down** (val loss 2.716→2.62), with none
of the disruption that made subset training null. Folds to a plain `nn.Linear` at export =
a drop-in NVFP4 checkpoint. (LoRA needs higher LR, 1e-4, since the adapter starts at zero.)

**Memory fixes that made any 8B QAT possible** (all in the harness): `--grad-checkpoint`,
`foreach=False` AdamW (kills the optimizer-step spike), `--train-every` (subset), and
`--lora-rank` (the real fix).

**Honest scope:** the 8B runs were time-boxed (512 ctx, ≤rank 32). Untried levers — **1024
ctx, rank 64** — would push past 46% (TinyLlama hit 70%). Diminishing returns observed:
r16/300 → r32/1500 (5× compute) moved 40.5% → 45.9%.

---

## 11. The "did we improve?" reconciliation (three baselines)

Improvement depends on *what you compare against*:
1. **QAT vs raw 4-bit PTQ → big win.** TinyLlama 10.73→9.77; 8B 9.54→8.81; acceptance up
   everywhere.
2. **QAT vs our best post-training correction stack → roughly a tie.** Both hit the same
   ~floor; they fix the same error → *redundant*. (This is the "we didn't improve" line, and
   it's *vs the stack*, not vs PTQ.)
3. **QAT (W4A4) vs FP16/FP8 → still short** (+0.87 on 8B). The verifier closes this for
   *output* quality; standalone it's lossy.

**QAT's real value isn't standalone PPL — it's (a) deployment simplicity** (a plain NVFP4
checkpoint, no inference side-channels) **and (b) acceptance + OOD robustness** for
spec-decode drafts.

---

## 12. Throughput reality (measured) & deployment economics

**Real Blackwell B200 (vLLM 0.23, stock ModelOpt checkpoints, Llama-8B decode tok/s):**

| batch | FP8 | FP4 | FP4/FP8 |
|---|---|---|---|
| 1 | 389.7 | 352.6 | 0.90× |
| 32 | 11235 | 10768 | 0.96× |
| 64 | 19787 | 19834 | 1.00× |

**Stock FP4 is *slower* than FP8 single-stream, reaching parity only ~batch 64.** Why:
batch-1 decode is memory/dequant-bound, so FP4's 2× tensor-core math sits idle; dual-scale
dequant overhead + immature vLLM kernels outweigh the smaller weights. FP4's real speed win
needs **big models (70B/405B) + mature kernels (TRT-LLM) + high concurrency** — an 8B on
vLLM hits none.

**Strategic consequence (locked in):** *we do not compete on raw speed.* We compete on:
- **Accuracy-per-byte** — same 4-bit memory as stock NVFP4 but more accurate (our A4 ~6.05
  vs stock ~6.26) → **higher acceptance-per-byte** → fewer verifications → faster
  *end-to-end* spec-decode.
- **Memory** — 4-bit = half FP8 bytes → bigger draft / longer KV / fit 70B on one card.

**Wins if productionized:**
- **vs FP16 (unambiguous):** ~4× smaller weights (8B: 16GB→4–5GB), lower decode latency
  (memory-bound, 4× less weight traffic), ≥ throughput (2× tensor-core math).
- **vs FP8 (subtler):** 2× smaller; quality parity *via the verifier*; throughput ~parity
  until you scale up.
- **Best case (spec-decode, QAT'd FP4 draft + verifier):** FP16-quality output, ~1.8×
  throughput, ¼ the draft memory.

---

## 13. TensorRT-LLM productionization (the interview's home turf)

**QAT is the most TRT-LLM-friendly approach we have**, *because* it pushes all correction
*into the weights* — leaving a bog-standard NVFP4 model. The LoRA adapters fold into the
weight at export → a plain NVFP4 checkpoint TRT-LLM ingests directly. **No custom kernels.**

**What TRT-LLM fuses on the NVFP4 (Blackwell) path:**
- **NVFP4 dequant fused into the GEMM** — the two-level scale (per-block fp8 + per-tensor
  fp32) decode and E2M1→bf16 happen in the GEMM prologue, not a separate pass.
- **RMSNorm + activation-quant** fused (the norm epilogue writes NVFP4 activations).
- **Previous GEMM epilogue → activation quant** (quantize-on-write).
- **RoPE + attention (FMHA/FlashAttention)** with FP8/FP4 KV cache.
- **Speculative decoding** (draft + target) is natively supported.

**The contrast that is the productionization argument:** our *correction-stack* alternative
(eq/SVD/QJL epilogues) would **not** fuse cleanly — it needs bespoke epilogue kernels. **QAT
needs none of that.** So the deployment hierarchy is: **QAT (drop-in) > W4A16+low-rank
(near-FP16, simple) > full correction stack (needs custom kernels).**

**Why no first-party TRT-LLM throughput benchmark:** it's expensive *engineering validation*
(engine builds, kernel autotuning, warmup — the cold vLLM install alone ate a whole box
window), not a research result. It's backed instead by our real B200 vLLM data + NVIDIA's
published TRT-LLM NVFP4 numbers. The 1.8× speedup uses an assumed cost_ratio.

---

## 13.5 Real-hardware validation (2026-07-01, RTX PRO 6000 box run)

A one-shot rented Blackwell box (vast.ai RTX PRO 6000, sm_120, torch 2.12/cu130) took the
pipeline end-to-end for the first time: QAT → ModelOpt NVFP4 export → deploy → measure.

**Accuracy reproduced on real hardware.** TinyLlama, 3000-step KL-QAT: FP16 9.358 / PTQ
10.734 / **QAT 9.766 (70.3% gap closed)**, acceptance 0.825→0.865 — matches the banked
9.768/70%.

**Deployed NVFP4 (ModelOpt `export_hf_checkpoint`) = 9.94 PPL.** A genuine, TRT-LLM/
vLLM-loadable checkpoint (770 MB, `quant_algo=NVFP4`, group_size 16, `exclude_modules=
[lm_head]`). Honest caveat: the deploy grid costs **+0.18 PPL vs fake-quant** — the CPU
grid-parity check covered weights but not ModelOpt's activation calibration, and the QAT
weights were tuned to *our* fake-quantizer. Deployed QAT still closes ~58% of the gap.

**Throughput — the regime is everything.** Real FP4 kernels, same GPU:
- *TinyLlama serving* (vLLM, FlashInfer NVFP4 kernel): FP4 is **3.4–4.8× slower** than BF16
  (b1 609 vs 2934 tok/s … b256 85.6k vs 289k). At 1.1B the matmuls are too small to fill
  the FP4 cores and dequant overhead dominates.
- *Large GEMM* (`cutlass_scaled_fp4_mm`, Llama-70B-layer shapes 8192–28672): FP4 is
  **3.2–3.8× faster** than BF16 (~1555 vs ~424 TFLOPS, relerr ~0.13). Same kernel — the
  only variable is matrix size. **FP4's speed win is a big-model / large-GEMM property;**
  a toy model cannot show it.

**Two corrections to earlier claims** (recorded for honesty): "ModelOpt export can't lose
accuracy" was optimistic (it costs +0.18); "FP4 throughput is dead on sm_120" was wrong
(vLLM's FlashInfer kernel works — the failure was pinning torch, which forced an ancient
vLLM; torchao 0.17 separately *emulates* FP4 on this stack).

Artifacts: `results/box_run/20260701_222113/` (SUMMARY, all logs, checkpoint config);
harness: `setup_box.sh`, `run_the_box.sh`, `export_nvfp4.py`, `measure_deploy.py`,
`fp4_gemm_bench_flashinfer.py`.

---

## 14. Honest limitations & what's unproven

- **Fake-quant simulation** (mostly). Accuracy runs are numerical fake-quant; **now
  partly superseded** — §13.5 ran a real ModelOpt NVFP4 checkpoint on an RTX PRO 6000
  (deployed 9.94 PPL) and measured real FP4 GEMM throughput (3.7× vs BF16 at scale). Big-
  model *end-to-end* FP4 serving is still unmeasured (box was TinyLlama-scale + one-shot).
- **Idealized theory.** (A1) is a high-resolution idealization; finite K=16, heuristic α,
  per-layer surrogate Hessians, diagonal-G approximations are all first-order.
- **Time-boxed 8B QAT** (512 ctx, ≤rank 32) — 1024 ctx / rank 64 untried.
- **No direct FP8 row at the 8B/512 QAT setting** (cheap; should add — closes the
  "FP16 vs FP8 baseline" question cleanly).
- **The equal-byte bake-off is unrun** — ours vs stock ModelOpt NVFP4 vs QuaRot/AWQ/Atom on
  PPL *and* acceptance. This is the #1 external-credibility gap.
- **GSM8K downstream** only has a noisy n=30 signal (rules out collapse, not 1–3pt drop).
- **The trained 8B QAT artifact was not saved** (disk-constrained; numbers banked, weights
  not persisted — reproducible from code).

---

## 15. Novelty (honest self-assessment)

- 4-bit inference / NVFP4 / QAT / spec-decode: **mainstream**, not ours to own (NVIDIA owns
  the format + hardware + TRT-LLM + ModelOpt; Google does Gemma QAT; Meta ships quantized
  Llama).
- **The two-lever theorem + the redundancy law** — the most *intellectually* distinctive
  piece: a predictive framework, validated 11 ways and across 5 models. Genuinely a
  contribution, even if the components are known.
- **QAT on the NVFP4 microscaling grid + LoRA-QAT to close the quant gap** — modestly novel
  *engineering* (LoRA-QAT here is the inverse of QLoRA: adapters trained to recover
  *quantization* error, not to fine-tune a task).
- **Acceptance-objective QAT generalizing better OOD** — a genuinely new empirical finding
  (small margin).

**Verdict: incremental novelty with one strong theoretical through-line.** Workshop /
empirical-paper material framed around *the mechanism* (the theorem + the acceptance
reframing), not around "yet another quantization method."

---

## 16. Hypothesis ledger — what we tried and what it taught

| # | Hypothesis | Outcome | Lesson |
|---|---|---|---|
| 1 | PolarQuant tightens NVFP4 | redundant / hurts | microscaling absorbs coarse-scale tricks |
| 2 | QuaRot rotation helps | harmful under MX4 | per-16 scales beat rotation's Gaussian floor |
| 3 | Per-block Hadamard helps | works *as a choice*, washes in stack | local-basis lever subsumed by additive |
| 4 | Channel permutation isolates outliers | wash (redundant w/ eq) | eq already handles outlier channels |
| 5 | Output-weighted scale select | wash | SVD mops up the residual regardless |
| 6 | GPTQ/AWQ fix the W4 cliff | only 13% / worse | scaling redundant for weights too |
| 7 | **Additive low-rank weight correction** | **58% of cliff** | the *additive* lever breaks the wall |
| 8 | Per-layer rank allocation | worse than uniform | rank is the lever, allocation isn't |
| 9 | Coordinated (full-serial) rounding | real but 3–50× latency, still < FP8 | not Pareto-viable |
| 10 | Trained activation correction | +0.032 deployed (deflates from +0.14) | residual is white → nearly irreducible |
| 11 | Trained weight factors | null | analytical LQER already near-FP16 |
| 12 | **QAT steps off the PTQ floor** | **62–70% (TinyLlama), generalizes OOD** | training is the open lever |
| 13 | **Acceptance-objective (kltv) QAT** | **beats KL, esp. OOD** | optimize the metric that becomes throughput |
| 14 | Full-weight 8B QAT | OOM | training all 6.5B weights doesn't fit 95GB |
| 15 | Subset (half-layer) 8B QAT | null | frozen half uncorrectable |
| 16 | **LoRA-QAT for 8B** | **46% closed, fits 66GB** | adapters: small enough to fit, complete enough to work |
| 17 | KV4 additive correction | real but not byte-legal | KV4 is the smallest, least-deployable leg |
| 18 | **QAT→ModelOpt export→deploy on real Blackwell** | **9.766 QAT; 9.94 deployed (+0.18)** | export preserves most, not all, of the QAT gain |
| 19 | **FP4 GEMM vs BF16 on sm_120** | **3.7× at 70B-layer shapes; slower at 1.1B** | FP4 speed is a large-GEMM property, not intrinsic |

---

## 17. Interview cheat-sheet (key numbers to have cold)

**The bars (8B):** FP16 ≈ quality ceiling; FP8 ≈ +0.03 (nearly lossless) = the incumbent.

**The codec (8B, 1024 ctx):** raw NVFP4 6.263 → eq+SVD+QJL **6.050** (68% of FP4→FP8 gap,
calibration-free). W4 cliff: naive 6.629 → GPTQ+low-rank **6.294** (58%). W4A4KV4 = **6.363**
(4-bit everything). W4A16+low-rank = **6.380, beats FP8**, standalone-deployable.

**The theorem:** under microscaling + equalization, only equalization + additive low-rank
reduce output error; rotation/permute/round/scale-objective are redundant. Confirmed 11
ways, 5 models.

**Spec-decode:** 4-bit draft + FP16 verifier = FP16 quality. Full W4A4KV4 draft α = **0.90**,
projected **1.85×**.

**QAT:** TinyLlama 70% gap closed, generalizes OOD. **8B: LoRA-QAT 46% closed, fits 66GB
(full-weight OOMs).** Acceptance-optimized (kltv) draft generalizes best OOD.

**Throughput (measured B200):** stock FP4 *ties* FP8 (parity ~batch 64) → compete on
**accuracy/byte + acceptance/byte + memory**, not raw speed.

**Real Blackwell run (RTX PRO 6000, §13.5):** QAT 9.766 (70%) reproduced on hardware;
deployed NVFP4 checkpoint 9.94 (+0.18 export cost); real FP4 GEMM **3.7× vs BF16 at
70B-layer shapes**, slower at TinyLlama shapes (speed win needs big GEMMs).

**TRT-LLM:** QAT'd NVFP4 = drop-in checkpoint, no custom kernels; dequant + RMSNorm-quant +
RoPE/attention all fuse. The correction stack would need bespoke kernels — which is *why* QAT
is the production answer.

---

## 18. Reproduce & file map

```bash
# A4 codec table (the 68% result)
python -m turboquant.validation.hf_perplexity --model unsloth/Meta-Llama-3.1-8B \
    --modes fp16 fp8 nvfp4_raw nvfp4_eqzp_svd_qjl
# W4A4 with GPTQ + additive low-rank weights
... --w4-gptq --w4-lowrank --w4-rank-div 6 --w4-lowrank-fp8
# TinyLlama heavy KL-QAT (70%) + acceptance three-way
python -m turboquant.validation.qat_nvfp4 --objective kl   --steps 3000 --lr 3e-5 --n-train 1500
python -m turboquant.validation.qat_nvfp4 --objective kltv --steps 3000 --lr 3e-5 --n-train 1500 --save-dir kltv_ckpt
python -m turboquant.validation.ood_eval --drafts orig:...,kl:...,kltv:kltv_ckpt
# 8B LoRA-QAT (the validated 8B path) — Blackwell-class GPU
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m turboquant.validation.qat_nvfp4 \
    --model unsloth/Meta-Llama-3.1-8B --objective kl --lora-rank 32 --steps 1500 --lr 1e-4 \
    --max-len 512 --grad-checkpoint
```

- **Theory:** [THEORY.md](THEORY.md) · **QAT detail:** [QAT_FINDINGS.md](QAT_FINDINGS.md) ·
  **Roadmap:** [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md) · **README:** [README.md](README.md)
- **Codec:** `turboquant/nvfp4.py`, `turboquant/gptq.py`, `turboquant/distill.py`
- **Harnesses:** `turboquant/validation/` — `hf_perplexity.py` (main), `qat_nvfp4.py`,
  `ood_eval.py`, `acceptance.py`, `kv4_correction.py`, `baseline_compare.py`,
  `vllm_fp4_bench.py`
- **Results:** `results/*.json`
