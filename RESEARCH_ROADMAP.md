# NVFP4 Research Roadmap

Status as of 2026-06-12. The old paper-survey roadmap is archived at the bottom;
most of it has been executed (see "What's done"). This version is organized
around **what we've established** and **the one active hypothesis worth testing
next** (per-block Hadamard), plus the path to publication.

## The one rule (governs everything)
**Pareto-clean: no technique is kept unless it improves an axis without
regressing any other** (accuracy, latency, memory, the pure-FP4 main GEMM path).
Corrections must be offline-folded into weights or applied in a fused epilogue,
and the whole stack must stay better-than-FP8 on every axis. See the
`nvfp4-pareto-constraint` memory.

---

## What's done (Llama-3.1-8B, full WikiText-2; FP16 5.918, FP8 5.948)

- **A4 (FP4 activations, FP16 weights): 6.050 = +0.10 vs FP8, 68% gap closed.**
  Stack = channel equalization + per-block best-of-{0,mid} zero-point × optclip +
  W-aware SVD side-channel + per-block QJL. Near-FP8 parity, the hard half.
- **W4A4 (FP4 weights + activations): best 6.294 = +0.35 vs FP8, 58% weight
  cliff recovered, Pareto-clean (0.75 byte/elem < FP8).** Winner = GPTQ +
  **additive low-rank residual correction** (in/8, fp8 factors).
- **Central finding:** under per-16 microscaling, **scaling/redistribution
  methods are redundant-to-harmful** (PolarQuant, global rotation, GPTQ ~13%,
  AWQ worse, rank allocation worse); **only additive side-channels help** (SVD,
  QJL, low-rank weight correction).
- **Negatives ruled out:** fp4 factors (< fp8), joint W+A Hessian (no signal —
  interaction negligible), per-layer rank allocation (< uniform).
- GSM8K 30-q probe: 50.0% → 46.7% (noise; rules out collapse, not parity).

---

## RESOLVED — per-block (block-diagonal) Hadamard (tested on real 8B, 2026-06-12)

**Idea (user):** apply a 16×16 Hadamard *within* each block before E2M1
quantization to spread within-block outliers → tighter scale. Distinct from the
*global* Hadamard we ruled out — it never mixes across blocks.

**Verdict: works in isolation *as a per-block choice*; redundant in the full
stack.** Implemented (`nvfp4_quantize_hwht`, deployable always-rotate folds
offline + multiply-free) and run on Llama-3.1-8B, full WikiText-2 (same box;
controls fp8 5.948, eqzp_svd_qjl reproduced 6.057):

| Test | PPL | vs baseline |
|---|---|---|
| nvfp4_zp (isolation baseline) | 6.186 | — |
| nvfp4_hwht (always-rotate) | 6.205 | **−0.019 (hurts)** |
| **nvfp4_hwht_bestof** | **6.129** | **+0.057 (wins)** |
| eqzp_svd_qjl (full-stack control) | 6.057 | — |
| eqzp_svd_qjl_hwht (composed) | 6.050 | +0.007 (**wash, within noise**) |

**What it confirmed:** the Hadamard is a *local basis option* — it helps only
when it's a per-block **choice** (forced always-rotate hurts smooth blocks),
validating the best-of-N philosophy (like zero-point/optclip) on a real model.
This matched the synthetic prediction exactly (always-rotate ~neutral/negative on
smooth distributions, best-of positive everywhere).

**Why it doesn't make the stack:** once eq + SVD + QJL are present, the
**additive SVD side-channel already drains the outlier-block pond** the Hadamard
targets — zero marginal gain. Another instance of the project's central thesis:
*additive correction subsumes basis/scaling tricks under microscaling.* A
publishable negative, not a disappointment.

**Caveats:** composed test used the deployable always-rotate; composed best-of
(the ceiling) was killed to save box time and isn't directly deployable anyway
(per-block choice breaks a uniform GEMM — would need a fixed per-position pattern
decided offline). Fake-quant Hadamard is slow in sim (~370s/mode); irrelevant on
real HW (multiply-free).

---

## RESOLVED — channel-level redistribution is redundant (a law, shown 3 ways)

**All "change-the-basis / regroup-channels" tricks collapse to the control once
equalization + additive side-channels are in place** (Llama-3.1-8B, full
WikiText-2; control `eqzp_svd_qjl` = 6.057):

| Lever | PPL | vs control |
|---|---|---|
| Hadamard always-rotate | 6.050 | wash |
| Hadamard fixed-mask (deployable) | 6.052 | wash |
| channel permutation (`cperm`) | 6.051 | wash |

This is one **robust law**, not three negatives: *under per-16 microscaling with
equalization, channel redistribution has nothing left to do — only additive
side-channels (SVD, QJL, low-rank weight correction) move PPL.* It mirrors the
earlier PolarQuant / rotation / GPTQ / AWQ redundancy results — same mechanism,
now airtight via three independent confirmations. **Publishable as a clean claim.**

Free side-finding (a unit test caught it before any box spend): naive
magnitude-sort permutation can *hurt* (+14% on abundant high-variance channels —
grouping them raises block maxima); only outlier-channel *isolation* helps, and
equalization already does that. The channel-regroup family is **closed**.

## RESOLVED — output-weighted scale selection is redundant too (the law, 3rd family)
`oda` (`nvfp4_eqzp_svd_qjl_oda`): pick the base block scale by `min (x−Q)ᵀdiag(WWᵀ)(x−Q)`
(downstream-sensitivity-weighted) instead of plain MSE. **Tested 8B: 6.052 vs 6.050 control = wash.**
The additive SVD+QJL side-channel absorbs the base-quant objective, so input-MSE vs output-weighted-MSE
doesn't matter. This is the **third independent family** to fall to the same law (after basis-change
rotation/WHT and channel-regroup permutation): *once additive correction is in place, refinements to the
base quantizer are redundant.* Gates the rest of the math survey — rank-allocation-by-output-error,
Wiener shrinkage, James-Stein all rest on this now-falsified premise → not pursued.

## The floor is PROVEN, not assumed (rounding-ceiling analysis, real 8B)
Tested computation-aware rounding (minimize output error `(x-q)ᵀG(x-q)`, G=WᵀW, vs
nearest) on **real captured Llama-8B activations**, at block widths 16/128/256/full,
±SVD. Real aggregate (output-error reduction vs nearest):

| | reduction |
|---|---|
| nearest + SVD (our stack) | +50.5% |
| G16 + SVD (within-block, "OBR") | +50.2% — **dead** |
| G128 + SVD | +49.4% |
| G256 + SVD | +49.2% |
| full-G + SVD (serial) | +68.4% |

Three facts: (1) within-block coordinated rounding is **inert** — additive SVD already
has it. (2) A real ceiling **exists** (full-G cuts our-stack error +36%) — but it's
**long-range**, captured by *no* block width, so reachable only via full serial
4096-channel rounding = **3–50× latency, not Pareto-viable**. (3) Even that ceiling
≈ **6.00 PPL, still above FP8 (5.948)** — dominated even if you paid the latency.
**A4 = 6.050 is the practical floor**; the additive side-channel captures all
parallel-deployable structure. Tooling: `capture_activations.py` + `analyze_rounding_ceiling.py`.

## The codec-trick space is exhausted — pivot to the paper
Three trick families ruled out (basis / regroup / scale-objective), plus the earlier
PolarQuant/rotation/GPTQ/AWQ redundancies. Everything left that could beat the current numbers
**breaks a premise**: light fine-tuning (not calibration-free) or lattice/trellis (leaves E2M1).
The contribution is the **mechanism**, and it's now exceptionally well-supported. Next work is
**breadth + baselines + composition + theory** (see "Path to publication"), not more codec tricks.

## Levers that break a premise (bigger gains, different paper)
- **Light fine-tuning** (LoRA on the correction) — likely ~FP8 parity, but not
  calibration-free; becomes per-model bespoke.
- **Lattice/trellis quant (QuIP#/QTIP)** — rate-distortion-optimal, but leaves the
  hardware-native E2M1 grid.

---

## Path to publication (none of this requires beating 6.294)
The contribution is the **mechanism** (microscaling redundancy + additive
correction + per-block local basis), not the absolute number.

**Non-negotiables:**
1. **Breadth** — 3–4 models across sizes (Llama-3.2-1B/3B, 8B) and families
   (Qwen2.5, Mistral, Gemma-2). Small models run on **free tier** (fake-quant
   sim, fits 16 GB).
2. **Full downstream suite** — MMLU, HellaSwag, ARC-C, WinoGrande, GSM8K (full
   1319), + C4 PPL. (The 30-q GSM8K was only a collapse check.)
3. **Head-to-head baselines** — QuaRot, SpinQuant, Atom, NVFP4-specific
   (ARCQuant, Four-Over-Six, ScaleSweep).

**Completeness / elevation:**
4. **W4A4KV4 composition** — combine W4 + A4 + OmniStack KV4 and measure (the
   "4-bit everything" headline; never run together yet).
5. **Real B200 throughput** (MLSys) or a rigorous roofline (ML venue).
6. **Theory** — ✅ DONE, see [THEORY.md](THEORY.md). Two-lever theorem: under
   microscaling + the high-resolution error model, output distortion is minimized by
   (i) optimal diagonal preconditioning (equalization) + (v) additive low-rank
   residual correction; orthogonal/permutation transforms, scale-objective changes,
   and coordinated rounding are provably redundant (the Gaussian fixed point of
   rotation + R-D optimality of the KLT residual). Predicts all ~11 experiments.

**Framing fix:** call it **light-calibration PTQ**, not calibration-free (we use
a few wikitext-train windows for eq scales and GPTQ Hessians).

**Venue fit:** TMLR (values honest, thorough work incl. the negative results) or
NeurIPS/ICML (needs breadth + baselines + theory); MLSys needs real kernels.

## Compute strategy (out of personal GPU budget)
- Small-model accuracy sims (breadth + tasks) on **Kaggle/Colab free tier** —
  fills the biggest publication gap for $0.
- Sponsorship for 8B/70B + B200: **NVIDIA Inception** / email the **ModelOpt /
  TensorRT-LLM team** (this work validates NVFP4 — you do their marketing),
  **ML Collective** (independent researchers), **Hugging Face** grants.
- The result + clean repo is what unlocks these — write the draft first.

---

## Archived: original paper-survey roadmap
The original literature survey (ARCQuant, Four-Over-Six, ScaleSweep, RaZeR,
SmoothQuant, QuaRot, SpinQuant, FlatQuant, SERQ/LQER/QERA, GPTQ/AWQ/MR-GPTQ,
Atom, QJL/TurboQuant/PolarQuant) and the original phase plan lived here. Most
phases executed: GPTQ/AWQ (redundant), low-rank correction (works), optclip/zp
(works), oda built, rank allocation (worse), joint W+A (no signal). The
remaining unexecuted survey item worth revisiting is **RaZeR redundant-zero
remapping** (a cheaper best-of-two zero-point). See git history for the full
original text.
