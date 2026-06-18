# Door C: escaping the two-lever law by unfreezing λ

A companion to `THEORY.md`. The two-lever law (Proposition 1 there) says that for
a party who takes the **trained model**, the **quantizer**, and the **white-error
model (A1)** as fixed, the only distortion-reducing levers are (i) diagonal
equalization and (v) additive low-rank correction. This note records what happens
when you are *allowed* to edit those frozen inputs — and shows, with data, that the
only self-consistent escape is a single change to the existing codec.

## The two hardware/training doors cancel

Editing the frozen inputs gives two candidate "doors," each owned by a different
party:

- **Door A (foundry — edit the quantizer):** stochastic rounding makes A1 *true by
  construction* (zero-mean, decorrelated error). But the residual covariance is
  `Σ_y = W·E[eeᵀ]·Wᵀ`; SR whitens `E[eeᵀ]`, which **destroys the low-rank coherence
  that lever (v) exists to remove**, converting a tight correctable residual into a
  high-rank floor. It fixes the *premise* while reshaping `Σ_y` against the
  correction. SR is only safe on a denser grid — i.e. a format change, not free
  silicon.
- **Door B (lab — edit the model):** QAT flattens the per-block peak-to-average
  ratio so the heavy tail never forms. But the tail that survives QAT is the
  **structural-outlier** tail (attention sinks / massive activations), which is
  sparse, persistent, and the *most low-rank-coherent residual imaginable* — exactly
  what lever (v) is best at. QAT hands (v) a cleaner target; it does not retire it.

The contradiction: the structural outlier that dominates its block's scale is the
**generator** of the coherence that lever (v) lives on. Door B protects that
generator (it is load-bearing); Door A's whole mechanism is to decorrelate it. The
two doors are **rival edits to the same matrix `Σ_y`** — one builds the structure
the other destroys. The "foundry + lab co-design" story eats itself.

## Door C: unfreeze the static parameters — and it collapses to one change

The quantizer's per-block absmax scale `s_b` is already *runtime-adaptive* (it reads
the live block), so it is robust to distribution shift. The brittleness lives
entirely in the two **static, calibration-fit** objects: the equalization scales `λ`
and the correction basis `u_j`. Door C = move those off their frozen values.

**The `u_j` half is already done — and already optimal.** The shipped side-channel
basis is *W's top input singular vectors*, computed offline from `W` with no
calibration data (`_svd_basis` in `turboquant/validation/hf_perplexity.py`), not a
calibration-fit residual PCA. The plan's instinct ("drop SVD → QJL for
obliviousness") would be a **downgrade**:

### Experiment 1 — `residual_basis.py` on real Llama-3.1-8B (7 layers)

Output-metric residual energy removed, equal rank, equal side-bits:

| basis | residual energy removed |
|---|---|
| **G-basis (oblivious, W-derived — what we ship)** | **+50.8%** |
| res-basis (calibration-fit residual PCA) | +32.0% |

Verdict: *"marginal — residual ≈ W structure."* The oblivious basis is not a
compromise; it **beats** the calibration-fit basis by 18.8pp (the latter overfits
~200 calibration rows). So `u_j` needs no change — it is oblivious *and* optimal.

That leaves **one** change: **runtime-adaptive, warm-started `λ`** — track the
per-channel scale from live activations (causal decaying-max), warm-started at the
calibration absmax so token 0 is identical to the shipped codec and the warm floor
fades into the live stream.

### Experiment 2 — `runtime_lambda_accept.py`, FP4 draft vs FP16 target

Acceptance `α = Σ min(p,q) = 1 − TV` between the FP4 draft and FP16 target,
teacher-forced single-stream, frozen-λ vs adaptive-λ (everything else identical):

| model / prompt | frozen α | adaptive α | Δα | worst sustained dip |
|---|---|---|---|---|
| gpt2 / OOD code | 0.9007 | 0.8974 | −0.0034 | small |
| gpt2 / in-dist prose | 0.9107 | 0.9210 | **+0.0103** | none (helps) |
| TinyLlama-1.1B / OOD code | 0.9479 | 0.9515 | **+0.0036** | +0.038 over 8 tok, self-heals |

At actual draft-model scale, letting `λ` move is **free-to-positive** on OOD, and
the worst case is a brief, bounded transient that self-heals — never a quality
regression, because the target verifies every token. At batch ≥ 32 the per-channel
stats are instantaneous (the FP4-wins regime); single-stream needs an EMA, but the
warm-start floor caps the downside at the status quo.

## What this is — and is not

Door C is a **robustness/variance** win, not an accuracy win. In-distribution
perplexity stays at the shipped 6.05 *by construction* (warm start). The payoff is
that the number **holds under distribution shift** instead of drifting up — it
prices deployment-robustness with data instead of asserting it. Consistent with the
Pareto constraint (no regression on any axis): the only axis that moves is a
transient acceptance dip on OOD cold-start, which the target backstops.

**Caveats (these establish mechanism + sign, not deployable magnitude):** fake-quant
only; small models (gpt2 124M, TinyLlama 1.1B); single short OOD prompt; single
decay (0.98); reduced SVD rank (d/64) on the TinyLlama run.

## Next (H100, not the Mac)

Tighten experiment 2 to a deployable claim: full d/16 basis; multiple OOD domains
(code / math / multilingual); a decay sweep; longer context; and the key addition —
**end-to-end OOD perplexity, frozen-λ vs adaptive-λ**, confirming no in-distribution
regression. The acceptance side exists; the PPL side is a small extension to the
`hf_perplexity` harness.
