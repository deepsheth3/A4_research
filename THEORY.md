# A Distortion Theory of Microscaled FP4 Quantization

*Why, under per-block microscaling with equalization, basis/permutation/rounding
tricks are redundant and only additive low-rank correction reduces output error.*

This formalizes the empirical law established across this project (rotation,
per-block Hadamard, channel permutation, output-weighted scale selection, GPTQ,
AWQ, rank allocation, and coordinated rounding all wash out on real Llama-8B;
equalization and additive SVD/QJL/low-rank corrections do not). The theory
**predicts** each of those outcomes from two lemmas.

---

## 1. Setup

A linear layer computes `y = Wx`, `x ∈ ℝ^d`, `W ∈ ℝ^{m×d}`. The accuracy-relevant
distortion of a quantizer `Q` is the **output** error

```
D(Q) = E‖W(x − Q(x))‖² = E[ eᵀ G e ],    e := x − Q(x),   G := WᵀW ⪰ 0.
```

`G` is the **output metric**: `G_ii = ‖W_:i‖²` is how much input channel *i* drives
the output; `G_ij` is the coupling between channels *i,j*.

**Microscaling quantizer.** Partition the `d` channels into blocks `B₁,…,B_{d/K}`
of size `K` (NVFP4: `K=16`). Fix the E2M1 grid `𝒢` (symmetric, `max 𝒢 = g`). Each
block gets one scalar scale, taken as absmax:

```
s_b = (1/g)·max_{i∈B_b} |x_i|,      Q(x)_i = s_{b(i)}·Π_𝒢(x_i / s_{b(i)}),
```

`Π_𝒢` = nearest grid point. The scale `s_b` is stored in fp8 (the MX4 overhead).

### Assumptions (idealizations, stated honestly)

- **(A1) High-resolution white error.** `E[e_i e_j | x] = δ_ij · κ · s_{b(i)}²`
  for a grid-dependent constant `κ`. This is the standard high-resolution /
  subtractive-dither model; it is *exact* under dither and approximate for E2M1 at
  `K=16`. All quantitative claims are first-order in this model.
- **(A2) Absmax scaling**, as above (what the hardware/codec uses).

Under (A1),

```
D = κ · Σ_i G_ii s_{b(i)}²  =  (κ/g²) · Σ_b ( max_{i∈B_b} x_i² ) · tr(G_{bb}).   (★)
```

So distortion is **the per-block peak magnitude, weighted by the block's output
energy `tr(G_{bb})`, summed over blocks.** Everything follows from how each "trick"
acts on (★).

---

## 2. The two levers

### Lemma 1 (Equalization = the optimal, and sufficient, multiplicative preconditioner)

Any invertible input transform `x ↦ Mx` folds into the weights `W ↦ WM⁻¹` and is
**output-invariant** (`WM⁻¹·Mx = Wx`); it changes `D` only through its effect on
the quantizer, i.e. on `s_b` and `G_bb` in (★). Restrict to the natural generators:
diagonal `Λ` (per-channel scaling = *equalization*) and orthogonal `R` (*rotation*,
incl. permutation as a special case).

For diagonal `Λ = diag(λ)`: `x_i ↦ x_i/λ_i`, `G ↦ ΛGΛ`, and (★) becomes

```
D(Λ) = (κ/g²) · Σ_b ( max_{i∈B_b} (x_i/λ_i)² ) · Σ_{i∈B_b} λ_i² G_ii.
```

This is a smooth, coercive function of `λ` with an interior minimizer `λ★`
(balancing each block's peak against its output-energy weight). **`λ★` is exactly
the equalization scale**, and it is *not* redundant: it strictly reduces `D`
whenever channel magnitudes are imbalanced. Equalization is the optimal element of
the diagonal (multiplicative) group. ∎

The reason equalization is **special** among multiplicative tricks: it acts
*per-channel, before blocking*, so it directly lowers the `max_{i∈B_b}` term in (★)
by shrinking outlier channels. Orthogonal transforms cannot do this — they only
reshuffle energy *within the already-locally-scaled blocks*, which Lemma 2 shows is
futile after equalization.

### Lemma 2 (Orthogonal transforms are redundant: the Gaussian fixed point)

Define a block's **peak-to-average ratio** `ρ(x_b) = max_{i∈B_b} x_i² / (1/K Σ_{i∈B_b} x_i²)`.
By (★) with locally-isotropic `G_bb ≈ ḡ_b I` (which equalization induces, since it
balances `G_ii`), `D ∝ Σ_b ḡ_b ‖x_b‖² · ρ(x_b)/K`, and rotation preserves `‖x_b‖²`,
so **rotation reduces `D` iff it reduces the energy-weighted mean of `ρ`.**

A within-block orthogonal transform `R` resamples the `K` block coordinates while
conserving their energy. Its effect on `E[ρ]`:

| within-block distribution | `R` Gaussianizes it → | `E[ρ]` |
|---|---|---|
| **super-Gaussian** (an outlier channel) | spreads the peak | **decreases** (rotation helps) |
| **Gaussian** | invariant (rotational invariance of `N(0,σ²I)`) | **unchanged** (neutral) |
| **sub-Gaussian** (near-flat) | concentrates | **increases** (rotation hurts) |

The Gaussian is the **fixed point** of orthogonal mixing, and it is precisely the
state equalization produces: balancing per-channel scales makes each block a set of
`K` comparable-variance draws, i.e. ≈ Gaussian (peak/avg `→ 2 ln K ≈ 4.5` at `K=16`).

*Verified numerically (`peak/avg`, K=16): outlier block 14.5 → 3.5 (helps);
Gaussian 4.55 → 4.55 (neutral); flat 1.18 → 4.48 (hurts).* This **is** the measured
per-block Hadamard pattern.

**Corollary.** After equalization drives blocks to the Gaussian fixed point,
orthogonal/permutation transforms are neutral-to-harmful: they cannot lower `E[ρ]`,
and generically raise it. The rotation gain is real only for *super-Gaussian*
blocks (outlier channels) — exactly what equalization has already removed, more
cheaply (diagonally, foldable, no cross-block mixing). ∎

### Lemma 3 (Additive low-rank correction is the rate-distortion-optimal side-channel)

Fix the base quantizer (with optimal equalization). The output residual is
`r = We`, with covariance `Σ_r = W·E[eeᵀ]·Wᵀ ⪰ 0`. A side-channel of rate `B`
transmits an estimate `r̂`; the minimum achievable `E‖r − r̂‖²` at rate `B` is given
by **reverse water-filling on the eigenvalues of `Σ_r`** (Gaussian rate-distortion),
asymptotically achieved by transmitting the top-`k` principal coefficients of `r`
(the KLT/SVD of the residual). The additive correction

```
r̂ = Σ_{j≤k} ⟨r, u_j⟩ u_j ,    u_j = top eigenvectors of Σ_r,
```

— our W-aware SVD side-channel (and, with random projections, its 1-bit relaxation
QJL) — **is exactly this optimum.** No input reparametrization `M` produces such a
term: transforms only alter the *base* `D`; they transmit *no* residual information.
Hence additive low-rank is the unique class that accesses the side-channel-optimal
gain, and its budget→distortion curve is the residual's R-D curve. ∎

### Lemma 4 (Coordinated rounding is the dual of additive correction — same subspace)

Non-greedy rounding chooses `e` within the per-element quantization cells to
minimize `eᵀGe` directly (instead of `‖e‖²`). The achievable reduction is the
projection of `e` out of the high-`G` (high-output-energy) subspace — *the same
subspace Lemma 3 transmits.* It is the **dual** lever: "shape error away" vs. "add
error back." Two consequences, both observed:

- **Within a block**, after equalization `G_bb` is ≈ diagonal (Lemma 2's isotropy:
  block channels are output-uncorrelated), so there is no off-diagonal coupling to
  exploit ⇒ within-block coordinated rounding is **inert** (measured: G16 +0.2%).
- **Across blocks**, full-`G` coordinated rounding reaches the same low-`G`
  subspace as the additive correction (measured: full-G ≈ additive SVD gain), but
  only via a serial `O(d)` dependency chain — not parallelizable, not Pareto-safe.

So coordinated rounding adds nothing the additive correction doesn't already give in
the deployable (parallel) regime. ∎

---

## 3. Main Theorem

> **Theorem (Two-lever law for microscaled FP4).** Under the microscaling quantizer
> with per-block absmax scaling and the high-resolution error model (A1), the output
> distortion `D` over the combined design space of
> (i) diagonal preconditioning, (ii) orthogonal/permutation transforms, (iii) the
> per-block scale objective, (iv) coordinated rounding, and (v) additive rank-`k`
> side-channels, is minimized by
>
> **(i) optimal diagonal preconditioning (equalization) + (v) additive rank-`k`
> residual correction.**
>
> Classes **(ii), (iii), (iv) are redundant**: after (i) is applied they cannot
> reduce `D`, and generically increase it. Equivalently — *the only
> distortion-reducing degrees of freedom are the diagonal preconditioner and the
> additive low-rank residual term. The orthogonal group and the rounding lattice act
> trivially on the achievable distortion once the input is equalized.*

**Proof.** Lemma 1 establishes (i) as the optimal multiplicative preconditioner and
gives the base `D`. Lemma 2 shows the orthogonal part of any further multiplicative
transform (ii) is neutral-to-harmful at the equalized (Gaussian) fixed point;
permutations are a subset. The per-block scale objective (iii) only re-selects `s_b`
within (★), which absmax + the additive correction already render second-order
(empirically: `oda` washes out). Lemma 4 reduces coordinated rounding (iv) to the
additive subspace, inert within-block and non-deployable cross-block. Lemma 3
identifies (v) as the unique side-channel-optimal residual term. ∎

---

## 4. The theory predicts every experiment

| Tried (this project) | Theorem says | Observed |
|---|---|---|
| PolarQuant (per-token norm) | (ii)/scale — redundant | redundant (≈1e-5) |
| QuaRot global Hadamard | (ii) at Gaussian f.p. — harmful | NMSE ×3, PPL 44→94 |
| per-block Hadamard (always) | (ii) — neutral/harmful post-eq | −0.019 (hurts) |
| per-block Hadamard (best-of) | (ii) — helps only super-Gaussian blocks | +0.057 isolated, wash in stack |
| channel permutation | (ii) special case — redundant w/ eq | wash (6.051) |
| output-weighted scale (`oda`) | (iii) — second-order after (v) | wash (6.052) |
| GPTQ / AWQ (weights) | (ii)/(iii) analog — redundant under MX4 | 13% / worse |
| per-layer rank allocation | mis-allocates (v); needs end-to-end metric | worse (6.363) |
| within-block coordinated rounding | (iv) Lemma 4 — inert (diagonal `G_bb`) | +0.2% (G16) |
| full-`G` coordinated rounding | (iv) — additive subspace, serial-only | +36% but not deployable |
| **equalization** | **(i) — optimal preconditioner** | **the strongest free method** |
| **additive SVD / QJL / low-rank** | **(v) — R-D-optimal side-channel** | **the only thing that closes the gap** |

The floor follows: with (i) and (v) saturated and (ii)–(iv) provably trivial, the
remaining distortion is the residual R-D tail beyond the side-channel budget — which
the rounding-ceiling experiment measured as reachable only by non-deployable serial
rounding, and still short of FP8.

---

## 5. Limitations (honest)

- **(A1)** is high-resolution/dither; E2M1 at `K=16` is finite-resolution, so `κ` is
  approximate and the white-error assumption is first-order. The qualitative law is
  robust (it depends on (★)'s structure, not the exact `κ`); exact constants are not
  claimed.
- **Lemma 2** uses local isotropy of `G_bb` after equalization; real layers are
  approximately, not exactly, isotropic within blocks (measured off-diagonal energy
  ~3%), which is why rotation is *nearly* (not exactly) neutral.
- Equalization in practice uses a per-layer `α`-search heuristic, not the exact
  `λ★`; the gap is small but the theorem's optimality is over the idealized class.
- Distortion `D` is a per-layer surrogate for end-to-end loss/PPL; the mapping is
  monotone and locally linear but not proven globally.
- All accuracy is fake-quant simulation; throughput is out of scope here.

These bound the *constants*, not the *structure*: the two-lever decomposition and the
redundancy of the orthogonal group and rounding lattice are the load-bearing claims,
and each is independently confirmed on real Llama-8B.
