# NVFP4 Research Roadmap — W4A4KV4 at FP8 Accuracy

## Current status
- **A4 is strong**: `eqzp_svd_qjl` = **6.050**, ≈ **+0.101 PPL from FP8** on Llama-3.1-8B / WikiText-2 (FP16 5.918, FP8 5.948/5.949, raw FP4 6.263). 68% of the FP4→FP8 gap closed, hardware-native main path, all corrections offline-folded or fused-epilogue.
- **W4A4 is NOT solved**: naive W4 pushed PPL to **6.629**; the extra **+0.579** came entirely from *raw* FP4 weight quantization (no error feedback).
- **Biggest unlock is not more activation tricks — it's making W4 stop destroying the model.**

---

## Papers to track

### A. NVFP4 / MXFP4 / FP4-specific
| Paper | Why it matters | What to take |
|---|---|---|
| **ARCQuant** (Augmented Residual Channels for NVFP4) | Direct NVFP4 competitor; residual channels with unified NVFP4 GEMM | Residual-channel design to fix A4/W4 without random QJL overhead |
| **Four Over Six** (Adaptive Block Scaling) | Very close to optclip; evaluates alternative per-block scale factors | Replace hand-designed optclip candidates with 4-over-6 scale selection |
| **ScaleSweep** (Accurate NVFP4 PTQ) | Block-scale init + importance-aware scale choice | Use weight/activation importance when choosing scales, not plain MSE |
| **RaZeR** (Redundant Zero Remapping) | FP4 redundant-zero representation adds useful values | Better version of best-of-two zero-point idea |
| **Adaptive Block-Scaled Data Types** | Adaptive FP4/INT4 block-scaled formats | Per-block format: FP4 where tails matter, INT4 where uniform helps |
| **MixFP4** (Adaptive FP4/INT4 blocks) | Selects FP4 micro-format per block | E2M1 vs E1M2/E2M2-like block choice as ablation |
| **AMXFP4** (Asymmetric Microscaling FP) | Asymmetric shared scales for activation outliers | Better asymmetric scale/zero handling than naive zero-point |
| **LLM-FP4** | Older but key FP4 W/A quant | Per-channel activation scaling + FP4 clipping search |

### B. Rotation / flattening / outlier / activation-shaping
| Paper | Why it matters | What to take |
|---|---|---|
| **SmoothQuant** | Classic outlier smoothing into weights | Keep equalization, tune α per layer/channel more systematically |
| **QuaRot** (rotated W4A4KV4) | Rotation baseline | Baseline only — our MX4 tests showed rotation *hurts* under NVFP4 |
| **SpinQuant** (learned rotations) | Learned > random rotations in W4A4KV4 | Revive rotation only as learned local/channel permutation, not global WHT |
| **FlatQuant** (learnable affine flattening, fused) | Flattens W+A with fused kernels | Strong candidate to improve A4 and W4 deployably |
| **InfoQuant** (distribution shaping) | Activation quant as distribution design | "Quantizer-facing distribution" idea to beat MSE for A4 |
| **OffQ** (offsetting structured outliers) | Offsets low-dim structured outliers | Structured offset extraction before NVFP4 scaling |
| **DuQuant** (dual transform) | Redistributes activation outliers | Activation-transformation baseline |

### C. Low-rank / residual / error-reconstruction
| Paper | Why it matters | What to take |
|---|---|---|
| **SERQ** (saliency-aware low-rank) | Very close to our SVD; saliency-aware compensation for W4A4 | Replace plain top-SVD with saliency-aware rank allocation |
| **Low-Rank Correction for Quantized LLMs** | Low-rank correction for activation error | Compare SVD side-channel vs full low-rank correction |
| **LQER** (low-rank error reconstruction) | Activation-induced scaling guides reconstruction | Use activation-weighted SVD, not plain SVD |
| **QERA** (analytical framework) | Closed-form low-rank error reconstruction | Closed-form objective to choose SVD/QJL basis |
| **ASER** (smoothing + low-rank recon) | Close to our equalization + SVD stack | Use as a direct baseline |
| **GlowQ** (group-shared low-rank) | Shares low-rank factors across input-sharing groups | Reduce SVD cost by sharing bases across q/k/v or gate/up |
| **Preserve-Then-Quantize** (rank budgets) | Preserves top singular subspace before quantizing residual | For W4: protect dominant directions before NVFP4 |

### D. Weight quantization
| Paper | Why it matters | What to take |
|---|---|---|
| **GPTQ** | The obvious fix for our raw-W4 failure | Hessian-aware NVFP4 weight quant with error feedback |
| **AWQ** | Protects salient weight channels via activation stats | AWQ-style channel protection before NVFP4 weight quant |
| **MR-GPTQ** (microscaling FP4 weight quant) | Directly relevant to MXFP4/NVFP4 weights | **Probably the most important paper for fixing W4** |
| **Atom** (low-bit serving system) | Full W4A4 serving: mixed precision, dynamic acts, KV quant, kernels | Copy the systems discipline: accuracy + throughput + kernels + batching |

### E. KV cache / TurboQuant / QJL
| Paper | Why it matters | What to take |
|---|---|---|
| **QJL** (1-bit JL for KV) | Our OmniStack/QJL lineage | Keep for KV + residual side-channel; don't rely on QJL alone for A4 |
| **TurboQuant** (online VQ) | PolarQuant + QJL theory | Keep theory for KV; activations need W-aware correction |
| **PolarQuant** (Hadamard Gaussian weight quant) | We found PolarQuant-style ideas fail under MX4 activations | Cite as related work; use our negative result to differentiate |

---

## Goal
**W4A4KV4 with FP8-like accuracy** — A4 near solved, W4 gets GPTQ/AWQ/MR-GPTQ treatment, KV4 uses OmniStack/QJL/TurboQuant.

## Phase 1 — Fix W4 first (the bottleneck)
Naive W4A4 went 6.050 → 6.629. Weight quant cost is enormous.

- **W4 v1**: GPTQ-style Hessian-aware NVFP4 weight quant; group/block=16; E2M1 + FP8 E4M3 scale; optclip / Four-Over-Six candidate scales; row-chunked to avoid OOM.
- **W4 v2**: AWQ-style activation-aware channel scaling; protect top 1–5% salient channels; fold scale into adjacent weights.
- **W4 v3**: MR-GPTQ-style FP4-specific error feedback; block-Hadamard for *weights only* (not activations); use activation Hessian **XXᵀ**, not WWᵀ.

**Targets** (from current 6.629): first ≤6.25 · strong ≤6.15 · very strong ≤6.10 · dream ≤6.05–6.08.

## Phase 2 — Saliency-aware SVD (replace plain SVD)
Plain Frobenius SVD isn't optimal (SERQ/LQER/QERA).
- Current: top singular vectors of W / equalized W.
- Better: top singular vectors of **activation-weighted error**, objective `||(XW) − (Q(X)W + correction)||`.
- Best: per-layer rank allocation — high rank for down_proj/o_proj/maybe gate_proj, low/zero for insensitive layers. Reduces side bits while improving quality.

## Phase 3 — Four-Over-Six + ScaleSweep objective for optclip
Replace fixed candidate MSE search with a weighted objective:
- Activations: choose scale by output-domain error `||(Q_s(X) − X)W||²`.
- Weights: choose scale by activation-Hessian error `trace((W − Q_s(W))ᵀ H (W − Q_s(W)))`.
- Candidates: max/6, max/4, percentile/6, ScaleSweep-style learned, best-of-two zero-remap.
- **One of the highest-return improvements.**

## Phase 4 — ARCQuant-style residual channels
Add residual channels into the reduction dim so compensation stays GEMM-friendly.
- A4 main: NVFP4 eqzp activation. Residual: top-r W-aware residual channels quantized to NVFP4/FP8, fused into GEMM reduction dim.
- Compare vs SVD side-channel / QJL / SVD+QJL. Switch if ARC beats them at equal bit budget.

## Phase 5 — FlatQuant / InfoQuant (only if fusable)
Don't add latency-destroying methods. Ablate: `eqzp_svd_qjl` vs FlatQuant+eqzp(/+svd) vs InfoQuant-transform+eqzp(/+svd). **Keep only if PPL improves ≥0.03 AND overhead is near-zero/fusable.**

## Phase 6 — KV cache (OmniStack/QJL first)
Don't overcomplicate KV yet. Baseline FP16 KV → OmniStack KV → QJL/TurboQuant KV → maybe NVFP4 KV. Long-context eval only after W4A4 PPL is not broken.

## Phase 7 — Evaluation
1. WikiText-2 full
2. GSM8K 150–200 q **with checkpoint/resume** (30-q probe only ruled out collapse)
3. MMLU small subset
4. HellaSwag or ARC-Challenge
5. Needle 8K/32K for KV

---

## "Best of all papers" target stack — OUR idea is the spine
**Framing (important):** our stack (`eqzp_svd_qjl` + OmniStack KV) is the base and stays. We do NOT replace it with any paper. Each paper technique is a candidate *drop-in for ONE component*, adopted only if it beats OUR version of that component AND passes the Pareto gate. Component-level tournament where our piece is the defending champion in every slot; everything else of ours is kept.

| Slot | Our champion (default) | Challenger — adopt only if it beats ours |
|---|---|---|
| Per-block scale | optclip best-of-8 (MSE) | 4-over-6 / ScaleSweep *objective* into our best-of-N |
| Zero-point | best-of-two zp | RaZeR redundant-zero |
| Side-channel | eq + SVD + QJL | saliency basis / per-layer rank grafted into our SVD; ARCQuant residual (if fusable) |
| Weights (open slot) | — (we had no W4) | GPTQ/MR-GPTQ/AWQ fills it |
| KV | OmniStack/QJL | TurboQuant comparison; NVFP4 KV |
| Transforms | SmoothQuant equalization | FlatQuant/InfoQuant only if fusable; NO global QuaRot under MX4 |

- **Hardware invariant**: keep NVFP4 main path; corrections offline-folded or fused-epilogue; no unfused high-precision branch.

## Priority order
1. GPTQ/MR-GPTQ-style W4 weight quant
2. Four-Over-Six / ScaleSweep objective for weight + activation scales
3. Saliency-aware SVD / QERA instead of plain SVD
4. ARCQuant-style residual-channel alternative
5. GSM8K checkpointed 200-question run
6. Comparison table vs ARCQuant, Four Over Six, ScaleSweep, SERQ, QuaRot, SpinQuant, FlatQuant
7. B200 / TRT-LLM kernel plan

**Bottom line:** the activation result is already strong. The biggest unlock is **making W4 stop destroying the model**. Fix W4 → real W4A4KV4 thesis (GPTQ/AWQ/MR-GPTQ W4 + near-FP8 A4 + OmniStack KV4).
