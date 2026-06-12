# TurboQuant: NVFP4 4-bit Quantization for LLMs (W4A4 + KV4)

A portable, pure-PyTorch quantization codec + accuracy harness for **NVFP4**
(E2M1 + MX4 per-16 fp8 microscaling). It started as the activation half of
[OmniStack_TurboQuant_Research.md](OmniStack_TurboQuant_Research.md) and now
covers the full **W4A4** path (4-bit weights *and* activations). The KV-cache
half (OmniStack) is validated separately in [`../Omnistack_RS`](../Omnistack_RS);
its QJL primitive is reused here.

## What this is (and isn't)

Runs on a Mac (CPU/MPS) for development and on a **single rented H100** for the
real-model runs. It validates the **accuracy** claim. NVFP4 is a **numerical
fake-quant simulation** (round to the E2M1 grid, matmul in fp16) — the H100 is
Hopper and has no FP4 tensor cores, so **throughput numbers are projected, not
measured.** Real-kernel speed, B200, TRT-LLM, MLPerf, and 70B are out of scope
(need Blackwell).

**Design constraint (the one rule):** every correction must be **Pareto-clean** —
folded into weights offline or applied in a fused epilogue, never regressing
latency, memory, or the pure-FP4 main GEMM path versus FP8. No technique is kept
unless it improves an axis without degrading another.

---

## Headline results (Llama-3.1-8B, full WikiText-2)

Reference points: **FP16 = 5.918**, **FP8 (per-token E4M3, the production bar) = 5.948**.

### A4 — FP4 activations, FP16 weights (the activation codec)

| Mode | PPL | vs FP8 | gap to FP8 closed |
|---|---|---|---|
| raw NVFP4 | 6.263 | +0.315 | 0% |
| + channel equalization + zero-point | 6.168 | +0.220 | 30% |
| + W-aware SVD side-channel | 6.065 | +0.117 | 63% |
| **+ QJL residual (`nvfp4_eqzp_svd_qjl`)** | **6.050** | **+0.102** | **68%** |

**A4 reaches near-FP8 parity (+0.10 PPL) — the hard half of the problem.**

### W4A4 — FP4 weights *and* FP4 activations

| Weight method (on top of the A4 stack) | PPL | vs FP8 | weight cliff recovered |
|---|---|---|---|
| naive nearest rounding | 6.629 | +0.681 | 0% |
| GPTQ (Hessian error feedback) | 6.552 | +0.604 | 13% |
| AWQ + GPTQ | 6.581 | +0.633 | *worse* |
| GPTQ + additive low-rank (in/16) | 6.376 | +0.428 | 44% |
| **GPTQ + additive low-rank (in/8, fp8 factors)** | **6.294** | **+0.346** | **58%** |

**Best deployable W4A4 = 6.294**, at **0.75 byte/elem — under FP8's 1 byte** on
every axis (FP4 weight 0.5B + fp8 low-rank factors 0.25B). fp8 factors are
lossless vs fp16 (both 6.294).

### GSM8K (8-shot CoT, 30-question probe)

| Mode | accuracy |
|---|---|
| FP16 | 50.0% (15/30) |
| A4 (`nvfp4_eqzp_svd_qjl`) | 46.7% (14/30) |

A 1-question difference — statistically noise at n=30. Rules out catastrophic
reasoning collapse; **does not** establish parity. Full GSM8K (1319 q) is pending.

---

## Key findings (the mechanistic story)

The central, consistent result across the whole project:

> **Under NVFP4's per-16 microscaling, coarse-scale "scaling / redistribution"
> methods are redundant-to-harmful; only *additive* side-channels help.**

Demonstrated repeatedly:

1. **PolarQuant** (per-token L2 norm) — redundant; the per-16 block scale absorbs
   it (`max|polar − raw| ≈ 1e-5` fp32; slightly *hurts* with fp8 scales).
2. **QuaRot global Hadamard rotation** — *harmful* under MX4 (NMSE ×3, gpt2 PPL
   44.7→94.2). The block scale already beats rotation's Gaussian floor.
3. **GPTQ** (Hessian error feedback on weights) — recovers only **13%** of the
   weight cliff. The per-16 microscaling already does most of what GPTQ does at
   coarse scale.
4. **AWQ** (salient-channel scaling) — *worse* than GPTQ (6.581 vs 6.552).
5. **Per-layer rank allocation** (water-filling) — *worse* than uniform (6.363 vs
   6.294): per-layer reconstruction error ≠ end-to-end importance, and starving
   layers to rank 0 costs more than it saves.

**What breaks the wall: additive low-rank correction** (LQER-style). Instead of
reshuffling error, it *adds information back* — an activation-Hessian-weighted
rank-r approximation of the residual `W − Q(W)`, riding a fused rank-r side
matmul. This is the weight-analog of the activation SVD side-channel that also
worked, and it is **immune to the microscaling redundancy** because it is
additive, not multiplicative. It recovers 44→58% of the weight cliff with rank.

QJL must also be applied **per sub-block** (per-128, `qjl_dim=64`, ~22% NMSE
reduction), not over the full hidden vector (~1%, useless).

---

## Negative results (tested, ruled out — these are findings)

| Tried | Outcome |
|---|---|
| PolarQuant, global rotation | redundant / harmful under microscaling |
| GPTQ, AWQ (weight scaling) | redundant — microscaling absorbs them |
| naive zero-point | hurts E2M1 (dense-near-zero grid); best-of-{0, mid} instead |
| W-aware rounding (no permutation) | 18.5% synthetic → ~3% real (no within-block channel correlation) |
| fp4 (E2M1) low-rank factors | worse than fp8 at equal bytes (E2M1 too coarse for factors) |
| joint W+A (quantized-activation Hessian) | no signal — interaction negligible (Q(x)≈x) |
| per-layer rank allocation | worse than uniform |

---

## Honest scope & caveats

- **Fake-quant simulation.** Accuracy is real; throughput (~1.5–1.8× FP8
  projected) is *not measured* — needs B200.
- **Light calibration, not calibration-free.** Uses a few wikitext-train windows
  for equalization scales (A4) and GPTQ Hessians (W4). Frame as *light-calibration
  PTQ*.
- **Single model / dataset for the strong numbers.** Llama-3.1-8B, WikiText-2 PPL.
  Cross-model breadth and the full downstream suite are pending.
- **W4A4 is not FP8-parity** (best +0.35). Within pure / light-calibration /
  hardware-native / Pareto-clean 4-bit, this is near the practical floor; reaching
  parity requires breaking a premise (learned rotations, fine-tuning, or
  lattice/trellis coding — each spends generality, purity, or the FP4 grid).
- **W4A4KV4 composition** (all three together — the "4-bit everything" headline)
  is **not yet run**.

---

## Layout

```
turboquant/
  nvfp4.py        E2M1 grid + MX4 block fake-quant; optclip; zero-point; W-aware
  gptq.py         GPTQ + AWQ + additive low-rank weight correction (NVFP4)
  act_codec.py    TurboQuantActQuantizer: NVFP4 + SVD side-channel + per-block QJL
  polarquant.py   magnitude/direction split (ablation only)
  rotation.py     Hadamard rotation (ablation only)
  config.py       TurboQuantConfig
  _omnistack.py   imports RademacherQJL from ../Omnistack_RS (reuse)
  tests/          pytest unit tests (CPU)
  validation/
    hf_perplexity.py   WikiText-2 PPL; A4 + W4A4 modes; weight-quant + Hessian
    gsm8k_eval.py      GSM8K 8-shot CoT, stop-at-####, per-mode checkpointing
    error_analysis.py  ablation + QJL sweep + outlier stress
  scripts/        run_gpu_session.sh, run_gsm8k_session.sh
```

## Run

```bash
pip install -r requirements.txt   # OmniStack-RS sibling repo, or set OMNISTACK_PATH

# Unit tests (Mac CPU, free) — 33 tests
pytest turboquant/tests -q

# A4 perplexity (one H100)
python -m turboquant.validation.hf_perplexity --model unsloth/Meta-Llama-3.1-8B \
  --modes fp16 fp8 nvfp4_raw nvfp4_eqzp_svd_qjl

# Best W4A4 (FP4 weights + activations, GPTQ + additive low-rank, fp8 factors)
python -m turboquant.validation.hf_perplexity --model unsloth/Meta-Llama-3.1-8B \
  --modes nvfp4_eqzp_svd_qjl --w4-gptq --w4-lowrank --w4-rank-div 8 --w4-lowrank-fp8

# GSM8K
python -m turboquant.validation.gsm8k_eval --model unsloth/Meta-Llama-3.1-8B --limit 150
```

Key flags: `--w4-gptq` (Hessian weight quant), `--awq`, `--w4-lowrank`
(additive correction), `--w4-rank-div N` (rank = in/N), `--w4-lowrank-fp8`
(Pareto-clean factor storage), `--w4-rank-alloc` (water-fill, *worse* — kept for
the ablation). See [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md) for next steps.

## What's needed for publication

Breadth (more models + full lm-eval task suite), head-to-head baselines (QuaRot /
SpinQuant / Atom / NVFP4-specific), the W4A4KV4 composition run, and real B200
throughput — none of which require beating 6.294. The contribution is the
**mechanism** (microscaling redundancy + additive correction), not the absolute
number. See [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md).

## Out of scope (need Blackwell + NVIDIA stack)

TRT-LLM / ModelOpt / CUDA kernels, in-register fusion, real FP4 tensor-core speed,
B200 throughput / concurrency, MLPerf, 70B. Re-validating the OmniStack KV codec
(done in `../Omnistack_RS`).
