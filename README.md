# NVFP4: Closing the 4-bit Accuracy Gap (Post-Training Codec + QAT)

A portable, pure-PyTorch quantization codec + accuracy harness for **NVFP4**
(E2M1 + per-16 fp8 microscaling), Blackwell's native 4-bit format — plus a
quantization-aware training (QAT) pipeline that trains directly against
NVIDIA's own ModelOpt quantizer and has been deployed and measured on real
Blackwell hardware (RTX PRO 6000 sm_120). The KV-cache codec building block
(QJL) is reused from [`../Omnistack_RS`](../Omnistack_RS).

Full writeup: [`paper/nvfp4_draft.tex`](paper/nvfp4_draft.tex) /
[`paper/nvfp4_draft.pdf`](paper/nvfp4_draft.pdf) (*"Microscaling Redundancy and
the Post-Training Floor: Quantization-Aware Training Closes the Rest of the
NVFP4 Accuracy Gap"*). [`NVFP4_MASTER_REPORT.md`](NVFP4_MASTER_REPORT.md) is
the complete, unedited research log (every technique, every result, every
failure).

**Design constraint (the one rule):** every correction must be **Pareto-clean** —
folded into weights offline, applied in a fused epilogue, or trained in (QAT),
never regressing latency, memory, or the pure-FP4 main GEMM path versus FP8.

---

## The story in two parts

**Part 1 — post-training codec (PTQ).** A distortion theory of the microscaling
quantizer (the "two-lever theorem", [`THEORY.md`](THEORY.md)) shows that under
per-block scaling, almost every popular correction — rotation, permutation,
per-block Hadamard, output-weighted scale selection, coordinated rounding — is
**redundant**. Only two levers survive: **per-channel equalization** and
**additive low-rank correction**. A codec built from exactly these two reaches
near-FP8 activation accuracy and a Pareto-clean W4A4 weight config. This part
also finds its own ceiling: the post-quantization *weight* residual is nearly
white, so additive low-rank correction of it is a dead end ([finding
#20](NVFP4_MASTER_REPORT.md)) — there is no more juice to extract post-hoc.

**Part 2 — quantization-aware training (QAT).** Training the model directly
against the deployment quantizer (not a proxy simulation) is the lever that
picks up where PTQ hits its floor. QAT closes most of the remaining W4A4 gap,
**reproduces with zero sim-to-deploy gap on real Blackwell hardware**, and
scaling it to "everything 4-bit" (W4A4KV4) turns out to make the KV-cache leg
**free**. Pushing further (adding equalization/AWQ on top of QAT) shows three
independent methods converge to the same wall — a **characterized, fundamental
floor** ~0.32 PPL above FP8, not a bug to keep chasing.

---

## Headline results

### Part 1 — post-training codec (Llama-3.1-8B, full WikiText-2)

Reference: **FP16 = 5.918**, **FP8 (per-token E4M3, the production bar) = 5.948**.

| Config | PPL | vs FP8 | gap closed |
|---|---|---|---|
| A4 — FP4 activations, FP16 weights, raw | 6.263 | +0.315 | 0% |
| **A4 — `eq + zero-point + SVD side-channel + per-block QJL`** | **6.050** | **+0.102** | **68%** |
| W4A4 — FP4 weights + activations, naive rounding | 6.629 | +0.681 | 0% |
| **W4A4 — GPTQ + additive low-rank (in/8, fp8 factors)** | **6.294** | **+0.346** | **58%** |

Best deployable W4A4 = 6.294 at **0.75 byte/elem — under FP8's 1 byte** on
every axis. Central finding: under NVFP4's per-16 microscaling, *coarse-scale
redistribution methods are redundant-to-harmful; only additive side-channels
help.* See [full negative-results table below](#negative-results-ptq-part-1).

### Part 2 — QAT (TinyLlama-1.1B, full WikiText-2)

Reference (fake-quant sim): **FP16 = 9.358**, FP8 (sim) = 9.388.

| Config | PPL | gap closed | spec-decode acceptance |
|---|---|---|---|
| PTQ W4A4 (no training) | 10.734 | — | 0.825 |
| **Heavy KL-QAT (3000 steps)** | **9.768** | **70%** | **0.862** |

### Part 2 — real Blackwell deployment (native ModelOpt QAT, `train == deploy`, TinyLlama, RTX PRO 6000)

The proxy fake-quant QAT above deployed at 9.94, not 9.77 — a real sim-to-deploy
gap. Training *inside* ModelOpt's own quantizer ([`modelopt_qat.py`](modelopt_qat.py))
closes it:

| Config (deployed, real ModelOpt NVFP4 export) | PPL | Notes |
|---|---|---|
| PTQ (deploy-grid) | 10.32 | |
| **Native QAT** | **9.709** | sim-to-deploy gap eliminated |
| **Native QAT + KV4** (W4A4KV4, everything 4-bit) | **9.705** | the KV-cache leg is *free* |
| Native QAT + AWQ equalization | 9.707 | AWQ adds nothing — two-lever theorem confirming itself |

**The Pareto-clean 4-bit floor is ~9.71 PPL, +0.32 over FP8 (9.388), and it is
fundamental.** Three independent, principled methods converge to the same
wall. The gap to FP8 can't be trained away: FP8 is already ~lossless, so
matching it means *lossless* 4-bit, and QAT can only make *weights* robust —
it can't remove the runtime activation/KV quantization noise that remains.
Crossing it needs mixed precision or W4A16 — both spend an axis the Pareto
constraint holds fixed.

### 8B (Llama-3.1-8B) — LoRA-QAT is the path that fits

Full-weight QAT OOMs at 8B (95GB exceeded). **LoRA-QAT** (adapters on every
layer, base frozen, `B` initialized to zero so training starts exactly at the
PTQ point) is the validated fix:

| Config | PPL | gap closed | acceptance |
|---|---|---|---|
| FP16 | 7.945 | — | — |
| PTQ W4A4 | 9.543 | — | 0.800 |
| **LoRA-QAT r32 / 1500 steps** | **8.810** | **45.9%** | 0.832 |

### Real-hardware throughput (measured, not projected)

- **GEMM**: real FP4 tensor-core matmul measured **3.7× faster than BF16 at
  70B-layer shapes**, but *slower* than BF16 at TinyLlama-sized shapes — the
  speed win is a large-GEMM property, not automatic.
- **End-to-end serving (B200)**: NVFP4 only **ties FP8** at single-stream / small
  batch; the throughput edge shows up at **large batch (parity ~batch 64,
  advantage grows past it)**. Conclusion: judge 4-bit methods on **accuracy per
  byte and acceptance per byte**, not single-stream speed. Reproducing this
  (paper Table 6) on a rented B200 is what [`run_throughput_bench.sh`](run_throughput_bench.sh)
  automates.

---

## Key findings

1. **Two-lever theorem** ([`THEORY.md`](THEORY.md)): under per-block
   microscaling, the *only* distortion-reducing levers are per-channel
   equalization and additive low-rank correction. Global/per-block rotation,
   channel permutation, output-weighted scale selection, and coordinated
   rounding are all provably redundant — and each was independently confirmed
   empirically. AWQ-on-top-of-QAT confirming this again at deploy time (§ above)
   is the same law showing up a second time, from a completely different angle.
2. **The post-training weight residual is white.** Once the model has been
   quantized, `W − Q(W)` carries almost no exploitable low-rank structure —
   additive correction recovers only 7–13% at rank 128. This is *why* Part 2
   exists: there's nothing more to extract post-hoc, so the fix has to happen
   during training.
3. **QAT must use the deployment quantizer, not a proxy.** A fake-quant
   simulation and ModelOpt's real NVFP4 quantizer disagree enough to matter
   (9.94 vs 9.709 deployed) — train==deploy is not optional for an honest
   accuracy number.
4. **The KV-cache leg is free under QAT.** Going from W4A4 to W4A4KV4 (weights,
   activations, *and* KV cache all 4-bit) costs 9.709 → 9.705 — i.e. nothing.
   Long-context memory savings essentially for free once QAT is already paying
   for the weight/activation correction.
5. **Acceptance, not standalone PPL, is the metric that decides the spec-decode
   win.** `modelopt_qat.py` was fixed (commit `72512e7`) to measure spec-decode
   acceptance (α = 1 − TV against an FP16 target) alongside PPL, since PPL alone
   doesn't tell you what a QAT'd model is worth as a *draft*.
6. **8B needs LoRA-QAT.** Full-weight QAT OOMs; naive subset (half-layer)
   training is null (frozen half's error is uncorrectable). LoRA with
   zero-initialized `B` starts training exactly at the PTQ point and moves
   monotonically down.

### Negative results (PTQ, Part 1)

| Tried | Outcome |
|---|---|
| PolarQuant, global Hadamard rotation | redundant / harmful under microscaling |
| GPTQ, AWQ (weight scaling) alone | redundant — microscaling absorbs them |
| naive zero-point | hurts E2M1's dense-near-zero grid; best-of-{0, mid} instead |
| per-block (16×16) Hadamard, fixed-mask, channel permutation, output-weighted scale selection | all wash or harmful — redundant with equalization |
| per-layer rank allocation (water-filling) | worse than uniform allocation |
| fp4 (E2M1) low-rank correction factors | worse than fp8 factors at equal bytes |
| **additive low-rank correction of the *post-quant weight residual*** | **dead — residual is already white (7–13% recovery at rank 128)** |

### Negative results (QAT, Part 2)

| Tried | Outcome |
|---|---|
| Full-weight QAT at 8B | OOM (95GB exceeded) |
| Subset (half-layer) QAT at 8B | null — frozen half's quant error is uncorrectable |
| KV4 additive low-rank residual correction | real but not byte-legal (blows the KV memory budget the leg exists to save) — only equalization folded into the projection is deployable |
| AWQ equalization on top of native QAT | no gain (9.707 vs 9.709) — confirms the two-lever theorem at deploy time |
| Chasing the FP8 line with more accuracy-only levers | floor at ~9.71 is fundamental — needs mixed precision or W4A16, which spend memory/compute |

---

## Honest scope & caveats

- **Part 1 (PTQ) accuracy is real; PTQ throughput is projected**, not measured
  (Hopper-class dev iteration has no FP4 tensor cores). **Part 2 (QAT) has now
  been measured on real Blackwell** (RTX PRO 6000 GEMM + accuracy; B200 serving
  throughput).
- **Light calibration, not calibration-free** for Part 1 (a few WikiText-train
  windows for equalization + GPTQ Hessians).
- **W4A4 (either part) is not FP8-parity.** Part 1 best PTQ = +0.346 PPL over
  FP8; Part 2's characterized floor is +0.32 PPL over FP8. Both are argued to
  be near the practical limit within their respective constraints (Pareto-clean,
  accuracy-only for Part 2's floor result).
- **Novelty is incremental, stated plainly**: QAT itself isn't new; QAT on the
  NVFP4 microscaling grid (vs int4) and LoRA-QAT to close a *quantization* gap
  (inverse of QLoRA) are modest engineering novelty. The two-lever theorem and
  the acceptance-objective (TV) QAT generalizing better OOD are the more
  genuinely new pieces. A near-identical NVIDIA Nemotron KL-distill NVFP4 QAT
  approach ("QAD") published independently — check novelty claims against it
  before submission.
- **8B LoRA-QAT runs were time-bounded** (rank 32, 1500 steps, 512 ctx) —
  untried levers (max-len 1024, rank 64) would likely push past 45.9%.

---

## Layout

```
turboquant/
  nvfp4.py                  E2M1 grid + MX4 block fake-quant; optclip; zero-point; W-aware
  gptq.py                   GPTQ + AWQ + additive low-rank weight correction (NVFP4)
  act_codec.py               TurboQuantActQuantizer: NVFP4 + SVD side-channel + per-block QJL
  distill.py                 trained-factor + acceptance-objective distillation primitives
  polarquant.py / rotation.py  ablation-only: magnitude/direction split, Hadamard rotation
  config.py                  TurboQuantConfig
  _omnistack.py               imports RademacherQJL from ../Omnistack_RS (reuse)
  tests/                      pytest unit tests (CPU), incl. ModelOpt-parity test
  validation/
    hf_perplexity.py          WikiText-2 PPL; A4 + W4A4 PTQ modes (Part 1)
    gsm8k_eval.py              GSM8K 8-shot CoT eval
    qat_nvfp4.py               fake-quant QAT harness (KL / TV-blend objectives, LoRA-QAT, KV4)
    export_nvfp4.py            export a QAT'd checkpoint through ModelOpt's real NVFP4 quantizer
    kv4_correction.py          KV-cache additive low-rank correction (ruled out, not byte-legal)
    ood_eval.py / ood_stack_vs_qat.py   OOD generalization + PTQ-stack-vs-QAT head-to-head
    measure_deploy.py / vllm_fp4_bench.py   real serving throughput (vLLM, sm_120/B200)
    weight_distill.py / activation_distill.py / runtime_lambda_*.py  Part 2 correction-basis experiments
    error_analysis.py          ablation + QJL sweep + outlier stress

modelopt_qat.py       native QAT *inside* ModelOpt's own quantizer (train==deploy; the deploy-accuracy source of truth)
gemm_bench.py          real FP4 vs FP8 vs BF16 GEMM throughput, production batches (Blackwell)
run_the_box.sh / setup_box.sh                  one-shot QAT+export+measure pipeline (RTX PRO 6000)
run_throughput_bench.sh / setup_box_throughput.sh   one-shot FP8-vs-NVFP4 decode throughput sweep (B200, Llama-3.1-8B)
BOX_RUNBOOK.md          exact sequence for the one-shot box run, with contingencies
```

## Run

```bash
pip install -r requirements.txt   # OmniStack-RS sibling repo, or set OMNISTACK_PATH

# Unit tests (Mac CPU, free)
pytest turboquant/tests -q

# Part 1 — A4 / W4A4 PTQ perplexity (one H100/GPU)
python -m turboquant.validation.hf_perplexity --model unsloth/Meta-Llama-3.1-8B \
  --modes fp16 fp8 nvfp4_raw nvfp4_eqzp_svd_qjl --w4-gptq --w4-lowrank --w4-rank-div 8 --w4-lowrank-fp8

# Part 2 — QAT (fake-quant sim harness)
python -m turboquant.validation.qat_nvfp4 --objective kl --steps 3000 --lr 3e-5 --n-train 1500

# Part 2 — 8B LoRA-QAT (needs a Blackwell-class GPU)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m turboquant.validation.qat_nvfp4 \
    --model unsloth/Meta-Llama-3.1-8B --objective kl --lora-rank 32 --steps 1500 --lr 1e-4 \
    --max-len 512 --grad-checkpoint

# Part 2 — native QAT inside ModelOpt (train==deploy; the real-deployment number)
python modelopt_qat.py            # add --kv4 for W4A4KV4

# Part 2 — one-shot box run: QAT -> ModelOpt export -> deployed PPL + throughput
bash setup_box.sh && bash run_the_box.sh   # see BOX_RUNBOOK.md

# Part 2 — B200 FP8-vs-NVFP4 decode throughput sweep (paper Table 6)
bash setup_box_throughput.sh && bash run_throughput_bench.sh
```

## What's needed for publication

The bake-off against external baselines (QuaRot / SpinQuant / AWQ / stock
ModelOpt NVFP4, equal-byte) is still unrun. A direct FP8 accuracy row at the
8B/512-ctx setting is cheap and missing. The recently-published NVIDIA
Nemotron "QAD" work is near-identical on the KL-distill QAT mechanism and
needs to be addressed directly in the novelty claims — the theorem and the
acceptance-objective OOD-generalization result are the pieces that remain
clearly ours. See [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md) for the
detailed, dated history of what's been tried.

## Out of scope

MLPerf, 70B, in-register kernel fusion beyond what ModelOpt/vLLM already
provide, and re-validating the OmniStack KV codec (done in `../Omnistack_RS`).
