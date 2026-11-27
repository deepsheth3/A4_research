# NVFP4 4-bit quantization research

Experiments on running LLMs in **NVFP4**, the 4-bit floating-point format that
NVIDIA Blackwell GPUs execute natively (4-bit E2M1 values, one FP8 scale per 16
values, one FP32 scale per tensor). The repo holds a pure-PyTorch NVFP4
simulator, post-training correction methods, a distillation (QAT) pipeline that
trains inside NVIDIA ModelOpt's own quantizer, Triton kernels for the extra
inference ops, and the logs and JSON results from every paid GPU run.

This README is the current, corrected summary.
[`NVFP4_MASTER_REPORT.md`](NVFP4_MASTER_REPORT.md) is the full research log,
written as the work happened. Where the two disagree, this file is right.

## Status (2026-11-26)

- **All experiments so far ran between 2026-06-11 and 2026-08-06** on rented
  GPUs (H100, RTX PRO 6000 Blackwell, B200). Models: Llama-3.1-8B,
  TinyLlama-1.1B, OLMoE-1B-7B.
- **A literature check on 2026-11-25 found that most of the method is prior
  art** ([list](#prior-art)) and that several headline numbers in earlier
  versions of this README were confounded
  ([list](#earlier-claims-that-do-not-hold-as-stated)).
- **Current experiment, not yet run:** all-4-bit distillation with no side
  channels, comparing three ways to treat activations during training
  ([details](#current-experiment)). The code is in
  [`modelopt_qat.py`](modelopt_qat.py) and has only been syntax-checked.
- **There is no submittable paper.** [`paper/nvfp4_draft.tex`](paper/nvfp4_draft.tex)
  predates the W4A8, fusion and MoE work and the literature check.
- **No result here has a seed sweep.** Every number is a single run, and
  calibration, training and evaluation all use WikiText-2 (train split for
  calibration and training, test split for evaluation).

## What is in the repo

| Work | When | Model, hardware | Code | Results |
|---|---|---|---|---|
| Post-training activation and weight correction | June | Llama-3.1-8B, H100 | [`turboquant/`](turboquant/), [`hf_perplexity.py`](turboquant/validation/hf_perplexity.py) | [`results_w4_dl/`](results_w4_dl/), `results/perplexity_*.json` |
| Distillation (QAT), simulated and ModelOpt-native | late June to July 1 | TinyLlama and 8B, RTX PRO 6000 | [`qat_nvfp4.py`](turboquant/validation/qat_nvfp4.py), [`modelopt_qat.py`](modelopt_qat.py) | [`results/box_run/20260701_222113/`](results/box_run/20260701_222113/) |
| Throughput | July | RTX PRO 6000, B200 | [`gemm_bench.py`](gemm_bench.py), [`vllm_fp4_bench.py`](turboquant/validation/vllm_fp4_bench.py) | master report §12 |
| 4-bit weights with FP8 activations (W4A8), head-to-head | Aug 5–6 | TinyLlama and 8B, RTX PRO 6000 | [`rotation_headtohead.py`](rotation_headtohead.py), [`gptq.py`](turboquant/gptq.py) | `results/rotation_headtohead_*.json` |
| Fused inference kernels | Aug 5–6 | RTX PRO 6000 | [`triton_fusion_suite.py`](triton_fusion_suite.py), [`fused_e2e.py`](fused_e2e.py), [`production_path_bench.py`](production_path_bench.py) | [`results/box_threads/`](results/box_threads/), `results/production_path_bench.json` |
| Mixture-of-experts | Aug 5–6 | OLMoE-1B-7B, RTX PRO 6000 | [`moe_expert_hessians.py`](turboquant/validation/moe_expert_hessians.py), [`moe_phase_b.py`](moe_phase_b.py), [`moe_e2e.py`](moe_e2e.py) | `results/moe_*.json`, [`results/box_session2/`](results/box_session2/) |
| All-4-bit distillation, three training arms | November | TinyLlama | [`modelopt_qat.py`](modelopt_qat.py) | none yet |

## Measured results

Perplexity (PPL) is on WikiText-2 test; lower is better. "Gap closed" is the
fraction of the distance from the untrained 4-bit model back to FP16.

### Post-training correction, 4-bit activations (Llama-3.1-8B, full test set)

FP16 = 5.918. FP8 = 5.948.

| Config | PPL | vs FP8 |
|---|---|---|
| 4-bit activations, FP16 weights, uncorrected | 6.263 | +0.315 |
| 4-bit activations with equalization, zero-point, SVD side channel and QJL | 6.050 | +0.102 |
| 4-bit weights and activations, plain rounding | 6.629 | +0.681 |
| 4-bit weights and activations, GPTQ plus low-rank correction (rank in/8, FP8 factors) | 6.294 | +0.346 |

A trained correction of the activation residual recovered only +0.032 PPL at
2048 context, so this leg has little left to gain after training-free methods.

### Distillation in a simulated quantizer (TinyLlama)

FP16 = 9.358. First 40k test tokens.

| Config | PPL | Gap closed |
|---|---|---|
| 4-bit weights and activations, no training | 10.734 | 0% |
| KL distillation, 3000 steps | 9.768 | 70% |

The simulator and ModelOpt disagree. The simulator-trained checkpoint scored
9.94 when exported through ModelOpt, not 9.77. The 70% figure is against the
simulator's own baseline; see the [caveat](#earlier-claims-that-do-not-hold-as-stated).

### Distillation inside ModelOpt's quantizer (TinyLlama, RTX PRO 6000)

Training and evaluation use the same quantizers that export ships, so the
reported number is the deployed one. FP16 = 9.358. First 40k test tokens.

| Config | Untrained PPL | Trained PPL | Gap closed |
|---|---|---|---|
| 4-bit weights and activations, 3000 steps | 10.185 | 9.709 | 57.6% |
| Same, plus 4-bit KV cache, 3000 steps | 10.590 | 9.705 | 71.8% |
| Same, plus KV cache and AWQ equalization, 4000 steps | 10.321 | 9.764 (best checkpoint 9.707) | 57.8% |

All three runs end near 9.71. That is a plateau for this training budget
(about 3 million tokens of WikiText-2), not a demonstrated limit. Published
recipes train on 300 million to 6 billion tokens.

### LoRA distillation at 8B (Llama-3.1-8B, simulated quantizer, 512 context)

Full-weight training ran out of memory on a 95 GB card. LoRA adapters on every
layer, with the base frozen, fit.

| Config | PPL | Gap closed |
|---|---|---|
| FP16 | 7.945 | — |
| 4-bit weights and activations, no training | 9.543 | 0% |
| LoRA rank 32, 1500 steps | 8.810 | 45.9% |

### 4-bit weights with FP8 activations, no training (August, RTX PRO 6000)

Every method uses the same Hessians, calibration and evaluation
([`rotation_headtohead.py`](rotation_headtohead.py)).

| Weight method | Llama-3.1-8B PPL | Gap closed | TinyLlama PPL | Gap closed |
|---|---|---|---|---|
| FP16 | 7.174 | — | 8.715 | — |
| Plain rounding | 8.035 | 0% | 9.292 | 0% |
| GPTQ | 7.744 | 33.8% | 9.028 | 45.7% |
| Rotation + GPTQ | 7.666 | 42.9% | 9.025 | 46.3% |
| Plain rounding + low-rank correction fitted in the Hessian metric (rank in/16, FP8 factors) | 7.479 | 64.6% | 8.921 | 64.2% |

Caveats: the baselines are our own implementations, not reference code. The
low-rank factors add about 1 bit per weight, so this is not an equal-byte
comparison (roughly 5.5 vs 4.5 bits). Evaluation is the first 100k (8B) or
200k (TinyLlama) test tokens, so FP16 differs from the tables above.
Equalization is not applied in this code path.

On the mixture-of-experts model (OLMoE-1B-7B, 64 experts) the same low-rank
method scored 8.758 against 8.805 for rotation + GPTQ. The least-used expert
received 345 calibration tokens for a 2048-dimensional Hessian. Two attempts to
fix that starvation (load-adaptive damping, shrinking toward the layer-mean
Hessian) did not help.

### Inference cost of the low-rank correction (RTX PRO 6000)

- With our own Triton kernels installed as model hooks, the TinyLlama W4A8
  forward pass dropped from 21.9 ms to 13.7 ms (1.60× faster than our unfused
  hooks). It is still 28% slower than the FP16 forward.
- Fused into the GEMM epilogue, the correction costs 1.06× against our own
  Triton GEMM at large shapes. That kernel runs at 0.54× to 1.13× the speed
  of cuBLAS, losing at large batch.
- Next to the production FP4 GEMM (`cutlass_scaled_fp4_mm`), running the
  correction as a separate kernel costs 1.76× on average
  (`results/production_path_bench.json`). No fused production kernel exists
  here.

### Throughput

- FP4 GEMM measured 3.7× faster than BF16 at 70B-layer shapes and slower than
  BF16 at TinyLlama-sized shapes.
- On a B200 serving Llama-3.1-8B through vLLM, NVFP4 ties FP8 at small batch
  and only pulls ahead past roughly batch 64.

## Earlier claims that do not hold as stated

These appeared in earlier versions of this README, the master report or the
slide deck. The logs were re-read on 2026-11-25.

| Claim | Problem |
|---|---|
| "QAT closes 70% of the gap" | Measured against the simulator's baseline (10.734). Inside ModelOpt the same recipe closes 57.6% of a smaller gap. |
| "The simulator and ModelOpt differ by 0.55 PPL" | The simulator quantizes all 155 linear layers including `lm_head`; ModelOpt leaves `lm_head` unquantized. The two baselines are not the same model. |
| "The 4-bit floor of ~9.71 is fundamental" | Three runs at one small training budget. Nothing shows more data would not lower it. |
| "The KV-cache leg is free under QAT" | 9.709 vs 9.705 from one run each, with no seeds. The honest reading is "no measurable cost in one run". |
| "The weight residual is white, so low-rank correction is dead" | True for a plain SVD of the residual (7–13% recovered at rank 128), wrong as a conclusion. The Hessian-metric fit at the same rank closes 64%. |
| "Naive 4-bit 6.676 → 6.369" (8B) and "8.003 → 7.563" (TinyLlama) | Both rows used the same corrected weights. The difference is 4-bit vs FP8 activations, not the weight method. |
| "GPTQ contributes nothing once low-rank correction is applied" | Rests on a single 0.01 PPL difference. The clean head-to-head has no GPTQ + low-rank arm. |
| "Rotation is provably redundant" | At 8B rotation improved GPTQ by 0.078 PPL. It still loses to the low-rank method. |
| "Speculative decoding gives 1.85×" (master report §9) | Assumes the 4-bit draft costs 0.3× the target. The B200 measurement puts FP4 vs FP8 at about parity, which gives no speedup. |
| "Low-rank correction costs 1.06× at inference" | Only against our own Triton GEMM. The production-path number is 1.76×. |

## Prior art

Do not present these as new. Each was found in the 2026-11-25 literature check.

- **Low-rank correction fitted in the activation metric:** LQER
  (arXiv 2402.02446), QERA (2410.06040), CALDERA (2405.18886), ASER
  (2411.07762).
- **Low-rank side channel next to NVFP4 with a fused Blackwell kernel:** BiLoCo
  (2610.32243), ResServe (2611.08713).
- **Rotation stops helping at small block sizes:** MR-GPTQ (2509.23202).
- **KL distillation into NVFP4, including a LoRA variant:** NVIDIA QAD
  (2601.20088), QATFactory (2609.39223).
- **Mixture-of-experts calibration:** MoEQuant (2505.03804), EAQuant
  (2506.13329, on OLMoE itself).

[`THEORY.md`](THEORY.md) argues that under per-block scaling only equalization
and additive low-rank correction reduce distortion. Treat it as this repo's
derivation of a result the papers above largely cover, with the 8B rotation
measurement as a counterexample to its strongest form.

## Current experiment

**Question.** QATFactory reports that NVFP4 models deploy better when trained
with weights quantized and activations left in full precision, even though
they are deployed with both quantized. They do not explain why. This
experiment compares three treatments of activations during training. Every arm
is evaluated and exported as the same all-4-bit model.

| `ACT_TRAIN` | Activations during training |
|---|---|
| `a4` | Quantized, straight-through gradient (the July recipe) |
| `a16` | Full precision (QATFactory's recommendation) |
| `noise` | Full precision plus the quantizer's own error with random signs |

Each run writes `results/qad_<tag>.json` with deployed PPL and per-token KL to
the FP16 teacher. A result counts only if it is larger than the spread across
seeds.

**Run it** on a Blackwell box (RTX PRO 6000 or 5090; about an hour, 40 GB disk):

```bash
cd /workspace
git clone -b w4a8-fusion-moe https://github.com/deepsheth3/A4_research.git NVFP4_Research
git clone https://github.com/deepsheth3/Omnistack-RS.git Omnistack_RS   # imported by turboquant
cd NVFP4_Research && source /venv/main/bin/activate
echo "torch==$(python -c "import torch;print(torch.__version__.split('+')[0])")" > /tmp/c.txt
pip install -c /tmp/c.txt "transformers>=4.44" "datasets>=2.20" accelerate sentencepiece "nvidia-modelopt[torch]"

# two-minute check of the new code path before paying for the full run
SEED=0 ACT_TRAIN=noise STEPS=500 python -u modelopt_qat.py

# nine runs: three arms, three seeds
nohup bash -c 'for s in 0 1 2; do for a in a4 a16 noise; do
  SEED=$s ACT_TRAIN=$a python -u modelopt_qat.py > results/qad_${a}_s${s}.log 2>&1
done; done' > /dev/null 2>&1 &
```

The repo must live at `/workspace/NVFP4_Research`; the script writes there.
Copy `results/qad_*` off the box before destroying it.

**Not done yet:** a prior-art check on the noise arm (injecting quantization
noise during training is an old idea), a training set larger than WikiText-2,
and any run on an 8B model with public benchmarks.

## Layout

```
turboquant/
  nvfp4.py                    E2M1 grid and per-16 block fake-quant
  gptq.py                     GPTQ, AWQ, rotation + GPTQ, Hessian-metric low-rank weight correction
  act_codec.py                activation codec: NVFP4 + SVD side channel + QJL
  distill.py                  trained-factor and acceptance-objective distillation primitives
  polarquant.py, rotation.py  ablations
  _omnistack.py               imports the QJL primitive from the sibling Omnistack_RS repo
  tests/                      CPU unit tests, including a ModelOpt parity test
  validation/
    hf_perplexity.py          WikiText-2 perplexity for every post-training mode
    qat_nvfp4.py              distillation in the simulated quantizer (KL, TV blend, LoRA, KV4)
    export_nvfp4.py           export through ModelOpt's NVFP4 quantizer
    moe_expert_hessians.py    per-expert token counts and Hessians for MoE models
    kv4_correction.py         KV-cache low-rank correction
    ood_eval.py, ood_stack_vs_qat.py        out-of-distribution checks
    measure_deploy.py, vllm_fp4_bench.py    serving throughput
    weight_distill.py, activation_distill.py, runtime_lambda_*.py   trained-correction experiments
    (others)                  one-off analyses referenced from the master report

modelopt_qat.py               distillation inside ModelOpt's quantizer; the current experiment
rotation_headtohead.py        W4A8 weight-method head-to-head
lowrank_recover.py            plain-SVD residual measurement
moe_phase_b.py, moe_e2e.py    MoE starvation mitigations and end-to-end comparison
triton_fusion_suite.py, triton_lowrank_fusion_bench.py, fused_e2e.py   fused inference kernels
production_path_bench.py      correction cost next to the production FP4 GEMM
gemm_bench.py, fp4_gemm_bench_flashinfer.py   FP4 vs FP8 vs BF16 GEMM throughput
kv_qjl_test.py                1-bit QJL correction for a 4-bit KV cache; no recorded result
setup_box*.sh, run_the_box*.sh, run_throughput_bench.sh   box setup and one-shot run scripts
BOX_RUNBOOK.md                sequence for the July one-shot box run

THEORY.md                     the two-lever derivation
NVFP4_MASTER_REPORT.md        full research log
QAT_FINDINGS.md, DOOR_C.md, RESEARCH_ROADMAP.md   dated notes from individual phases
paper/                        stale draft
results/, results_w4_dl/, results_crossmodel_dl/   logs and JSON from every run
```

## Other run commands

`turboquant` imports the QJL primitive from
[Omnistack-RS](https://github.com/deepsheth3/Omnistack-RS). Clone it next to
this repo or set `OMNISTACK_PATH`.

```bash
pip install -r requirements.txt

# unit tests (CPU)
pytest turboquant/tests -q

# post-training perplexity, 4-bit activations and weights (one GPU)
python -m turboquant.validation.hf_perplexity --model unsloth/Meta-Llama-3.1-8B \
  --modes fp16 fp8 nvfp4_raw nvfp4_eqzp_svd_qjl --w4-gptq --w4-lowrank --w4-rank-div 8 --w4-lowrank-fp8

# W4A8 weight-method head-to-head
python rotation_headtohead.py --help

# distillation in the simulated quantizer
python -m turboquant.validation.qat_nvfp4 --objective kl --steps 3000 --lr 3e-5 --n-train 1500

# 8B LoRA distillation
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m turboquant.validation.qat_nvfp4 \
    --model unsloth/Meta-Llama-3.1-8B --objective kl --lora-rank 32 --steps 1500 --lr 1e-4 \
    --max-len 512 --grad-checkpoint

# distillation inside ModelOpt (env vars: MODEL, STEPS, LR, KV4=1, CFG, SEED, ACT_TRAIN, GSM8K=1)
python modelopt_qat.py

# July one-shot box run and B200 throughput sweep
bash setup_box.sh && bash run_the_box.sh
bash setup_box_throughput.sh && bash run_throughput_bench.sh
```
