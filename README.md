# TurboQuant: 4-bit Activation Quantization (NVFP4 + per-block QJL)

Implementation of the activation-quantization half of
[OmniStack_TurboQuant_Research.md](OmniStack_TurboQuant_Research.md). The KV-cache
half (OmniStack) is already built and validated in the sibling repo
[`../Omnistack_RS`](../Omnistack_RS); here we **reuse** its QJL primitive and
build the novel activation codec on top.

## What this is (and isn't)

A portable, pure-PyTorch codec + accuracy harness that runs on a Mac (CPU/MPS)
for development and on a **single rented H100** for the real-model run. It
validates the research's **accuracy** claim. It does **not** implement the
B200/TRT-LLM throughput stack — the H100 is Hopper and has no FP4 tensor cores,
so NVFP4 is a **numerical fake-quant simulation** (round to the E2M1 grid, matmul
in fp16). Throughput, MLPerf, 70B, and CUDA kernels are out of scope (need
Blackwell).

## Key findings (the design deviates from the paper — on purpose)

Two stages of the paper's pipeline were tested and found not to work as written:

1. **PolarQuant is redundant with MX4 microscaling — dropped.** NVFP4's per-16
   block scale already normalizes each group; per-token L2 normalization is a
   global constant the block scale absorbs exactly (`max|polar − raw| ≈ 1e-5`
   with fp32 scales). With real fp8 scales it slightly *hurts*. So PolarQuant is
   off by default (kept as an ablation flag).

2. **QJL must be applied per sub-block, not over the full hidden vector.** QJL's
   MSE reduction is `qjl_dim·2/(π·block)`. OmniStack gets ~31.8% by applying it
   per 128-dim head; on a full 4096-dim activation vector, `qjl_dim=64` corrects
   only ~1%. We apply QJL per 128-element block (`qjl_dim=64`, ratio 0.5) →
   ~22% activation-NMSE reduction, at 4.5 bits/element.

So the implemented codec is **NVFP4(MX4) + per-block QJL**, fully data-oblivious
(no calibration, no codebook — see the Lloyd-Max note in the research doc).

3. **QuaRot-style Hadamard rotation is also harmful under MX4 — rejected.**
   Rotation (`rotation.py`, `use_hadamard` flag, kept for ablation) helps
   *coarse-scaled* quantizers (13× NMSE win at per-token scales — QuaRot's
   regime) but NVFP4's per-16 block scales already contain outliers better than
   rotation's Gaussian floor; under MX4 it raises NMSE 3× and gpt2 PPL 44.7→94.2.
   Caveat: the research doc's §3.2 *cost* argument against rotation was wrong
   (the rotation folds into weights offline for free, as QuaRot deploys); the
   conclusion was right for a different reason — MX4 supersedes it. Pattern:
   **outlier remedies designed for coarse-scaled quantization (PolarQuant,
   rotation) are redundant-to-harmful under fine-grained microscaling.**

### End-to-end perplexity so far (local, WikiText-2)

| Model | baseline | fp8 (target) | nvfp4_raw | turboquant | QJL recovers |
|---|---|---|---|---|---|
| gpt2 (Conv1D, 3k tok) | 38.15 | — | 44.68 | 45.41 | **−11%** (hurt) |
| TinyLlama-1.1B (Llama nn.Linear, 6k tok) | 10.196 | 10.260 | 10.792 | 10.573 | **+37%** |

Two honest takeaways:
1. **Architecture matters.** On gpt2 (Conv1D) the correction slightly hurt; on a
   real Llama-arch model (the H100 8B target's architecture) per-block QJL clearly
   helps — it recovers ~37% of the raw-NVFP4 gap.
2. **But it does NOT yet reach the FP8 production target.** FP8 is near-lossless
   (+0.064 PPL); turboquant is +0.377, still +0.313 short of FP8 — above the plan's
   ≤0.1 PPL bar on this 1.1B model. So on small models the method improves raw
   NVFP4 but is not yet FP8-parity.

**Open question for the H100 run:** does the gap to FP8 close on a real 8B model
(larger models, true fp16, where NVFP4 activation error bites harder and the
correction may matter more)? A null result is still a legitimate, publishable
outcome ("why FP4 activation quantization remains hard"). NVFP4/FP8 here are
fake-quant simulations and gpt2/TinyLlama ran in fp32 — not the final word.

## Layout

```
turboquant/
  nvfp4.py        NVFP4 E2M1 grid + MX4 block fake-quant
  polarquant.py   magnitude/direction split (ablation only)
  act_codec.py    TurboQuantActQuantizer: (PolarQuant) + NVFP4 + per-block QJL
  config.py       TurboQuantConfig (defaults = corrected design)
  _omnistack.py   imports RademacherQJL from ../Omnistack_RS (reuse, not reimpl)
  tests/          pytest unit tests (CPU)
  validation/
    error_analysis.py   ablation + QJL sweep + outlier stress -> results/
    hf_perplexity.py    Experiment B: WikiText-2 PPL, 3 modes
  scripts/run_gpu_session.sh   one-shot H100 run
```

## Run

```bash
pip install -r requirements.txt
# OmniStack-RS must be a sibling repo (or set OMNISTACK_PATH).

# 1. Unit tests (Mac CPU, free)
pytest turboquant/tests -q

# 2. Numerical error analysis (Mac CPU, free) -> results/
python -m turboquant.validation.error_analysis

# 3. End-to-end smoke (Mac MPS/CPU, free)
python -m turboquant.validation.hf_perplexity --model gpt2 --limit 3000

# 4. Real run — ONE H100 (the only step that costs money)
bash turboquant/scripts/run_gpu_session.sh meta-llama/Llama-3.1-8B
```

## Out of scope (need Blackwell + NVIDIA stack)

TRT-LLM / ModelOpt / CUDA `IPluginV3` kernels, in-register fusion, real FP4
tensor-core speed, B200 throughput / concurrency, MLPerf LoadGen, Nsight, 70B.
Re-validating the OmniStack KV codec (already done in `../Omnistack_RS`).
