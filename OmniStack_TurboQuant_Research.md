# Beyond NVFP4: Near-Lossless 4-bit Activation Quantization via TurboQuant Residual Correction

**Deep Vivek Sheth**
Founding AI Engineer, DotNex | deepsheth3@gmail.com | github.com/deepsheth3

---

## Abstract

NVIDIA's NVFP4 format achieves 4-bit weight quantization on Blackwell GPUs with near-FP8 accuracy, but full FP4 activation quantization remains undeployed in production — TRT-LLM currently requires FP8 activations alongside NVFP4 KV cache precisely because FP4 activation accuracy is insufficient for production use. We propose a two-part system that closes this gap. For KV cache, we extend our prior OmniStack-RS work — WHT rotation followed by INT4 Lloyd-Max quantization and a 1-bit QJL residual correction — achieving 3.37× compression over BF16 with 0.0024 maximum reconstruction error validated on Criteo Day 23. For linear layer activations, we propose applying Google's TurboQuant framework — PolarQuant transformation followed by QJL residual correction — directly to runtime activations in registers, enabling near information-theoretic optimal 3-bit activations without calibration data. Together these form a fully sub-FP8 inference stack on Blackwell that addresses NVIDIA's documented accuracy gap in FP4 activation quantization, particularly for long-context and reasoning workloads where quantization error compounds.

---

## 1. Motivation

### 1.1 The Production Gap in NVFP4

As of June 2026, TRT-LLM ships NVFP4 (E2M1, MX4 two-level scaling) for weights and KV cache on Blackwell GPUs. However, a critical constraint exists in the official documentation:

> "Currently TRT-LLM only supports FP8 weight/activation quantization when NVFP4 KV cache is enabled. Therefore `--quant fp8` is required."

This means the full FP4 inference stack — FP4 weights, FP4 activations, FP4 KV cache — is not production-ready. NVIDIA's own documentation flags known accuracy failures:

- Complex reasoning tasks where quantization errors compound through chains
- Long-context tasks where small per-token errors accumulate over 100K+ tokens
- Very large models where per-layer errors stack across 80+ layers

FP4 activation quantization without residual correction is not accurate enough for these workloads. This is the gap we target.

### 1.2 Why Activation Accuracy Matters More Than Weight Accuracy

Weights are fixed. A calibrated offline quantization scheme like AWQ can optimize scale factors once and validate accuracy before deployment. Activations are dynamic — they vary per input, per token, and cannot be pre-calibrated to the same degree. The information-theoretic challenge is harder: you need an online, data-oblivious quantizer that works on any input distribution without preprocessing.

### 1.3 The TurboQuant Opportunity

Google's TurboQuant (ICLR 2026) proves that combining PolarQuant transformation with QJL residual correction achieves near-optimal distortion rate for KV cache vectors — within a constant factor of 2.7 from the information-theoretic lower bound. It is explicitly data-oblivious and online. Our insight is that these properties make it ideal for linear layer activations, which must be quantized online in registers without calibration data.

---

## 2. Background

### 2.1 OmniStack-RS: KV Cache Quantization

Our prior work, OmniStack-RS, implements a three-stage KV cache codec:

**Stage 1 — WHT Rotation:** Apply the Walsh-Hadamard Transform to KV activations before quantization. Because WHT is an orthogonal matrix (H × H^T = I), the rotation cancels in the attention dot product: (HQ)(HK)^T = Q × K^T. The rotation is applied once at KV cache write time with zero inference overhead. The WHT guarantees a roughly Gaussian output distribution regardless of the input distribution — a consequence of the Johnson-Lindenstrauss lemma in high dimensions.

**Stage 2 — INT4 Lloyd-Max Quantization:** With WHT guaranteeing Gaussian distribution, we fit a 16-centroid Lloyd-Max codebook per KV head (8 heads × 16 centroids × 4 bytes = 512 bytes total). Lloyd-Max iteratively finds optimal centroid positions for the actual data distribution, concentrating resolution where data is dense. The codebook is computed once offline from calibration data and stored permanently.

**Stage 3 — 1-bit QJL Residual:** After INT4 quantization, a residual remains. We project this residual onto 64 random ±1 directions (the Johnson-Lindenstrauss transform), storing only the sign of each projection — 64 bits per KV element. At reconstruction, the random matrix G is regenerated from a stored seed (not the matrix itself) and multiplied by the stored signs to recover an approximation of the residual.

**Validated Results on A10 GPU:**
- Compression: 3.37× over BF16 (4.75 bits/element)
- Maximum reconstruction error: 0.0024 (FP32 parity)
- Throughput: 104,000 users/sec at 0.69ms P99
- Dataset: Criteo Day 23, 5M rows, 745K unique users
- Benchmark: MLPerf LoadGen SingleStream scenario

### 2.2 TurboQuant: Near-Optimal Online Vector Quantization

TurboQuant (Ashkboos et al., ICLR 2026) treats KV cache quantization as a rate-distortion problem. It combines two prior contributions from the same research line:

**PolarQuant:** Convert vectors to polar coordinates, separating magnitude from direction. The direction (unit sphere surface) is quantized separately from the magnitude. This representation is more amenable to low-bit quantization because the magnitude normalization removes the outlier problem that plagues standard quantization.

**QJL:** Identical in principle to our OmniStack QJL — random ±1 projections of the residual, sign bits stored, reconstruction via the same random matrix. This independently validates our approach from a different research direction.

**TurboQuant results:**
- 3-bit compression with zero accuracy loss on LongBench, ZeroSCROLLS, RULER, L-Eval
- Matches full-precision performance up to 104,000 tokens under 4× compression (Needle-In-A-Haystack)
- Near information-theoretic optimal: within factor 2.7 of lower bound
- Data-oblivious: no calibration required
- Online: works per-vector at inference time

### 2.3 NVFP4: The Hardware Target

NVFP4 uses E2M1 format (1 sign, 2 exponent, 1 mantissa bit) giving 16 representable values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. MX4 microscaling adds:
- One float8 local scale per 16 elements (12.5% overhead)
- One float32 global scale per tensor
- Hardware-native consumption by Blackwell tensor cores at ~18 PFLOPS on B200

NVFP4 currently ships for weights only in production TRT-LLM. KV cache NVFP4 requires ModelOpt calibration and mandates FP8 activations.

---

## 3. Proposed System

### 3.1 KV Cache: OmniStack (Existing, Validated)

We retain OmniStack as-is for KV cache compression. The WHT rotation cancels in attention, making it uniquely suited for this application. The validated 0.0024 maximum error provides a strong accuracy baseline that neither NVFP4 nor TurboQuant alone has demonstrated on a real production benchmark (Criteo Day 23).

On Blackwell (B200, 8 GPUs):
- OmniStack KV per request: 671MB × (4.75/16) = 199MB
- Concurrent requests: 1,496GB / 199MB = 7,520
- vs pure FP16 KV: 1,496GB / 671MB = 2,230 concurrent requests
- OmniStack improvement: 3.4× more concurrent users

### 3.2 Linear Layer Activations: TurboQuant Online Quantization (Novel)

For linear layer activations Y = X × W, WHT is not viable — rotating X without also rotating W produces incorrect output, and rotating W requires an expensive un-rotation of Y at every layer (160 WHT operations per forward pass for 80 layers).

We propose applying TurboQuant's PolarQuant + QJL framework directly to activations in registers:

> **Implementation update (2026-06-10):** the implemented codec is **NVFP4(MX4) + per-block QJL**;
> Stage 1 (PolarQuant) is disabled by default because it is empirically redundant with MX4
> microscaling, and Stage 3 (QJL) is applied per 128-element sub-block rather than over the full
> hidden vector. See §4.2 / §4.3 findings and `README.md`. The compute/overhead estimates below
> assume the full pipeline; PolarQuant's L2-norm step is removed in the implementation.

**Stage 1 — PolarQuant Transform (in registers):**
For each activation vector X at layer input:
1. Compute magnitude: ||X|| (one reduction)
2. Normalize to unit sphere: X_unit = X / ||X||
3. Store magnitude separately (float32 or float8)
4. Quantize X_unit to FP4/INT4 (unit sphere has bounded range, fewer outliers)

The key insight: separating magnitude from direction removes the outlier problem. The unit sphere surface has bounded values by definition. Low-bit quantization of direction vectors is much more accurate than quantizing the raw activation.

**Stage 2 — NVFP4 Quantization of Direction:**
Quantize the normalized X_unit using NVFP4 (E2M1 + MX4 local scale). Because X_unit is bounded, NVFP4's limited dynamic range (max value 6) is sufficient. The MX4 local scale handles the remaining group-level variation.

**Stage 3 — QJL Residual Correction (in registers):**
After NVFP4 quantization of the direction, a residual remains:
residual = X_unit_true - X_unit_quantized

Apply 64 random ±1 projections:
b = sign(G × residual), where G is 64 × hidden_dim random ±1 matrix

Store 64 sign bits. At matmul time, recover residual estimate:
residual_hat = G^T × b_signed × scale

Correct the activation before feeding to tensor cores:
X_corrected = (X_unit_quantized + residual_hat) × magnitude

Feed X_corrected in FP4 to NVFP4 tensor cores.

**Compute overhead:**
- PolarQuant: one L2 norm + division per activation vector, O(hidden_dim) = 8,192 ops
- QJL projection: 64 × 8,192 = 524,288 ops per layer
- 80 layers: 41M operations per token
- B200 CUDA cores: ~50 TFLOPS
- Time: 41M / 50T = 0.0008ms per token
- Decode time per token: 0.619ms
- QJL overhead: 0.13% — negligible

**Memory overhead:**
- Activations are computed in registers and never stored to HBM between layers
- QJL correction happens entirely in registers within one forward pass
- Memory saving from activation quantization: ZERO (activations don't come from HBM)
- The benefit is compute: enabling FP4 tensor cores with near-lossless accuracy

### 3.3 Weights: NVFP4 (Standard)

We use standard NVFP4 weight quantization via ModelOpt + TRT-LLM pipeline:
- 70B model: 70B × 4.5 bits / 8 = 39.375GB (vs 140GB FP16)
- Per GPU (8×): 4.92GB
- Decode load time: 4.92GB / 8 TB/s = 0.615ms per token (bottleneck)

---

## 4. Why TurboQuant Works for Activations (Where NVFP4 Alone Fails)

### 4.1 The Outlier Problem

LLM activations have systematic outliers in specific dimensions — values 100× larger than average, appearing consistently across all tokens. NVFP4's MX4 local scale (one float8 per 16 elements) handles group-level variation but not within-group outliers. One extreme value in a group of 16 forces the local scale large, quantizing all other values in the group to near-zero. This is the root cause of NVFP4's documented accuracy failures on reasoning tasks.

### 4.2 PolarQuant Removes the Outlier Problem

By normalizing to unit sphere before quantization, we guarantee all direction components are in [-1, 1]. The outlier — which was large in magnitude — is now captured entirely in the separate magnitude scalar, not in the direction vector. The direction vector has uniform magnitude per element, making NVFP4 quantization significantly more accurate.

> **Implementation finding (2026-06-10): PolarQuant is redundant with MX4 and was dropped.**
> This rationale does not survive contact with the implementation. NVFP4's MX4 microscaling already
> divides every 16-element block by its own max, i.e. it already performs per-group normalization.
> Per-token L2 normalization is just a global per-vector constant, which the block scale absorbs
> *exactly*: with fp32 scales, `polar+NVFP4` and `raw NVFP4` are bit-identical (`max|diff| ≈ 1.5e-5`).
> With real fp8-e4m3 block scales, normalizing pushes scales into the fp8 subnormal range and
> *increases* NMSE (4.46e-3 → 6.53e-3 on synthetic outlier activations). PolarQuant is therefore off
> by default in the codec (retained only as an ablation flag). Note also that per-token normalization
> cannot address *per-channel* outliers (specific dims large across all tokens — the documented LLM
> pattern); only a rotation (WHT) or channel scaling (SmoothQuant) would, and §3.2 rules out WHT for
> linear-layer activations.

### 4.3 QJL Provides Near-Optimal Residual Recovery

Whatever error remains after PolarQuant + NVFP4 is captured by QJL. The theoretical guarantee from TurboQuant: this combination is within a constant factor of the information-theoretic optimum for online, data-oblivious vector quantization. No offline calibration approach can do asymptotically better for online activation quantization.

> **Implementation finding (2026-06-10): QJL must be applied per sub-block, not over the full hidden
> vector.** QJL's MSE reduction is `qjl_dim·2/(π·dim)`. OmniStack achieves ~31.8% because it applies
> QJL per *128-dim head* (qjl_dim=64, ratio 0.5). Applied to a full 4096-dim activation vector,
> qjl_dim=64 gives ratio 0.02 → only ~1% reduction (empirically 4.46e-3 → 4.42e-3, negligible). The
> codec therefore applies QJL per 128-element block (qjl_dim=64, ratio 0.5), recovering the ~22%
> reduction at 4.5 bits/element. **Caveat for §6.1:** on a gpt2 smoke test this NMSE reduction did
> *not* translate to lower end-to-end perplexity (raw NVFP4 added +6.5 PPL vs fp16; per-block QJL did
> not recover it, despite reducing activation NMSE in 49/49 layers). Whether it helps a real 8B model
> is the open question Experiment B on the H100 is designed to answer.

---

## 5. Theoretical Performance on 8× B200

### 5.1 Memory

| Component | Size |
|-----------|------|
| NVFP4 weights (70B) | 39.4GB |
| Total HBM (8×B200) | 1,536GB |
| Free for KV cache | 1,496.6GB |
| OmniStack KV per request (2048 tokens) | 199MB |
| Concurrent requests | 7,520 |

### 5.2 Latency

| Operation | Time |
|-----------|------|
| Weight loading per token (NVFP4, 8×B200) | 0.615ms |
| KV cache loading per token (OmniStack) | 0.003ms |
| TurboQuant activation correction | 0.0008ms |
| Total decode per token | 0.619ms |
| Prefill (1024 tokens, compute-bound) | ~1ms |

### 5.3 Throughput

| System | Concurrent | Decode/token | Throughput |
|--------|-----------|-------------|------------|
| H100 FP16 baseline | 745 | 5.22ms | 139 RPS |
| H100 W4A8 (current SOTA) | 1,696 | 1.30ms | 1,304 RPS |
| B200 NVFP4 weights only | 2,230 | 0.615ms | 3,626 RPS |
| **B200 proposed system** | **7,520** | **0.619ms** | **11,842 RPS** |

Note: 85× improvement over H100 FP16 baseline; 9× over current H100 SOTA.

### 5.4 Caveats

- B200 bandwidth assumed at theoretical peak (real: ~80%, ~6.4 TB/s effective)
- TurboQuant accuracy on activations unproven — validated only for KV cache
- NVFP4 FP4 kernels still being optimized in CUTLASS/vLLM
- NVLink all-reduce overhead not modeled

Applying 80% efficiency: ~9,474 RPS practical estimate.

---

## 6. Key Research Questions

### 6.1 Primary Validation

Does PolarQuant + NVFP4 + QJL match FP8 activation accuracy on:
- Standard benchmarks (MMLU, HellaSwag, WinoGrande)
- Reasoning benchmarks (GSM8K, MATH, HumanEval)
- Long-context benchmarks (RULER, Needle-In-A-Haystack at 100K tokens)
- OmniStack's own benchmark (Criteo Day 23 recommendation accuracy)

### 6.2 Optimal QJL Budget for Activations

OmniStack uses 64 projections for KV cache. Activations may require more or fewer:
- 32 projections: lower overhead, lower accuracy
- 64 projections: OmniStack default
- 128 projections: higher overhead, higher accuracy

Experiment: sweep projection count, measure accuracy vs overhead tradeoff per benchmark.

### 6.3 Per-Layer Budget Allocation

Not all layers need the same QJL budget. Early layers have mild outliers; later layers have severe ones. Adaptive budget allocation:
- Run profiling pass: measure per-layer activation quantization error without correction
- Allocate more QJL projections to high-error layers
- Reduce projections for low-error layers
- Total compute budget stays constant, accuracy improves

### 6.4 Interaction with Speculative Decoding

EAGLE-3 draft model sees target model hidden states. If target model activations are TurboQuant-quantized, does this degrade draft acceptance rate? Hypothesis: no significant degradation since PolarQuant + QJL preserves inner products near-optimally (TurboQuant's core property).

---

## 7. Comparison to Related Work

| Method | Target | Bits | Calibration | Hardware | Accuracy Guarantee |
|--------|--------|------|-------------|----------|-------------------|
| SmoothQuant | Weight + Activation | INT8 | Required | Any | Empirical |
| AWQ | Weights | INT4 | Required | Any | Empirical |
| GPTQ | Weights | INT4 | Required | Any | Empirical |
| NVFP4 (current) | Weights + KV | 4.5 bits | Required | Blackwell only | <1% loss claimed |
| OmniStack | KV cache | 4.75 bits | Required | Any | 0.0024 max error, validated |
| TurboQuant | KV cache | 3 bits | None | Any | Near-optimal theoretical |
| **This work** | **KV + Activations** | **3-4.75 bits** | **None (activations)** | **Blackwell** | **Near-optimal (activations) + validated (KV)** |

---

## 8. Implementation Plan

### Phase 1 — Baseline (2 weeks)
- Set up B200 benchmark environment with TRT-LLM v1.x
- Establish FP16, FP8, NVFP4-weights baselines on LLaMA-3 70B
- Confirm OmniStack KV cache integration with TRT-LLM's KvCacheConfig API

### Phase 2 — TurboQuant Activation Prototype (4 weeks)
- Implement PolarQuant + QJL as a CUDA kernel operating in-register
- Integrate as a pre-matmul hook in TRT-LLM's IPluginV3 interface
- Validate on small model (LLaMA-3 8B) before scaling to 70B

### Phase 3 — Accuracy Validation (3 weeks)
- Run full benchmark suite: reasoning, long-context, standard
- Compare vs FP8 activations (current TRT-LLM default)
- Identify failure modes and per-layer error distribution

### Phase 4 — Optimization (3 weeks)
- Implement adaptive per-layer QJL budget allocation
- Fuse PolarQuant + QJL correction into attention kernel epilogue
- Profile with Nsight Compute: verify QJL overhead stays below 1%

### Phase 5 — Full System Integration (2 weeks)
- Combine OmniStack KV + TurboQuant activations + NVFP4 weights
- End-to-end benchmarking on 8× B200
- Ablation study: contribution of each component

---

## 9. Why This Matters to TRT Team

NVIDIA's current production constraint — FP8 activations required alongside NVFP4 KV cache — exists because FP4 activation accuracy is insufficient. This research proposes the missing component: a near-optimal online residual correction that makes FP4 activations viable for production.

The OmniStack approach was independently validated as producing the same core technique (QJL) as Google's TurboQuant, predating the TurboQuant publication by several months. This suggests the direction is correct. The extension to linear layer activations is the natural next step — data-oblivious, online, near-optimal, and directly targeting the gap NVIDIA documents in their own TRT-LLM codebase.

For Daisy's Edge-LLM team specifically: Jetson Orin has no NVFP4 tensor cores. OmniStack's hardware-agnostic INT4 + QJL approach works on any GPU, providing the same KV cache compression benefits for edge deployment that NVFP4 provides only on Blackwell.

---

## References

1. OmniStack-RS: github.com/deepsheth3/Omnistack-RS
2. TurboQuant: Ashkboos et al., "TurboQuant: Online vector quantization with near-optimal distortion rate," ICLR 2026
3. QJL: "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with No Retraining," June 2024
4. PolarQuant: "PolarQuant: Optimal Gaussian Weight Quantization via Hadamard Rotation for LLM Compression," 2025
5. SmoothQuant: Xiao et al., 2022
6. AWQ: Lin et al., 2023
7. GPTQ: Frantar et al., ICLR 2023
8. TRT-LLM Quantization Docs: nvidia.github.io/TensorRT-LLM/features/quantization.html
9. NVFP4 Production Status: build.nvidia.com/spark/nvfp4-quantization
