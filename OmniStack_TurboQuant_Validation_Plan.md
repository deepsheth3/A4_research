# Validation Plan: OmniStack KV + TurboQuant Activation Quantization

**Deep Vivek Sheth**
deepsheth3@gmail.com | github.com/deepsheth3

---

## Overview

This document defines the complete validation methodology for the proposed system combining OmniStack KV cache compression with TurboQuant activation quantization on Blackwell B200 GPUs. The goal is to prove that FP4 activations with TurboQuant residual correction match FP8 activation accuracy — closing the gap NVIDIA documents in TRT-LLM production.

**Core claim to validate:**
> PolarQuant + NVFP4 + QJL residual correction enables FP4 activations with accuracy indistinguishable from FP8 across standard, reasoning, and long-context benchmarks.

---

## Hardware and Software Configuration

```
Hardware:
  GPUs:           8× NVIDIA B200 SXM (192GB HBM each, 8 TB/s)
  Interconnect:   NVLink 4.0 (900 GB/s bidirectional)
  Total HBM:      1,536GB
  
Software:
  TRT-LLM:        v1.x (latest, NVFP4 support from v0.17+)
  CUDA:           12.9+
  ModelOpt:       latest (for NVFP4 weight quantization)
  MLPerf LoadGen: v4.0 (for throughput benchmarking)
  
Model:
  Primary:   LLaMA-3 70B Instruct
  Secondary: LLaMA-3 8B Instruct (faster iteration during development)
  
Precision:
  Weights:     NVFP4 (E2M1, MX4 scaling via ModelOpt)
  Activations: TurboQuant (PolarQuant + NVFP4 + 64-projection QJL)
  KV Cache:    OmniStack (WHT + INT4 Lloyd-Max + 1-bit QJL)
```

---

## Baseline Establishment (Week 1 — Before Testing Our System)

Run all baselines on identical hardware before any experiments. These are ground truth. If baselines don't match published numbers, stop and debug.

### Baseline 1 — FP16 (Ground Truth)

```
Config:   FP16 weights, FP16 activations, FP16 KV cache
Purpose:  Ground truth for all accuracy comparisons
Expected results (LLaMA-3 70B):
  MMLU:           ~79.0%
  GSM8K:          ~90.0%
  HumanEval:      ~62.0%
  WikiText-2 PPL: ~2.8
  Tokens/sec:     ~192 (decode, batch=1, 8×B200)
```

### Baseline 2 — FP8 (Current TRT-LLM Production)

```
Config:   FP8 weights, FP8 activations, FP8 KV cache
Purpose:  The accuracy target our system must match
Expected results:
  MMLU:           ~78.8% (within 0.2% of FP16)
  GSM8K:          ~89.2% (within 1% of FP16)
  HumanEval:      ~61.5% (within 1% of FP16)
  WikiText-2 PPL: ~2.82
  Tokens/sec:     ~384 (2× FP16)
```

### Baseline 3 — NVFP4 Weights + FP8 Activations (Current NVIDIA Best)

```
Config:   NVFP4 weights, FP8 activations, NVFP4 KV cache
Purpose:  Current production ceiling — what we improve upon
Expected results:
  MMLU:           ~78.5%
  GSM8K:          ~87.9% (known degradation vs FP8)
  HumanEval:      ~61.0%
  WikiText-2 PPL: ~2.85
  Tokens/sec:     ~769 (4× FP16)
```

### Baseline 4 — NVFP4 Weights + FP4 Activations (No Correction)

```
Config:   NVFP4 weights, raw NVFP4 activations (no TurboQuant), NVFP4 KV
Purpose:  Quantify the gap TurboQuant must close
Expected results:
  MMLU:           ~77-78% (degraded)
  GSM8K:          ~80-85% (significantly degraded — NVIDIA's known failure)
  HumanEval:      ~58-60% (degraded)
  WikiText-2 PPL: ~3.0+ (noticeably worse)
  
This baseline proves the problem exists
The gap between Baseline 4 and Baseline 2 is what TurboQuant must recover
```

---

## Ablation Study (Weeks 2-4 — One Component at a Time)

Never test the full system before testing each component independently. If the full system fails, you need to know which component caused it.

### Experiment A — OmniStack KV Only

```
Config:   FP16 weights + FP16 activations + OmniStack KV cache
Purpose:  Confirm KV compression alone preserves accuracy
          Replicate your existing Criteo validation in this new context
          
Benchmarks:
  Standard:      MMLU, GSM8K, HumanEval
  Long-context:  Needle-In-A-Haystack (8K, 32K, 64K, 128K tokens)
  Custom:        Criteo Day 23 recommendation accuracy
  
Pass criteria:
  Standard benchmarks: within 0.3% of FP16 baseline
  Needle-128K:         within 1% recall of FP16
  Criteo:              max reconstruction error ≤ 0.0024 (your validated result)
  
If this fails: OmniStack integration is broken. Fix before proceeding.
If this passes: OmniStack confirmed, proceed to Experiment B.
```

> **Implementation note (2026-06-10):** the codec under test is the *corrected* design —
> NVFP4(MX4) + per-block QJL, with PolarQuant disabled (redundant with MX4) and QJL applied per
> 128-element sub-block (full-vector QJL@64 corrects only ~1%). See the research doc §4.2/§4.3
> findings. A local CPU/MPS harness (`turboquant/validation/`) already covers WikiText-2 and the
> ablation; this experiment is the H100 confirmation on a real 8B model. Early gpt2 smoke: per-block
> QJL lowers activation NMSE in every layer but did not lower perplexity — treat a null result as a
> valid outcome (see the plan's closing note on publishable negative results).

### Experiment B — TurboQuant Activations Only (Key Novel Experiment)

```
Config:   FP16 weights + TurboQuant FP4 activations + FP16 KV cache
Purpose:  Confirm TurboQuant activation correction works for linear layers
          This is the unpublished contribution — validate it first on 8B model
          
Sub-experiment B1 (LLaMA-3 8B, faster iteration):
  Run on 8B before 70B to validate approach cheaply
  If 8B fails, debug before scaling to 70B
  
Sub-experiment B2 (LLaMA-3 70B):
  Full scale validation
  
Benchmarks:
  Fast sanity:   WikiText-2 perplexity (run first, 30 minutes)
  Standard:      MMLU, GSM8K, HumanEval
  Reasoning:     MATH (Hendrycks), ARC-Challenge
  
Pass criteria:
  WikiText-2 PPL:  within 0.05 of FP8 baseline (≤2.87)
  MMLU:            within 0.5% of FP8 baseline
  GSM8K:           within 1.0% of FP8 baseline ← critical
  HumanEval:       within 1.0% of FP8 baseline
  
If WikiText-2 fails: TurboQuant activation correction is broken
  Debug: check PolarQuant implementation, QJL projection count
  
If WikiText-2 passes but GSM8K fails:
  Correction works for simple tasks but not reasoning
  Need more QJL projections for later layers
  Proceed to per-layer budget analysis
  
If all pass: TurboQuant activation correction validated
  Proceed to Experiment C.
```

### Experiment C — NVFP4 Weights Only

```
Config:   NVFP4 weights + FP16 activations + FP16 KV cache
Purpose:  Replicate NVIDIA's published NVFP4 weight results
          Confirm our environment matches their claims
          
Benchmarks:
  MMLU, GSM8K, HumanEval, WikiText-2
  
Pass criteria:
  Must match NVIDIA's published numbers within 0.5%
  If not: environment misconfigured, fix before proceeding
```

### Experiment D — TurboQuant Activations + NVFP4 Weights

```
Config:   NVFP4 weights + TurboQuant FP4 activations + FP16 KV cache
Purpose:  Test interaction between weight and activation quantization
          Error from weight quantization may compound with activation error
          
Benchmarks:
  Full suite: MMLU, GSM8K, HumanEval, MATH, WikiText-2
  
Pass criteria:
  MMLU:      within 0.5% of FP8 baseline
  GSM8K:     within 1.5% of FP8 baseline (slight relaxation for combined)
  WikiText-2: within 0.08 of FP8 baseline
  
If interaction causes degradation beyond threshold:
  Increase QJL projections from 64 to 128
  Apply adaptive per-layer allocation (see Section 6)
  Rerun until pass criteria met
```

### Experiment E — Full System (Run Last)

```
Config:   NVFP4 weights + TurboQuant FP4 activations + OmniStack KV
Purpose:  Full system validation
          Only run after A, B, C, D all pass independently
          
Benchmarks: Complete suite (see Section 4)
Pass criteria: See Section 5
```

---

## Complete Benchmark Suite

### Tier 1 — Standard Accuracy (Must Pass)

| Benchmark | Metric | FP16 Expected | Pass Threshold |
|-----------|--------|---------------|----------------|
| MMLU | 5-shot accuracy | ~79.0% | ≥78.0% |
| HellaSwag | 0-shot accuracy | ~85.0% | ≥84.0% |
| WinoGrande | 0-shot accuracy | ~81.0% | ≥80.0% |
| ARC-Challenge | 0-shot accuracy | ~67.0% | ≥66.0% |
| WikiText-2 | Perplexity ↓ | ~2.80 | ≤2.90 |

### Tier 2 — Reasoning (Critical — NVIDIA's Known Failure Mode)

| Benchmark | Metric | FP16 Expected | FP8 Expected | Pass Threshold |
|-----------|--------|---------------|--------------|----------------|
| GSM8K | 8-shot accuracy | ~90.0% | ~89.2% | ≥88.5% |
| MATH | 4-shot accuracy | ~41.0% | ~40.5% | ≥39.5% |
| HumanEval | pass@1 | ~62.0% | ~61.5% | ≥60.5% |
| MBPP | pass@1 | ~65.0% | ~64.5% | ≥63.5% |

**Why GSM8K is the most important single number:**

NVIDIA documents reasoning tasks as NVFP4's failure mode. If your system passes GSM8K within 1% of FP8 while using FP4 activations, the core claim is validated. This is the headline result.

### Tier 3 — Long Context (Tests Error Accumulation Over Time)

| Benchmark | Context | Metric | Pass Threshold |
|-----------|---------|--------|----------------|
| Needle-In-A-Haystack | 8K tokens | Recall % | ≥99.0% |
| Needle-In-A-Haystack | 32K tokens | Recall % | ≥98.5% |
| Needle-In-A-Haystack | 64K tokens | Recall % | ≥97.5% |
| Needle-In-A-Haystack | 128K tokens | Recall % | ≥97.0% |
| RULER | 128K tokens | Composite score | Within 2% of FP16 |
| LongBench | Various | Composite score | Within 2% of FP16 |
| L-Eval | 32K tokens | Task accuracy | Within 2% of FP16 |

**Direct comparison point:** TurboQuant published that their system matches full-precision performance at 104K tokens under 4× compression on Needle-In-A-Haystack. Your system must match or exceed this.

### Tier 4 — Custom Benchmark (OmniStack Validation)

| Benchmark | Metric | Existing Result | Pass Threshold |
|-----------|--------|-----------------|----------------|
| Criteo Day 23 | Max reconstruction error | 0.0024 | ≤0.0024 |
| Criteo Day 23 | Throughput | 104K users/sec | ≥100K users/sec |
| Criteo Day 23 | P99 latency | 0.69ms | ≤0.75ms |

This is your unique benchmark. It validates the combined system on a real production dataset that no other paper has results on.

---

## Pass/Fail Criteria — Full System

The system passes validation if ALL of the following hold:

```
Accuracy conditions (all must pass):
  ✓ MMLU within 1.0% of FP8 baseline
  ✓ GSM8K within 1.0% of FP8 baseline        ← primary claim
  ✓ HumanEval within 1.0% of FP8 baseline
  ✓ Needle-128K within 2% of FP16 baseline   ← long-context claim
  ✓ WikiText-2 PPL within 0.1 of FP8 baseline
  ✓ Criteo max error ≤ 0.0024                ← OmniStack claim

Performance conditions (all must pass):
  ✓ Decode time per token ≤ 0.70ms (within 13% of theoretical 0.619ms)
  ✓ TurboQuant overhead ≤ 1% of total decode time
  ✓ Concurrent requests ≥ 7,000 (within 7% of theoretical 7,520)
  ✓ P99 TTFT ≤ 5ms
```

---

## Error Analysis (Intermediate Diagnostics)

Beyond final benchmark accuracy, measure these intermediate signals to understand where errors come from and guide debugging.

### Per-Layer Reconstruction Error

```
After each of the 80 transformer layers:
  Compare FP16 activation vs TurboQuant-quantized activation
  Metric: cosine similarity and L2 error
  
  Expected pattern:
    Layers 1-20:   low error (simple representations)
    Layers 40-60:  moderate error (complex semantics)
    Layers 70-80:  higher error (output distribution)
    
  If error spikes at specific layers:
    Those layers need more QJL projections
    Trigger adaptive budget allocation
    
  Plot: layer index vs reconstruction error
  Save as: layer_error_profile.png
```

### Attention Score Distortion

```
Metric: KL divergence between FP16 attention weights
        and quantized attention weights
        
  Per head, per layer, per sequence position
  
  TurboQuant's theoretical guarantee:
    PolarQuant + QJL minimizes dot product distortion
    Attention scores should be preserved near-optimally
    
  If attention distortion is high at specific heads:
    Those heads may need higher KV precision
    Consider mixed precision: some heads OmniStack, some FP8
    
  Pass threshold: mean KL divergence < 0.01 across all heads
```

### Perplexity vs Sequence Length

```
Measure perplexity at increasing context lengths:
  1K, 4K, 8K, 16K, 32K, 64K, 128K tokens
  
  If perplexity degrades sharply at longer contexts:
    KV compression error is accumulating
    Reduce OmniStack compression or increase QJL projections
    
  Expected: flat perplexity curve (error not accumulating)
  Failure mode: perplexity increases with sequence length
```

### QJL Projection Count Sweep

```
Sweep projection count: 16, 32, 64, 128, 256
Measure accuracy on GSM8K and WikiText-2 at each count

  Expected:
    16 projections:  accuracy gap vs FP8 is large
    64 projections:  gap mostly closed (our default)
    128 projections: marginal improvement, 2× overhead
    256 projections: diminishing returns
    
  Goal: find minimum projection count that passes all criteria
  Ideal: 32 projections (2× less overhead than our default)
  
  Plot: projection count vs GSM8K accuracy and overhead
```

---

## Performance Benchmarking Protocol

### Latency Measurement

```
Tool: Nsight Systems (nsys)
Command:
  nsys profile \
    --output=turboquant_profile \
    --trace=cuda,nvtx \
    --stats=true \
    python inference_benchmark.py

What to measure:
  Total decode time per token (wall clock, GPU synchronized)
  Weight loading time (should dominate at 0.615ms)
  TurboQuant kernel time (should be < 0.001ms)
  KV cache load time (should be < 0.005ms)
  All-reduce communication time (NVLink, 8 GPUs)
  
Acceptance criteria:
  Weight loading: 0.55-0.65ms (within 10% of theoretical)
  TurboQuant:     < 0.001ms (< 0.2% of total)
  Total decode:   < 0.70ms
```

### Throughput Measurement

```
Tool: MLPerf LoadGen v4.0
Scenario: SingleStream (P99 latency)
Scenario: Server (throughput at QOS target)

SingleStream config:
  Target QOS: P99 latency < 2ms TTFT
  Min queries: 1,000
  Report: P99 latency in ms
  
Server config:
  Target QOS: P99 latency < 5ms TTFT
  Load range: 1,000 to 15,000 RPS
  Report: maximum RPS at target QOS
  
Concurrent request test:
  Gradually increase concurrent requests from 100 to 8,000
  Measure HBM utilization at each step
  Confirm HBM fills at theoretical limit (7,520 requests)
  
Expected results:
  P99 TTFT:           ~2ms
  Max throughput:     ~9,000-12,000 RPS (80% of theoretical 11,842)
  Concurrent at full: ~7,200-7,500 requests
```

### Nsight Compute Kernel Analysis

```
Tool: ncu (Nsight Compute)
Target: TurboQuant activation kernel

Command:
  ncu --kernel-name turboquant_activation_kernel \
      --metrics \
        sm__throughput.avg.pct_of_peak_sustained_elapsed,\
        l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
        sm__warps_active.avg.pct_of_peak_sustained_active \
      python inference_benchmark.py

What to verify:
  Kernel is CUDA core bound (not tensor core, not memory bound)
  Memory throughput low (kernel operates in registers)
  Compute throughput moderate (64 projections per activation)
  Warp occupancy reasonable
  
If kernel shows high memory traffic:
  Activation data is being spilled to SMEM or HBM
  Refactor to keep everything in registers
  This is a correctness issue — fix before claiming overhead is negligible
```

---

## Adaptive Per-Layer QJL Budget

If the uniform 64-projection budget fails accuracy on certain layers, implement adaptive allocation:

### Profiling Pass

```
Step 1: Run inference on 100 calibration samples
Step 2: For each of 80 layers, measure:
  activation_error[layer] = L2(FP16_activation, NVFP4_activation)
  
Step 3: Sort layers by error magnitude
  High error layers: layers 60-80 typically
  Low error layers:  layers 1-20 typically
```

### Budget Allocation

```
Total QJL budget: 80 layers × 64 projections = 5,120 projections

Adaptive allocation:
  Top 20 high-error layers:  128 projections each = 2,560
  Middle 40 medium-error:     64 projections each = 2,560
  Bottom 20 low-error:         0 projections each = 0
  Total: 5,120 (same budget, better allocation)
  
Store per-layer projection count as config file
Load at engine build time
Different projection counts = different kernel launches per layer
```

---

## Reproducibility Requirements

All results must be reproducible. Document and fix the following:

```
Random seeds:
  QJL random matrix G generated from fixed seed per (layer, head)
  Same seed = identical G matrix = identical results
  Store seed configuration in model checkpoint
  
Environment:
  Docker container with pinned TRT-LLM version
  requirements.txt with all package versions locked
  CUDA version, driver version, hardware model documented
  
Scripts:
  One-command reproduction: bash run_full_validation.sh
  Individual benchmark scripts for each tier
  Baseline scripts separate from system scripts
  
Reporting variance:
  Run each benchmark 3 times with different input orderings
  Report mean ± std deviation
  If std deviation > 0.3%: investigate non-determinism
```

---

## Expected Results Table (Target)

This is what the paper's main results table should look like if the system works:

| Method | MMLU | GSM8K | HumanEval | Needle-128K | PPL | Tokens/sec | Memory |
|--------|------|-------|-----------|-------------|-----|-----------|--------|
| FP16 baseline | 79.0 | 90.0 | 62.0 | 99.2% | 2.80 | 192 | 1× |
| FP8 (TRT-LLM default) | 78.8 | 89.2 | 61.5 | 98.8% | 2.82 | 384 | 2× |
| NVFP4 w + FP8 act | 78.5 | 87.9 | 61.0 | 97.2% | 2.85 | 769 | 4× |
| NVFP4 w + FP4 act (no correction) | 77.8 | 83.1 | 58.5 | 93.0% | 3.10 | 1,828 | 4× |
| **Ours (full system)** | **78.6** | **89.0** | **61.4** | **98.5%** | **2.83** | **~9,500** | **~5×** |

Key claims demonstrated:
1. Our system matches FP8 accuracy on GSM8K (89.0 vs 89.2) while NVFP4 without correction fails (83.1)
2. Our system matches TurboQuant on long context (98.5% vs published 97%+)
3. Our system achieves ~9,500 tokens/sec (49× FP16) with near-FP8 accuracy

---

## Timeline

| Week | Activity |
|------|----------|
| 1 | Environment setup, baseline establishment |
| 2 | Experiment A (OmniStack KV only) |
| 3-4 | Experiment B (TurboQuant activations, LLaMA-3 8B first) |
| 5 | Experiment B scale to LLaMA-3 70B |
| 6 | Experiment C (NVFP4 weights) + Experiment D (combined) |
| 7 | Adaptive per-layer QJL budget if needed |
| 8 | Experiment E (full system) |
| 9 | Performance benchmarking (Nsight, MLPerf) |
| 10 | Error analysis, variance measurement |
| 11 | Ablation table completion |
| 12 | Write-up, code release, reproducibility verification |

---

## Failure Modes and Debug Plan

| Symptom | Likely Cause | Debug Step |
|---------|-------------|------------|
| WikiText-2 PPL >> FP8 | PolarQuant broken | Check normalization, unit sphere bounds |
| GSM8K degrades, MMLU ok | Later layer errors accumulate | Increase QJL projections for layers 60-80 |
| Needle fails at 128K | KV compression loses information | Reduce OmniStack compression ratio |
| Throughput < 50% theoretical | TurboQuant kernel spills to SMEM | Refactor kernel to stay in registers |
| Results non-reproducible | QJL seed not fixed | Pin per-layer seeds in checkpoint |
| Baseline ≠ published numbers | Environment misconfigured | Check CUDA, TRT-LLM, ModelOpt versions |

---

## What a Successful Validation Proves

If all experiments pass:

1. **FP4 activations are production-viable** with TurboQuant correction — closing NVIDIA's documented gap
2. **OmniStack KV compression is robust** — maintained accuracy when combined with FP4 activations
3. **TurboQuant generalizes** from KV cache (published) to linear layer activations (novel)
4. **The combined system achieves** near-FP8 accuracy at near-FP4 speed — first fully sub-FP8 inference stack

If Experiment B fails but others pass:

Still publishable as: "Why FP4 activation quantization remains hard and what the theoretical limits are — an empirical study using near-optimal correction methods."

Negative results with theoretical grounding are publishable at MLSys and ICLR.
