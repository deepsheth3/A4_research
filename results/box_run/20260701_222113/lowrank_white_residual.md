# Negative result: the NVFP4 weight residual is white — low-rank correction is dead

> **Correction, 2026-11-25. The measurements below stand; the Finding and
> Consequence do not.** This test took a plain (Frobenius) SVD of `W − Q(W)` and
> scored weight-space error. On 2026-08-05 the same residual, at the same rank,
> fitted in the activation-Hessian metric closed 64.2% of the RTN→FP16
> perplexity gap on this same model with FP8 activations, and 64.6% on
> Llama-3.1-8B (`results/rotation_headtohead_*_fp8.json`). So post-hoc weight
> correction is not dead and QAT is not the only weight lever. The sentence
> below that the low-rank lever is "inert on the microscaled NVFP4 residual" was
> also wrong when written: the 58% result it cites was itself measured on NVFP4
> weights. Whether the residual's output-weighted energy is concentrated (the
> likely explanation) has not been measured directly.

Measured on real Llama (TinyLlama) weights, RTX PRO 6000 (sm_120), 2026-07-01.
Additive low-rank fp8 residual correction on top of the NVFP4 base weight:

| layer (out,in) | NVFP4 relerr | +rank64 | +rank128 | recovered @128 |
|---|---|---|---|---|
| q_proj (2048,2048)   | 0.0989 | 0.0915 | 0.0856 | 13% (+1.0 b/param) |
| gate_proj (5632,2048)| 0.1026 | 0.0987 | 0.0950 |  7% (+0.68 b/param) |
| down_proj (2048,5632)| 0.1059 | 0.1018 | 0.0980 |  7% (+0.68 b/param) |

**Finding.** NVFP4's per-16 microscaling already whitens the quantization residual:
a rank-128/2048 (6%) fp8 subspace recovers only 7-13% of the error — the signature of
near-white noise. So the two-lever theorem's additive-low-rank lever, which recovers ~58%
of the *naive* W4 weight cliff, is **inert on the microscaled NVFP4 residual**.

**Consequence.** No post-hoc weight correction can fix NVFP4 weights. The only lever that
moves the number is training the weights to pre-absorb the rounding (QAT). This is why
QAT closing 70% of the gap is not incremental — it is the *only* admissible weight lever.
Repro: `lowrank_recover.py`.
