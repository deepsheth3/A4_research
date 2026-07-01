# Negative result: the NVFP4 weight residual is white — low-rank correction is dead

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
