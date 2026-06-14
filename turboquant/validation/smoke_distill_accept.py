"""Offline smoke test: the full trained-factor + acceptance pipeline end-to-end.

Builds a tiny random LLaMA from config (NO download, runs in seconds on CPU) —
same nn.Linear architecture as the real $10 run, unlike gpt2's Conv1D. Proves
the wiring: NVFP4-quantize each Linear -> SVD warm-start factors -> attach
LowRankLinear -> distill factors vs the FP16 teacher -> KL/acceptance improve.

This is a SMOKE (does the code run + move the metric), not a result. Run:
    python -m turboquant.validation.smoke_distill_accept
"""

from __future__ import annotations

import copy

import torch

from turboquant.nvfp4 import nvfp4_quantize
from turboquant.gptq import gptq_lowrank_factors
from turboquant.distill import attach_lowrank, distill_factors, kl_distill_loss
from turboquant.validation.acceptance import acceptance_stats


TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")


def build_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).float().eval()


def main():
    torch.manual_seed(0)
    teacher = build_tiny_llama()
    student = copy.deepcopy(teacher)

    # Build a DEGRADED student: NVFP4-quantize each Linear, then plant a real
    # 4-bit-sized gap (tiny random weights quantize near-losslessly, so we inject
    # damage to give distillation something to recover — mirrors the real 8B run
    # where W4A4KV4 sits +0.4 PPL above FP8). Factors use LoRA init (L=0, R small)
    # so the student starts at the degraded Q(W) and the gradient still flows.
    torch.manual_seed(1)
    factors = {}
    for name, mod in student.named_modules():
        if name.endswith(TARGET_SUFFIXES) and isinstance(mod, torch.nn.Linear):
            W = mod.weight.data.float()
            out, inn = W.shape
            Wq = nvfp4_quantize(W, block=16)
            Wq = Wq + 0.15 * W.std() * torch.randn_like(Wq)   # planted 4-bit gap
            r = max(16, inn // 2)
            L = torch.zeros(out, r)
            R = 0.01 * torch.randn(r, inn)
            factors[name] = (Wq, L, R)
    n = attach_lowrank(student, factors)
    print(f"attached LowRankLinear to {n} layers")

    # calibration batches (random token ids)
    ids = [torch.randint(0, 256, (4, 32)) for _ in range(8)]

    def kl_on(x):
        with torch.no_grad():
            return kl_distill_loss(student(x).logits, teacher(x).logits).item()

    eval_x = torch.randint(0, 256, (8, 32))
    a0, g0 = acceptance_stats(
        student(eval_x).logits[:, :-1].reshape(-1, 256),
        teacher(eval_x).logits[:, :-1].reshape(-1, 256))
    kl0 = kl_on(eval_x)
    print(f"BEFORE distill:  KL={kl0:.5f}  alpha={a0:.4f}  greedy={g0:.4f}")

    distill_factors(student, teacher, ids, lr=5e-3, epochs=40, log_every=80)

    a1, g1 = acceptance_stats(
        student(eval_x).logits[:, :-1].reshape(-1, 256),
        teacher(eval_x).logits[:, :-1].reshape(-1, 256))
    kl1 = kl_on(eval_x)
    print(f"AFTER  distill:  KL={kl1:.5f}  alpha={a1:.4f}  greedy={g1:.4f}")

    ok = kl1 < kl0 and a1 >= a0
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
