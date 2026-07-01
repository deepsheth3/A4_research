import torch
from transformers import AutoModelForCausalLM
from turboquant.nvfp4 import nvfp4_quantize, round_e4m3

m = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)
sd = dict(m.named_parameters())

def relerr(a, b):
    return (a - b).norm().item() / b.norm().item()

names = ["model.layers.10.self_attn.q_proj", "model.layers.10.mlp.gate_proj", "model.layers.10.mlp.down_proj"]
print("Additive low-rank residual correction on real Llama weights (NVFP4 base + fp8 rank-r side-channel)")
for nm in names:
    W = sd[nm + ".weight"].data.float().cuda()      # (out, in)
    Wq = nvfp4_quantize(W, block=16).cuda()          # NVFP4 4-bit base (deploy grid)
    R = (W - Wq)
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    base = relerr(Wq, W)
    out, inn = W.shape
    print(f"\n{nm}  {tuple(W.shape)}   NVFP4 base relerr = {base:.4f}  (4.5 bits/param)")
    for r in (16, 32, 64, 128):
        # rank-r residual, factors stored in fp8-e4m3 (real side-channel cost)
        L = round_e4m3((U[:, :r] * S[:r]))           # (out, r)
        Vt = round_e4m3(Vh[:r])                       # (r, in)
        Rr = L @ Vt
        e = relerr(Wq + Rr, W)
        extra = r * (out + inn) * 8 / W.numel()      # fp8 factors -> bits/param
        print(f"   + rank {r:>3} (fp8): relerr {e:.4f}   ({100*(1-e/base):.0f}% of NVFP4 error recovered)   +{extra:.2f} bits/param")
