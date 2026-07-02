"""CPU smoke for KV4-aware QAT: codec fake-quant on K/V shape, STE hook, grad flow.
Does NOT run a real model — just validates the mechanism added to qat_nvfp4.py."""
import torch, torch.nn as nn
from turboquant.config import TurboQuantConfig
from turboquant.act_codec import TurboQuantActQuantizer
from turboquant.validation.qat_nvfp4 import install_kv4_qat

torch.manual_seed(0)


class Toy(nn.Module):
    """Mimics an attention block's K/V projections (GQA: 4 kv-heads * 64 = 256)."""
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(2048, 256)
        self.v_proj = nn.Linear(2048, 256)
        self.q_proj = nn.Linear(2048, 2048)   # should NOT be hooked

    def forward(self, x):
        return self.k_proj(x).sum() + self.v_proj(x).sum() + self.q_proj(x).sum()


m = Toy()
cfg = TurboQuantConfig(mx_block=16, qjl_block=128, qjl_dim=64, use_optclip=True)
codec = TurboQuantActQuantizer(cfg)

# 1) codec fake-quantizes a K/V-shaped tensor and actually changes it
kv = torch.randn(1, 8, 256)
q = codec.fake_quantize(kv.float(), layer_idx=0)
assert q.shape == kv.shape, q.shape
err = (q - kv).abs().mean().item()
print(f"[1] fake_quantize ok  shape={tuple(q.shape)}  mean|Δ|={err:.4f}  (nonzero => quant active)")
assert err > 0, "quantizer was a no-op"

# 2) hooks land only on k_proj/v_proj, not q_proj
handles, n_kv = install_kv4_qat(m, codec)
print(f"[2] installed KV4 hooks on {n_kv} modules (expected 2: k_proj, v_proj)")
assert n_kv == 2, n_kv

# 3) forward + backward: STE lets grads flow to the K/V weights despite quantization
x = torch.randn(1, 8, 2048)
loss = m(x)
loss.backward()
gk = m.k_proj.weight.grad
print(f"[3] grad flows through STE: k_proj.grad norm={gk.norm():.3f}  finite={torch.isfinite(gk).all().item()}")
assert gk is not None and torch.isfinite(gk).all() and gk.norm() > 0

# 4) the hook is actually quantizing (output differs from un-hooked path)
for h in handles:
    h.remove()
ref = m.k_proj(x)
handles2, _ = install_kv4_qat(m, codec)
with torch.no_grad():
    hooked = m.k_proj(x)
d = (hooked - ref).abs().mean().item()
print(f"[4] hook changes k_proj output: mean|Δ|={d:.4f}  (>0 => KV4 active in forward)")
assert d > 0

print("\nSMOKE PASS — KV4-aware QAT mechanism is wired correctly.")
