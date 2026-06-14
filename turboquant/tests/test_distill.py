"""CPU tests for trained low-rank factors (distill.py).

Proves the mechanism end-to-end on a tiny model: gradients flow ONLY to the
factors, and KL to a teacher actually decreases. This is what de-risks the GPU
run — if it works here, the box only spends minutes on the real 8B.
"""

import torch
import torch.nn as nn

from turboquant.distill import (
    LowRankLinear, attach_lowrank, trainable_factor_params,
    kl_distill_loss, distill_factors,
)


class TinyNet(nn.Module):
    def __init__(self, d=32, v=40):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, v)

    def forward(self, x):
        from types import SimpleNamespace
        h = torch.relu(self.fc1(x))
        return SimpleNamespace(logits=self.fc2(h))


def test_lowrank_linear_shapes_and_zero_init():
    Wq = torch.randn(8, 6)
    L = torch.zeros(8, 3)
    R = torch.zeros(3, 6)
    m = LowRankLinear(Wq, L, R)
    x = torch.randn(4, 6)
    # zero factors -> pure Wq path
    assert torch.allclose(m(x), x @ Wq.t(), atol=1e-5)


def test_only_factors_are_trainable():
    net = TinyNet()
    Wq = net.fc1.weight.detach().clone()
    L = torch.zeros(32, 4)
    R = torch.zeros(4, 32)
    n = attach_lowrank(net, {"fc1": (Wq, L, R)})
    assert n == 1
    params = trainable_factor_params(net)
    assert len(params) == 2                       # exactly L and R
    trainable = [nm for nm, p in net.named_parameters() if p.requires_grad]
    assert all("fc1.L" in nm or "fc1.R" in nm for nm in trainable)
    assert "fc2.weight" not in trainable


def test_distill_decreases_kl():
    torch.manual_seed(0)
    teacher = TinyNet()
    # student = teacher with fc1 weight perturbed (the "quantization error"),
    # corrected by trainable factors initialized at zero.
    student = TinyNet()
    student.load_state_dict(teacher.state_dict())
    Wq = teacher.fc1.weight.detach().clone()
    Wq = Wq + 0.3 * torch.randn_like(Wq)          # planted weight error
    # full-rank factors (r=in) CAN exactly cancel the error -> KL should ~vanish.
    # LoRA-style init: L=0, R=random -> correction starts at 0 but grad flows
    # (zero-init of BOTH factors is a zero-gradient saddle; the SVD warm-start
    # in the real pipeline avoids it).
    L0 = torch.zeros(32, 32)
    R0 = 0.01 * torch.randn(32, 32)
    attach_lowrank(student, {"fc1": (Wq, L0, R0)})

    batches = [torch.randn(16, 32) for _ in range(6)]
    eval_x = torch.randn(64, 32)                   # fixed held-out eval
    def kl_on(x):
        with torch.no_grad():
            return kl_distill_loss(student(x).logits, teacher(x).logits).item()
    before = kl_on(eval_x)
    distill_factors(student, teacher, batches, lr=5e-2, epochs=60)
    after = kl_on(eval_x)
    assert after < before * 0.25, f"KL did not drop enough: {before:.4f} -> {after:.4f}"
