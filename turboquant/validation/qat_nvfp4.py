"""QAT for NVFP4: does training the model to be quantization-aware recover the W4A4
gap that post-training quantization leaves? (The Door-B / Gemma-QAT-style escape.)

We put the NVFP4 fake-quantizer in the FORWARD pass on weights AND activations with a
straight-through estimator, and fine-tune the weights (distilling against the FP16
teacher). Clean head-to-head, plain NVFP4 (no eq/SVD/QJL), so the only variable is the
training:

  FP16        : teacher, no quant
  PTQ-W4A4    : frozen weights, fake-quant forward  (= what PTQ gives, the baseline)
  QAT-W4A4    : same fake-quant forward, weights trained to survive it

If QAT-W4A4 beats PTQ-W4A4, quantization-aware training recovers the gap PTQ can't.

Run (RTX 6000 / H100, ~1h on TinyLlama):
  python -m turboquant.validation.qat_nvfp4 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
     --steps 800 --lr 2e-5
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from turboquant.nvfp4 import nvfp4_quantize
from turboquant.distill import kl_distill_loss
from turboquant.config import TurboQuantConfig
from turboquant.act_codec import TurboQuantActQuantizer
from turboquant.validation.hf_perplexity import _is_linear, perplexity, calibrate_eq_scales
from turboquant.validation.acceptance import acceptance_stats

RESULTS = Path(__file__).resolve().parents[2] / "results"


def install_kv4_qat(student, codec) -> list:
    """Make QAT KV-cache-aware: fake-quant (NVFP4 + QJL) the K/V projection outputs in the
    forward with a straight-through estimator, so the trained weights learn to be robust to
    KV4 too. Hooks the k_proj/v_proj leaves (now QATLinear after replace_linears) by name.
    KV keeps the QJL side-channel (cheap, 1-bit); only the SVD side-channel is dropped."""
    handles = []
    kv = [(n, m) for n, m in student.named_modules() if n.endswith(("k_proj", "v_proj"))]

    def _make(idx):
        def h(module, inp, out):
            q = codec.fake_quantize(out.float(), layer_idx=idx).to(out.dtype)
            return out + (q - out).detach()          # STE: forward=quantized, grad=identity
        return h

    for i, (_, m) in enumerate(kv):
        handles.append(m.register_forward_hook(_make(i)))
    return handles, len(kv)


def _fq(x, block):
    """NVFP4 fake-quant with straight-through gradient (forward=quantized, grad=identity)."""
    if x.shape[-1] % block:
        return x
    q = nvfp4_quantize(x.float(), block=block).to(x.dtype)
    return x + (q - x).detach()


class QATLinear(nn.Module):
    """nn.Linear with NVFP4 fake-quant (STE) on weight + input; weight is trainable."""

    def __init__(self, lin: nn.Linear, block: int = 16, quant_act: bool = True):
        super().__init__()
        self.weight = nn.Parameter(lin.weight.data.clone())
        self.register_buffer("bias", lin.bias.data.clone() if lin.bias is not None else None)
        self.block = block
        self.quant_act = quant_act

    def forward(self, x):
        w = _fq(self.weight, self.block)
        xq = _fq(x, self.block) if self.quant_act else x
        return F.linear(xq, w, self.bias)


class LoRAQATLinear(nn.Module):
    """Frozen NVFP4 base weight + trainable low-rank adapter; the COMBINED weight is
    fake-quantized (STE). Only the adapter trains -> ~50x smaller optimizer state than
    full-weight QAT, so ALL layers fit at 8B (full-weight 8B QAT OOMs at 95GB). B inits to
    zero => BA=0 => starts exactly at PTQ (no initial disruption); the adapter learns to
    pre-distort the weight so post-quant error shrinks. Folds to a plain nn.Linear at
    export (drop-in NVFP4 checkpoint for TRT-LLM)."""

    def __init__(self, lin: nn.Linear, block: int = 16, quant_act: bool = True,
                 rank: int = 16, alpha: float | None = None):
        super().__init__()
        out_f, in_f = lin.weight.shape
        self.register_buffer("weight", lin.weight.data.clone())   # frozen base
        self.register_buffer("bias", lin.bias.data.clone() if lin.bias is not None else None)
        self.block = block
        self.quant_act = quant_act
        self.scaling = (alpha if alpha is not None else rank) / rank
        dev, dt = lin.weight.device, lin.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))     # B=0 -> adapter starts at 0

    def merged_weight(self):
        return self.weight + self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x):
        w = _fq(self.merged_weight(), self.block)
        xq = _fq(x, self.block) if self.quant_act else x
        return F.linear(xq, w, self.bias)


def _layer_indices(model) -> list[int]:
    """Decoder-block indices present in the module tree (from '...layers.{i}...')."""
    import re
    idx = {int(m.group(1)) for n, _ in model.named_modules()
           if (m := re.search(r"\.layers\.(\d+)\.", n))}
    return sorted(idx)


def boundary_ignore(model) -> list[str]:
    """The ModelOpt-style high-precision recipe: keep the first + last decoder blocks
    and lm_head in BF16. lm_head is already excluded by ModelOpt's NVFP4 default, so
    including it here just makes our fake-quant mirror what actually deploys."""
    idx = _layer_indices(model)
    pats = ["lm_head"]
    if idx:
        pats += [f".layers.{idx[0]}.", f".layers.{idx[-1]}."]
    return pats


def replace_linears(model, block=16, quant_act=True, lora_rank=0, lora_alpha=None,
                    ignore=None) -> int:
    """Swap eligible nn.Linear for (LoRA)QATLinear. ``ignore`` is a list of substrings;
    a Linear whose module name contains any of them is left in BF16 (unquantized) —
    the deployable layer-precision recipe (e.g. boundary blocks + lm_head)."""
    ignore = ignore or []
    n = 0
    name_to_mod = dict(model.named_modules())
    for name, m in list(name_to_mod.items()):
        if isinstance(m, nn.Linear) and m.weight.shape[-1] % block == 0:
            if any(pat in name for pat in ignore):
                continue
            parent_name, _, child = name.rpartition(".")
            parent = name_to_mod[parent_name] if parent_name else model
            new = (LoRAQATLinear(m, block, quant_act, lora_rank, lora_alpha) if lora_rank > 0
                   else QATLinear(m, block, quant_act))
            setattr(parent, child, new.to(m.weight.device))
            n += 1
    return n


@torch.no_grad()
def _eval(student, teacher, ppl_ids, accept_w, max_len, device):
    V = teacher.config.vocab_size
    ppl = perplexity(student, ppl_ids, max_len, max_len, device)
    a = g = 0.0
    for x in accept_w:
        t = teacher(x).logits[:, :-1, :].reshape(-1, V)
        d = student(x).logits[:, :-1, :].reshape(-1, V)
        ai, gi = acceptance_stats(d, t)
        a += ai; g += gi
    n = max(len(accept_w), 1)
    return round(ppl, 4), round(a / n, 4), round(g / n, 4)


def tv_loss(student_logits, teacher_logits):
    """Total-variation distillation = directly maximize spec-decode acceptance.
    acceptance alpha = 1 - TV(student, teacher), so minimizing TV maximizes accepted
    tokens — the exact metric that becomes throughput (vs KL which only correlates)."""
    V = student_logits.size(-1)
    q = F.softmax(student_logits.float().reshape(-1, V), dim=-1)
    p = F.softmax(teacher_logits.float().reshape(-1, V), dim=-1)
    return 0.5 * (q - p).abs().sum(-1).mean()


@torch.no_grad()
def _val_loss(model, batches):
    return sum(model(x, labels=x).loss.item() for x in batches) / max(len(batches), 1)


@torch.no_grad()
def _val_tv(student, teacher, batches):
    """Mean TV (= 1 - acceptance) on held-out — the metric that matters, for selection."""
    return sum(tv_loss(student(x).logits, teacher(x).logits).item()
               for x in batches) / max(len(batches), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--n-accept", type=int, default=8)
    ap.add_argument("--ppl-limit", type=int, default=40000)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="activation checkpointing — needed to fit large (8B) QAT in GPU mem")
    ap.add_argument("--train-every", type=int, default=1,
                    help="train only every Nth QATLinear (memory lever for 8B; rest stay frozen+fake-quant)")
    ap.add_argument("--lora-rank", type=int, default=0,
                    help=">0 enables LoRA-QAT: freeze base weights, train only rank-r adapters on ALL "
                         "layers (tiny optimizer state -> full 8B QAT fits). The real 8B path.")
    ap.add_argument("--lora-alpha", type=float, default=None, help="LoRA scaling alpha (default = rank)")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--no-quant-act", action="store_true", help="W4A16 (weights only)")
    ap.add_argument("--save-dir", default=None, help="fold QAT weights back to nn.Linear and save_pretrained here")
    ap.add_argument("--objective", choices=["kl", "tv", "kltv"], default="kl",
                    help="kl=match teacher; tv=accept(1-TV) direct (weak grad); "
                         "kltv=KL backbone + TV acceptance pressure (recommended)")
    ap.add_argument("--tv-weight", type=float, default=1.0, help="TV term weight for kltv")
    ap.add_argument("--mx-block", type=int, default=16,
                    help="MX4 microscaling group size (16=NVFP4 spec; 32=MXFP4-grained, "
                         "cheaper scale overhead but coarser — outliers hurt 2x, lean on --eq-fold)")
    ap.add_argument("--eq-fold", action="store_true",
                    help="calibrate per-channel equalization scales and apply as activation "
                         "pre-hooks before QAT — lets QAT absorb the correction that SVD "
                         "provides post-hoc, at zero inference cost (no side-channel at deploy)")
    ap.add_argument("--kv4", action="store_true",
                    help="make QAT KV-cache-aware: fake-quant K/V projection outputs to "
                         "NVFP4 + QJL (STE) in the forward, so weights learn KV4 robustness. "
                         "Full W4A4KV4 trained, SVD dropped, QJL kept on KV.")
    ap.add_argument("--qjl-block", type=int, default=128, help="QJL sub-block size (KV4)")
    ap.add_argument("--qjl-dim", type=int, default=64, help="QJL sign projections per block (KV4); 0=off")
    ap.add_argument("--ignore", default="", help="comma-sep name substrings kept in BF16 (unquantized)")
    ap.add_argument("--ignore-boundary", action="store_true",
                    help="ModelOpt-style recipe: keep first+last decoder block and lm_head in BF16")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32
    quant_act = not args.no_quant_act
    print(f"device={device} dtype={dtype} model={args.model} W4A{'4' if quant_act else '16'} "
          f"steps={args.steps} lr={args.lr}")

    def load():
        try:
            return AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    L = args.max_len
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    test_ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
    ppl_ids = test_ids[:, : args.ppl_limit]
    accept_w = [test_ids[:, i * L:(i + 1) * L].to(device) for i in range(test_ids.size(1) // L)][: args.n_accept]
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    tr_ids = tok("\n\n".join(train["text"]), return_tensors="pt").input_ids   # full train (heavy QAT)
    tr_w = [tr_ids[:, i * L:(i + 1) * L].to(device) for i in range(tr_ids.size(1) // L)]
    train_b, val_b = tr_w[args.n_val:args.n_val + args.n_train], tr_w[:args.n_val]
    print(f"  train {len(train_b)}  val {len(val_b)}  accept {len(accept_w)} windows")

    teacher = load().to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = load().to(device).eval()

    # EQ-fold: calibrate per-channel scales while student still has nn.Linear (before replace_linears).
    # The scales fold equalization into the forward pass so QAT sees uniform activations without
    # any SVD side-channel — the model learns to quantize-robustly in already-equalized space.
    eq_scales = {}
    if args.eq_fold:
        calib_ids = tr_ids[:, :4 * L]
        eq_scales = calibrate_eq_scales(student, calib_ids, L, device, mx_block=args.mx_block,
                                        weight_aware=True)
        print(f"  EQ-fold: calibrated {len(eq_scales)} layer scales")

    ignore = [p for p in args.ignore.split(",") if p]
    if args.ignore_boundary:
        ignore += boundary_ignore(student)
    if ignore:
        print(f"  ignore (kept BF16): {ignore}")
    nrep = replace_linears(student, args.mx_block, quant_act, args.lora_rank, args.lora_alpha,
                           ignore=ignore)

    # Install eq pre-hooks on the now-replaced QATLinear/LoRAQATLinear modules.
    # Iterate model.modules() in the same order as calibrate_eq_scales used, matching by index.
    eq_hooks = []
    if args.eq_fold:
        def _is_any_linear(m):
            return isinstance(m, (QATLinear, LoRAQATLinear)) or _is_linear(m)
        for i, m in enumerate(mod for mod in student.modules() if _is_any_linear(mod)):
            if i not in eq_scales:
                continue
            s = eq_scales[i].to(device)
            # Fold s into the weight's input columns so equalization is identity in full
            # precision: (W·s) @ (x/s) = W @ x. The activation pre-hook alone divides x but
            # never compensates W -> a corrupted forward (35k ppl). With the fold, the
            # quantizer sees equalized (uniform) activations AND weights, which is the point.
            m.weight.data.mul_(s.to(m.weight.dtype))
            def _make_eq_hook(scale):
                def h(module, args_):
                    x = args_[0]
                    return (x / scale.to(x.dtype),) + args_[1:]
                return h
            eq_hooks.append(m.register_forward_pre_hook(_make_eq_hook(s)))
        print(f"  EQ-fold: installed hooks on {len(eq_hooks)} student layers")

    # KV4-aware QAT: fake-quant K/V to NVFP4+QJL in the forward (STE) on the student only
    # (teacher stays FP16). Hooks persist through training AND eval, so PTQ and QAT are both
    # measured under KV4 — an honest W4A4KV4 head-to-head.
    if args.kv4:
        kv_cfg = TurboQuantConfig(mx_block=args.mx_block, qjl_block=args.qjl_block,
                                  qjl_dim=args.qjl_dim, use_optclip=True)
        kv_codec = TurboQuantActQuantizer(kv_cfg)
        _, n_kv = install_kv4_qat(student, kv_codec)
        print(f"  KV4: fake-quant (NVFP4+QJL dim={args.qjl_dim}) on {n_kv} K/V projections")

    if args.grad_checkpoint:
        student.config.use_cache = False
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in student.parameters():
        p.requires_grad_(False)
    if args.lora_rank > 0:
        # LoRA-QAT: only the adapters train (all layers covered, tiny optimizer state).
        params = [p for m in student.modules() if isinstance(m, LoRAQATLinear)
                  for p in (m.lora_A, m.lora_B)]
        for p in params:
            p.requires_grad_(True)
        print(f"  LoRA-QAT rank {args.lora_rank} on {nrep} linears; "
              f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable adapter params")
    else:
        qlins = [m for m in student.modules() if isinstance(m, QATLinear)]
        # train-every>1: only every Nth QATLinear is trainable (grads+Adam states scale with
        # trainable params) — the memory lever for fitting 8B QAT. Untrained layers are still
        # fake-quantized in the forward, so the student is fully W4A4; we just don't update them.
        train_q = qlins[::args.train_every]
        params = [m.weight for m in train_q]
        for p in params:
            p.requires_grad_(True)
        print(f"  replaced {nrep} linears with QATLinear; training {len(params)}/{len(qlins)} "
              f"weights (every {args.train_every})")

    out = {"model": args.model, "w4a4": quant_act, "steps": args.steps, "lr": args.lr,
           "objective": args.objective, "eq_fold": args.eq_fold, "mx_block": args.mx_block,
           "kv4": args.kv4, "qjl_dim": args.qjl_dim if args.kv4 else 0}

    fp16_ppl = round(perplexity(teacher, ppl_ids, L, L, device), 4)
    out["fp16_ppl"] = fp16_ppl
    print(f"  FP16 ppl = {fp16_ppl}")

    # PTQ baseline = student before any training (fake-quant forward, original weights)
    p, a, g = _eval(student, teacher, ppl_ids, accept_w, L, device)
    out["ptq"] = {"ppl": p, "alpha": a, "greedy": g}
    print(f"  [PTQ ] ppl={p} alpha={a} greedy={g}")

    # QAT: train + SELECT on the objective that matters (tv=acceptance, not PPL)
    if args.objective == "tv":
        loss_fn = tv_loss
    elif args.objective == "kltv":
        loss_fn = lambda s, t: kl_distill_loss(s, t) + args.tv_weight * tv_loss(s, t)
    else:
        loss_fn = lambda s, t: kl_distill_loss(s, t)
    accept_obj = args.objective in ("tv", "kltv")           # select on acceptance for both
    val_fn = ((lambda: _val_tv(student, teacher, val_b)) if accept_obj
              else (lambda: _val_loss(student, val_b)))
    metric = "val TV(1-accept)" if accept_obj else "val loss"
    # foreach=False: update params one at a time. foreach/fused Adam allocates a transient
    # full-size copy of the optimizer state (e.g. _foreach_sqrt) which OOMs at 8B scale.
    opt = torch.optim.AdamW(params, lr=args.lr, foreach=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    best = val_fn()
    best_state = [p.detach().clone() for p in params]
    print(f"  objective={args.objective}  PTQ {metric} = {best:.4f}")
    step = 0
    t0 = time.time()
    while step < args.steps:
        for x in train_b:
            if step >= args.steps:
                break
            loss = loss_fn(student(x).logits, teacher(x).logits)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            if step % args.eval_every == 0:
                v = val_fn()
                tag = ""
                if v < best:
                    best, tag = v, "  <- best"
                    best_state = [p.detach().clone() for p in params]
                print(f"  step {step:4d}  train {args.objective} {loss.item():.4f}  {metric} {v:.4f}{tag}", flush=True)
    with torch.no_grad():
        for p_, b_ in zip(params, best_state):
            p_.copy_(b_)
    out["train_seconds"] = round(time.time() - t0, 1)

    p, a, g = _eval(student, teacher, ppl_ids, accept_w, L, device)
    out["qat"] = {"ppl": p, "alpha": a, "greedy": g}
    print(f"  [QAT ] ppl={p} alpha={a} greedy={g}")

    out["ppl_recovered"] = round(out["ptq"]["ppl"] - out["qat"]["ppl"], 4)
    out["alpha_gain"] = round(out["qat"]["alpha"] - out["ptq"]["alpha"], 4)
    gap_ptq = out["ptq"]["ppl"] - fp16_ppl
    gap_qat = out["qat"]["ppl"] - fp16_ppl
    out["gap_closed_frac"] = round(1 - gap_qat / gap_ptq, 3) if gap_ptq > 1e-6 else None
    verdict = ("QAT WINS — recovers the PTQ gap" if out["ppl_recovered"] > 0.02
               else "no recovery (QAT ~= PTQ)")
    out["verdict"] = verdict
    print(f"\n  FP16 {fp16_ppl} | PTQ {out['ptq']['ppl']} (+{gap_ptq:.3f}) | "
          f"QAT {out['qat']['ppl']} (+{gap_qat:.3f})")
    print(f"  *** ACCEPTANCE (the metric that matters): {out['ptq']['alpha']} -> "
          f"{out['qat']['alpha']}  ({out['alpha_gain']:+.4f}) ***")
    print(f"  (PPL sanity: recovered {out['ppl_recovered']:+.4f}, gap closed {out['gap_closed_frac']})")
    print(f"  VERDICT: {verdict}")

    if args.save_dir:                                 # fold QATLinear -> nn.Linear, save HF model
        name_to_mod = dict(student.named_modules())
        for name, m in list(name_to_mod.items()):
            if isinstance(m, (QATLinear, LoRAQATLinear)):
                w = m.merged_weight().detach() if isinstance(m, LoRAQATLinear) else m.weight.data
                lin = nn.Linear(w.shape[1], w.shape[0],
                                bias=m.bias is not None).to(w.device, w.dtype)
                lin.weight.data.copy_(w)
                if m.bias is not None:
                    lin.bias.data.copy_(m.bias)
                parent_name, _, child = name.rpartition(".")
                setattr(name_to_mod[parent_name] if parent_name else student, child, lin)
        student.save_pretrained(args.save_dir)
        tok.save_pretrained(args.save_dir)
        print(f"  saved QAT model -> {args.save_dir}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    eq_tag = "_eq" if args.eq_fold else ""
    kv_tag = "_kv4" if args.kv4 else ""
    blk_tag = f"_b{args.mx_block}" if args.mx_block != 16 else ""
    pth = RESULTS / f"qat_nvfp4_{args.objective}{eq_tag}{kv_tag}{blk_tag}_{args.model.replace('/', '_')}.json"
    pth.write_text(json.dumps(out, indent=2))
    print(f"saved -> {pth}")


if __name__ == "__main__":
    main()
