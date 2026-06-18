"""Experiment C: does *runtime-adaptive* equalization (lambda) cost acceptance?

Door C of the co-design discussion collapses to a single change: move the one
calibration-dependent knob — the per-channel equalization scales `s_i` — off its
frozen value and let it track the live activation stream, warm-started from the
calibrated value. The SVD correction basis is already data-oblivious (it is W's
top input singular vectors; see `residual_basis.py`, which shows the W-basis beats
a calibration-fit residual basis), so lambda is the *only* thing left to unfreeze.

The worry is not in-distribution accuracy — warm-start makes token 0 identical to
the shipped codec by construction. The worry is *adaptation wander*: on an
out-of-distribution prompt, does letting lambda move ever make the FP4 draft a
worse match to the FP16 target than just freezing it, and for how many tokens
before it self-heals?

This measures exactly that, single-stream (batch 1), per token position:

  frozen   : shipped calibrated eq scales s_i = amax_i^alpha   (the baseline)
  adaptive : s_i,t = r_i,t^alpha, r_i,t = causal decaying-max of |x|, warm-started
             at the calibration amax  (decay^(t-j) weighting; warm floor fades as
             decay^(t+1), so t=0 == frozen and live stats take over)

Acceptance is the standard spec-decoding quantity a(p,q) = sum_x min(p,q) = 1-TV
(see acceptance.py) between FP4 draft and FP16 target next-token distributions,
teacher-forced over the SAME prompt — exact and hardware-agnostic. Everything
except lambda (zp+optclip base, W-SVD side-channel anchored at warm-start, per-
block QJL) is held fixed across the two regimes.

Run (Mac, CPU/MPS):
    python -m turboquant.validation.runtime_lambda_accept --model gpt2 --max-pos 192
    python -m turboquant.validation.runtime_lambda_accept \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-pos 256
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from turboquant.act_codec import TurboQuantActQuantizer
from turboquant.config import TurboQuantConfig
from turboquant.nvfp4 import nvfp4_quantize_zp, round_e4m3
from turboquant.validation.hf_perplexity import (
    _EQ_ALPHAS, _is_linear, _precompute_aux,
)

RESULTS = Path(__file__).resolve().parents[2] / "results"

# Out-of-distribution prompt (Python source) vs an in-distribution control
# (English prose, matching the WikiText calibration domain). The control should
# show ~no drawdown; the OOD prompt is where adaptation has to work.
_OOD_PROMPT = '''\
import torch
def fused_attention(q, k, v, scale, mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, seq_len):
        t = torch.arange(seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)
'''

_CTRL_PROMPT = '''\
The history of the printing press is often told as a single moment of invention, \
but it was in truth the convergence of several older crafts. Movable type required \
a metal alloy soft enough to cast cleanly yet hard enough to survive thousands of \
impressions, an oil-based ink that would adhere to metal rather than bead away, and \
a screw press adapted from the making of wine and paper. Each of these existed in \
some form before the middle of the fifteenth century, and the achievement lay less \
in any one device than in fitting them into a single repeatable process.'''


def _per_pos_acceptance(draft: torch.Tensor, target: torch.Tensor):
    """Per-position alpha = sum_x min(p,q) (=1-TV) and greedy argmax agreement.

    draft, target: (T, V) logits. Returns (alpha (T,), greedy (T,))."""
    q = F.softmax(draft.float(), dim=-1)
    p = F.softmax(target.float(), dim=-1)
    alpha = torch.minimum(p, q).sum(dim=-1)
    greedy = (draft.argmax(-1) == target.argmax(-1)).float()
    return alpha, greedy


def _adaptive_scale(absx: torch.Tensor, warm: torch.Tensor, alpha: float,
                    decay: float) -> torch.Tensor:
    """Causal decaying-max per-channel scale, warm-started at ``warm``.

    r_t = max over j in {-1,0,..,t} of decay^(t-j) * v_j, with v_{-1}=warm (the
    calibration amax) and v_j=|x_j|. In log space this is a cummax:
        log r_t = t*log(decay) + max_{j<=t}( log v_j - j*log(decay) ).
    s_t = r_t^alpha. The warm term contributes warm*decay^(t+1), so it equals the
    frozen scale at t=0 and fades into the live stream. ``absx`` is (T, D).
    """
    T = absx.shape[0]
    logdec = math.log(decay)
    t = torch.arange(T, device=absx.device, dtype=absx.dtype).unsqueeze(1)  # (T,1)
    b = absx.clamp_min(1e-12).log() - t * logdec                            # (T,D)
    # cummax is unimplemented on MPS; hop to CPU for it (cf. _svd_basis linalg).
    cm = torch.cummax(b.cpu(), dim=0)[0].to(b) if b.device.type == "mps" \
        else torch.cummax(b, dim=0)[0]                                      # data terms
    warm_b = warm.clamp_min(1e-5).log() + logdec                           # j=-1 floor
    cm = torch.maximum(cm, warm_b.unsqueeze(0))
    r = (t * logdec + cm).exp().clamp_min(1e-5)
    return r ** alpha


def _make_hook(layer_idx, codec, mx_block, cache, adaptive, decay):
    """Full eq->zp+optclip->W-SVD(+QJL) stack; only lambda differs by ``adaptive``."""
    def hook(module, args):
        x = args[0]
        if x.shape[-1] % mx_block:
            return None
        xf = x.float()
        flat = xf.reshape(-1, xf.shape[-1])                       # (T, D), B=1
        if adaptive:
            s = _adaptive_scale(flat.abs(), cache["warm"], cache["alpha"], decay)
        else:
            s = cache["frozen"].to(flat).unsqueeze(0)             # (1,D) broadcast
        xe = flat / s
        base = nvfp4_quantize_zp(xe, block=mx_block, optclip=True)
        basis = cache["basis"]
        coeff = (xe - base) @ basis
        cs = coeff.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
        coeff = round_e4m3(coeff / cs) * cs
        xh = base + coeff @ basis.T
        if codec.cfg.qjl_dim > 0:
            xh = xh + codec.qjl_correct(xe - xh, layer_idx)
        return ((xh * s).reshape_as(xf).to(x.dtype), *args[1:])
    return hook


@torch.no_grad()
def calibrate_eq_full(model, calib_ids, max_len, device, mx_block=16):
    """Per-layer (frozen_scale, warm_amax, alpha) — mirrors calibrate_eq_scales
    but also returns the raw amax and chosen alpha the adaptive path needs."""
    print("  calibrating eq (frozen scale + warm amax + alpha)...", end=" ", flush=True)
    t0 = time.time()
    amax, samples, handles = {}, {}, []
    for i, m in enumerate(mod for mod in model.modules() if _is_linear(mod)):
        def make(idx):
            def h(module, args):
                flat = args[0].detach().float().reshape(-1, args[0].shape[-1])
                a = flat.abs().amax(dim=0)
                amax[idx] = torch.maximum(amax[idx], a) if idx in amax else a
                if idx not in samples:
                    samples[idx] = flat[:128].clone()
            return h
        handles.append(m.register_forward_pre_hook(make(i)))
    for begin in range(0, calib_ids.size(1), max_len):
        model(calib_ids[:, begin:begin + max_len].to(device))
    for h in handles:
        h.remove()
    frozen, warm, alpha = {}, {}, {}
    for i, a in amax.items():
        if a.numel() % mx_block:                                  # can't NVFP4 this proj
            continue
        sample = samples[i]
        best_alpha = best_e = None
        for al in _EQ_ALPHAS:
            s = a.clamp_min(1e-5) ** al
            e = ((sample - nvfp4_quantize_zp(sample / s, block=mx_block, optclip=True) * s)
                 ** 2).sum().item()
            if best_e is None or e < best_e:
                best_alpha, best_e = al, e
        frozen[i] = a.clamp_min(1e-5) ** best_alpha
        warm[i] = a.clamp_min(1e-5)
        alpha[i] = best_alpha
    print(f"done ({len(frozen)} layers, {time.time() - t0:.1f}s)", flush=True)
    return frozen, warm, alpha


@torch.no_grad()
def _logits(model, ids, hooks_fn=None):
    handles = hooks_fn() if hooks_fn else []
    try:
        return model(ids).logits[0]                               # (T, V)
    finally:
        for h in handles:
            h.remove()


def _summarize(name, alpha_f, alpha_a, greedy_f, greedy_a, eps):
    """Print per-bucket alpha + drawdown, return a JSON-able summary."""
    drawdown = (alpha_f - alpha_a)                                # +ve = adaptive worse
    T = drawdown.numel()
    # per-token TV is spiky; the decision-relevant signal is *sustained* drawdown,
    # so smooth with a centered moving average before max/self-heal.
    w = min(8, T)
    c = torch.cat([torch.zeros(1), drawdown.cumsum(0)])
    idx = torch.arange(T)
    lo = (idx - w // 2).clamp_min(0)
    hi = (idx + w // 2 + 1).clamp_max(T)
    sm = (c[hi] - c[lo]) / (hi - lo).float()                      # (T,) windowed drawdown
    imax = int(sm.argmax())
    # self-heal: first t at/after the smoothed peak from which it stays < eps
    healed = T
    for t in range(imax, T):
        if bool((sm[t:] < eps).all()):
            healed = t
            break
    print(f"\n=== {name} (T={T}) — alpha=1-TV(draft,target), higher=better ===")
    print(f"  mean alpha   frozen {alpha_f.mean():.4f}   adaptive {alpha_a.mean():.4f}"
          f"   delta {alpha_a.mean() - alpha_f.mean():+.4f}  <- headline")
    print(f"  mean greedy  frozen {greedy_f.mean():.4f}   adaptive {greedy_a.mean():.4f}")
    print(f"  worst sustained drawdown (w={w}) {sm[imax]:+.4f} at pos {imax}"
          f"   (raw-token max {drawdown.max():+.4f}); self-heal pos "
          f"{healed if healed < T else 'none'}")
    nb = min(8, T)
    edges = torch.linspace(0, T, nb + 1).long()
    print("  position bucket :   a_frozen  a_adapt   drawdown")
    for b in range(nb):
        lo, hi = edges[b].item(), edges[b + 1].item()
        if hi <= lo:
            continue
        print(f"   [{lo:4d},{hi:4d})    :   {alpha_f[lo:hi].mean():.4f}   "
              f"{alpha_a[lo:hi].mean():.4f}   {drawdown[lo:hi].mean():+.4f}")
    return {
        "T": T,
        "mean_alpha_frozen": round(alpha_f.mean().item(), 5),
        "mean_alpha_adaptive": round(alpha_a.mean().item(), 5),
        "mean_alpha_delta": round((alpha_a.mean() - alpha_f.mean()).item(), 5),
        "mean_greedy_frozen": round(greedy_f.mean().item(), 5),
        "mean_greedy_adaptive": round(greedy_a.mean().item(), 5),
        "worst_sustained_drawdown": round(sm[imax].item(), 5),
        "worst_sustained_pos": imax,
        "raw_token_max_drawdown": round(drawdown.max().item(), 5),
        "self_heal_pos": None if healed >= T else healed,
        "drawdown_eps": eps,
        "smooth_window": w,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--max-pos", type=int, default=192, help="prompt length cap (tokens)")
    ap.add_argument("--decay", type=float, default=0.98, help="per-token EMA decay for adaptive lambda")
    ap.add_argument("--qjl-block", type=int, default=128)
    ap.add_argument("--qjl-dim", type=int, default=64)
    ap.add_argument("--eps", type=float, default=0.005, help="drawdown self-heal threshold")
    ap.add_argument("--max-len", type=int, default=512, help="calibration window")
    ap.add_argument("--ood-file", default=None,
                    help="path to OOD prompt text; overrides the embedded code prompt")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32
    print(f"device={device} dtype={dtype} model={args.model} decay={args.decay}")

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.model)

    # Calibrate eq on WikiText-train (the in-distribution domain) — frozen scale,
    # warm amax, per-layer alpha.
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    calib_ids = tok("\n\n".join(train["text"][:2000]),
                    return_tensors="pt").input_ids[:, : 4 * args.max_len]
    frozen, warm, alpha = calibrate_eq_full(model, calib_ids, args.max_len, device)

    # W-SVD bases in the warm-start equalized space (data-oblivious; anchored).
    aux = _precompute_aux(model, 16, device, need_basis=True, need_comp=False,
                          eq_scales=frozen)
    cache = {}
    for i in frozen:
        if i in aux and "basis" in aux[i]:
            cache[i] = {"frozen": frozen[i].to(device), "warm": warm[i].to(device),
                        "alpha": alpha[i], "basis": aux[i]["basis"]}
    print(f"  active layers (eq+basis): {len(cache)}")

    cfg = TurboQuantConfig(qjl_block=args.qjl_block, qjl_dim=args.qjl_dim)
    codec = TurboQuantActQuantizer(cfg)

    def hooks(adaptive):
        def install():
            handles = []
            for i, m in enumerate(mod for mod in model.modules() if _is_linear(mod)):
                if i in cache:
                    handles.append(m.register_forward_pre_hook(
                        _make_hook(i, codec, 16, cache[i], adaptive, args.decay)))
            return handles
        return install

    summary = {"model": args.model, "decay": args.decay, "max_pos": args.max_pos,
               "qjl_block": args.qjl_block, "qjl_dim": args.qjl_dim, "prompts": {}}
    ood_text = Path(args.ood_file).read_text() if args.ood_file else _OOD_PROMPT
    for name, text in (("ood_code", ood_text), ("control_prose", _CTRL_PROMPT)):
        ids = tok(text, return_tensors="pt").input_ids[:, : args.max_pos].to(device)
        target = _logits(model, ids)                              # FP16, no hooks
        draft_f = _logits(model, ids, hooks(adaptive=False))
        draft_a = _logits(model, ids, hooks(adaptive=True))
        # next-token distributions at positions [0, T-1) (drop final, like collect_acceptance)
        af, gf = _per_pos_acceptance(draft_f[:-1], target[:-1])
        aa, ga = _per_pos_acceptance(draft_a[:-1], target[:-1])
        summary["prompts"][name] = _summarize(name, af.cpu(), aa.cpu(),
                                              gf.cpu(), ga.cpu(), args.eps)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"runtime_lambda_accept_{args.model.replace('/', '_')}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
