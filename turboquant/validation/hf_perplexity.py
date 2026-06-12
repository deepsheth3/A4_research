"""End-to-end Experiment B: TurboQuant FP4 *activations* on a real model.

Registers forward-pre-hooks on every linear projection (nn.Linear and GPT-2's
Conv1D) that fake-quantize the layer's *input activation*, then measures
WikiText-2 perplexity in three modes:

  fp16        — no hook (baseline)
  nvfp4_raw   — raw NVFP4 activations, no correction (the "problem" baseline)
  turboquant  — NVFP4 + per-block QJL correction (the proposed fix)

Weights and KV cache stay in the model's native precision — only activations are
quantized, isolating the activation-quantization question.

Smoke test (Mac, CPU/MPS):
    python -m turboquant.validation.hf_perplexity --model gpt2 --limit 50
Real run (one H100):
    python -m turboquant.validation.hf_perplexity --model meta-llama/Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from turboquant import nvfp4_quantize
from turboquant.act_codec import TurboQuantActQuantizer
from turboquant.nvfp4 import nvfp4_quantize_zp, round_e4m3, waware_comp
from turboquant.config import TurboQuantConfig

RESULTS = Path(__file__).resolve().parents[2] / "results"
_LINEAR_TYPES = ("Linear", "Conv1D")  # Conv1D == GPT-2's linear layers


def _is_linear(module) -> bool:
    return type(module).__name__ in _LINEAR_TYPES


_FP8_MAX = 448.0  # E4M3 max representable magnitude


def fp8_quantize(x: torch.Tensor) -> torch.Tensor:
    """FP8-E4M3 fake-quant with per-token absmax scaling (TRT-LLM-style dynamic
    activation FP8 — the production accuracy target NVFP4+QJL must match)."""
    scale = (x.abs().amax(dim=-1, keepdim=True) / _FP8_MAX).clamp_min(1e-12)
    return round_e4m3(x / scale) * scale


@torch.no_grad()
def collect_weight_hessians(model, calib_ids, max_len: int, device, n_seq: int = 64) -> dict:
    """Per-linear input Hessian H = E[xᵀx] over calibration windows (for GPTQ).

    Accumulated on GPU via forward-pre-hooks; one pass over a few windows of
    wikitext train. Offline — never runs at deploy time."""
    print("  collecting weight Hessians (offline)...", end=" ", flush=True)
    t0 = time.time()
    hess: dict = {}
    count: dict = {}
    handles = []
    for i, m in enumerate(mod for mod in model.modules() if _is_linear(mod)):
        def make(idx):
            def h(module, args):
                x = args[0].detach().float().reshape(-1, args[0].shape[-1])
                g = x.t() @ x
                hess[idx] = g if idx not in hess else hess[idx] + g
                count[idx] = count.get(idx, 0) + x.shape[0]
            return h
        handles.append(m.register_forward_pre_hook(make(i)))
    n = 0
    for begin in range(0, calib_ids.size(1), max_len):
        if n >= n_seq:
            break
        model(calib_ids[:, begin:begin + max_len].to(device))
        n += 1
    for h in handles:
        h.remove()
    for i in hess:
        hess[i] /= max(count[i], 1)
    print(f"done ({len(hess)} layers, {n} windows, {time.time() - t0:.1f}s)", flush=True)
    return hess


@torch.no_grad()
def quantize_weights_gptq(model, hessians: dict, block: int = 16, awq: bool = False,
                          lowrank: bool = False, rank_div: int = 16,
                          fp8_factors: bool = False, fp4_factors: bool = False) -> int:
    """GPTQ-quantize every linear weight to NVFP4 using its activation Hessian.

    ``awq`` adds AWQ-style salient-channel protection; ``lowrank`` adds an
    additive low-rank residual correction (LQER-style, rank = in/``rank_div``)."""
    from turboquant.gptq import (gptq_quantize_weight, awq_gptq_quantize_weight,
                                 lowrank_corrected_weight)
    if lowrank:
        def quant(W, H, block):
            return lowrank_corrected_weight(W, H, block=block, rank_div=rank_div,
                                            fp8_factors=fp8_factors, fp4_factors=fp4_factors)
        ftag = ',fp4' if fp4_factors else (',fp8' if fp8_factors else '')
        tag = f"GPTQ+lowrank(in/{rank_div}{ftag})"
    elif awq:
        quant, tag = awq_gptq_quantize_weight, "AWQ+GPTQ"
    else:
        quant, tag = gptq_quantize_weight, "GPTQ"
    print(f"  {tag}-quantizing weights to NVFP4 (W4)...", end=" ", flush=True)
    t0 = time.time()
    n = 0
    for i, m in enumerate(mod for mod in model.modules() if _is_linear(mod)):
        W = m.weight.data
        if W.shape[-1] % block != 0 or i not in hessians:
            continue
        m.weight.data = quant(
            W.float(), hessians[i].to(W.device), block=block).to(W.dtype)
        n += 1
    print(f"done ({n} layers, {time.time() - t0:.1f}s)", flush=True)
    return n


@torch.no_grad()
def quantize_weights_gptq_alloc(model, hessians: dict, block: int = 16, rank_div: int = 8,
                                max_rank_div: int = 2, fp8_factors: bool = True) -> int:
    """GPTQ + low-rank correction with per-layer rank ALLOCATION (water-filling).

    Same total side-channel budget as a uniform in/``rank_div`` allocation, but
    distributed across layers by marginal singular-value energy per byte — more
    rank where weight quantization actually hurts the output, ~none where it
    doesn't. Pure Pareto move: identical total cost, better accuracy."""
    from turboquant.gptq import gptq_lowrank_factors, apply_lowrank
    print(f"  GPTQ+lowrank ALLOC (budget in/{rank_div}, fp8={fp8_factors})...",
          end=" ", flush=True)
    t0 = time.time()
    layers = [(i, m) for i, m in enumerate(mod for mod in model.modules() if _is_linear(mod))
              if m.weight.shape[-1] % block == 0 and i in hessians]

    # Phase A: GPTQ + spectrum per layer (factors stashed on CPU to bound GPU mem)
    parts, budget = {}, 0
    for i, m in layers:
        out_d, in_d = m.weight.shape
        maxr = (in_d // max_rank_div) // block * block
        W = m.weight.data.float()
        Wq, Lfac, Rfac, S = gptq_lowrank_factors(W, hessians[i].to(W.device), block, maxr)
        parts[i] = (Wq.cpu(), Lfac.cpu(), Rfac.cpu(), S.cpu(), m, out_d, in_d)
        budget += (in_d // rank_div) * (out_d + in_d)        # uniform-equivalent bytes

    # Water-fill: allocate rank in `block` chunks to max marginal energy / byte
    import heapq
    alloc = {i: 0 for i, _ in layers}
    heap = []
    for i, (_, _, _, S, _, out_d, in_d) in parts.items():
        cost = block * (out_d + in_d)
        nchunks = S.numel() // block
        if nchunks:
            gain = (S[:block] ** 2).sum().item()
            heapq.heappush(heap, (-gain / cost, i, 1))
    spent = 0
    while heap and spent < budget:
        neg, i, c = heapq.heappop(heap)
        _, _, _, S, _, out_d, in_d = parts[i]
        cost = block * (out_d + in_d)
        if spent + cost > budget:
            break
        alloc[i] = c * block
        spent += cost
        if (c + 1) * block <= S.numel():
            gain = (S[c * block:(c + 1) * block] ** 2).sum().item()
            heapq.heappush(heap, (-gain / cost, i, c + 1))

    # Phase B: apply allocated rank per layer
    for i, (Wq, Lfac, Rfac, S, m, out_d, in_d) in parts.items():
        dev = m.weight.device
        m.weight.data = apply_lowrank(Wq.to(dev), Lfac.to(dev), Rfac.to(dev),
                                      alloc[i], block=block,
                                      fp8_factors=fp8_factors).to(m.weight.dtype)
    ranks = [alloc[i] for i, _ in layers]
    print(f"done ({len(layers)} layers, ranks {min(ranks)}-{max(ranks)}, "
          f"{time.time() - t0:.1f}s)", flush=True)
    return len(layers)


@torch.no_grad()
def quantize_weights_nvfp4(model, block: int = 16, row_chunk: int = 512) -> int:
    """Fake-quantize every linear weight to NVFP4 (E2M1 + per-16 fp8 block
    scales), in place — making the GEMM W4A4: FP4 weight x FP4 activation,
    FP16 accumulate. Weights are static, so this is a one-time OFFLINE pass with
    zero deploy cost (uses the best static quantizer: joint zp x optclip, which
    is free here). SVD bases / eq scales computed afterward see the deployed FP4
    weights, as they would in production.

    Quantized in row-chunks: the joint search expands an 8-candidate axis, which
    on a full 8B weight matrix would spike many GB at once — chunking bounds it.
    """
    n = 0
    for m in model.modules():
        if _is_linear(m):
            W = m.weight.data
            if W.shape[-1] % block != 0:
                continue
            out = torch.empty_like(W)
            for i in range(0, W.shape[0], row_chunk):
                out[i:i + row_chunk] = nvfp4_quantize_zp(
                    W[i:i + row_chunk].float(), block=block, optclip=True).to(W.dtype)
            m.weight.data = out
            n += 1
    print(f"  quantized {n} linear weights to NVFP4 (W4)", flush=True)
    return n


def _svd_basis(module, x_dim: int, k: int, row_scale=None) -> torch.Tensor:
    """Top-k input-side singular vectors of the layer weight (offline in deploy).

    nn.Linear weight is (out, in) with y = x Wᵀ -> effective A = Wᵀ (in, out);
    GPT-2's Conv1D weight is already (in, out). With channel equalization the
    residual lives in the equalized space, so the basis comes from diag(s)·A.
    """
    W = module.weight.detach().float()
    if W.device.type != "cuda":  # MPS lacks linalg_qr; fall back to CPU there
        W = W.cpu()
    A = W.T if W.shape[1] == x_dim else W
    if row_scale is not None:
        A = row_scale.to(A).unsqueeze(1) * A
    U, _, _ = torch.svd_lowrank(A, q=min(k + 8, min(A.shape)), niter=4)
    return U[:, :k].contiguous()


def _make_hook(mode: str, layer_idx: int, codec: TurboQuantActQuantizer, mx_block: int,
               precomputed_cache: dict | None = None):
    cache = precomputed_cache if precomputed_cache is not None else {}

    def hook(module, args):
        x = args[0]
        if x.shape[-1] % mx_block != 0:  # can't NVFP4 this projection; leave it
            return None
        if mode == "fp8":
            xq = fp8_quantize(x.float()).to(x.dtype)
        elif mode == "nvfp4_raw":
            xq = nvfp4_quantize(x.float(), block=mx_block).to(x.dtype)
        elif mode == "nvfp4_zp":
            xq = nvfp4_quantize_zp(x.float(), block=mx_block).to(x.dtype)
        elif mode in ("nvfp4_eq", "nvfp4_eqzp"):
            s = cache["eq"]
            xe = x.float() / s
            qe = (nvfp4_quantize_zp(xe, block=mx_block, optclip=True)
                  if mode == "nvfp4_eqzp" else nvfp4_quantize(xe, block=mx_block))
            xq = (qe * s).to(x.dtype)
        elif mode.startswith("nvfp4_eqzp_svd"):  # eq fold + zp x optclip + SVD (+ QJL)
            s = cache["eq"]
            xe = x.float() / s
            base = nvfp4_quantize_zp(xe, block=mx_block, optclip=True, imp=cache.get("imp"))
            coeff = (xe - base) @ cache["basis"]
            cs = coeff.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
            coeff = round_e4m3(coeff / cs) * cs
            xh = base + coeff @ cache["basis"].T
            if codec.cfg.qjl_dim > 0:
                xh = xh + codec.qjl_correct(xe - xh, layer_idx)
            xq = (xh * s).to(x.dtype)
        elif "war" in mode:
            xq = codec.fake_quantize_war(x.float(), cache["comp"], cache.get("basis"),
                                         layer_idx=layer_idx).to(x.dtype)
        elif mode.startswith("turboquant_svd"):
            xq = codec.fake_quantize_svd(x.float(), cache["basis"],
                                         layer_idx=layer_idx).to(x.dtype)
        else:  # turboquant / turboquant_opt / rotation variants
            xq = codec.fake_quantize(x.float(), layer_idx=layer_idx).to(x.dtype)
        return (xq, *args[1:])
    return hook


def _war_comp(module, x_dim: int, block: int) -> torch.Tensor:
    """Per-layer W-aware rounding feedback matrix (offline, from W only)."""
    W = module.weight.detach().float()
    if W.device.type != "cuda":  # MPS lacks linalg ops; fall back to CPU there
        W = W.cpu()
    A = W.T if W.shape[1] == x_dim else W
    return waware_comp(A, block)


def _precompute_aux(model, mx_block: int, device, need_basis: bool, need_comp: bool,
                    eq_scales: dict | None = None, need_imp: bool = False) -> dict:
    """Precompute per-layer side data (SVD bases / feedback matrices) upfront."""
    print("  precomputing per-layer aux (offline)...", end=" ", flush=True)
    t0 = time.time()
    aux = {}
    for i, m in enumerate(mod for mod in model.modules() if _is_linear(mod)):
        x_dim = m.weight.shape[1] if type(m).__name__ == "Linear" else m.weight.shape[0]
        if x_dim % mx_block != 0:
            continue
        entry = {}
        if need_basis:
            k = max(8, x_dim // 16)  # equal side-bit budget as QJL (fp8 coeffs, k=d/16)
            row_scale = eq_scales.get(i) if eq_scales else None
            entry["basis"] = _svd_basis(m, x_dim, k, row_scale).to(device)
        if need_comp:
            entry["comp"] = _war_comp(m, x_dim, mx_block).to(device)
        if need_imp and eq_scales is not None and i in eq_scales:
            # output-domain importance per input channel (equalized space):
            # imp_i = s_i^2 * ||W[:,i]||^2 = diag(W_eq W_eqᵀ)_i
            W = m.weight.detach().float()
            colnorm2 = ((W ** 2).sum(dim=0) if type(m).__name__ == "Linear"
                        else (W ** 2).sum(dim=1))
            s = eq_scales[i].to(colnorm2)
            entry["imp"] = ((s ** 2) * colnorm2).to(device)
        aux[i] = entry
    print(f"done ({len(aux)} layers, {time.time() - t0:.1f}s)", flush=True)
    return aux


_EQ_ALPHAS = (0.25, 0.5, 0.75)


def _pick_eq_scale(sample: torch.Tensor, amax: torch.Tensor, block: int) -> torch.Tensor:
    """Pick the per-layer equalization strength alpha that minimizes actual
    quantization error on a stashed activation sample (offline)."""
    best_s = best_e = None
    for alpha in _EQ_ALPHAS:
        s = amax.clamp_min(1e-5) ** alpha
        e = ((sample - nvfp4_quantize_zp(sample / s, block=block, optclip=True) * s) ** 2).sum()
        if best_e is None or e < best_e:
            best_s, best_e = s, e
    return best_s


@torch.no_grad()
def calibrate_eq_scales(model, calib_ids, max_len: int, device, mx_block: int = 16) -> dict:
    """Per-channel absmax over a few calibration windows -> equalization scales.

    s_i = amax_i^alpha (SmoothQuant-style), with alpha chosen *per layer* from
    ``_EQ_ALPHAS`` by measured quantization error on a stashed sample. Offline
    in deploy — s folds into the producing layer's weights, zero inference cost.
    """
    print("  calibrating per-channel eq scales (offline)...", end=" ", flush=True)
    t0 = time.time()
    amax: dict = {}
    samples: dict = {}
    handles = []
    for i, m in enumerate(mod for mod in model.modules() if _is_linear(mod)):
        def make(idx):
            def h(module, args):
                x = args[0].detach().float()
                flat = x.reshape(-1, x.shape[-1])
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
    scales = {i: _pick_eq_scale(samples[i], a, mx_block) for i, a in amax.items()}
    print(f"done ({len(scales)} layers, {time.time() - t0:.1f}s)", flush=True)
    return scales


def install_hooks(model, mode: str, cfg: TurboQuantConfig, eq_scales: dict | None = None):
    if mode == "fp16":
        return []
    if mode == "nvfp4_rot":      # rotation only, no QJL
        cfg = TurboQuantConfig(mx_block=cfg.mx_block, qjl_dim=0, use_hadamard=True)
    elif mode == "turboquant_rot":  # rotation + per-block QJL
        cfg = TurboQuantConfig(mx_block=cfg.mx_block, qjl_block=cfg.qjl_block,
                               qjl_dim=cfg.qjl_dim, use_hadamard=True)
    elif mode == "turboquant_opt":  # optimal-clip scales + per-block QJL
        cfg = TurboQuantConfig(mx_block=cfg.mx_block, qjl_block=cfg.qjl_block,
                               qjl_dim=cfg.qjl_dim, use_optclip=True)
    elif mode == "turboquant_svd":  # optimal-clip + SVD side-channel (no QJL)
        cfg = TurboQuantConfig(mx_block=cfg.mx_block, qjl_dim=0, use_optclip=True)
    elif mode == "turboquant_svd_qjl":  # SVD side-channel + QJL on the leftover residual
        cfg = TurboQuantConfig(mx_block=cfg.mx_block, qjl_block=cfg.qjl_block,
                               qjl_dim=cfg.qjl_dim, use_optclip=True)
    elif "war" in mode:  # nvfp4_war (zero side bits) / turboquant_war_svd (+SVD)
        cfg = TurboQuantConfig(mx_block=cfg.mx_block, qjl_dim=0)
    elif mode.startswith("nvfp4_eqzp_svd"):  # composed mode; "_qjl" enables QJL,
        cfg = TurboQuantConfig(             # "_oda" enables output-domain scale select
            mx_block=cfg.mx_block, qjl_block=cfg.qjl_block,
            qjl_dim=cfg.qjl_dim if "qjl" in mode else 0)
    codec = TurboQuantActQuantizer(cfg)

    need_basis = "svd" in mode
    need_comp = "war" in mode
    need_imp = "oda" in mode
    aux = {}
    if need_basis or need_comp or need_imp:
        device = next(model.parameters()).device
        aux = _precompute_aux(model, cfg.mx_block, device, need_basis, need_comp,
                              eq_scales if mode.startswith("nvfp4_eq") else None,
                              need_imp=need_imp)
    if mode.startswith("nvfp4_eq"):
        if not eq_scales:
            raise ValueError(f"mode {mode} needs calibrated eq_scales")
        for i, s in eq_scales.items():
            aux.setdefault(i, {})["eq"] = s

    handles = []
    for i, m in enumerate(mod for mod in model.modules() if _is_linear(mod)):
        handles.append(m.register_forward_pre_hook(
            _make_hook(mode, i, codec, cfg.mx_block, aux.get(i))
        ))
    return handles


@torch.no_grad()
def perplexity(model, input_ids, max_len, stride, device) -> float:
    """Standard sliding-window perplexity (HF recipe)."""
    nlls, n_tokens = [], 0
    seq_len = input_ids.size(1)
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(device)
        targets = ids.clone()
        targets[:, :-trg_len] = -100
        out = model(ids, labels=targets)
        # out.loss is mean over (trg_len-1) target tokens
        valid = trg_len - 1
        nlls.append(out.loss.float() * valid)
        n_tokens += valid
        prev_end = end
        if end == seq_len:
            break
    return torch.exp(torch.stack(nlls).sum() / n_tokens).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="cap test tokens (0 = full) for quick smoke")
    ap.add_argument("--qjl-block", type=int, default=128)
    ap.add_argument("--qjl-dim", type=int, default=64)
    ap.add_argument("--w4", action="store_true",
                    help="also quantize linear WEIGHTS to NVFP4 (true W4A4) for quantized modes")
    ap.add_argument("--w4-gptq", action="store_true",
                    help="use GPTQ (Hessian-aware) weight quant instead of naive rounding (implies --w4)")
    ap.add_argument("--awq", action="store_true",
                    help="add AWQ-style salient-channel protection to GPTQ weight quant")
    ap.add_argument("--w4-lowrank", action="store_true",
                    help="add additive low-rank residual correction to GPTQ weight quant (LQER-style)")
    ap.add_argument("--w4-rank-div", type=int, default=16,
                    help="low-rank correction rank = in/this (smaller = higher rank/more capacity)")
    ap.add_argument("--w4-lowrank-fp8", action="store_true",
                    help="store low-rank factors in fp8 (Pareto-honest: keeps the memory axis bounded)")
    ap.add_argument("--w4-lowrank-fp4", action="store_true",
                    help="store low-rank factors in fp4/NVFP4 (keeps '4-bit everywhere'; higher rank at same bytes)")
    ap.add_argument("--w4-rank-alloc", action="store_true",
                    help="water-fill the low-rank budget across layers by sensitivity (same total cost)")
    ap.add_argument("--modes", nargs="+", default=["fp16", "fp8", "nvfp4_raw", "turboquant"])
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32
    print(f"device={device} dtype={dtype} model={args.model}")

    def load_model():
        # transformers>=5 renamed torch_dtype -> dtype; support both.
        try:
            return AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
    if args.limit:
        ids = ids[:, : args.limit]
    print(f"test tokens: {ids.size(1)}")

    use_w4 = args.w4 or args.w4_gptq
    has_quant_mode = any(m not in ("fp16", "fp8") for m in args.modes)

    eq_scales = None
    if any(m.startswith("nvfp4_eq") for m in args.modes):
        train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        calib_ids = tok("\n\n".join(train["text"][:2000]),
                        return_tensors="pt").input_ids[:, : 4 * args.max_len]
        model = load_model().to(device).eval()
        eq_scales = calibrate_eq_scales(model, calib_ids, args.max_len, device)
        del model

    hessians = None
    if args.w4_gptq and has_quant_mode:
        train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        h_calib = tok("\n\n".join(train["text"][:4000]),
                      return_tensors="pt").input_ids[:, : 32 * args.max_len]
        model = load_model().to(device).eval()
        hessians = collect_weight_hessians(model, h_calib, args.max_len, device, n_seq=32)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    cfg = TurboQuantConfig(qjl_block=args.qjl_block, qjl_dim=args.qjl_dim, use_polarquant=False)
    results = {}
    for mode in args.modes:
        model = load_model().to(device).eval()
        if use_w4 and mode not in ("fp16", "fp8"):  # keep fp16/fp8 as clean references
            if args.w4_gptq and args.w4_rank_alloc:
                quantize_weights_gptq_alloc(model, hessians, cfg.mx_block,
                                            rank_div=args.w4_rank_div,
                                            fp8_factors=args.w4_lowrank_fp8)
            elif args.w4_gptq:
                quantize_weights_gptq(model, hessians, cfg.mx_block,
                                      awq=args.awq, lowrank=args.w4_lowrank,
                                      rank_div=args.w4_rank_div,
                                      fp8_factors=args.w4_lowrank_fp8,
                                      fp4_factors=args.w4_lowrank_fp4)
            else:
                quantize_weights_nvfp4(model, cfg.mx_block)
        handles = install_hooks(model, mode, cfg, eq_scales)
        t0 = time.time()
        ppl = perplexity(model, ids, args.max_len, args.stride, device)
        results[mode] = {"perplexity": ppl, "seconds": round(time.time() - t0, 1)}
        print(f"  {mode:12s} PPL = {ppl:8.3f}   ({results[mode]['seconds']}s)")
        for h in handles:
            h.remove()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    RESULTS.mkdir(exist_ok=True)
    payload = {"model": args.model, "config": vars(args), "results": results}
    out = RESULTS / f"perplexity_{args.model.replace('/', '_')}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    if {"fp16", "nvfp4_raw", "turboquant"} <= results.keys():
        f, r, t = (results[m]["perplexity"] for m in ("fp16", "nvfp4_raw", "turboquant"))
        print(f"\nGap closed by QJL: raw NVFP4 +{r - f:.3f} PPL over fp16, "
              f"TurboQuant +{t - f:.3f} PPL  (recovered {100 * (r - t) / max(r - f, 1e-9):.1f}%)")
        if "fp8" in results:
            p8 = results["fp8"]["perplexity"]
            print(f"FP8 target: {p8:.3f} PPL (+{p8 - f:.3f} vs fp16). "
                  f"TurboQuant vs FP8: {t - p8:+.3f} PPL "
                  f"({'matches/beats' if t <= p8 else 'short of'} the production target).")


if __name__ == "__main__":
    main()
