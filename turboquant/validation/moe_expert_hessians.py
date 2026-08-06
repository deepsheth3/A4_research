"""Token-starved MoE experts: does low routing load wreck the calibration Hessian?

THE HYPOTHESIS
--------------
Our method (equalization + additive low-rank correction) needs a per-layer Hessian
H = E[xᵀx] to solve for the rank-r correction that is optimal in the OUTPUT metric.
On a dense model every token flows through every matrix, so H is estimated from
hundreds of thousands of rows — plenty. In an MoE with top-k routing, each expert
only sees the tokens routed to IT.

OLMoE-1B-7B: 64 experts/layer, top-8 routing, expert input dim 2048.
  -> an average expert sees 8/64 = 12.5% of tokens
  -> H is (2048, 2048), so an expert needs >=2048 routed tokens just to be full rank
  -> at 32x1024 calib tokens the average expert gets ~4k rows: barely 2x the
     threshold, and any below-average expert falls UNDER it.

A rank-deficient H makes its inverse mostly damping-term noise, so the correction
solved against it is fit to noise rather than signal. This script measures whether
that actually happens, and how hard, BEFORE spending effort on mitigations
(load-adaptive damping / shrinkage toward the layer-mean H / skipping the correction
on the worst experts).

WHAT IT DOES
------------
transformers >= 5.x stores MoE experts as batched 3D nn.Parameters inside a single
`*Experts` module (NOT per-expert nn.Linear), so the repo's usual
`model.modules()` + `_is_linear` walk finds ZERO expert weights. Instead we hook the
`Experts` module itself: its forward already receives (hidden_states, top_k_index,
top_k_weights), i.e. the router's decisions — so ONE hook yields both the per-expert
token count and the exact token slice needed for that expert's Hessian.

Run:
  python -m turboquant.validation.moe_expert_hessians --model allenai/OLMoE-1B-7B-0924
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parents[2] / "results"


def find_expert_modules(model):
    """Return [(name, module)] for every batched-expert module in the model.

    Matches transformers' `*Experts` classes (OlmoeExperts, Qwen2MoeExperts, ...),
    identified structurally (a 3D gate_up_proj/down_proj parameter) rather than by
    class name, so it survives renames across architectures.
    """
    found = []
    for name, m in model.named_modules():
        gu = getattr(m, "gate_up_proj", None)
        dp = getattr(m, "down_proj", None)
        if isinstance(gu, torch.nn.Parameter) and gu.dim() == 3:
            found.append((name, m))
        elif isinstance(dp, torch.nn.Parameter) and dp.dim() == 3:
            found.append((name, m))
    return found


@torch.no_grad()
def collect_expert_hessians(model, calib_ids, max_len, device, n_windows, num_experts):
    """Per (layer, expert): routed-token count and H = sum(xᵀx) over ITS tokens.

    One forward-pre-hook per Experts module. `top_k_index` is the router's choice,
    so no separate router hook is needed.
    """
    mods = find_expert_modules(model)
    if not mods:
        raise RuntimeError(
            "no batched-expert modules found — is this an MoE, and does this "
            "transformers version use the 3D-parameter Experts layout?")
    print(f"  found {len(mods)} expert modules: {mods[0][0]} ...", flush=True)

    counts: dict = {}   # name -> (E,) token counts
    hess: dict = {}     # name -> (E, d, d) accumulated xᵀx

    handles = []
    for name, m in mods:
        d = m.gate_up_proj.shape[-1] if hasattr(m, "gate_up_proj") else m.down_proj.shape[-1]

        def mk(nm, dim):
            def hook(module, args, kwargs):
                # forward(hidden_states, top_k_index, top_k_weights) — positional or kw
                if len(args) >= 2:
                    hs, idx = args[0], args[1]
                else:
                    hs = args[0] if args else kwargs.get("hidden_states")
                    idx = kwargs.get("top_k_index")
                if hs is None or idx is None:
                    return None
                x = hs.reshape(-1, hs.shape[-1]).float()
                flat = idx.reshape(-1)                       # (tokens*top_k,)
                tok = torch.arange(idx.shape[0], device=idx.device).repeat_interleave(
                    idx.shape[-1]) if idx.dim() > 1 else torch.arange(
                    flat.numel(), device=idx.device)
                if nm not in hess:
                    hess[nm] = torch.zeros(num_experts, dim, dim, device=x.device,
                                           dtype=torch.float32)
                    counts[nm] = torch.zeros(num_experts, device=x.device,
                                             dtype=torch.long)
                for e in flat.unique().tolist():
                    sel = tok[flat == e]
                    if sel.numel() == 0:
                        continue
                    xe = x[sel]
                    hess[nm][e] += xe.t() @ xe
                    counts[nm][e] += xe.shape[0]
                return None
            return hook
        handles.append(m.register_forward_pre_hook(mk(name, d), with_kwargs=True))

    t0 = time.time()
    n = 0
    for begin in range(0, calib_ids.size(1), max_len):
        if n >= n_windows:
            break
        model(calib_ids[:, begin:begin + max_len].to(device))
        n += 1
        print(f"    window {n}/{n_windows} ({time.time()-t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()
    print(f"  collected over {n} windows in {time.time()-t0:.1f}s", flush=True)
    return counts, hess


def conditioning(H: torch.Tensor, count: int) -> dict:
    """Rank/conditioning diagnostics for one expert's Hessian."""
    d = H.shape[0]
    ev = torch.linalg.eigvalsh(H.double().cpu())
    ev = ev.clamp_min(0)
    top = ev[-1].item()
    if top <= 0:
        return {"tokens": count, "dim": d, "top_eig": 0.0, "eff_rank": 0,
                "cond": float("inf"), "rank_deficient": True}
    # "effective rank": eigenvalues that carry real signal relative to the largest.
    eff = int((ev > top * 1e-6).sum().item())
    smallest_nonzero = ev[ev > top * 1e-12]
    cond = (top / smallest_nonzero[0].item()) if smallest_nonzero.numel() else float("inf")
    return {"tokens": int(count), "dim": d, "top_eig": top, "eff_rank": eff,
            "cond": cond,
            # the headline: fewer routed tokens than dimensions => H CANNOT be full rank
            "rank_deficient": bool(count < d or eff < d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--n-windows", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4,
                    help="how many MoE layers to analyze (eigendecomps are the cost)")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype} model={args.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(device).eval()

    cfg = model.config
    E = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts")
    topk = getattr(cfg, "num_experts_per_tok", 0)
    print(f"  experts={E} top_k={topk} layers={cfg.num_hidden_layers}", flush=True)

    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    calib = tok("\n\n".join(train["text"][:4000]),
                return_tensors="pt").input_ids[:, : args.n_windows * args.max_len]

    counts, hess = collect_expert_hessians(model, calib, args.max_len, device,
                                           args.n_windows, E)

    total_tokens = args.n_windows * args.max_len
    report = {"model": args.model, "num_experts": E, "top_k": topk,
              "calib_tokens": total_tokens, "layers": {}}

    names = sorted(hess.keys())[: args.layers]
    for nm in names:
        c, H = counts[nm], hess[nm]
        print(f"\n  {nm}: analyzing {E} experts (dim {H.shape[-1]})", flush=True)
        per = []
        for e in range(E):
            per.append(conditioning(H[e], int(c[e].item())))
        report["layers"][nm] = per
        cs = sorted(per, key=lambda r: r["tokens"])
        d = per[0]["dim"]
        starved = [r for r in per if r["tokens"] < d]
        print(f"    tokens/expert: min {cs[0]['tokens']} / median "
              f"{cs[len(cs)//2]['tokens']} / max {cs[-1]['tokens']}  (dim={d})")
        print(f"    experts with FEWER tokens than dimensions: {len(starved)}/{E}")
        print(f"    rank-deficient H: {sum(r['rank_deficient'] for r in per)}/{E}")
        q = len(cs) // 4 or 1
        lo = sum(r["eff_rank"] for r in cs[:q]) / q
        hi = sum(r["eff_rank"] for r in cs[-q:]) / q
        print(f"    eff_rank  bottom-quartile {lo:.0f}  vs  top-quartile {hi:.0f}"
              f"   (of {d})")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"moe_expert_hessians_{args.model.replace('/', '_')}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
