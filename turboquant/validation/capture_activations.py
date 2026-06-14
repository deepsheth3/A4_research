"""Capture real activations + output-metric G for the rounding-ceiling analysis.

Cheap box step (one forward pass, ~2-3 min): for a spread of representative linear
layers, save (a) a sample of the REAL input activations and (b) G = WᵀW (the output
metric) and the per-channel eq scale. Everything heavy (the ceiling computation) then
runs OFFLINE for free via analyze_rounding_ceiling.py — no GPU, infinite re-analysis.

We save G (d_in x d_in), not W, since every ceiling only needs G and x:
  output error = ||(x-q)Wᵀ||² = (x-q) G (x-q)ᵀ.

Run (box):
  python -m turboquant.validation.capture_activations \
    --model unsloth/Meta-Llama-3.1-8B --out caps.pt
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def _is_linear(m) -> bool:
    return type(m).__name__ in ("Linear", "Conv1D")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B")
    ap.add_argument("--n-tokens", type=int, default=256, help="activation rows to save/layer")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--every", type=int, default=28, help="keep every Nth linear (spread)")
    ap.add_argument("--max-din", type=int, default=8192, help="skip layers with larger d_in (G size)")
    ap.add_argument("--max-layers", type=int, default=10)
    ap.add_argument("--out", default="caps.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()

    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(train["text"][:500]), return_tensors="pt").input_ids[:, : args.max_len].to(device)

    # pick a spread of linears (every Nth, square-ish d_in, capped)
    linears = [(n, m) for n, m in model.named_modules()
               if _is_linear(m) and m.weight.shape[1] <= args.max_din]
    chosen = linears[:: args.every][: args.max_layers]
    print(f"capturing {len(chosen)} layers (of {len(linears)} eligible):")
    for n, _ in chosen:
        print(f"  {n}  d_in={dict(chosen)[n].weight.shape[1]}")

    caps: dict = {}
    handles = []
    for name, m in chosen:
        W = m.weight.detach().float()                       # (out, in)
        G = (W.t() @ W).half().cpu()                         # (in, in) output metric
        store = {"G": G, "amax": None, "x": None}

        def make(nm, st):
            @torch.no_grad()
            def h(module, inp):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                a = x.abs().amax(0)
                st["amax"] = a if st["amax"] is None else torch.maximum(st["amax"], a.to(st["amax"]))
                if st["x"] is None:                          # save first n_tokens rows once
                    st["x"] = x[: args.n_tokens].half().cpu()
            return h
        caps[name] = store
        handles.append(m.register_forward_pre_hook(make(name, store)))

    with torch.no_grad():
        for begin in range(0, ids.size(1), args.max_len):
            model(ids[:, begin: begin + args.max_len])
    for hd in handles:
        hd.remove()

    for st in caps.values():                                # eq scale from full-pass amax
        st["amax"] = st["amax"].half().cpu()
    torch.save({"model": args.model, "caps": caps}, args.out)
    mb = sum(st["G"].numel() * 2 + st["x"].numel() * 2 for st in caps.values()) / 1e6
    print(f"\nwrote {args.out}  (~{mb:.0f} MB, {len(caps)} layers) — scp it down for offline analysis")


if __name__ == "__main__":
    main()
