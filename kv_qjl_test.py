r"""Does a 1-bit QJL residual correction pay for itself on an NVFP4 KV cache?

Context. The OmniStack KV codec is WHT + INT4 Lloyd-Max + 1-bit QJL. Two of those
stages do not survive contact with NVFP4:

  WHT rotation   -> redundant. Per-16 block scales already isolate outliers locally;
                    this project measured rotation as neutral-to-harmful under
                    microscaling (QuaRot global rotation, per-block Hadamard,
                    channel permutation all washed or hurt).
  Lloyd-Max      -> undeployable. Blackwell's FP4 tensor cores only decode the fixed
                    E2M1 grid, so a fitted codebook cannot be shipped. (And the fixed
                    grid is only ~4% worse than a Gaussian-optimal codebook anyway.)

That leaves NVFP4 + 1-bit QJL. QJL is the only stage that *adds information* rather
than rearranging it, and unlike the rank-r SVD correction it is calibration-free
(seed-regenerated projection, no stored basis) and far cheaper to store: qjl_dim bits
per token per layer, versus 64 fp8 values for the SVD version.

The open question this answers: the rank-64 SVD correction recovered ~56% of the KV4
error. QJL is a 1-bit relaxation of the same idea, so it should recover less. How much
less decides whether wiring it into an attention kernel is worth starting.

Runs on CPU/MPS — QJL needs no linalg, so none of the MPS gaps that block the SVD path
apply here.

  python kv_qjl_test.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from turboquant._omnistack import RademacherQJL
from turboquant.nvfp4 import nvfp4_quantize_zp
from turboquant.validation.hf_perplexity import _is_linear, perplexity

RESULTS = Path(__file__).resolve().parent / "results"


def kv_modules(model):
    return [(n, m) for n, m in model.named_modules()
            if n.endswith(("k_proj", "v_proj")) and _is_linear(m)]


def install_kv(model, block: int, qjl_dim: int = 0, qjl_block: int = 128):
    """NVFP4 the K/V projection outputs; optionally add a 1-bit QJL residual.

    qjl_dim=0 -> plain NVFP4 KV (the uncorrected baseline).
    qjl_dim>0 -> quantize, then encode the residual to 1-bit signs + norm over
                 qjl_block-sized chunks and add the reconstruction back.
    """
    handles = []
    for _name, m in kv_modules(model):
        def hook(module, inp, output):
            if not torch.is_tensor(output) or output.shape[-1] % block:
                return None
            o = output.float()
            q = nvfp4_quantize_zp(o, block=block, optclip=True)
            if qjl_dim > 0:
                r = o - q
                d = r.shape[-1]
                blk = qjl_block if d % qjl_block == 0 else d
                rb = r.reshape(*r.shape[:-1], d // blk, blk)
                qjl = RademacherQJL(head_dim=blk, qjl_dim=qjl_dim)
                signs, norms = qjl.encode(rb)
                q = q + qjl.reconstruct(signs, norms).reshape_as(r)
            return q.to(output.dtype)
        handles.append(m.register_forward_hook(hook))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=60000)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--qjl-block", type=int, default=128)
    ap.add_argument("--qjl-dims", type=int, nargs="+", default=[16, 32, 64])
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32
    print(f"device={device} model={args.model}", flush=True)

    def load():
        try:
            return AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[:, : args.limit]
    L = args.max_len
    print(f"eval tokens: {ids.size(1)}", flush=True)

    results = {}

    def run(tag, qjl_dim):
        model = load().to(device).eval()
        h = install_kv(model, args.block, qjl_dim=qjl_dim, qjl_block=args.qjl_block)
        t0 = time.time()
        ppl = perplexity(model, ids, L, L, device)
        for x in h:
            x.remove()
        results[tag] = {"ppl": ppl, "qjl_dim": qjl_dim,
                        "seconds": round(time.time() - t0, 1)}
        print(f"  {tag:16s} PPL = {ppl:8.4f}  ({results[tag]['seconds']}s)", flush=True)
        del model
        return ppl

    # FP16 reference: no KV hooks at all.
    model = load().to(device).eval()
    t0 = time.time()
    fp16 = perplexity(model, ids, L, L, device)
    results["fp16"] = {"ppl": fp16, "seconds": round(time.time() - t0, 1)}
    print(f"  {'fp16':16s} PPL = {fp16:8.4f}  ({results['fp16']['seconds']}s)", flush=True)
    del model

    raw = run("kv4_raw", 0)
    for d in args.qjl_dims:
        run(f"kv4_qjl{d}", d)

    cost = raw - fp16
    summary = {"fp16": round(fp16, 4), "kv4_raw": round(raw, 4),
               "kv4_cost": round(cost, 4)}
    for d in args.qjl_dims:
        p = results[f"kv4_qjl{d}"]["ppl"]
        summary[f"qjl{d}_recovered_frac"] = round((raw - p) / cost, 4) if abs(cost) > 1e-9 else None
        # bits/token/layer for the side channel vs the KV vector it corrects
        summary[f"qjl{d}_overhead_bits"] = d
    print("\n  " + json.dumps(summary, indent=2).replace("\n", "\n  "), flush=True)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"kv_qjl_{args.model.replace('/', '_')}.json"
    out.write_text(json.dumps({"model": args.model, "block": args.block,
                               "qjl_block": args.qjl_block,
                               "results": results, "summary": summary}, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
