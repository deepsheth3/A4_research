"""OOD head-to-head: the calibrated EQ/SVD/QJL *stack* vs *QAT*, in-dist + OOD.

The hypothesis (user's): QAT generalizes because it pushes correction into the
weights, while the post-hoc stack (eq scales + SVD side-channel + QJL) is *fit* to
the calibration distribution and may degrade out-of-distribution. This harness tests
it directly: both corrections are derived from WikiText, then both are evaluated on
WikiText (in-dist), GSM8K (OOD math), and C4 (OOD web) — reporting *both* perplexity
and spec-decode acceptance (1 - TV vs the FP16 target), which is the metric that
becomes throughput.

Three drafts, all W4A4 (+ optional KV4):
  - ptq   : raw NVFP4 round, no correction        (the floor)
  - stack : eq (calibrated on wiki) + SVD + QJL    (the post-hoc correction)
  - qat   : a trained QAT checkpoint               (correction in the weights)

The stack is calibrated ONLY on wiki, so its OOD rows measure transfer — same as QAT,
which is trained only on wiki. A fair, like-for-like generalization test.

Run (block 16 and 32 for the sweep):
  python -m turboquant.validation.ood_stack_vs_qat --mx-block 16 \
      --target TinyLlama/TinyLlama-1.1B-Chat-v1.0 --qat-ckpt qat_heavy_ckpt
  python -m turboquant.validation.ood_stack_vs_qat --mx-block 32 ... (same)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from turboquant.config import TurboQuantConfig
from turboquant.validation.qat_nvfp4 import replace_linears
from turboquant.validation.hf_perplexity import (
    calibrate_eq_scales, install_hooks, install_kv_hooks, perplexity,
)
from turboquant.validation.acceptance import acceptance_stats, speedup

RESULTS = Path(__file__).resolve().parents[2] / "results"


def _load_corpora(tok, lim: int):
    """wiki (in-dist), gsm8k (OOD math), c4 (OOD web). C4 is best-effort."""
    from datasets import load_dataset
    corp = {}
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    corp["wiki"] = tok("\n\n".join(wiki["text"]), return_tensors="pt").input_ids[:, :lim]
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    corp["gsm8k"] = tok("\n\n".join(x["question"] + "\n" + x["answer"]
                                    for x in list(gsm)[:1500]),
                        return_tensors="pt").input_ids[:, :lim]
    try:  # C4 is large/streamed; skip cleanly if unavailable offline
        c4 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        txt = []
        for i, x in enumerate(c4):
            txt.append(x["text"])
            if i >= 2000:
                break
        corp["c4"] = tok("\n\n".join(txt), return_tensors="pt").input_ids[:, :lim]
    except Exception as e:  # noqa: BLE001 — offline / gated is fine, just report
        print(f"  (c4 unavailable: {type(e).__name__}; skipping OOD-web)", flush=True)
    return corp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--qat-ckpt", default=None, help="path to a folded QAT nn.Linear checkpoint")
    ap.add_argument("--mx-block", type=int, default=16, help="MX4 block (16=NVFP4, 32=MXFP4-grained)")
    ap.add_argument("--stack-mode", default="nvfp4_eqzp_svd_qjl",
                    help="install_hooks mode for the post-hoc stack draft")
    ap.add_argument("--kv4", action="store_true", help="also NVFP4 the K/V cache for all drafts")
    ap.add_argument("--qjl-block", type=int, default=128)
    ap.add_argument("--qjl-dim", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=40000)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--cost-ratio", type=float, default=0.3)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32
    tok = AutoTokenizer.from_pretrained(args.target)
    L = args.max_len

    def load(mid):
        try:
            return AutoModelForCausalLM.from_pretrained(mid, dtype=dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype)

    corpora = _load_corpora(tok, args.limit)
    windows = {k: [v[:, i * L:(i + 1) * L].to(device) for i in range(v.size(1) // L)]
               for k, v in corpora.items()}
    print(f"  block={args.mx_block} corpora={ {k: v.size(1) for k, v in corpora.items()} }", flush=True)

    # Target FP16 logits, cached per corpus (the acceptance reference).
    target = load(args.target).to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    V = target.config.vocab_size

    @torch.no_grad()
    def logits_flat(model, corp):
        return [model(x).logits[:, :-1, :].reshape(-1, V) for x in windows[corp]]
    tgt = {c: logits_flat(target, c) for c in corpora}

    # Calibrate the stack's eq scales on WIKI only (in-dist), so OOD rows test transfer.
    calib_ids = corpora["wiki"][:, :4 * L]
    eq_scales = calibrate_eq_scales(target, calib_ids, L, device, mx_block=args.mx_block)

    cfg = TurboQuantConfig(mx_block=args.mx_block, qjl_block=args.qjl_block,
                           qjl_dim=args.qjl_dim, use_polarquant=False)

    drafts = ["ptq", "stack"] + (["qat"] if args.qat_ckpt else [])
    out = {"target": args.target, "mx_block": args.mx_block, "stack_mode": args.stack_mode,
           "kv4": args.kv4, "qat_ckpt": args.qat_ckpt, "gamma": args.gamma,
           "cost_ratio": args.cost_ratio, "drafts": {}}

    @torch.no_grad()
    def evaluate(model, label):
        row = {}
        for c in corpora:
            a = sum(acceptance_stats(d, t)[0] for d, t in zip(logits_flat(model, c), tgt[c])) / len(windows[c])
            ppl = perplexity(model, corpora[c], L, L, device)
            row[c] = {"alpha": round(float(a), 4), "ppl": round(ppl, 4),
                      "speedup": round(speedup(float(a), args.gamma, args.cost_ratio), 3)}
        out["drafts"][label] = row
        line = "  ".join(f"{c} α{row[c]['alpha']:.4f}/ppl{row[c]['ppl']:.3f}" for c in corpora)
        print(f"  [{label:5s}] {line}", flush=True)

    for label in drafts:
        if label == "qat":
            model = load(args.qat_ckpt).to(device).eval()
            replace_linears(model, args.mx_block, quant_act=True)
        else:
            model = load(args.target).to(device).eval()
            if label == "ptq":
                replace_linears(model, args.mx_block, quant_act=True)
            else:  # stack: post-hoc correction via forward hooks
                install_hooks(model, args.stack_mode, cfg, eq_scales=eq_scales)
        if args.kv4:
            install_kv_hooks(model, mx_block=args.mx_block)
        evaluate(model, label)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n  === OOD GENERALIZATION (acceptance: in-dist wiki vs OOD) ===")
    for label, row in out["drafts"].items():
        ood = [c for c in corpora if c != "wiki"]
        deltas = "  ".join(f"{c} {row[c]['alpha'] - row['wiki']['alpha']:+.4f}" for c in ood)
        print(f"  {label:6s} wiki α{row['wiki']['alpha']:.4f} | OOD drop  {deltas}")

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"ood_stack_vs_qat_b{args.mx_block}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
