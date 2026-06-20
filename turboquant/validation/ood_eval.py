"""Acceptance-first eval: does QAT raise spec-decode ACCEPTANCE, in-dist AND OOD?

Acceptance (1 - TV vs the FP16 target) is the metric that matters — it becomes
throughput, with quality guaranteed by the verifier. We compare W4A4 drafts (original
vs KL-QAT vs acceptance/TV-QAT) against the FP16 target, on WikiText and GSM8K (OOD).
PPL is reported only as a sanity check.

Run:
  python -m turboquant.validation.ood_eval --target TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
     --drafts orig:TinyLlama/TinyLlama-1.1B-Chat-v1.0,kl:/root/A4/qat_heavy_ckpt,tv:/root/A4/qat_tv_ckpt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from turboquant.validation.qat_nvfp4 import replace_linears
from turboquant.validation.hf_perplexity import perplexity
from turboquant.validation.acceptance import acceptance_stats, speedup

RESULTS = Path(__file__).resolve().parents[2] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--drafts", default="orig:TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=40000)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--cost-ratio", type=float, default=0.3)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = "cuda", torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.target)
    L, lim = args.max_len, args.limit

    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wiki_ids = tok("\n\n".join(wiki["text"]), return_tensors="pt").input_ids[:, :lim]
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    ood_ids = tok("\n\n".join(x["question"] + "\n" + x["answer"] for x in list(gsm)[:1500]),
                  return_tensors="pt").input_ids[:, :lim]
    corpora = {"wiki": wiki_ids, "ood_gsm8k": ood_ids}
    windows = {k: [v[:, i * L:(i + 1) * L].to(device) for i in range(v.size(1) // L)]
               for k, v in corpora.items()}
    print(f"  wiki {wiki_ids.size(1)} tok, ood {ood_ids.size(1)} tok", flush=True)

    def load(mid):
        try:
            return AutoModelForCausalLM.from_pretrained(mid, dtype=dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype)

    target = load(args.target).to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    V = target.config.vocab_size

    @torch.no_grad()
    def tgt_logits(corp):                                  # cache target logits per corpus
        return [target(x).logits[:, :-1, :].reshape(-1, V) for x in windows[corp]]
    tgt = {c: tgt_logits(c) for c in corpora}

    out = {"target": args.target, "gamma": args.gamma, "cost_ratio": args.cost_ratio, "drafts": {}}
    for spec in args.drafts.split(","):
        label, path = spec.split(":", 1)
        d = load(path).to(device).eval()
        replace_linears(d, 16, quant_act=True)            # W4A4 draft
        row = {}
        with torch.no_grad():
            for c in corpora:
                a = sum(acceptance_stats(d(x).logits[:, :-1, :].reshape(-1, V), t)[0]
                        for x, t in zip(windows[c], tgt[c])) / len(windows[c])
                ppl = perplexity(d, corpora[c], L, L, device)
                row[c] = {"alpha": round(a, 4), "ppl": round(ppl, 4),
                          "speedup": round(speedup(a, args.gamma, args.cost_ratio), 3)}
        out["drafts"][label] = row
        del d
        torch.cuda.empty_cache()
        print(f"  [{label:5s}] wiki alpha {row['wiki']['alpha']} (spd {row['wiki']['speedup']}x) | "
              f"ood alpha {row['ood_gsm8k']['alpha']} (spd {row['ood_gsm8k']['speedup']}x)", flush=True)

    print("\n  === ACCEPTANCE (W4A4 draft vs FP16 target) ===")
    for label, row in out["drafts"].items():
        print(f"  {label:6s}  wiki {row['wiki']['alpha']:.4f}  ood {row['ood_gsm8k']['alpha']:.4f}")
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "ood_accept_eval.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
