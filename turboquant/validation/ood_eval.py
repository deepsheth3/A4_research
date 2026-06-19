"""Does NVFP4-QAT generalize, or just overfit WikiText?

Plain W4A4 perplexity of the ORIGINAL vs the QAT'd model, on in-distribution
(WikiText) AND out-of-distribution (GSM8K math). The decisive comparison:
  orig-W4A4 vs QAT-W4A4 on OOD.
If QAT-W4A4 beats orig-W4A4 on OOD too -> QAT generalized (the robustness we want).
If QAT helps WikiText but HURTS OOD -> QAT overfit WikiText (the failure mode).

Run:
  python -m turboquant.validation.ood_eval \
     --orig TinyLlama/TinyLlama-1.1B-Chat-v1.0 --qat /root/A4/qat_heavy_ckpt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from turboquant.validation.qat_nvfp4 import replace_linears
from turboquant.validation.hf_perplexity import perplexity

RESULTS = Path(__file__).resolve().parents[2] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--qat", default="/root/A4/qat_heavy_ckpt")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=40000)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = "cuda", torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.orig)
    L, lim = args.max_len, args.limit

    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wiki_ids = tok("\n\n".join(wiki["text"]), return_tensors="pt").input_ids[:, :lim]
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    ood_text = "\n\n".join(x["question"] + "\n" + x["answer"] for x in list(gsm)[:1500])
    ood_ids = tok(ood_text, return_tensors="pt").input_ids[:, :lim]
    print(f"  wiki tokens {wiki_ids.size(1)}  ood(gsm8k) tokens {ood_ids.size(1)}", flush=True)

    def load(mid):
        try:
            return AutoModelForCausalLM.from_pretrained(mid, dtype=dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype)

    @torch.no_grad()
    def evalm(mid):
        m = load(mid).to(device).eval()
        fp16 = {"wiki": round(perplexity(m, wiki_ids, L, L, device), 4),
                "ood": round(perplexity(m, ood_ids, L, L, device), 4)}
        replace_linears(m, 16, quant_act=True)            # plain W4A4 fake-quant
        w4 = {"wiki": round(perplexity(m, wiki_ids, L, L, device), 4),
              "ood": round(perplexity(m, ood_ids, L, L, device), 4)}
        del m
        torch.cuda.empty_cache()
        return fp16, w4

    o_fp16, o_w4 = evalm(args.orig)
    print(f"  [orig] fp16 wiki {o_fp16['wiki']} ood {o_fp16['ood']} | "
          f"W4A4 wiki {o_w4['wiki']} ood {o_w4['ood']}", flush=True)
    q_fp16, q_w4 = evalm(args.qat)
    print(f"  [qat ] fp16 wiki {q_fp16['wiki']} ood {q_fp16['ood']} | "
          f"W4A4 wiki {q_w4['wiki']} ood {q_w4['ood']}", flush=True)

    out = {"orig_fp16": o_fp16, "orig_w4a4": o_w4, "qat_fp16": q_fp16, "qat_w4a4": q_w4}
    out["w4a4_wiki_gain"] = round(o_w4["wiki"] - q_w4["wiki"], 4)   # +ve = QAT better in-dist
    out["w4a4_ood_gain"] = round(o_w4["ood"] - q_w4["ood"], 4)      # +ve = QAT better OOD
    verdict = ("GENERALIZES — QAT-W4A4 beats orig-W4A4 on OOD too" if out["w4a4_ood_gain"] > 0.02
               else "OVERFIT — QAT helps WikiText but not OOD" if out["w4a4_wiki_gain"] > 0.02
               else "no clear effect")
    out["verdict"] = verdict
    print(f"\n  W4A4 in-dist gain (orig-qat) = {out['w4a4_wiki_gain']:+.4f}")
    print(f"  W4A4 OOD     gain (orig-qat) = {out['w4a4_ood_gain']:+.4f}")
    print(f"  VERDICT: {verdict}")

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "ood_eval_qat.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
