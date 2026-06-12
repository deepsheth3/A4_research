"""GSM8K accuracy evaluation with TurboQuant FP4 activation quantization.

Two modes only:
  fp16               — no quantization (accuracy ceiling)
  nvfp4_eqzp_svd_qjl — full TurboQuant stack (the paper's candidate)

8-shot chain-of-thought prompting (first 8 train examples), greedy decoding,
final-answer exact match. Calibration reuses wikitext-2 train (same as PPL eval).

Smoke test (limit to 50 problems):
    python -m turboquant.validation.gsm8k_eval --model gpt2 --limit 50
Real run (H100):
    python -m turboquant.validation.gsm8k_eval --model unsloth/Meta-Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from turboquant.config import TurboQuantConfig
from turboquant.validation.hf_perplexity import (
    calibrate_eq_scales, install_hooks, quantize_weights_nvfp4)

RESULTS = Path(__file__).resolve().parents[2] / "results"

_ANS_RE = re.compile(r"####\s*([\d,]+)")


class _StopAtHash(StoppingCriteria):
    """Stop generation as soon as '####' appears in the new tokens."""
    def __init__(self, hash_token_ids: list[int]):
        self._ids = hash_token_ids
        self._n = len(hash_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        if input_ids.shape[1] < self._n:
            return False
        return input_ids[0, -self._n:].tolist() == self._ids


def _extract_answer(text: str) -> str | None:
    """Pull the answer after '####', falling back to the last bare integer."""
    m = _ANS_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"\b\d+\b", text)
    return nums[-1] if nums else None


def _build_prompt(exemplars: list, question: str) -> str:
    parts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in exemplars]
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


@torch.no_grad()
def evaluate_gsm8k(model, tokenizer, problems: list, exemplars: list,
                   device: str, max_new_tokens: int = 256,
                   stop_criteria: StoppingCriteriaList | None = None) -> tuple[float, int, int]:
    correct = total = 0
    for prob in problems:
        prompt = _build_prompt(exemplars, prob["question"])
        gold = prob["answer"].split("####")[-1].strip().replace(",", "")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stop_criteria,
        )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = _extract_answer(gen)
        if pred == gold:
            correct += 1
        total += 1
        if total % 25 == 0:
            print(f"    {total}/{len(problems)}  running acc={100*correct/total:.1f}%", flush=True)
    return correct / total if total else 0.0, correct, total


def _load_model(model_name: str, dtype):
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B")
    ap.add_argument("--limit", type=int, default=0, help="cap test problems (0 = full 1319)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--qjl-block", type=int, default=128)
    ap.add_argument("--qjl-dim", type=int, default=64)
    ap.add_argument("--w4", action="store_true",
                    help="also quantize linear WEIGHTS to NVFP4 (true W4A4) for quantized modes")
    ap.add_argument("--modes", nargs="+", default=["fp16", "nvfp4_eqzp_svd_qjl"])
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype} model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gsm = load_dataset("openai/gsm8k", "main")
    exemplars = list(gsm["train"].select(range(8)))
    test_problems = list(gsm["test"])
    if args.limit:
        test_problems = test_problems[: args.limit]
    print(f"GSM8K: {len(test_problems)} test problems, 8-shot CoT, greedy decoding")

    # Stop as soon as '####' appears — answers land at ~80-100 tokens, not 256
    hash_ids = tok.encode("####", add_special_tokens=False)
    stop_criteria = StoppingCriteriaList([_StopAtHash(hash_ids)])

    eq_scales = None
    if any(m.startswith("nvfp4_eq") for m in args.modes):
        print("\ncalibrating eq scales (wikitext-2 train)...")
        wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        calib_ids = tok(
            "\n\n".join(wikitext["text"][:2000]), return_tensors="pt"
        ).input_ids[:, : 4 * 1024]
        model = _load_model(args.model, dtype).to(device).eval()
        eq_scales = calibrate_eq_scales(model, calib_ids, 1024, device)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    cfg = TurboQuantConfig(qjl_block=args.qjl_block, qjl_dim=args.qjl_dim, use_polarquant=False)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"gsm8k_{args.model.replace('/', '_')}.json"
    results = {}
    for mode in args.modes:
        print(f"\n--- {mode} ---")
        model = _load_model(args.model, dtype).to(device).eval()
        if args.w4 and mode not in ("fp16", "fp8"):  # keep fp16/fp8 as clean references
            quantize_weights_nvfp4(model, cfg.mx_block)
        handles = install_hooks(model, mode, cfg, eq_scales)
        t0 = time.time()
        acc, correct, total = evaluate_gsm8k(
            model, tok, test_problems, exemplars, device, args.max_new_tokens, stop_criteria
        )
        elapsed = time.time() - t0
        results[mode] = {"accuracy": round(acc * 100, 2), "correct": correct,
                         "total": total, "seconds": round(elapsed, 1)}
        print(f"  {mode}: {acc*100:.1f}%  ({correct}/{total})  ({elapsed:.0f}s)")
        # Flush to disk after EVERY mode so a mid-run box death still saves what's done.
        out.write_text(json.dumps(
            {"model": args.model, "config": vars(args), "results": results}, indent=2))
        print(f"  wrote {out}", flush=True)
        for h in handles:
            h.remove()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if "fp16" in results and "nvfp4_eqzp_svd_qjl" in results:
        b = results["fp16"]["accuracy"]
        q = results["nvfp4_eqzp_svd_qjl"]["accuracy"]
        print(f"\nFP16:          {b:.1f}%")
        print(f"TurboQuant FP4: {q:.1f}%")
        print(f"Delta:          {q - b:+.1f} pts  "
              f"({'within 1 pt — parity' if abs(q - b) <= 1.0 else 'meaningful drop'})")


if __name__ == "__main__":
    main()
