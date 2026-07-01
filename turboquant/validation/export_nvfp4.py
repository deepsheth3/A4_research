"""Export a (QAT'd) HF model to a real NVFP4 checkpoint via NVIDIA ModelOpt.

This is the piece that turns our fake-quant *research* into a *deployable* artifact:
our `qat_nvfp4.py --save-dir` writes BF16 weights trained to survive NVFP4; this
script re-quantizes them with ModelOpt's blessed NVFP4 and writes a checkpoint that
TensorRT-LLM / vLLM load directly (packed E2M1 + fp8-e4m3 block scales + config).

Grid parity is proven on CPU in tests/test_modelopt_parity.py: ModelOpt's NVFP4
(E2M1, block 16, e4m3 scale) matches our fake-quant, so this export preserves the
accuracy we measured (at worst the deploy grid is slightly *better*).

Needs a Blackwell box with modelopt installed:
    pip install nvidia-modelopt[torch]
    python -m turboquant.validation.export_nvfp4 \
        --model qat_ckpt_dir --out nvfp4_ckpt_dir --kv4 --ignore-boundary

Verified API (NVIDIA/Model-Optimizer @ main):
    import modelopt.torch.quantization as mtq
    mtq.NVFP4_DEFAULT_CFG   # {"algorithm":"max","quant_cfg":[...]}, lm_head already excluded
    mtq.NVFP4_KV_CFG        # partial cfg that enables NVFP4 KV-cache quantizers
    mtq.quantize(model, cfg, forward_loop)
    from modelopt.torch.export import export_hf_checkpoint
    export_hf_checkpoint(model, dtype=torch.bfloat16, export_dir=out)
"""
from __future__ import annotations

import argparse
import copy
import re


def layer_indices_from_names(names) -> list[int]:
    """Decoder-block indices present in a list of module names ('...layers.{i}...')."""
    idx = {int(m.group(1)) for n in names if (m := re.search(r"\.layers\.(\d+)\.", n))}
    return sorted(idx)


def build_quant_cfg(base_cfg: dict, *, boundary_layers=None, kv_cfg: dict | None = None) -> dict:
    """Compose the final ModelOpt QuantizeConfig (pure dict manip — CPU-testable).

    - starts from ``base_cfg`` (e.g. mtq.NVFP4_DEFAULT_CFG),
    - optionally merges ``kv_cfg`` (mtq.NVFP4_KV_CFG) to add KV-cache quantizers,
    - appends ``enable: False`` entries for each boundary decoder block so it stays
      BF16. Entries appended last take precedence, matching ModelOpt's own
      default_disabled_quantizers pattern. (lm_head is already disabled by the
      NVFP4 default, so it needs no entry here.)
    """
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("quant_cfg", [])
    if kv_cfg:
        cfg["quant_cfg"] = cfg["quant_cfg"] + list(kv_cfg.get("quant_cfg", []))
    for i in boundary_layers or []:
        cfg["quant_cfg"].append({"quantizer_name": f"*layers.{i}.*", "enable": False})
    return cfg


def _eval_ppl(model, tok, device, max_len=1024, limit=40000):
    """WikiText PPL of the ModelOpt-quantized model, via our validated perplexity().
    Run *after* mtq.quantize (fake-quant on the deploy grid) — this is the headline:
    the deployed QAT'd NVFP4 model's real accuracy, using the same harness that
    produced our fake-quant numbers, so it's directly comparable."""
    from datasets import load_dataset
    from turboquant.validation.hf_perplexity import perplexity

    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[:, :limit]
    return round(perplexity(model, ids, max_len, max_len, device), 4)


def _calib_loop(model, tok, device, n_samples=32, max_len=512):
    """Small WikiText forward loop for ModelOpt amax calibration."""
    import torch
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if t.strip()][:n_samples]

    def forward_loop(m):
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len)
            with torch.no_grad():
                m(ids["input_ids"].to(device))

    return forward_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF dir/name of the (QAT'd) BF16 model")
    ap.add_argument("--out", required=True, help="output dir for the NVFP4 checkpoint")
    ap.add_argument("--kv4", action="store_true", help="also quantize the KV cache (NVFP4_KV_CFG)")
    ap.add_argument("--ignore-boundary", action="store_true",
                    help="keep first+last decoder block in BF16 (lm_head already excluded)")
    ap.add_argument("--calib-samples", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--eval-ppl", action="store_true",
                    help="measure deployed-QAT WikiText PPL on the quantized model (headline)")
    args = ap.parse_args()

    import torch
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "ModelOpt NVFP4 export needs a CUDA (Blackwell) GPU"
    device = "cuda"

    print(f"loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device
    )

    boundary = []
    if args.ignore_boundary:
        idx = layer_indices_from_names([n for n, _ in model.named_modules()])
        boundary = [idx[0], idx[-1]] if idx else []
        print(f"  keeping decoder blocks {boundary} + lm_head in BF16")

    cfg = build_quant_cfg(
        mtq.NVFP4_DEFAULT_CFG,
        boundary_layers=boundary,
        kv_cfg=mtq.NVFP4_KV_CFG if args.kv4 else None,
    )

    print("calibrating + quantizing (NVFP4) ...", flush=True)
    mtq.quantize(model, cfg, _calib_loop(model, tok, device, args.calib_samples, args.max_len))
    mtq.print_quant_summary(model)

    if args.eval_ppl:
        ppl = _eval_ppl(model, tok, device)
        print(f"*** DEPLOYED QAT NVFP4 WikiText PPL: {ppl} ***", flush=True)

    print(f"exporting NVFP4 checkpoint -> {args.out}", flush=True)
    export_hf_checkpoint(model, dtype=torch.bfloat16, export_dir=args.out)
    tok.save_pretrained(args.out)
    print("done. load in TRT-LLM/vLLM directly from", args.out, flush=True)


if __name__ == "__main__":
    main()
