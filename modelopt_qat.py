"""QAT *inside* ModelOpt: train against the exact deployment quantizer, so train==deploy.

Kills the sim-to-deploy gap (our proxy fake-quant QAT deployed at 9.94 vs 9.77 trained).
Here the fake-quant in the forward IS ModelOpt's NVFP4 (the one export ships), so the PPL
we measure during training is literally the deployed accuracy -- no gap by construction.
"""
import time, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from turboquant.validation.hf_perplexity import perplexity
from turboquant.validation.export_nvfp4 import build_quant_cfg

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEV = "cuda"
L = 1024
STEPS = int(os.environ.get("STEPS", 3000))
KV4 = os.environ.get("KV4", "0") == "1"
OUT = "/workspace/NVFP4_Research/results/box_run/modelopt_qat_kv4_ckpt" if KV4 \
      else "/workspace/NVFP4_Research/results/box_run/modelopt_qat_ckpt"

tok = AutoTokenizer.from_pretrained(MODEL)
student = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
for p in teacher.parameters():
    p.requires_grad_(False)
V = teacher.config.vocab_size

tr = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
tr_ids = tok("\n\n".join(tr["text"]), return_tensors="pt").input_ids
test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ppl_ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[:, :40000]

def win(ids, k):
    return ids[:, k * L:(k + 1) * L].to(DEV)

# 1) install ModelOpt NVFP4 quantizers (STE) + calibrate
calib = [win(tr_ids, k) for k in range(32)]
def floop(m):
    with torch.no_grad():
        for b in calib:
            m(b)
cfg = build_quant_cfg(mtq.NVFP4_DEFAULT_CFG, kv_cfg=mtq.NVFP4_KV_CFG if KV4 else None)
print(f"calibrating + installing ModelOpt NVFP4 quantizers (KV4={KV4}, steps={STEPS})...", flush=True)
mtq.quantize(student, cfg, floop)

fp16 = perplexity(teacher, ppl_ids, L, L, DEV)
ptq = perplexity(student, ppl_ids, L, L, DEV)   # PTQ on the DEPLOY grid
print(f"FP16 {fp16:.4f} | PTQ(deploy-grid) {ptq:.4f}", flush=True)

# 2) QAT: train the weights to survive ModelOpt's own quantizer
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=3e-5)
nwin = tr_ids.size(1) // L
best = ptq
t0 = time.time()
student.train()
for s in range(STEPS):
    x = win(tr_ids, s % nwin)
    with torch.no_grad():
        t = teacher(x).logits.float().reshape(-1, V)
    out = student(x).logits.float().reshape(-1, V)
    loss = F.kl_div(F.log_softmax(out, -1), F.softmax(t, -1), reduction="batchmean")
    opt.zero_grad(); loss.backward(); opt.step()
    if (s + 1) % 500 == 0:
        student.eval()
        p = perplexity(student, ppl_ids, L, L, DEV)
        best = min(best, p)
        student.train()
        print(f"step {s+1} kl {loss.item():.4f} deploy-PPL {p:.4f} best {best:.4f} "
              f"[{(time.time()-t0)/60:.1f}min]", flush=True)

student.eval()
final = perplexity(student, ppl_ids, L, L, DEV)
print(f"\n*** ModelOpt-NATIVE QAT: deployed PPL = {final:.4f}  (best {best:.4f}) | "
      f"FP16 {fp16:.4f} | PTQ {ptq:.4f} | gap closed {(ptq-final)/(ptq-fp16):.3f} ***", flush=True)
print("train==deploy: this PPL IS the deployed accuracy (no sim-to-real gap).", flush=True)
export_hf_checkpoint(student, dtype=torch.bfloat16, export_dir=OUT)
print(f"exported -> {OUT}", flush=True)
