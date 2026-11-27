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

MODEL = os.environ.get("MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEV = "cuda"
L = 1024
STEPS = int(os.environ.get("STEPS", 3000))
KV4 = os.environ.get("KV4", "0") == "1"
GSM8K = os.environ.get("GSM8K", "0") == "1"
GSM8K_FEWSHOT = int(os.environ.get("GSM8K_FEWSHOT", 5))
GSM8K_LIMIT = int(os.environ.get("GSM8K_LIMIT", 0))  # 0 = full lm-eval test set
# Skip the FP16 teacher's GSM8K pass when a published score (same task/fewshot config)
# is already the comparison point -- the teacher model is still loaded/used as the
# QAT distillation target either way, this only skips its (expensive) generation eval.
SKIP_FP16_GSM8K = os.environ.get("SKIP_FP16_GSM8K", "0") == "1"
SEED = os.environ.get("SEED")   # unset = original sequential window order; set = shuffled order
# Activation treatment DURING TRAINING. Deployment (eval + export) is always the configured
# quantizers, so every arm is scored as the same all-NVFP4 model.
#   a4    quantize activations with STE (original behaviour)
#   a16   weights-only training; activation quantizers on only for eval/export
#   noise a16 + the quantizer's own error with random signs (zero-mean, exact gradient)
ACT_TRAIN = os.environ.get("ACT_TRAIN", "a4")
assert ACT_TRAIN in ("a4", "a16", "noise"), ACT_TRAIN
_MODEL_TAG = MODEL.replace("/", "_")
if ACT_TRAIN != "a4" or SEED is not None:      # default run keeps its original paths
    _MODEL_TAG += f"_{ACT_TRAIN}_s{SEED}"
OUT = f"/workspace/NVFP4_Research/results/box_run/modelopt_qat_kv4_ckpt_{_MODEL_TAG}" if KV4 \
      else f"/workspace/NVFP4_Research/results/box_run/modelopt_qat_ckpt_{_MODEL_TAG}"

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

if GSM8K:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    def gsm8k_score(m):
        """Standard lm-evaluation-harness GSM8K score (the same benchmark + scoring used
        on model cards / leaderboards) on the model already in memory — no hand-rolled
        CoT/regex eval, no reload, no PTQ pass."""
        was_training = m.training
        m.eval()
        # max_length capped: "auto" batch-sizing estimates worst-case memory against
        # the model's advertised max context (e.g. Phi-3.5-mini claims 131072), which
        # tanks the estimate to batch=1 even though real 5-shot GSM8K prompts are a
        # few hundred tokens. Cap it to what these prompts actually need.
        lm = HFLM(pretrained=m, tokenizer=tok, batch_size="auto", max_length=2048)
        out = simple_evaluate(model=lm, tasks=["gsm8k"], num_fewshot=GSM8K_FEWSHOT,
                               limit=GSM8K_LIMIT or None)
        if was_training:
            m.train()
        return out["results"]["gsm8k"]

# 1) install ModelOpt NVFP4 quantizers (STE) + calibrate
calib = [win(tr_ids, k) for k in range(32)]
def floop(m):
    with torch.no_grad():
        for b in calib:
            m(b)
CFG_NAME = os.environ.get("CFG", "NVFP4_DEFAULT_CFG")   # e.g. NVFP4_AWQ_LITE_CFG (equalization)
base_cfg = getattr(mtq, CFG_NAME)
cfg = build_quant_cfg(base_cfg, kv_cfg=mtq.NVFP4_KV_CFG if KV4 else None)
print(f"calibrating + installing {CFG_NAME} (KV4={KV4}, steps={STEPS})...", flush=True)
mtq.quantize(student, cfg, floop)

# Activation (input) quantizers that calibration left enabled; lm_head's stay off.
act_q = {n: m for n, m in student.named_modules()
         if n.endswith("input_quantizer") and m.is_enabled}

def act_quant(on):
    for m in act_q.values():
        m.enable() if on else m.disable()

noise_on = [False]
def _noise_hook(module, args):
    """Add the activation quantizer's own error with random signs: same magnitude as
    deployment, zero mean, and the gradient is exact (the error is a constant)."""
    if not noise_on[0]:
        return None
    x, iq = args[0], module.input_quantizer
    with torch.no_grad():
        iq.enable(); err = iq(x) - x; iq.disable()
        sign = torch.where(torch.rand_like(x, dtype=torch.float32) < 0.5, -1.0, 1.0)
    return (x + err * sign.to(x.dtype),) + args[1:]

def train_mode(on):
    """on = the training-time activation treatment; off = the deployed quantizers."""
    if ACT_TRAIN != "a4":
        act_quant(not on)
    noise_on[0] = on and ACT_TRAIN == "noise"
    student.train(on)

if ACT_TRAIN != "a4":   # fail in seconds, not after a paid training run
    assert act_q, "no enabled input quantizers found (ModelOpt layout changed?)"
    with torch.no_grad():
        q_on = student(calib[0]).logits
        act_quant(False)
        q_off = student(calib[0]).logits
    assert not torch.equal(q_on, q_off), "switching activation quantizers off changed nothing"
    act_quant(True)
    if ACT_TRAIN == "noise":
        for n in act_q:
            student.get_submodule(n.rsplit(".", 1)[0]).register_forward_pre_hook(_noise_hook)
print(f"ACT_TRAIN={ACT_TRAIN} SEED={SEED}: {len(act_q)} activation quantizers", flush=True)

@torch.no_grad()
def kl_to_teacher(m):
    """Mean per-token KL(teacher || m) in nats on the held-out eval windows. Paired with
    the teacher token by token, so it is far less noisy than a perplexity difference."""
    n = ppl_ids.size(1) // L
    tot = sum(F.kl_div(F.log_softmax(m(win(ppl_ids, k)).logits.float(), -1),
                       F.softmax(teacher(win(ppl_ids, k)).logits.float(), -1),
                       reduction="sum").item() for k in range(n))
    return tot / (n * L)

fp16 = perplexity(teacher, ppl_ids, L, L, DEV)
ptq = perplexity(student, ppl_ids, L, L, DEV)   # internal only: best-of baseline, never printed
ptq_kl = kl_to_teacher(student)                 # internal only: saved to the JSON record
print(f"FP16 {fp16:.4f}", flush=True)

if GSM8K and not SKIP_FP16_GSM8K:
    gsm_fp16 = gsm8k_score(teacher)
    print(f"GSM8K (lm-eval, {GSM8K_FEWSHOT}-shot) FP16: {gsm_fp16}", flush=True)
elif GSM8K:
    gsm_fp16 = "SKIPPED (SKIP_FP16_GSM8K=1) -- compare QAT below to the published score directly"
    print(f"GSM8K FP16: {gsm_fp16}", flush=True)

# 2) QAT: train the weights to survive ModelOpt's own quantizer.
#    cosine LR with warmup — converges to a lower basin than constant LR.
LR = float(os.environ.get("LR", 3e-5))
# Train ONLY the model weights — leave quantizer / AWQ pre_quant_scale params FIXED
# (from calibration). Training the equalization scales destroys them (divergence).
trainable = [p for n, p in student.named_parameters()
             if p.requires_grad and all(k not in n for k in
             ("quantizer", "_scale", "pre_quant", "input_quant", "weight_quant"))]
print(f"training {len(trainable)} weight tensors (quantizer/AWQ-scale params frozen)", flush=True)
opt = torch.optim.AdamW(trainable, lr=LR)
warmup = 100
def lr_at(s):
    if s < warmup:
        return (s + 1) / warmup
    import math as _m
    return 0.5 * (1 + _m.cos(_m.pi * (s - warmup) / max(1, STEPS - warmup)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
nwin = tr_ids.size(1) // L
order = list(range(nwin)) if SEED is None else \
    torch.randperm(nwin, generator=torch.Generator().manual_seed(int(SEED))).tolist()
best = ptq
t0 = time.time()
train_mode(True)
for s in range(STEPS):
    x = win(tr_ids, order[s % nwin])
    with torch.no_grad():
        t = teacher(x).logits.float().reshape(-1, V)
    out = student(x).logits.float().reshape(-1, V)
    loss = F.kl_div(F.log_softmax(out, -1), F.softmax(t, -1), reduction="batchmean")
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if (s + 1) % 500 == 0:
        train_mode(False)
        p = perplexity(student, ppl_ids, L, L, DEV)
        best = min(best, p)
        train_mode(True)
        print(f"step {s+1} kl {loss.item():.4f} deploy-PPL {p:.4f} best {best:.4f} "
              f"[{(time.time()-t0)/60:.1f}min]", flush=True)

train_mode(False)
final = perplexity(student, ppl_ids, L, L, DEV)
final_kl = kl_to_teacher(student)
print(f"\n*** ModelOpt-NATIVE QAT: deployed PPL = {final:.4f}  (best {best:.4f}) | "
      f"FP16 {fp16:.4f} | KL to FP16 {final_kl:.4f} nats/token ***", flush=True)
import json
with open(f"/workspace/NVFP4_Research/results/qad_{_MODEL_TAG}{'_kv4' if KV4 else ''}.json", "w") as f:
    json.dump({"model": MODEL, "act_train": ACT_TRAIN, "seed": SEED, "kv4": KV4, "steps": STEPS,
               "lr": LR, "fp16_ppl": fp16, "ptq_ppl": ptq, "ptq_kl": ptq_kl,
               "qad_ppl": final, "qad_best_ppl": best, "qad_kl": final_kl}, f, indent=2)
print("train==deploy: these numbers ARE the deployed model's (no sim-to-real gap).", flush=True)

if GSM8K:
    gsm_qat = gsm8k_score(student)
    print(f"*** GSM8K (lm-eval, {GSM8K_FEWSHOT}-shot, standalone quality): "
          f"FP16 {gsm_fp16} -> QAT {gsm_qat} ***", flush=True)

export_hf_checkpoint(student, dtype=torch.bfloat16, export_dir=OUT)
print(f"exported -> {OUT}", flush=True)
