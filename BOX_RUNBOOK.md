# One-Shot Box Runbook (RTX Pro 6000, sm_120)

**There is one box and no second box, ever.** This is the exact sequence to run so the
session completes unattended and every artifact survives teardown. QAT is the
deliverable — there is no PTQ path here.

Target: **TinyLlama-1.1B**, **RTX Pro 6000 (sm_120)**. Everything else is pre-tested on CPU.

---

## What we get out of the box (and what can't be gotten any other way)

| Result | How | Box-only? |
|---|---|---|
| **Deployed-QAT NVFP4 WikiText PPL** (headline) | `export_nvfp4 --eval-ppl` (ModelOpt quant + our `perplexity()`) | accuracy already known in fake-quant; this confirms it on the deploy grid |
| **Real FP4 GEMM throughput** | `gemm_bench.py` | **yes** — never measured |
| **Real end-to-end FP4 serving throughput** | `measure_deploy.py` (vLLM/TRT-LLM) | **yes** — never measured on our artifact |
| **A permanent deployable NVFP4 checkpoint** | exported to `results/box_run/<ts>/nvfp4_ckpt` | **yes — save/upload it before teardown** |

The accuracy headline does **not** depend on the serving runtime, so it lands even if
throughput tooling misbehaves.

---

## Step 0 — Environment (bare box: only CUDA + PyTorch present)

Everything else is installed by us. Copy the repo onto the box, then:

```bash
cd NVFP4_Research
bash setup_box.sh          # pins torch, installs deps, pre-downloads model+data, preflights
```

`setup_box.sh` **pins the box's Blackwell PyTorch** so no dependency can replace it.
Core deps (accuracy headline) are mandatory; throughput deps (torchao/vllm) are
best-effort and cannot break the headline if they fail to install.

**TRT-LLM is not used on a bare sm_120 box** — pip TRT-LLM isn't supported on Blackwell
and there's no container here. Throughput therefore runs on **vLLM** (NVFP4 verified on
sm_120). The accuracy headline needs neither — only torch + transformers + modelopt.

Setup ends by running the 5 CPU parity tests; if they pass, the environment is sound
before you commit to the ~4h QAT.

---

## Step 1 — Run it (one command)

```bash
bash run_the_box.sh
```

That runs, fail-soft and fully logged to `results/box_run/<timestamp>/`:
0. preflight (SM + CPU tests) → 1. `gemm_bench` → 2. QAT (3000-step KL, the 70%-gap
recipe) → 3. export NVFP4 + deployed PPL → 4. throughput (auto backend).

Overrides via env, e.g. `EXTRA_QAT="--kv4" EXTRA_EXPORT="--kv4" bash run_the_box.sh`
for the W4A4KV4 variant, or `STEPS=800` for a faster smoke.

Read `results/box_run/<ts>/SUMMARY.txt` at the end.

---

## Serving backend on a bare sm_120 box

TRT-LLM is not installed here (pip TRT-LLM unsupported on Blackwell, no container). The
throughput step uses **vLLM** (`measure_deploy.py --backend auto` will land on vLLM since
trtllm won't import). If vLLM didn't install, step 4 is skipped and the accuracy headline
(step 3, runtime-independent) is unaffected.

---

## Contingencies (so a mid-run failure still banks value)

| If this fails | You still have | Do |
|---|---|---|
| `gemm_bench` SUSPECT/emulating | everything else (non-blocking) | note the FP4 kernel gap; proceed |
| QAT step | gemm number | check GPU mem / dataset download; rerun step 2 only |
| export | QAT ckpt saved at `qat_ckpt/` | export is minutes — rerun `export_nvfp4` alone |
| vLLM didn't install | accuracy headline + exported ckpt | throughput skipped; ckpt still deployable later |

## Before you tear the box down
```bash
# SAVE THE ARTIFACT — there is no second box.
tar czf nvfp4_ckpt.tgz results/box_run/<ts>/nvfp4_ckpt
# upload nvfp4_ckpt.tgz + results/box_run/<ts>/SUMMARY.txt + *.log somewhere durable
```
