# V100 Benchmarks 2026 — 108 LLMs + diffusion + video on 2× Tesla V100 32GB

> **Companion repo** to the article *"AI lab for 200 000 ₽: 2× Tesla V100 build, full stack notes, 108-model benchmark"* on Habr.
> **Russian version:** see [README.ru.md](README.ru.md).

What's here: every raw result, every script, every command we actually ran on V100 SXM2 (cc 7.0, Volta) — so you can re-check our numbers, copy a working config, or skip the dead ends we already paid for.

- **108 LLMs** — Ollama Q4_K_M, single-stream, N=1 (variance CV ≤ 0.10%), real call-center transcripts at 4 context levels (~3K / 5K / 10K / 14K tokens) + a fixed evaluation prompt.
- **5 image-gen runs** — Z-Image-Turbo 1024×1024 PNG (sd.cpp, `--type bf16` workaround for [issue #1292](https://github.com/leejet/stable-diffusion.cpp/issues/1292)).
- **4+5 PNG** — FLUX.1-dev and SDXL via sd.cpp.
- **3 video clips** — Wan2.2 TI2V 5B (sd.cpp) + 3 clips — CogVideoX-5B (diffusers, VAE FP32 safeguard).
- **Hardware** — 5× V100 32GB pods on vast.ai, mix of SXM2+NVLink and SYS topology (lottery, see notes).

---

## TL;DR — what V100 can and cannot do in 2026

| Stack | Verdict | Notes |
|---|---|---|
| **AWQ INT4 (vLLM, 1cat fork)** | ✅ works | 7B–70B; SHM 96KB wall on 32B+ long-prompt models with `head_dim=128` |
| **GGUF Q4_K_M (Ollama)** | ✅ works | Our main runner — covers 108 models, no SHM wall, single-card and dual-card transparently |
| **FP16 (vLLM, transformers)** | ✅ works | Native, but 14B fp16 = ~28 GB, fits one card only at low context |
| **FP8 / NVFP4 / MXFP4** | ❌ blocked | No hardware support on cc 7.0. Hopper (FP8) / Blackwell (NVFP4/MXFP4) only. |
| **BF16 (PyTorch native)** | ❌ partial | CPU emulation on Volta → NaN in VAE; use FP16 or `--type bf16` in sd.cpp |
| **Flash Attention 2** | ❌ blocked | Requires sm_80+. Use eager attention or vLLM 1cat fork. |
| **Z-Image / FLUX / SDXL** | ✅ works | sd.cpp + `--type bf16` workaround. 64 s/1024² Z-Image, 26 s/1024² SDXL. |
| **Wan2.2 / CogVideoX video** | ✅ works | sd.cpp (Wan) and diffusers (CogVideoX with `vae.to(float32)`). Slow but runs. |
| **LTX-2.3 / Sora-class video** | ❌ blocked | sd.cpp lacks the arch; diffusers hits Volta cuBLAS overflow. |

Top-10 LLMs by decode throughput on V100 (Ollama Q4_K_M, single-stream, avg across 4 context levels):

| Model | Params | Avg tps | Pod |
|---|---:|---:|---|
| `smollm2:135m` | 0.135 B | **424** | pod3 |
| `llama3.2:1b-instruct-q4_K_M` | 1.0 B | **300** | pod5 |
| `smollm2:360m` | 0.36 B | **299** | pod3 |
| `glm-ocr` | — | **263** | pod4 |
| `gemma3:1b` | 1.0 B | **197** | pod3 |
| `qwen3:0.6b-q4_K_M` | 0.6 B | **190** | pod4 |
| `smollm2:1.7b` | 1.7 B | **175** | pod3 |
| `qwen3:1.7b-q4_K_M` | 1.7 B | **166** | pod4 |
| `llama3.2:3b-instruct-q4_K_M` | 3.0 B | **157** | pod5 |
| `gemma3:4b` | 4.0 B | **119** | pod3 |

Full table (all 108 models) — [`docs/FULL_REPORT.md`](docs/FULL_REPORT.md).

---

## Repo layout

```
repo/
├── README.md                  # this file
├── README.ru.md               # Russian mirror
├── docs/
│   ├── FULL_REPORT.md         # 108-model master table, sorted by avg tps
│   ├── MODEL_CARDS.md         # per-model card: config, 4-level results, sample output
│   ├── MODEL_NAMES_MAPPING.md # Ollama tag → upstream HF repo (DeepSeek-R1 etc.)
│   ├── VARIANCE_FINDINGS.md   # why N=1 is enough at temperature=0 (CV ≤ 0.10%)
│   ├── V100_CHECKLIST.md      # 17 Volta-specific traps + fixes
│   └── COSTS.md               # vast.ai pricing, time spent, total spend
├── benchmarks/
│   ├── ollama/
│   │   ├── stats_interactive.html  # main dashboard (open in browser)
│   │   ├── stats_wide.csv          # all results, wide format
│   │   ├── stats_long.csv          # all results, long format (one row per level)
│   │   ├── data.json + detail.json # JSON dumps for the dashboard
│   │   └── raw/pod{1..5}/*.json    # raw per-model results, ~108 files
│   ├── diffusion/
│   │   ├── zimage/                 # 5 PNG + results.jsonl
│   │   ├── flux/                   # 4 PNG + timings
│   │   └── sdxl/                   # 5 PNG + timings
│   └── video/
│       ├── wan22/                  # 3 mp4 (Wan2.2 TI2V 5B)
│       └── cogvideox/              # 3 mp4 (CogVideoX-5B)
├── scripts/
│   ├── ollama_bench_real.py        # the runner — Ollama + 4-level corpus + N=1
│   ├── build_dashboard.py          # regenerate stats_interactive.html
│   ├── build_model_cards.py        # regenerate MODEL_CARDS.md
│   ├── run_wan22_sdcpp.sh          # Wan2.2 video on V100 via sd.cpp
│   ├── cogvideox_v100.py           # CogVideoX-5B with VAE FP32 fix
│   ├── zimage_inference.py         # Z-Image-Turbo via diffusers (failed, see V100_CHECKLIST)
│   └── setup_pod.sh                # vast.ai pod bootstrap (CUDA 12.8, Ollama 0.24)
└── raw/
    ├── eval_prompt.txt             # the BVM evaluation prompt (anonymized)
    ├── transcripts/                # 4 anonymized call-center transcripts (short/small/medium/large)
    └── models_queue.md             # per-pod model schedule (which model ran where, when)
```

---

## How to read MODEL_CARDS.md

Each model card looks like this:

```markdown
### `qwen2.5:7b-instruct-q4_K_M`

**Family:** Qwen2.5 · **Params:** 7.0 B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Run config:**
ollama run qwen2.5:7b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0

**Results (BVM transcript + eval prompt, avg = 102.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
| short | 2684 | 1100 | 2875 | 113.2 | 5530 | stop |
| ... | ... | ... | ... | ... | ... | ... |

**Sample answer (level=short):** [model output snippet]
```

The card answers four questions you'd ask before running this model yourself:

1. **Will it run on V100?** — `Status: OK` means it completed 4/4 context levels without OOM/timeout.
2. **What command did we use?** — exact `ollama run` invocation + options + env.
3. **How fast?** — decode tps at 4 context lengths, so you know if it slows down on long prompts.
4. **Does the output make sense?** — one short sample, so you can sanity-check the model is alive.

---

## Reproducing the benchmark

You need: a V100 32GB (single or dual), Ubuntu 22.04, CUDA 12.8+, Ollama 0.24+.

```bash
# 1. Install (vast.ai template: `cuda-12.8.1-auto`, `amd64`, 200 GB disk)
bash scripts/setup_pod.sh

# 2. Set Volta-friendly env
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
ollama serve &

# 3. Run the bench against a list of models
python scripts/ollama_bench_real.py \
  --models models_queue.txt \
  --transcripts raw/transcripts/ \
  --prompt raw/eval_prompt.txt \
  --out results/

# 4. Regenerate dashboard
python scripts/build_dashboard.py
open benchmarks/ollama/stats_interactive.html
```

**Cost reference (vast.ai, May 2026):** 1× V100 32GB SXM2 ≈ $0.45/hr. The full 108-model sweep ran on 5 pods in parallel for ~10 hours → ~$22 in compute. See [`docs/COSTS.md`](docs/COSTS.md) for the full breakdown.

---

## Known traps (a.k.a. "do not repeat our mistakes")

Short version — full list with line-by-line fixes in [`docs/V100_CHECKLIST.md`](docs/V100_CHECKLIST.md).

1. **vLLM stock build does not boot on Volta.** Use the [1cat fork](https://github.com/sasha0552/pascal-pkgs-ci) or compile with `TORCH_CUDA_ARCH_LIST=7.0` and `VLLM_ATTENTION_BACKEND=XFORMERS`.
2. **AWQ on 32B+ models hits a 96 KB shared-memory wall** in `flash_attn_v100`'s `prefill_paged_fwd` when `head_dim=128` (Qwen2.5, Llama-3.3). Long prompts crash; short prompts work. Workaround: drop to Ollama GGUF.
3. **BF16 emulation produces NaN** in VAE for FLUX / Z-Image → ~3 KB white PNG. Fix in sd.cpp: `--type bf16` (it routes through FP32 accum, not native BF16).
4. **`torch.bfloat16` in diffusers** silently falls back to CPU emulation on Volta. Always use `torch.float16`, and force VAE to FP32: `pipe.vae.to(torch.float32)`.
5. **vast.ai NVLink is a lottery.** `nvidia-smi topo -m` shows either `SYS` (PCIe) or `NV2` (NVLink). 4 out of 5 of our pods on May 14 turned out NV2 — your mileage will vary.
6. **DeepSeek-R1 in Ollama is NOT the real 671B R1.** All `deepseek-r1:*` tags are distilled into Llama/Qwen. See [`docs/MODEL_NAMES_MAPPING.md`](docs/MODEL_NAMES_MAPPING.md).

---

## Why N=1 is enough (and why we don't repeat each run 5 times)

At `temperature=0.0`, single-stream, Ollama gives a coefficient of variation **≤ 0.10 %** across reruns (we measured: see [`docs/VARIANCE_FINDINGS.md`](docs/VARIANCE_FINDINGS.md)). At that level, N=5 just multiplies cost without changing the answer. The numbers in this repo are N=1 — and we're not hiding that.

What N=1 does NOT cover: thermal throttling on long sustained loads, scheduler jitter under multi-tenant noise on vast.ai, batched serving. If you need any of those — re-run with your own load profile.

---

## License and reuse

- **Scripts, dashboards, docs:** MIT. Copy, fork, embed in your own write-up — credit appreciated, not required.
- **Transcripts (`raw/transcripts/`):** released under CC-BY-4.0, anonymized before publication — names, phone numbers, addresses and client-company identifiers were replaced with placeholders. Originals were real BVM call-center recordings; what's in this repo is the cleaned corpus.
- **Model outputs (in MODEL_CARDS):** each model's own license applies. We're publishing outputs as benchmark evidence, not as training data. A handful of model responses that echoed input PII back at us were redacted (search for `[redacted: model echoed PII`).

---

## Authors and contact

This bench was built by the team behind [**BVM**](https://bvm.ai) — we do AI for sales-call analytics and on-prem LLM deployments (this V100 box runs our own production call-evaluation pipeline). The article and this repo are the engineering write-up of one weekend of running the rig at its limits.

- **Telegram (notes from the CTO):** [@notes_from_cto](https://t.me/notes_from_cto) — irregular technical posts on what we're building.
- **BVM site:** [bvm.ai](https://bvm.ai)
- **Issues / PRs:** open one here — fastest way to reach us about the bench itself.
- **Email for collaborations:** see GitHub profile.

If you found a bug in our numbers, please **open an issue with the exact model, pod, and the JSON file from `benchmarks/ollama/raw/`** — we want the table to be correct.

---

*Generated 2026-05-16. Last full sweep: 2026-05-15. Dashboard auto-updates when new results land in `benchmarks/ollama/raw/`.*
