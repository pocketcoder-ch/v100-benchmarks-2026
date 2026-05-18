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

**Interactive table (sort, filter by family/pod/size, search) — [pocketcoder-ch.github.io/v100-benchmarks-2026](https://pocketcoder-ch.github.io/v100-benchmarks-2026/).** Wider columns (TTFT, prefill, VRAM, runs spread) and per-model cards — [`docs/FULL_REPORT.md`](docs/FULL_REPORT.md) / [`docs/MODEL_CARDS.md`](docs/MODEL_CARDS.md).

<details open><summary><b>Full table — all 108 LLMs, sorted by avg decode tps</b></summary>

Rows are sorted by `avg` decode tps (mean of 4 context levels). Six FAIL rows at the bottom — see [`docs/V100_CHECKLIST.md`](docs/V100_CHECKLIST.md) for the cause of each failure.

| # | Model | Family | B | Pod | short | small | medium | large | **avg tps** | Status |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `smollm2:135m` | SmolLM-2 | — | pod3 | 433.6 | 420.3 | 421.6 | 421.4 | **424.2** | OK |
| 2 | `llama3.2:1b-instruct-q4_K_M` | Llama-3.2 | 1.0 | pod5 | 352.5 | 322.5 | 274.6 | 249.0 | **299.6** | OK |
| 3 | `smollm2:360m` | SmolLM-2 | — | pod3 | 305.3 | 297.8 | 296.9 | 297.9 | **299.4** | OK |
| 4 | `glm-ocr` | GLM-OCR | — | pod4 | 303.4 | 247.6 | 257.6 | 243.8 | **263.1** | OK |
| 5 | `gemma3:1b` | Gemma-3 | 1.0 | pod3 | 203.2 | 172.1 | 207.4 | 204.2 | **196.7** | OK |
| 6 | `qwen3:0.6b-q4_K_M` | Qwen3 | 0.6 | pod4 | 204.8 | 195.5 | 177.3 | 184.4 | **190.5** | OK |
| 7 | `smollm2:1.7b` | SmolLM-2 | 1.7 | pod3 | 183.4 | 172.5 | 172.2 | 171.7 | **175.0** | OK |
| 8 | `qwen3:1.7b-q4_K_M` | Qwen3 | 1.7 | pod4 | 163.7 | 182.2 | 163.2 | 156.5 | **166.4** | OK |
| 9 | `llama3.2:3b-instruct-q4_K_M` | Llama-3.2 | 3.0 | pod5 | 180.6 | 167.3 | 147.0 | 134.4 | **157.3** | OK |
| 10 | `qwen:1.8b` | Qwen | 1.8 | pod5 | 274.4 | 248.8 | — | — | **130.8** | OK |
| 11 | `qwen:1.8b` | Qwen | 1.8 | pod4 | 272.4 | 248.3 | — | — | **130.2** | OK |
| 12 | `gemma3:4b` | Gemma-3 | 4.0 | pod3 | 107.4 | 123.4 | 123.7 | 122.8 | **119.3** | OK |
| 13 | `moondream` | Moondream | — | pod5 | 182.2 | 30.0 | 113.4 | 113.5 | **109.8** | OK |
| 14 | `gpt-oss:20b` | GPT-OSS | 20.0 | pod4 | 110.8 | 107.9 | 105.3 | 103.1 | **106.8** | OK |
| 15 | `glm4:9b` | GLM-4 | 9.0 | pod3 | 108.5 | 107.8 | 103.0 | 100.4 | **104.9** | OK |
| 16 | `starcoder2:7b` | StarCoder-2 | 7.0 | pod3 | — | 177.3 | 159.1 | 82.7 | **104.8** | OK |
| 17 | `qwen2.5-coder:7b-instruct-q4_K_M` | Qwen2.5-Coder | 7.0 | pod3 | 113.4 | 108.0 | 98.7 | 92.5 | **103.2** | OK |
| 18 | `qwen2.5:7b-instruct-q4_K_M` | Qwen2.5 | 7.0 | pod4 | 113.2 | 109.1 | 97.3 | 91.4 | **102.8** | OK |
| 19 | `dolphin3:8b` | Dolphin-3 | 8.0 | pod3 | 110.0 | 105.8 | 98.0 | 93.2 | **101.8** | OK |
| 20 | `starcoder2:7b` | StarCoder-2 | 7.0 | pod4 | — | 177.8 | 151.8 | 71.0 | **100.2** | OK |
| 21 | `glm4:9b` | GLM-4 | 9.0 | pod2 | 103.2 | 101.9 | 95.2 | 91.5 | **97.9** | OK |
| 22 | `falcon3:7b` | Falcon-3 | 7.0 | pod3 | 106.8 | 98.1 | 88.4 | 85.5 | **94.7** | OK |
| 23 | `nemotron-mini:4b-instruct-q4_K_M` | Nemotron-Mini | 4.0 | pod5 | 153.1 | 97.8 | 13.7 | 113.7 | **94.6** | OK |
| 24 | `nemotron-mini:4b` | Nemotron-Mini | 4.0 | pod4 | 152.4 | 97.4 | 13.7 | 113.1 | **94.1** | OK |
| 25 | `magicoder:7b` | Magicoder | 7.0 | pod5 | 113.6 | 102.5 | 82.7 | 71.4 | **92.6** | OK |
| 26 | `codellama:7b` | CodeLlama | 7.0 | pod5 | 113.9 | 102.2 | 82.2 | 70.9 | **92.3** | OK |
| 27 | `olmo2:7b` | OLMo-2 | 7.0 | pod3 | 91.3 | 91.2 | 91.3 | 90.8 | **91.2** | OK |
| 28 | `starling-lm:7b-beta` | Starling | 7.0 | pod5 | 106.6 | 94.6 | 80.2 | 80.1 | **90.4** | OK |
| 29 | `qwen:7b` | Qwen | 7.0 | pod5 | 106.6 | 96.9 | 81.9 | 71.8 | **89.3** | OK |
| 30 | `openchat:7b` | OpenChat | 7.0 | pod5 | 107.2 | 94.8 | 77.4 | 72.2 | **87.9** | OK |
| 31 | `deepseek-r1:8b` | DeepSeek-R1 | 8.0 | pod3 | 90.7 | 90.3 | 84.5 | 80.0 | **86.4** | OK |
| 32 | `neural-chat:7b` | Neural-Chat | 7.0 | pod5 | 107.5 | 95.1 | 75.5 | 64.9 | **85.8** | OK |
| 33 | `zephyr:7b-beta` | Zephyr | 7.0 | pod5 | 106.9 | 94.7 | 75.3 | 65.2 | **85.5** | OK |
| 34 | `qwen3:4b-q4_K_M` | Qwen3 | 4.0 | pod4 | 97.5 | 92.8 | 78.4 | 69.0 | **84.4** | OK |
| 35 | `llama3.1:8b` | Llama-3.1 | 8.0 | pod2 | 100.0 | 90.2 | 77.0 | 69.7 | **84.2** | OK |
| 36 | `dolphin3:8b` | Dolphin-3 | 8.0 | pod2 | 99.7 | 90.5 | 76.9 | 69.6 | **84.2** | OK |
| 37 | `llama3:8b` | Llama-3 | 8.0 | pod5 | 107.5 | 96.9 | 63.7 | 64.6 | **83.2** | OK |
| 38 | `llama3.1:8b-instruct-q4_K_M` | Llama-3.1 | 8.0 | pod5 | 98.1 | 88.6 | 76.1 | 68.7 | **82.9** | OK |
| 39 | `mistral:7b-instruct-v0.3-q4_K_M` | Mistral | 7.0 | pod5 | 98.0 | 87.2 | 71.1 | 61.9 | **79.5** | OK |
| 40 | `mistral:7b` | Mistral | 7.0 | pod5 | 96.9 | 86.9 | 70.5 | 61.7 | **79.0** | OK |
| 41 | `llama3.2-vision:11b` | Llama-3.2 | 11.0 | pod2 | 93.8 | 73.8 | 74.9 | 66.8 | **77.3** | OK |
| 42 | `nemotron-mini:4b-instruct-fp16` | Nemotron-Mini | 4.0 | pod2 | 89.2 | 72.9 | 69.6 | 73.0 | **76.2** | OK |
| 43 | `starcoder2:15b` | StarCoder-2 | 15.0 | pod3 | 66.2 | 117.6 | 59.3 | 58.4 | **75.4** | OK |
| 44 | `alibilge/Huihui-GLM-4.6V-Flash-abliterated:q4_k_m` | Huihui-GLM-4.6V | — | pod4 | 78.6 | 76.9 | 74.2 | 71.6 | **75.3** | OK |
| 45 | `mixtral:8x7b` | Mixtral | 56.0 | pod3 | 81.2 | 77.7 | 72.5 | 68.5 | **75.0** | OK |
| 46 | `gpt-oss:120b` | GPT-OSS | 120.0 | pod1 | 76.4 | 74.8 | 72.8 | — | **74.7** | OK |
| 47 | `granite3-dense:8b` | Granite | 8.0 | pod2 | 71.5 | 72.8 | 71.1 | 76.3 | **72.9** | OK |
| 48 | `vicuna:7b` | Vicuna | 7.0 | pod5 | 114.4 | 58.9 | 58.7 | 58.9 | **72.7** | OK |
| 49 | `llama2:7b` | Llama-2 | 7.0 | pod5 | 113.9 | 60.0 | 57.8 | 57.8 | **72.4** | OK |
| 50 | `qwen3:8b-q4_K_M` | Qwen3 | 8.0 | pod4 | 78.3 | 76.7 | 63.9 | 58.7 | **69.4** | OK |
| 51 | `solar:10.7b` | Solar | 10.7 | pod5 | 72.2 | 66.3 | 66.3 | 66.0 | **67.7** | OK |
| 52 | `falcon3:10b` | Falcon-3 | 10.0 | pod3 | 76.6 | 70.5 | 61.1 | 61.0 | **67.3** | OK |
| 53 | `gemma2:9b-instruct-q4_K_M` | Gemma-2 | 9.0 | pod2 | 73.8 | 69.2 | 61.6 | 60.2 | **66.2** | OK |
| 54 | `codegemma:7b` | codegemma | 7.0 | pod5 | 100.9 | 96.3 | 31.3 | 31.6 | **65.0** | OK |
| 55 | `phi3:3.8b` | Phi-3 | 3.8 | pod5 | 95.0 | 73.6 | 47.8 | 37.2 | **63.4** | OK |
| 56 | `granite3.1-dense:8b-instruct-q4_K_M` | Granite | 8.0 | pod5 | 78.5 | 69.9 | 55.4 | 47.7 | **62.9** | OK |
| 57 | `phi3:3.8b-mini-128k-instruct-q4_K_M` | Phi-3 | 3.8 | pod5 | 93.8 | 71.4 | 46.8 | 36.7 | **62.2** | OK |
| 58 | `gemma3:12b` | Gemma-3 | 12.0 | pod3 | 63.7 | 61.9 | 61.3 | 60.5 | **61.9** | OK |
| 59 | `dolphin-mixtral:8x7b` | Dolphin-Mixtral | 56.0 | pod5 | 72.4 | 66.8 | 56.6 | 50.4 | **61.6** | OK |
| 60 | `nous-hermes2-mixtral` | Nous-Hermes-Mixtral | — | pod5 | 71.8 | 66.3 | 56.3 | 50.2 | **61.1** | OK |
| 61 | `mistral-nemo:12b-instruct-2407-q4_K_M` | Mistral-Nemo | 12.0 | pod5 | 68.7 | 62.8 | 54.4 | 49.8 | **58.9** | OK |
| 62 | `glm-4.7-flash:latest` | GLM-4.7-Flash | — | pod4 | 61.7 | 58.6 | 54.1 | 50.4 | **56.2** | OK |
| 63 | `olmo2:13b` | OLMo-2 | 13.0 | pod3 | 55.3 | 55.2 | 55.2 | 55.2 | **55.2** | OK |
| 64 | `codellama:13b` | CodeLlama | 13.0 | pod5 | 66.9 | 60.0 | 49.6 | 42.8 | **54.8** | OK |
| 65 | `qwen:14b` | Qwen | 14.0 | pod5 | 63.8 | 58.6 | 49.1 | 43.2 | **53.7** | OK |
| 66 | `qwen2.5:14b-instruct-q4_K_M` | Qwen2.5 | 14.0 | pod4 | 59.5 | 56.1 | 49.9 | 45.9 | **52.9** | OK |
| 67 | `qwen2.5-coder:14b-instruct-q4_K_M` | Qwen2.5-Coder | 14.0 | pod3 | 59.0 | 55.4 | 49.5 | 45.9 | **52.5** | OK |
| 68 | `qwen3:14b-q4_K_M` | Qwen3 | 14.0 | pod4 | 57.2 | 55.5 | 50.0 | 46.6 | **52.3** | OK |
| 69 | `deepseek-r1:14b` | DeepSeek-R1 | 14.0 | pod3 | 58.6 | 55.2 | 49.4 | 45.7 | **52.2** | OK |
| 70 | `phi3:14b-medium-128k-instruct-q4_K_M` | Phi-3 | 14.0 | pod5 | 59.9 | 54.3 | 45.4 | 40.1 | **49.9** | OK |
| 71 | `deepseek-coder-v2:16b-lite-instruct-q4_K_M` | DeepSeek-Coder | 16.0 | pod3 | 80.2 | 56.7 | 33.0 | 25.2 | **48.8** | OK |
| 72 | `llama2:13b` | Llama-2 | 13.0 | pod5 | 66.6 | 37.4 | 47.8 | 43.2 | **48.7** | OK |
| 73 | `phi4:14b-q4_K_M` | Phi-4 | 14.0 | pod2 | 57.7 | 51.4 | 42.2 | 27.9 | **44.8** | OK |
| 74 | `vicuna:13b` | Vicuna | 13.0 | pod4 | 67.7 | 40.9 | 42.6 | 9.7 | **40.2** | OK |
| 75 | `vicuna:13b` | Vicuna | 13.0 | pod5 | 66.9 | 40.6 | 42.4 | 9.7 | **39.9** | OK |
| 76 | `mistral-small:24b-instruct-2501-q4_K_M` | Mistral-Small | 24.0 | pod2 | 42.0 | 39.7 | 36.4 | 33.7 | **37.9** | OK |
| 77 | `codestral:22b-v0.1-q4_K_M` | Codestral | 22.0 | pod3 | 41.0 | 38.3 | 33.1 | 30.1 | **35.6** | OK |
| 78 | `gemma3:27b` | Gemma-3 | 27.0 | pod3 | 33.4 | 33.0 | 32.6 | 32.2 | **32.8** | OK |
| 79 | `gemma2:27b-instruct-q4_K_M` | Gemma-2 | 27.0 | pod2 | 35.0 | 33.8 | 31.0 | 31.1 | **32.8** | OK |
| 80 | `qwen:32b` | Qwen | 32.0 | pod5 | 33.9 | 32.3 | 29.5 | 27.5 | **30.8** | OK |
| 81 | `command-r:35b-08-2024-q4_K_M` | Command-R | 35.0 | pod5 | 32.5 | 31.2 | 29.3 | 27.9 | **30.2** | OK |
| 82 | `yi:34b` | Yi | 34.0 | pod2 | 29.1 | 28.9 | 28.4 | 28.9 | **28.8** | OK |
| 83 | `aya-expanse:32b-q4_K_M` | Aya | 32.0 | pod5 | 32.5 | 31.2 | 25.3 | 25.6 | **28.7** | OK |
| 84 | `aya-expanse:32b` | Aya | 32.0 | pod5 | 32.5 | 31.2 | 25.3 | 25.6 | **28.6** | OK |
| 85 | `qwq:32b-preview-q4_K_M` | QwQ | 32.0 | pod2 | 30.2 | 29.1 | 26.5 | 24.9 | **27.7** | OK |
| 86 | `qwen2.5:32b-instruct-q4_K_M` | Qwen2.5 | 32.0 | pod1 | 30.1 | 28.9 | 26.4 | 24.8 | **27.5** | OK |
| 87 | `qwen2.5-coder:32b-instruct-q4_K_M` | Qwen2.5-Coder | 32.0 | pod3 | 29.9 | 28.6 | 26.5 | 25.1 | **27.5** | OK |
| 88 | `deepseek-r1:32b` | DeepSeek-R1 | 32.0 | pod3 | 29.6 | 28.4 | 26.2 | 24.8 | **27.3** | OK |
| 89 | `exaone-deep:32b` | EXAONE | 32.0 | pod5 | 29.0 | 27.4 | 24.1 | 22.0 | **25.6** | OK |
| 90 | `wizardcoder:33b` | WizardCoder | 33.0 | pod5 | 30.5 | 28.1 | 23.4 | 17.4 | **24.8** | OK |
| 91 | `qwen3:32b-q4_K_M` | Qwen3 | 32.0 | pod2 | 27.3 | 26.2 | 23.7 | 22.0 | **24.8** | OK |
| 92 | `codellama:70b` | CodeLlama | 70.0 | pod5 | 17.5 | 17.4 | 17.4 | 17.5 | **17.5** | OK |
| 93 | `llama2:70b` | Llama-2 | 70.0 | pod5 | 17.6 | 16.6 | 16.3 | 16.4 | **16.7** | OK |
| 94 | `hermes3:70b` | Hermes | 70.0 | pod5 | 17.4 | 16.8 | 15.4 | 14.7 | **16.1** | OK |
| 95 | `llama3.1:70b` | Llama-3.1 | 70.0 | pod3 | 16.1 | 15.9 | 15.4 | 15.1 | **15.6** | OK |
| 96 | `nemotron:70b` | Nemotron | 70.0 | pod3 | 16.1 | 15.9 | 15.4 | 15.1 | **15.6** | OK |
| 97 | `llama3:70b` | Llama-3 | 70.0 | pod5 | 17.5 | 16.8 | 14.1 | 13.7 | **15.5** | OK |
| 98 | `llama3.3:70b-instruct-q4_K_M` | Llama-3.3 | 70.0 | pod1 | 15.8 | 15.2 | 14.1 | 13.5 | **14.6** | OK |
| 99 | `deepseek-r1:70b-llama-distill-q4_K_M` | DeepSeek-R1 | 70.0 | pod1 | 15.6 | 15.0 | 14.0 | 13.3 | **14.5** | OK |
| 100 | `qwen2.5:72b-instruct-q4_K_M` | Qwen2.5 | 72.0 | pod1 | 14.7 | 14.2 | 13.2 | 12.5 | **13.7** | OK |
| 101 | `mixtral:8x22b-instruct-v0.1-q3_K_M` | Mixtral | 176.0 | pod4 | 7.2 | 6.2 | 4.6 | 3.6 | **5.4** | OK |
| 102 | `mistral-large:123b-instruct-2407-q3_K_M` | Mistral-Large | 123.0 | pod1 | 3.0 | 2.6 | 1.7 | — | **2.4** | OK |
| 103 | `command-r-plus:104b-q4_K_S` | Command-R+ | 104.0 | pod1 | — | — | — | — | — | FAIL |
| 104 | `deepseek-r1:14b-q4_K_M` | DeepSeek-R1 | 14.0 | pod3 | — | — | — | — | — | FAIL |
| 105 | `deepseek-r1:32b-q4_K_M` | DeepSeek-R1 | 32.0 | pod3 | — | — | — | — | — | FAIL |
| 106 | `deepseek-r1:8b-q4_K_M` | DeepSeek-R1 | 8.0 | pod3 | — | — | — | — | — | FAIL |
| 107 | `phi:2.7b` | Phi | 2.7 | pod5 | — | — | — | — | — | FAIL |
| 108 | `qwen:72b` | Qwen | 72.0 | pod5 | — | — | — | — | — | FAIL |

</details>

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

This bench was built by the team behind [**BVM**](https://bvmax.ru/ai) — we do AI for sales-call analytics and on-prem LLM deployments (this V100 box runs our own production call-evaluation pipeline). The article and this repo are the engineering write-up of one weekend of running the rig at its limits.

- **Telegram (notes from the CTO):** [@notes_from_cto](https://t.me/notes_from_cto) — irregular technical posts on what we're building.
- **BVM site:** [bvmax.ru/ai](https://bvmax.ru/ai)
- **Issues / PRs:** open one here — fastest way to reach us about the bench itself.
- **Email for collaborations:** see GitHub profile.

If you found a bug in our numbers, please **open an issue with the exact model, pod, and the JSON file from `benchmarks/ollama/raw/`** — we want the table to be correct.

---

*Generated 2026-05-16. Last full sweep: 2026-05-15. Dashboard auto-updates when new results land in `benchmarks/ollama/raw/`.*
