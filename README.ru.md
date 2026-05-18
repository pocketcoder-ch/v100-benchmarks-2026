# V100 Benchmarks 2026 — 108 LLM + diffusion + видео на 2× Tesla V100 32GB

> **Сопроводительный репозиторий** к статье *«AI-лаборатория за 200 000 ₽: сборка на 2× Tesla V100, полный стек, бенч 108 моделей»* на Хабре.
> **English version:** see [README.md](README.md).

Что внутри: все сырые результаты, скрипты и команды, которые мы реально запускали на V100 SXM2 (cc 7.0, Volta) — чтобы вы могли перепроверить наши цифры, скопировать рабочий конфиг или не наступить на грабли, на которые уже наступили мы.

- **108 LLM** — Ollama Q4_K_M, single-stream, N=1 (variance CV ≤ 0.10%), реальные транскрипции звонков колл-центра на 4 уровнях контекста (~3K / 5K / 10K / 14K токенов) + фиксированный eval prompt.
- **5 image-gen** — Z-Image-Turbo 1024×1024 PNG (sd.cpp, workaround `--type bf16` для [issue #1292](https://github.com/leejet/stable-diffusion.cpp/issues/1292)).
- **4+5 PNG** — FLUX.1-dev и SDXL через sd.cpp.
- **3 видео** — Wan2.2 TI2V 5B (sd.cpp) + 3 видео — CogVideoX-5B (diffusers, safeguard VAE FP32).
- **Железо** — 5× V100 32GB на vast.ai, смесь SXM2+NVLink и SYS-топологии (лотерея, см. ниже).

---

## TL;DR — что V100 умеет и что не умеет в 2026

| Стек | Вердикт | Комментарий |
|---|---|---|
| **AWQ INT4 (vLLM, форк 1cat)** | ✅ работает | 7B–70B; SHM 96KB wall на 32B+ моделях с `head_dim=128` на длинных prompt'ах |
| **GGUF Q4_K_M (Ollama)** | ✅ работает | Наш основной раннер — 108 моделей, нет SHM wall, прозрачно single/dual GPU |
| **FP16 (vLLM, transformers)** | ✅ работает | Нативно, но 14B fp16 = ~28 ГБ, лезет только в одну карту на низком контексте |
| **FP8 / NVFP4 / MXFP4** | ❌ нельзя | Нет hardware-поддержки на cc 7.0. Только Hopper (FP8) / Blackwell (NVFP4/MXFP4). |
| **BF16 (нативный PyTorch)** | ❌ частично | CPU-эмуляция на Volta → NaN в VAE; используйте FP16 или `--type bf16` в sd.cpp |
| **Flash Attention 2** | ❌ нельзя | Требует sm_80+. Eager attention или vLLM 1cat fork. |
| **Z-Image / FLUX / SDXL** | ✅ работает | sd.cpp + workaround `--type bf16`. 64 сек/1024² Z-Image, 26 сек/1024² SDXL. |
| **Wan2.2 / CogVideoX видео** | ✅ работает | sd.cpp (Wan) и diffusers (CogVideoX с `vae.to(float32)`). Медленно, но крутится. |
| **LTX-2.3 / Sora-класс видео** | ❌ нельзя | sd.cpp не поддерживает архитектуру; diffusers упирается в Volta cuBLAS overflow. |

Топ-10 LLM по decode-скорости на V100 (Ollama Q4_K_M, single-stream, среднее по 4 уровням контекста):

| Модель | Params | Avg tps | Pod |
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

**Интерактивная таблица (сортировка, фильтр по семье/поду/размеру, поиск) — [pocketcoder-ch.github.io/v100-benchmarks-2026](https://pocketcoder-ch.github.io/v100-benchmarks-2026/).** Расширенные колонки (TTFT, prefill, VRAM, runs spread) и карточки моделей — [`docs/FULL_REPORT.md`](docs/FULL_REPORT.md) / [`docs/MODEL_CARDS.md`](docs/MODEL_CARDS.md).

<details><summary><b>Полная таблица — все 108 LLM, сорт по avg decode tps (кликни чтобы раскрыть)</b></summary>

Строки отсортированы по `avg` decode tps (среднее по 4 уровням контекста). Шесть FAIL-строк внизу — причины каждого падения см. в [`docs/V100_CHECKLIST.md`](docs/V100_CHECKLIST.md).

| # | Модель | Семья | B | Pod | short | small | medium | large | **avg tps** | Статус |
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

## Структура репозитория

```
repo/
├── README.md                  # английская версия
├── README.ru.md               # этот файл
├── docs/
│   ├── FULL_REPORT.md         # мастер-таблица 108 моделей, сорт по avg tps
│   ├── MODEL_CARDS.md         # карточка на каждую модель: конфиг, 4 уровня, пример ответа
│   ├── MODEL_NAMES_MAPPING.md # Ollama-тег → upstream HF репо (DeepSeek-R1 и др.)
│   ├── VARIANCE_FINDINGS.md   # почему N=1 хватает при temperature=0 (CV ≤ 0.10%)
│   ├── V100_CHECKLIST.md      # 17 Volta-специфичных граблей и фиксов
│   └── COSTS.md               # vast.ai цены, потраченное время, общий бюджет
├── benchmarks/
│   ├── ollama/
│   │   ├── stats_interactive.html  # главный дашборд (открыть в браузере)
│   │   ├── stats_wide.csv          # все результаты, wide-формат
│   │   ├── stats_long.csv          # все результаты, long-формат (строка на уровень)
│   │   ├── data.json + detail.json # JSON для дашборда
│   │   └── raw/pod{1..5}/*.json    # сырые per-model результаты, ~108 файлов
│   ├── diffusion/
│   │   ├── zimage/                 # 5 PNG + results.jsonl
│   │   ├── flux/                   # 4 PNG + тайминги
│   │   └── sdxl/                   # 5 PNG + тайминги
│   └── video/
│       ├── wan22/                  # 3 mp4 (Wan2.2 TI2V 5B)
│       └── cogvideox/              # 3 mp4 (CogVideoX-5B)
├── scripts/
│   ├── ollama_bench_real.py        # сам раннер — Ollama + 4-уровневый корпус + N=1
│   ├── build_dashboard.py          # пересобрать stats_interactive.html
│   ├── build_model_cards.py        # пересобрать MODEL_CARDS.md
│   ├── run_wan22_sdcpp.sh          # Wan2.2 видео на V100 через sd.cpp
│   ├── cogvideox_v100.py           # CogVideoX-5B с VAE FP32 фиксом
│   ├── zimage_inference.py         # Z-Image-Turbo через diffusers (не взлетело, см. чеклист)
│   └── setup_pod.sh                # bootstrap vast.ai пода (CUDA 12.8, Ollama 0.24)
└── raw/
    ├── eval_prompt.txt             # eval prompt (обезличен)
    ├── transcripts/                # 4 обезличенные транскрипции (short/small/medium/large)
    └── models_queue.md             # расписание моделей по подам (где, когда, что)
```

---

## Как читать MODEL_CARDS.md

Каждая карточка выглядит так:

```markdown
### `qwen2.5:7b-instruct-q4_K_M`

**Семья:** Qwen2.5 · **Params:** 7.0 B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
ollama run qwen2.5:7b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0

**Замеры (BVM транскрипция + eval prompt, avg = 102.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
| short | 2684 | 1100 | 2875 | 113.2 | 5530 | stop |
| ... | ... | ... | ... | ... | ... | ... |

**Пример ответа модели** (level=short): [фрагмент вывода]
```

Карточка отвечает на четыре вопроса, которые вы задаёте себе перед тем как запускать модель:

1. **Запустится ли на V100?** — `Status: OK` = прошла все 4 уровня контекста без OOM/таймаута.
2. **Какой командой запускали?** — точная команда `ollama run` + options + env.
3. **Как быстро?** — decode tps на 4 длинах контекста, чтобы видеть деградацию на длинных prompt'ах.
4. **Адекватен ли ответ?** — короткий пример вывода, чтобы убедиться что модель живая.

---

## Как воспроизвести бенч

Понадобится: V100 32GB (одна или две), Ubuntu 22.04, CUDA 12.8+, Ollama 0.24+.

```bash
# 1. Установка (vast.ai template: `cuda-12.8.1-auto`, `amd64`, 200 GB диск)
bash scripts/setup_pod.sh

# 2. Volta-friendly env
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
ollama serve &

# 3. Прогон по списку моделей
python scripts/ollama_bench_real.py \
  --models models_queue.txt \
  --transcripts raw/transcripts/ \
  --prompt raw/eval_prompt.txt \
  --out results/

# 4. Пересобрать дашборд
python scripts/build_dashboard.py
open benchmarks/ollama/stats_interactive.html
```

**Стоимость (vast.ai, май 2026):** 1× V100 32GB SXM2 ≈ $0.45/час. Полный sweep на 108 моделей крутился на 5 подах параллельно ~10 часов → ~$22 на compute. Разбивка — [`docs/COSTS.md`](docs/COSTS.md).

---

## Известные грабли (или: «не повторяйте наши ошибки»)

Краткая версия — полный разбор по строчкам в [`docs/V100_CHECKLIST.md`](docs/V100_CHECKLIST.md).

1. **vLLM из коробки не стартует на Volta.** Берите [форк 1cat](https://github.com/sasha0552/pascal-pkgs-ci) или компилируйте с `TORCH_CUDA_ARCH_LIST=7.0` и `VLLM_ATTENTION_BACKEND=XFORMERS`.
2. **AWQ на 32B+ моделях упирается в SHM 96 КБ wall** в `flash_attn_v100` (`prefill_paged_fwd`), если `head_dim=128` (Qwen2.5, Llama-3.3). Длинные prompt'ы падают, короткие работают. Workaround: переходите на Ollama GGUF.
3. **BF16-эмуляция даёт NaN** в VAE у FLUX / Z-Image → белый PNG ~3 КБ. Фикс в sd.cpp: `--type bf16` (он гоняет через FP32 accum, не нативный BF16).
4. **`torch.bfloat16` в diffusers** молча уходит в CPU-эмуляцию на Volta. Только `torch.float16`, плюс VAE форсом во FP32: `pipe.vae.to(torch.float32)`.
5. **NVLink на vast.ai — лотерея.** `nvidia-smi topo -m` показывает либо `SYS` (PCIe), либо `NV2` (NVLink). 14 мая из 5 наших подов 4 оказались NV2 — у вас будет иначе.
6. **DeepSeek-R1 в Ollama — это НЕ настоящий 671B R1.** Все теги `deepseek-r1:*` — distilled в Llama/Qwen. См. [`docs/MODEL_NAMES_MAPPING.md`](docs/MODEL_NAMES_MAPPING.md).

---

## Почему N=1 хватает (и зачем мы НЕ прогоняли каждую модель 5 раз)

При `temperature=0.0`, single-stream, Ollama даёт коэффициент вариации **≤ 0.10 %** между перепрогонами (замеряли: см. [`docs/VARIANCE_FINDINGS.md`](docs/VARIANCE_FINDINGS.md)). На таком уровне N=5 просто умножает счёт без изменения ответа. Все цифры здесь — N=1, и мы это не скрываем.

Что N=1 НЕ покрывает: thermal throttling на длинных нагрузках, scheduler jitter на multi-tenant подах, batched serving. Если это критично — перепрогоняйте под свой профиль нагрузки.

---

## Лицензия и переиспользование

- **Скрипты, дашборды, документация:** MIT. Копируйте, форкайте, встраивайте в свои публикации — атрибуция приветствуется, но не обязательна.
- **Транскрипции (`raw/transcripts/`):** CC-BY-4.0, обезличены перед публикацией — имена, телефоны, адреса и идентификаторы компаний-клиентов заменены на placeholder'ы. Оригиналы — реальные записи звонков колл-центра BVM; в репо лежит уже почищенный корпус.
- **Ответы моделей (в MODEL_CARDS):** действуют лицензии конкретных моделей. Мы публикуем выводы как доказательство бенча, не как training-данные. У трёх моделей response_preview, где модель эхом процитировала PII из контекста, был замазан (грепаемо: `[redacted: model echoed PII`).

---

## Авторы и контакты

Этот бенч собрала команда [**BVM**](https://bvmax.ru/ai) — мы делаем AI для аналитики звонков отдела продаж и on-prem LLM-деплои (этот V100-бокс крутит наш собственный prod-пайплайн оценки звонков). Статья и репо — инженерный разбор одного уикенда на пределе возможностей этого железа.

- **Telegram (заметки CTO):** [@notes_from_cto](https://t.me/notes_from_cto) — нерегулярные технические посты о том, что мы строим.
- **Сайт BVM:** [bvmax.ru/ai](https://bvmax.ru/ai)
- **Issues / PR:** открывайте здесь — самый быстрый канал по бенчу.
- **Email для коллабораций:** см. GitHub-профиль.

Если нашли баг в наших цифрах — **откройте issue с указанием конкретной модели, пода и JSON-файла из `benchmarks/ollama/raw/`**. Мы хотим, чтобы таблица была корректной.

---

*Сгенерировано 2026-05-16. Последний полный sweep: 2026-05-15. Дашборд автоматически обновляется при появлении новых результатов в `benchmarks/ollama/raw/`.*
