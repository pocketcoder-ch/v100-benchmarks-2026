# V100 Checklist — 17 граблей и их фиксы

Технический хендбук по Volta sm_70 в 2026 году. Каждый пункт — реально словленная грабля при прогоне бенча 15.05.2026 (5 podов × Tesla V100-SXM2-32GB на Vast.ai). Для каждой грабли: симптом → причина → фикс → ссылка.

> Контекст: V100 = compute capability **7.0** (Volta GV100, 2017). Hardware ограничения:
> 96 KB shared memory per SM, нет BF16/FP8/FP4 path, нет async copy (`cp.async`),
> нет sm_80-instructions. PyTorch wheels под cu128/cu129 уже без sm_70 (PyTorch issue #157517).
> CUDA 12.8 — потолок. После CUDA 13 sm_70 удалён из toolchain.

---

## 1. vLLM stock не стартует на V100

**Симптом:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
или silent `Killed` через 12–22 секунды после `vllm serve`, без traceback.

**Причина:** PyTorch 2.8+/cu128 wheels собраны без sm_70. Upstream vLLM 0.x под Volta — кернелы FlashAttention/marlin требуют cc 7.5+ (Turing+).

**Фикс:** использовать **1Cat-vLLM fork** (https://github.com/1CatAI/1Cat-vLLM) — релиз 1.0.0 от 13.05.2026 с собственным FLASH_ATTN_V100 backend и WMMA AWQ INT4 kernel (портирован из LMDeploy TurboMind SM70).

```bash
pip install \
  https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/flash_attn_v100-1.0.0-cp312-cp312-linux_x86_64.whl \
  https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/vllm-1.0.0-cp312-cp312-linux_x86_64.whl
export VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100
```

Если очень нужно собирать самим — `TORCH_CUDA_ARCH_LIST=7.0` перед pip install + локальный билд PyTorch с sm_70.

**Ссылка:** https://github.com/1CatAI/1Cat-vLLM, vLLM issue #6173 (V100 не поддерживает upstream flash-attn).

---

## 2. FlashAttention 2 не работает на Volta

**Симптом:** падение при импорте `flash_attn`:
```
RuntimeError: FlashAttention only supports Ampere GPUs or newer.
```
или silent NaN, если кернел всё же подгрузился через дев-сборку.

**Причина:** FlashAttention 2 требует sm_80+ (async copy `cp.async`, mma.sync шире, чем есть на Volta). Все попытки «принудительно собрать под sm_70» либо не компилируются, либо падают на старте.

**Фикс:**
- В diffusion-стеке (sd.cpp): **не передавать** `--diffusion-fa`.
- В vLLM: использовать FLASH_ATTN_V100 backend (1cat fork) — кастомная реализация под Volta WMMA.
- В transformers: `attn_implementation="eager"` или `"sdpa"`, никогда не `"flash_attention_2"`.

**Ссылка:** FlashAttention README (https://github.com/Dao-AILab/flash-attention) — Volta явно не поддерживается. FA3 отключён даже на sm_8.6/8.9 — Volta тем более.

---

## 3. SHM 96 KB Wall — `flash_attn_v100` prefill_paged_fwd падает на длинных prompt

**Симптом:**
```
RuntimeError: Shared memory exceeds 96KB: 101888 bytes
```
на моделях с `head_dim=128` (Qwen2.5, Llama-3.3, Mistral-Small) при prompt длиннее ~1100–1800 токенов.

**Причина:** V100 hardware limit = 96 KB shared memory per block. В 1cat-vLLM 1.0.0 kernel зашит с `BLOCK_M=BLOCK_N=128` для `head_dim=128`:
- Q tile: 128×128×2 = 32 768 B
- K tile: 32 768 B
- V tile: 32 768 B
- meta + warp reduction: ~3 584 B
- **Σ = 101 888 B > 96 KB → mandatory crash**

Для `head_dim=256` (Qwen3.5/3.6 family) kernel выбирает меньшие tiles → ~64 KB → работает без лимита.

**Эмпирический safe range (15.05.2026, single-stream):**

| Конфиг | Safe max prompt | Decode t/s |
|---|---|---|
| head_dim=256 single (Qwen3.5/3.6) | без лимита | 9–12 |
| head_dim=128 single GPU (Qwen2.5-32B) | ~1100–1500 | 23–33 |
| head_dim=128 TP=2 NVLink (Qwen2.5-72B, Llama-3.3-70B) | ~1800–1900 | 12–16 |

**Фикс:** нужна пересборка `flash_attn_v100_cuda.cpython-*.so` c `BLOCK_N=64` для head_dim=128 (PR в 1Cat-vLLM). **Workarounds (всё что НЕ помогает):**
- ❌ `--max-num-batched-tokens 2048` (SHM не зависит от chunk)
- ❌ `--block-size 8` (это KV cache page, не prefill tile)
- ❌ `--kv-cache-dtype fp8` (нет HW)
- ❌ `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (отрезан в 1.0.0)
- ❌ Python patch `flash_attn_interface.py` (константы в `.so`)

**Практический обход:**
- Для long-context на V100 — **Qwen3.5/3.6 family** (head_dim=256, без wall'а).
- Для Qwen2.5/Llama-3.3 AWQ — только chat (короткие prompt).
- Альтернатива: уйти на Ollama (llama.cpp под GGUF), там SHM-wall неактуален.

**Ссылка:** Volta Tuning Guide https://docs.nvidia.com/cuda/volta-tuning-guide/index.html. Аналогичная проблема на Turing (sm_75, 64 KB!): vLLM issue #38918 (Gemma4 на Turing).

---

## 4. AWQ vs compressed-tensors — `--quantization awq` FORCE

**Симптом:**
```
ValueError: Quantization scheme not supported. Min capability: 75. Current: 70.
```
при попытке загрузить community AWQ-репо (например `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`), где в `config.json` стоит `quant_method: compressed-tensors`.

**Причина:** upstream vLLM реестр `compressed-tensors` (Marlin kernel) требует sm_75+. На V100 cc=7.0 этот backend отказывается стартовать, даже если веса физически — обычный AWQ INT4.

**Фикс:** всегда явно передавать `--quantization awq` (FORCE), не полагаться на auto-detect из `quantization_config.quant_method`:

```bash
vllm serve cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --quantization awq \           # ← FORCE, не compressed-tensors
  --tensor-parallel-size 2 \
  --max-model-len 32768
```

В 1cat-vLLM это перенаправляет загрузку через WMMA SM70 AWQ kernel.

**Ссылка:** v100_quant_compatibility memory, грабли 4 и 13 в `benchmarks/CHECKLIST.md`.

---

## 5. FP8 / NVFP4 / MXFP4 — нет hardware на cc 7.0

**Симптом:** запуск падает при инициализации весов, либо silent NaN/garbage output. Конкретные сообщения:
```
RuntimeError: torch.float8_e4m3fn is not supported on this device
```
или
```
NotImplementedError: FP8 quantization requires compute capability >= 8.9
```

**Причина:** V100 (cc 7.0) hardware path:

| Формат | Минимальный cc | V100 |
|---|---|---|
| FP16 / INT8 / INT4 | 7.0+ | ✅ |
| BF16 | 8.0+ (Ampere) | ❌ |
| FP8 (e4m3/e5m2) | 8.9+ (Hopper) | ❌ |
| NVFP4 | 10.0+ (Blackwell) | ❌ |
| MXFP4 (gpt-oss) | 8.9+ | ❌ |

**Фикс:** проверять `quantization_config.quant_method` в `config.json` модели **до** скачивания:

```bash
curl -sS -L "https://huggingface.co/<author>/<repo>/raw/main/config.json" | \
  python3 -c "import sys,json; c=json.load(sys.stdin); qc=c.get('quantization_config',{}); \
  print('quant_method:', qc.get('quant_method','none'), '| format:', qc.get('format','-'))"
```

Если выводит `fp8` / `nvfp4` / `mxfp4` — на V100 не запустится, не тратьте диск. Искать AWQ/GPTQ INT4 эквиваленты (авторы: `cpatonn`, `cyankiwi`, `casperhansen`, `hugging-quants`, `kishizaki-sci`, `cognitivecomputations`).

**Ссылка:** v100_quant_compatibility memory.

---

## 6. BF16 в diffusers → CPU emulation → NaN

**Симптом:** `torch_dtype=torch.bfloat16` на V100 либо:
- падает с `RuntimeError: BFloat16 is not supported on this device`;
- silent fallback на CPU (10–100× медленнее, без warning);
- NaN на выходе пайплайна (особенно VAE).

**Причина:** Volta tensor cores не имеют BF16 path. PyTorch может эмулировать BF16 software-side, но (а) только частично, (б) на FP16 операциях VAE происходит overflow (max FP16 ≈ 65 504).

**Фикс:**
- В diffusers ВСЕГДА `torch_dtype=torch.float16` на V100, не bfloat16.
- Для BF16-native моделей (FLUX, SD3, FLUX.2) — VAE форсить в FP32 (см. грабли 7).
- Для cuBLAS GEMM в sd.cpp — `--type bf16` workaround (см. грабли 8).

**Ссылка:** PyTorch issue #157517, HF blog Gregor Koch FLUX.2-on-8×V100 (https://huggingface.co/blog/cronos3k/h100-not-required-32b-flux2-dev-running-on-2017-ha).

---

## 7. FLUX / Z-Image / SD3 — VAE FP16 даёт NaN PNG ~3 KB

**Симптом:** PIL save проходит без ошибки, но PNG = ~3 KB (пустой/чёрный/белый). Warning:
```
RuntimeWarning: invalid value encountered in cast
```
при `(images * 255).round().astype("uint8")`.

**Причина:** На Volta sm_70 FP16 tensor cores имеют ограниченный диапазон, VAE decode операции переполняются → Inf → NaN после нескольких слоёв. Подтверждено на 3 моделях: FLUX.1-dev (sd.cpp), Z-Image-Turbo (diffusers), FLUX.2-dev. На Ampere/Hopper не воспроизводится.

**Фикс (diffusers):**
```python
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                     torch_dtype=torch.float16)
pipe.to("cuda")
pipe.vae.to(dtype=torch.float32)   # ← VAE в FP32 после .to('cuda')
```
Цена: +0.5–1 sec на decode, качество корректное.

**Фикс (stable-diffusion.cpp):**
```bash
sd-cli ... --vae-on-cpu --clip-on-cpu
```
Цена замера 15.05.2026 (FLUX.1-dev Q5_K_S, 1024×1024, 28 steps):
- Без `--vae-on-cpu`: 134 сек, blur/white output ❌
- С `--vae-on-cpu`: 245 сек, sharp output ✅
- SDXL FP16 (для сравнения, FP16-native): 26 сек, sharp ✅

**Ссылка:** Z-Image issue #14 (FP16 NaN в latents) https://github.com/Tongyi-MAI/Z-Image/issues/14. SDXL fixed VAE — известный FP16 NaN баг.

---

## 8. Z-Image / Wan2.2 / FLUX в sd.cpp — cuBLAS COMPUTE_16F overflow

**Симптом:** Z-Image-Turbo Q4_0/Q8_0 GGUF на V100 в stable-diffusion.cpp → полностью чёрные PNG ~3 KB. На RTX 4060 Ti / Ampere+ те же файлы дают нормальную картинку.

**Причина:** sd.cpp по умолчанию использует `CUBLAS_COMPUTE_16F` для matmul. На V100 Tensor Cores промежуточные accumulator-ы переполняются за FP16 range (~65 504) → NaN → чёрный пиксель буфер. На Ampere+ есть hardware-level overflow handling.

**Фикс — `--type bf16` workaround:**
```bash
sd-cli --diffusion-model z_image_turbo-Q8_0.gguf \
  --vae ae.safetensors \
  --llm Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -p "..." --cfg-scale 1.0 --steps 8 \
  --type bf16 \                  # ← BF16 эмуляция через FP32 accum в cuBLAS
  -H 1024 -W 1024 -s 42 -o out.png
```

`--type bf16` форсит BF16 эмуляцию через FP32 accumulation, не требует ggml-cuda.cu патчить. Замер 15.05.2026: **64 сек / 1024² PNG**, всё на GPU, 5/5 промптов прошли.

Полный fix (Volta в whitelist для FP32 accum) — issue #1292, patch в `ggml/src/ggml-cuda/ggml-cuda.cu` line ~1310:
```c
if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc) || cc == GGML_CUDA_CC_VOLTA)
```
По данным автора (tylike): 6.23s → 1.99s (3× быстрее), VRAM 20.3 GB → 7.7 GB. На master sd.cpp **может быть ещё не merged** — проверять руками. До merge — workaround через `--type bf16` работает.

**Ссылка:** https://github.com/leejet/stable-diffusion.cpp/issues/1292.

---

## 9. NVLink topology lottery на Vast.ai — SYS vs NV2

**Симптом:** TP=2 запускается, но decode на 70B AWQ медленный (5–8 t/s вместо ожидаемых 12–16) или вообще hang на shm_broadcast.

**Причина:** Vast.ai не гарантирует NVLink между картами в одном поде. После `nvidia-smi topo -m`:

| Topo | Что значит | Частота |
|---|---|---|
| **SYS** | PCIe + cross-NUMA | ~50% |
| NODE | PCIe в одной NUMA | редко |
| NV1 | 1 NVLink канал (25 GB/s) | редко |
| **NV2** | Полный NVLink (50 GB/s) | 30–50% |

Из 5 podов 14.05.2026: 2 NV2 + 3 SYS. Из 5 podов 15.05.2026: 4 NV2 + 1 SYS (везучая партия).

**Фикс:**
1. Сразу после rent: `nvidia-smi topo -m | head -8`.
2. Если NV2 — гонять TP=2 на 70B+.
3. Если SYS — для TP=2 принудительно: `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 --disable-custom-all-reduce`. Иначе hang на shm_broadcast.
4. Если SYS — лучше брать TP=1 модели (≤32B AWQ).

**Ссылка:** vastai_v100_template, vastai_topology_lottery memories. NVLink топология не отражена в карточке пода до rent.

---

## 10. TP cross-pod не работает — только внутри одной ноды

**Симптом:** идея «5 podов × 64 GB = 320 GB суммарного пула под Qwen3-235B AWQ» не работает. vLLM либо вообще не стартует с remote workers, либо падает на all_reduce.

**Причина:** vLLM TP реализован через NCCL/MPI внутри одной ноды (общая RAM/PCIe). Cross-pod (через интернет) NCCL не масштабируется — latency 10–100 ms vs нужных <50 µs.

**Фикс:** один pod = один worker. Распределять разные модели по разным podам, не одну модель через TP. Для бенча 15.05.2026: 5 podов = 5 параллельных моделей одновременно, не общий пул.

**Ссылка:** vLLM docs (distributed inference), feedback_pod_distribution memory.

---

## 11. Ollama AWQ не запускает (это vLLM формат) — только GGUF

**Симптом:** попытка `ollama create my-awq -f Modelfile` с `FROM ./awq-folder/` либо падает, либо silent fallback на CPU (1–2 t/s = неюзабельно).

**Причина:** Ollama — обёртка над llama.cpp, llama.cpp кернелы работают только с GGUF (single-file packed). AWQ — это vLLM/TensorRT-LLM формат: `.safetensors` + AWQ-scaling метаданные. Форматы несовместимы на уровне CUDA kernel.

**Фикс:**
- Для Ollama использовать только GGUF (Q2/Q3/Q4/Q5/Q6/Q8 от bartowski / unsloth / lmstudio-community).
- Если очень нужен AWQ-чекпойнт в Ollama — конвертация AWQ → fp16 safetensors → GGUF через `llama.cpp/convert_hf_to_gguf.py`. Лишняя возня, лучше сразу искать GGUF-вариант.
- Для AWQ моделей на V100 — путь через 1cat-vLLM.

**Ссылка:** awq_vs_gguf_ollama memory. См. также: на 2× V100 64 GB суммарно **не лезет** Qwen3-235B даже в Q2_K (85.7 GB) — никакой quant не помогает.

---

## 12. RunPod disk quota trap — `df` врёт, vllm падает молча

**Симптом:** `vllm serve` завершается без traceback, 0-байтный лог. `df -h /workspace` показывает 364T / 162T free — вроде места дофига.

**Причина:** RunPod использует NFS для `/workspace`. `df` показывает размер pool (общее хранилище провайдера), а не **per-pod квоту** (~300 GB). При превышении квоты — `OSError: [Errno 122] Disk quota exceeded` (EDQUOT). Возникает при попытке записать даже маленький commit-hash файл в `hf_cache`. vLLM от этого падает без stack trace.

**Фикс:**
1. Контролировать через `du`, не `df`:
   ```bash
   du -sh /workspace/.venv-* /workspace/hf_cache
   ```
2. Держать `/workspace` ≤ 260 GB (буфер 40 GB).
3. После каждого бенча: `rm -rf /workspace/hf_cache/hub/models--<model>` (удалять забенченную модель).
4. `df` на RunPod NFS — бесполезен для контроля квоты.

**Ссылка:** runpod_disk_quota memory.

---

## 13. RunPod CUDA filter — нужен CUDA ≥ 12.8 при деплое

**Симптом:** случайные node-level баги (kernel mismatch, странные `CUDA error: unknown error`, NCCL crash на старте). На некоторых нодах PyTorch 2.8+ просто отказывается стартовать.

**Причина:** RunPod не фильтрует ноды по версии CUDA / драйвера. Старые ноды (driver 525 / CUDA 12.0) бракованные для всего, что собрано под cu128 (PyTorch wheels 2.8/2.9, vLLM 0.18+, 1Cat-vLLM 1.0.0).

**Фикс:** при деплое pod явно ставить filter:
- **RunPod:** в Pod template указать CUDA Version `>= 12.8`.
- **Vast.ai:** в Extra Filters
  ```
  cpu_arch in ['amd64'] cuda_max_good>=12.8
  ```
- Image: `vastai/pytorch:cuda-12.8.1-auto` (НЕ `latest` — он на cu130/13.0, ломает совместимость c cu128-wheels).

**Ссылка:** runpod_cuda_filter, vastai_v100_template memories.

---

## 14. Pod patience — не убивать vllm pkill сразу, оставляет orphan VRAM

**Симптом:** после `pkill -9 vllm` следующий запуск падает:
```
CUDA out of memory. Tried to allocate ... GB
```
хотя `nvidia-smi` показывает 30+ GB занято непонятно чем.

**Причина:** SIGKILL не даёт CUDA runtime освободить VRAM. Allocations остаются как orphan на драйвере. На Volta это особенно неприятно (нет MIG, нет нормального cgroup cleanup), процесс мог уже сделать sync_loop bind на /dev/nvidia*.

**Фикс:**
1. При hang vllm — **ждать ошибку в логе**, не убивать сразу. Часто vllm всё-таки докрутит инициализацию за 5–15 минут (особенно на TP=2 с долгим NCCL warmup).
2. Если убивать необходимо — `SIGTERM` сначала, только потом `SIGKILL`.
3. После SIGKILL — очистка orphan:
   ```bash
   fuser -k -9 /dev/nvidia*       # форсит refcount cleanup
   nvidia-smi --gpu-reset         # если есть права (на vast.ai обычно нет)
   ```
4. Если ничего не помогло — restart pod-а (cheaper than $0.45/час waste на orphan VRAM).

**Ссылка:** feedback_pod_patience memory, грабля 1 в `benchmarks/CHECKLIST.md`.

---

## 15. DeepSeek-R1 в Ollama — distilled, не настоящий R1 671B

**Симптом:** `ollama pull deepseek-r1:70b` скачивает 42 GB, запускается на V100, выдаёт нормальные ответы. Создаёт впечатление «о, R1 671B работает на V100!». Это не так.

**Причина:** Ollama library `deepseek-r1` tag-и — это **distilled** модели:
- `deepseek-r1:1.5b` / `:7b` / `:8b` / `:14b` / `:32b` → DeepSeek-R1-Distill-Qwen-{1.5,7,14,32}B
- `deepseek-r1:70b` → DeepSeek-R1-Distill-Llama-70B
- Настоящий R1 671B — это `deepseek-r1:671b` (FP16 = 1.4 TB, Q4 ~370 GB, **не лезет ни в какую V100-конфигурацию**).

Distilled — это SFT-fine-tune Qwen/Llama на R1-trajectories. Архитектура — base Qwen2 / Llama-3, не deepseek_v3.

**Фикс:** в статьях / отчётах писать **полное имя** модели:
- ❌ «DeepSeek-R1 работает на V100 за 12 t/s»
- ✅ «DeepSeek-R1-Distill-Llama-70B (Q4) на 2× V100 NVLink — 12 t/s»

**Ссылка:** `docs/MODEL_NAMES_MAPPING.md` в этом репо.

---

## 16. Neural Magic → Red Hat AI миграция (FP8/W8A8 переехали)

**Симптом:** при попытке скачать FP8-квант от Neural Magic:
```bash
huggingface-cli download neuralmagic/Llama-3.3-70B-Instruct-FP8-dynamic
# → HTTP 404
```

**Причина:** Neural Magic куплен Red Hat (2024). Официальные FP8/W8A8-кванты Llama/Qwen/Mistral переехали с `neuralmagic/*` на `RedHatAI/*`. Старые 3.1-серии репо ещё живут, новые 3.3+ — только под RedHatAI namespace.

**Подтверждено 2026-05-07:**
- ❌ `neuralmagic/Llama-3.3-70B-Instruct-FP8-dynamic` → 404
- ✅ `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` → 200
- ✅ `RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8` → 200
- ✅ `nvidia/Llama-3.3-70B-Instruct-FP8` → 200 (NVIDIA-выпуск, отдельный namespace)
- ✅ `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8` → 200 (старая 3.1, ещё живёт)

**Фикс:** в скриптах/Modelfile всегда писать `RedHatAI/...` для FP8/W8A8 от Neural Magic, не `neuralmagic/...`.

**Внимание:** для V100 это всё равно не пригодится — **FP8 нет HW path на cc 7.0** (грабли 5). Записываем как «общая референс-инфа», чтобы не залипать на 404 при просмотре старых HOWTO.

**Ссылка:** neuralmagic_redhat_migration memory.

---

## 17. HF model 404 — некоторые модели сняты с раздачи

**Симптом:**
```
huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error.
Repository Not Found for url: https://huggingface.co/<author>/<repo>/...
```

**Причина:** На HuggingFace модель может стать недоступной по нескольким причинам:
- Автор удалил репо (rename, миграция в org-namespace, см. грабли 16).
- Gated access — нужно принять лицензию через web-интерфейс (Llama, Mistral, Gemma).
- HF убрал модель за TOS violation (редко).
- Модель перемещена в другую организацию.

**Фикс:**
1. Pre-launch audit — curl-проверка `config.json` через HF API **до** того, как класть в shortlist:
   ```bash
   curl -sS -L -o /dev/null -w "%{http_code}\n" \
     "https://huggingface.co/<author>/<repo>/raw/main/config.json"
   ```
   Если `200` — репо живо, `401` — gated, `404` — нет.
2. Для gated: получить HF token (`HF_TOKEN=hf_...`), принять лицензию через web.
3. Для исчезнувших — искать reupload по тегам `quantization_config.quant_method` через HF search API.
4. Держать локальный список «verified HF repos» (см. `[[verified_hf_repos]]` memory).

**Ссылка:** грабля 5 и 14 в `benchmarks/CHECKLIST.md`.

---

## Сводная команда запуска (template для V100)

```bash
# AWQ модель single GPU (32B и меньше, или MoE с активацией <14B)
SLUG=<slug> HF_REPO=<author>/<model> VENV=1cat TP=1 CUDA_DEV=0 MML=<native> \
  QUANT_FLAG='--quantization awq' \         # ВСЕГДА awq, не compressed-tensors!
  GPU_UTIL=0.85 DURATION=300 PORT=8000 \
  nohup /root/sweep_one_model.sh > /root/sweep_<slug>.log 2>&1 &

# 70B+ AWQ TP=2 на NVLink-pod (только если topo показал NV1/NV2)
SLUG=<slug> HF_REPO=<author>/<70b-awq> VENV=1cat TP=2 CUDA_DEV=0,1 MML=16384 \
  QUANT_FLAG='--quantization awq' \
  GPU_UTIL=0.90 DURATION=300 PORT=8000 \
  NCCL_P2P_DISABLE=0 \
  nohup /root/sweep_one_model.sh > /root/sweep_<slug>.log 2>&1 &
```

## Что не лезет на 2× V100 NVLink (64 GB суммарно)

| Модель | Params | Min quant size | На 2× V100 |
|---|---|---|---|
| Mixtral-8x22B (141B) | 141B | AWQ 78 GB / Q3 60 GB | ⚠ Q3 впритык |
| gpt-oss-120b | 120B | AWQ 66 GB | ⚠ KV режется в ноль |
| Qwen3-235B-A22B | 235B | Q2_K 85.7 GB | ❌ wall |
| DeepSeek-V3 / R1 | 671B | Q4 ~370 GB | ❌ wall |
| Llama-4-Scout (109B) | 109B | AWQ 55 GB | ⚠ KV режется |
| Mistral-Large 123B | 123B | Q3_K_M 59 GB | ✅ впритык |
| Qwen3-Coder-480B-A35B | 480B | AWQ 264 GB | ❌ wall |

Подробнее: `docs/MODEL_CARDS.md` и `awq_vs_gguf_ollama` memory.

## Ссылки

- vLLM fork под Volta: https://github.com/1CatAI/1Cat-vLLM
- stable-diffusion.cpp issue #1292 (cuBLAS overflow fix): https://github.com/leejet/stable-diffusion.cpp/issues/1292
- HF blog «FLUX.2 on 8× V100»: https://huggingface.co/blog/cronos3k/h100-not-required-32b-flux2-dev-running-on-2017-ha
- vLLM SHM-wall analog на Turing: https://github.com/vllm-project/vllm/issues/38918
- Volta Tuning Guide: https://docs.nvidia.com/cuda/volta-tuning-guide/index.html
- Ollama V100 flash-attn issue: https://github.com/ollama/ollama/issues/10859

---

_Хендбук собран командой BVM по итогам прогона 108 моделей на 5× V100 SXM2 (Vast.ai) 15.05.2026. Полные числовые результаты — `docs/FULL_REPORT.md` и `benchmarks/`._
