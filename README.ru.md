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

Полная таблица (все 108 моделей) — [`docs/FULL_REPORT.md`](docs/FULL_REPORT.md).

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

Этот бенч собрала команда [**BVM**](https://bvm.ai) — мы делаем AI для аналитики звонков отдела продаж и on-prem LLM-деплои (этот V100-бокс крутит наш собственный prod-пайплайн оценки звонков). Статья и репо — инженерный разбор одного уикенда на пределе возможностей этого железа.

- **Telegram (заметки CTO):** [@notes_from_cto](https://t.me/notes_from_cto) — нерегулярные технические посты о том, что мы строим.
- **Сайт BVM:** [bvm.ai](https://bvm.ai)
- **Issues / PR:** открывайте здесь — самый быстрый канал по бенчу.
- **Email для коллабораций:** см. GitHub-профиль.

Если нашли баг в наших цифрах — **откройте issue с указанием конкретной модели, пода и JSON-файла из `benchmarks/ollama/raw/`**. Мы хотим, чтобы таблица была корректной.

---

*Сгенерировано 2026-05-16. Последний полный sweep: 2026-05-15. Дашборд автоматически обновляется при появлении новых результатов в `benchmarks/ollama/raw/`.*
