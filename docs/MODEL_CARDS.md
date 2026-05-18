# Карточки моделей — V100 Ollama Real Bench

**Дата:** 2026-05-15 · **Моделей:** 108 · **Pods:** 5× V100 32GB SXM2 NVLink (vast.ai)

**Корпус:** реальные BVM транскрипции звонков колл-центра (4 уровня контекста ~3K/5K/10K/14K tok) + eval prompt «диагностика боли клиента».
**Стек:** Ollama 0.24.0, Q4_K_M (если не указано иное), N=1 (CV ≤ 0.10%, see VARIANCE_FINDINGS.md).
**Env vars:** `OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0`

---

## Содержание

- [Aya](#aya) — 2 моделей
- [CodeLlama](#codellama) — 3 моделей
- [Codestral](#codestral) — 1 моделей
- [Command-R](#command-r) — 1 моделей
- [Command-R+](#command-r-plus) — 1 моделей
- [DeepSeek-Coder](#deepseek-coder) — 1 моделей
- [DeepSeek-R1](#deepseek-r1) — 7 моделей
- [Dolphin-3](#dolphin-3) — 2 моделей
- [Dolphin-Mixtral](#dolphin-mixtral) — 1 моделей
- [EXAONE](#exaone) — 1 моделей
- [Falcon-3](#falcon-3) — 2 моделей
- [GLM-4](#glm-4) — 2 моделей
- [GLM-4.7-Flash](#glm-47-flash) — 1 моделей
- [GLM-OCR](#glm-ocr) — 1 моделей
- [GPT-OSS](#gpt-oss) — 2 моделей
- [Gemma-2](#gemma-2) — 2 моделей
- [Gemma-3](#gemma-3) — 4 моделей
- [Granite](#granite) — 2 моделей
- [Hermes](#hermes) — 1 моделей
- [Huihui-GLM-4.6V](#huihui-glm-46v) — 1 моделей
- [Llama-2](#llama-2) — 3 моделей
- [Llama-3](#llama-3) — 2 моделей
- [Llama-3.1](#llama-31) — 3 моделей
- [Llama-3.2](#llama-32) — 3 моделей
- [Llama-3.3](#llama-33) — 1 моделей
- [Magicoder](#magicoder) — 1 моделей
- [Mistral](#mistral) — 2 моделей
- [Mistral-Large](#mistral-large) — 1 моделей
- [Mistral-Nemo](#mistral-nemo) — 1 моделей
- [Mistral-Small](#mistral-small) — 1 моделей
- [Mixtral](#mixtral) — 2 моделей
- [Moondream](#moondream) — 1 моделей
- [Nemotron](#nemotron) — 1 моделей
- [Nemotron-Mini](#nemotron-mini) — 3 моделей
- [Neural-Chat](#neural-chat) — 1 моделей
- [Nous-Hermes-Mixtral](#nous-hermes-mixtral) — 1 моделей
- [OLMo-2](#olmo-2) — 2 моделей
- [OpenChat](#openchat) — 1 моделей
- [Phi](#phi) — 1 моделей
- [Phi-3](#phi-3) — 3 моделей
- [Phi-4](#phi-4) — 1 моделей
- [QwQ](#qwq) — 1 моделей
- [Qwen](#qwen) — 6 моделей
- [Qwen2.5](#qwen25) — 4 моделей
- [Qwen2.5-Coder](#qwen25-coder) — 3 моделей
- [Qwen3](#qwen3) — 6 моделей
- [SmolLM-2](#smollm-2) — 3 моделей
- [Solar](#solar) — 1 моделей
- [StarCoder-2](#starcoder-2) — 3 моделей
- [Starling](#starling) — 1 моделей
- [Vicuna](#vicuna) — 3 моделей
- [WizardCoder](#wizardcoder) — 1 моделей
- [Yi](#yi) — 1 моделей
- [Zephyr](#zephyr) — 1 моделей
- [codegemma](#codegemma) — 1 моделей

---

## Aya

### `aya-expanse:32b-q4_K_M`

**Семья:** Aya · **Params:** 32.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run aya-expanse:32b-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 28.7 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2684 | 3743 | 717.1 | **32.53** | 7062 | stop |
| small | 4646 | 6192 | 750.3 | **31.19** | 9812 | stop |
| medium | 8192 | 11607 | 705.8 | **25.31** | 15884 | stop |
| large | 8192 | 11606 | 705.8 | **25.6** | 16206 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Повторные повреждения кровли, необходимость надежного ремонта ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ

---

### `aya-expanse:32b`

**Семья:** Aya · **Params:** 32.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run aya-expanse:32b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 28.6 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2684 | 3735 | 718.7 | **32.51** | 7040 | stop |
| small | 4646 | 6182 | 751.6 | **31.21** | 9811 | stop |
| medium | 8192 | 11567 | 708.2 | **25.29** | 15837 | stop |
| large | 8192 | 11600 | 706.2 | **25.56** | 16210 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Повторные повреждения кровли, необходимость надежного ремонта ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ

---

## CodeLlama

### `codellama:7b`

**Семья:** CodeLlama · **Params:** 7.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run codellama:7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 92.3 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3533 | 1186 | 2979.2 | **113.89** | 2423 | stop |
| small | 6103 | 2159 | 2826.1 | **102.24** | 3600 | stop |
| medium | 11667 | 4690 | 2487.6 | **82.24** | 7030 | stop |
| large | 16125 | 7078 | 2278.3 | **70.93** | 8931 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: 21 мансардное окно ПРОСРОЧКА: нет БОЛЬ: вздутия стяжки под рулонным покрытием, трещины и периодические локальные ремонты ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: нет КЦД: ДА

---

### `codellama:13b`

**Семья:** CodeLlama · **Params:** 13.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run codellama:13b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 54.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3533 | 2219 | 1591.9 | **66.89** | 3885 | stop |
| small | 6103 | 4014 | 1520.5 | **60.0** | 7185 | stop |
| medium | 11667 | 8535 | 1367.0 | **49.64** | 10783 | stop |
| large | 16125 | 12787 | 1261.1 | **42.8** | 15461 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: НЕ ВЫЯВЛЕНА ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 0 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ

---

### `codellama:70b`

**Семья:** CodeLlama · **Params:** 70.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run codellama:70b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 17.5 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2048 | 4996 | 409.9 | **17.54** | 14141 | stop |
| small | 2048 | 5028 | 407.3 | **17.45** | 12529 | stop |
| medium | 2048 | 5044 | 406.0 | **17.44** | 14267 | stop |
| large | 2048 | 5039 | 406.4 | **17.53** | 14147 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> DOLG: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: НЕ ВЫЯВЛЕНА ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 0 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ  Please note that the provided transcript is not a complete conversation, and some information may be missing. Additionally, it's important to respect people's privacy and avoid sharing perso

---

## Codestral

### `codestral:22b-v0.1-q4_K_M`

**Семья:** Codestral · **Params:** 22.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run codestral:22b-v0.1-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 35.6 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3803 | 4202 | 905.0 | **40.96** | 9128 | stop |
| small | 6606 | 7378 | 895.4 | **38.29** | 12326 | stop |
| medium | 12855 | 16016 | 802.6 | **33.09** | 23610 | stop |
| large | 17872 | 24187 | 738.9 | **30.13** | 30782 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Повреждение рулонной гидроизоляции и вздутие стяжки на кровле ВОПРОС_О_БОЛИ: ДА (вопросы о проблемах с кровлей, её состоянии) ПРИСОЕДИНЕНИЕ: 2 (клиент открыт к рекомендациям и уточняет специфику кровли) ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: ДА (предложена консультация для начала,

---

## Command-R

### `command-r:35b-08-2024-q4_K_M`

**Семья:** Command-R · **Params:** 35.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run command-r:35b-08-2024-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 30.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2648 | 3751 | 706.0 | **32.54** | 6990 | stop |
| small | 4610 | 6200 | 743.6 | **31.21** | 9523 | stop |
| medium | 8658 | 12367 | 700.1 | **29.3** | 15981 | stop |
| large | 11864 | 17937 | 661.4 | **27.9** | 21834 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Повторные повреждения кровли ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 1 ВОЗРАЖЕНИЯ: Жидкая резина, бюджет КЦД: НЕТ

---

## Command-R+

### `command-r-plus:104b-q4_K_S`

**Семья:** Command-R+ · **Params:** 104.0 B · **Pod:** pod1 · **Status:** FAIL (0/4 levels)

**Конфиг запуска:**
```
ollama run command-r-plus:104b-q4_K_S
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры:** FAIL — модель не запустилась (см. JSON pod1).

---

## DeepSeek-Coder

### `deepseek-coder-v2:16b-lite-instruct-q4_K_M`

**Семья:** DeepSeek-Coder · **Params:** 16.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run deepseek-coder-v2:16b-lite-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 48.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3713 | 2742 | 1354.3 | **80.18** | 4488 | stop |
| small | 6498 | 4863 | 1336.3 | **56.68** | 7325 | stop |
| medium | 12760 | 10692 | 1193.4 | **32.99** | 15080 | stop |
| large | 17492 | 16088 | 1087.3 | **25.21** | 20551 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Ремонт кровли и гидроизоляции (3 слова) ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: ДА

---

## DeepSeek-R1

### `deepseek-r1:8b-q4_K_M`

**Семья:** DeepSeek-R1 · **Params:** 8.0 B · **Pod:** pod3 · **Status:** FAIL (0/4 levels)

**Конфиг запуска:**
```
ollama run deepseek-r1:8b-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры:** FAIL — модель не запустилась (см. JSON pod3).

---

### `deepseek-r1:8b`

**Семья:** DeepSeek-R1 · **Params:** 8.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run deepseek-r1:8b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 86.4 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3139 | 1229 | 2554.4 | **90.7** | 5068 | length |
| small | 5396 | 2080 | 2593.9 | **90.32** | 5884 | length |
| medium | 10185 | 4292 | 2373.0 | **84.5** | 8346 | length |
| large | 14021 | 6358 | 2205.2 | **79.96** | 10640 | length |

---

### `deepseek-r1:14b-q4_K_M`

**Семья:** DeepSeek-R1 · **Params:** 14.0 B · **Pod:** pod3 · **Status:** FAIL (0/4 levels)

**Конфиг запуска:**
```
ollama run deepseek-r1:14b-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры:** FAIL — модель не запустилась (см. JSON pod3).

---

### `deepseek-r1:14b`

**Семья:** DeepSeek-R1 · **Params:** 14.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run deepseek-r1:14b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 52.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3135 | 2080 | 1507.3 | **58.61** | 8060 | length |
| small | 5392 | 3748 | 1438.5 | **55.25** | 10015 | length |
| medium | 10181 | 7539 | 1350.4 | **49.36** | 14534 | length |
| large | 14017 | 11170 | 1254.9 | **45.66** | 18658 | length |

---

### `deepseek-r1:32b-q4_K_M`

**Семья:** DeepSeek-R1 · **Params:** 32.0 B · **Pod:** pod3 · **Status:** FAIL (0/4 levels)

**Конфиг запуска:**
```
ollama run deepseek-r1:32b-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры:** FAIL — модель не запустилась (см. JSON pod3).

---

### `deepseek-r1:32b`

**Семья:** DeepSeek-R1 · **Params:** 32.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run deepseek-r1:32b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 27.3 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3135 | 4545 | 689.7 | **29.63** | 15518 | length |
| small | 5392 | 8161 | 660.7 | **28.42** | 19584 | length |
| medium | 10181 | 15936 | 638.9 | **26.25** | 28261 | length |
| large | 14017 | 23042 | 608.3 | **24.83** | 36080 | length |

---

### `deepseek-r1:70b-llama-distill-q4_K_M`

**Семья:** DeepSeek-R1 · **Params:** 70.0 B · **Pod:** pod1 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run deepseek-r1:70b-llama-distill-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 14.5 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2843 | 6752 | 421.0 | **15.61** | 27015 | length |
| small | 4968 | 11358 | 437.4 | **15.03** | 32327 | length |
| medium | 9451 | 22542 | 419.3 | **14.02** | 44968 | length |
| large | 12911 | 32197 | 401.0 | **13.34** | 55815 | length |

---

## Dolphin-3

### `dolphin3:8b`

**Семья:** Dolphin-3 · **Params:** 8.0 B · **Pod:** pod2 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run dolphin3:8b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 84.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2863 | 975 | 2935.7 | **99.7** | 2762 | stop |
| small | 4988 | 1733 | 2878.4 | **90.48** | 3393 | stop |
| medium | 9471 | 3596 | 2633.6 | **76.9** | 5895 | stop |
| large | 12931 | 5254 | 2461.0 | **69.56** | 7425 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Необходимо долговременное решение для капитального ремонта кровли в центре города. ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: отсутствие надежного долгосрочного решения, периодические локальные ремонты, трещины и вздутия стяжки. КЦД: НЕТ

---

### `dolphin3:8b`

**Семья:** Dolphin-3 · **Params:** 8.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run dolphin3:8b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 101.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2863 | 1001 | 2861.4 | **110.04** | 2315 | stop |
| small | 4988 | 1771 | 2817.2 | **105.78** | 2978 | stop |
| medium | 9471 | 3617 | 2618.2 | **98.0** | 5417 | stop |
| large | 12931 | 5285 | 2446.6 | **93.25** | 6935 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Необходимо долговременное решение для ремонта кровли ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: отсутствие надежного решения, проблемы с гидроизоляцией и стяжкой КЦД: НЕТ

---

## Dolphin-Mixtral

### `dolphin-mixtral:8x7b`

**Семья:** Dolphin-Mixtral · **Params:** 56.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run dolphin-mixtral:8x7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 61.6 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3826 | 5388 | 710.0 | **72.41** | 6930 | stop |
| small | 6629 | 9395 | 705.6 | **66.83** | 10968 | stop |
| medium | 12878 | 19237 | 669.4 | **56.58** | 21297 | stop |
| large | 17895 | 27686 | 646.3 | **50.44** | 29796 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Ремонт плоской кровли ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ

---

## EXAONE

### `exaone-deep:32b`

**Семья:** EXAONE · **Params:** 32.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run exaone-deep:32b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 25.6 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 4355 | 6482 | 671.8 | **29.04** | 17147 | length |
| small | 7373 | 11417 | 645.8 | **27.38** | 22708 | length |
| medium | 14413 | 24199 | 595.6 | **24.12** | 37002 | length |
| large | 20020 | 35307 | 567.0 | **22.04** | 48063 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> <thought>  ДОЛГ: НЕТ (требования к бюджете не упоминаются, оценка не дана) ПРОСРОЧКА: НЕТ (сроков не согласовано) БОЛЬ: регулярные проблемы с вздутием стяжки и трещинами в эксплуатируемой кровле ВОПРОС_О_БОЛИ: ДА (клиент описывает стресс из-за регулярных ремонтов и потенциальных последствий) ПРИСОЕД

---

## Falcon-3

### `falcon3:7b`

**Семья:** Falcon-3 · **Params:** 7.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run falcon3:7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 94.7 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 7267 | 2932 | 2478.9 | **106.85** | 4935 | stop |
| small | 12401 | 5639 | 2199.3 | **98.13** | 9149 | stop |
| medium | 22000 | 12189 | 1804.9 | **88.38** | 12589 | stop |
| large | 22000 | 12195 | 1804.0 | **85.47** | 16587 | length |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: НЕ ВЫЯВЛЕНА ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 1 ВОЗРАЖЕНИЯ: Уточнил адрес, описал кровлю, интересуется современным решениям. КЦД: НЕТ

---

### `falcon3:10b`

**Семья:** Falcon-3 · **Params:** 10.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run falcon3:10b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 67.3 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 7267 | 4157 | 1748.3 | **76.6** | 6453 | stop |
| small | 12401 | 7995 | 1551.1 | **70.45** | 11021 | stop |
| medium | 22000 | 17201 | 1279.0 | **61.13** | 19461 | stop |
| large | 22000 | 17207 | 1278.6 | **61.02** | 20804 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: НЕ ВЫЯВЛЕНА ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: наплавляемые рулонные материалы, ПВХ-мембрана КЦД: ДА

---

## GLM-4

### `glm4:9b`

**Семья:** GLM-4 · **Params:** 9.0 B · **Pod:** pod2 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run glm4:9b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 97.9 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2615 | 1017 | 2570.6 | **103.25** | 2355 | stop |
| small | 4546 | 1815 | 2504.4 | **101.86** | 3263 | stop |
| medium | 8571 | 3728 | 2299.4 | **95.18** | 5181 | stop |
| large | 11755 | 5423 | 2167.4 | **91.45** | 6894 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Повреждение рулонной гидроизоляции и вздутие стяжки ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: ДА

---

### `glm4:9b`

**Семья:** GLM-4 · **Params:** 9.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run glm4:9b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 104.9 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2615 | 1021 | 2561.1 | **108.54** | 2183 | stop |
| small | 4546 | 1818 | 2500.4 | **107.78** | 3196 | stop |
| medium | 8571 | 3713 | 2308.4 | **102.99** | 4895 | stop |
| large | 11755 | 5399 | 2177.3 | **100.37** | 6619 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Повреждение рулонной гидроизоляции и вздутие стяжки ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: ДА

---

## GLM-4.7-Flash

### `glm-4.7-flash:latest`

**Семья:** GLM-4.7-Flash · **Params:** — B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run glm-4.7-flash:latest
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 56.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2588 | 3082 | 839.7 | **61.69** | 8776 | length |
| small | 4509 | 5294 | 851.8 | **58.65** | 11151 | length |
| medium | 8441 | 11311 | 746.3 | **54.09** | 17679 | length |
| large | 11561 | 16978 | 680.9 | **50.42** | 23784 | length |

---

## GLM-OCR

### `glm-ocr`

**Семья:** GLM-OCR · **Params:** — B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run glm-ocr
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 263.1 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2842 | 441 | 6439.6 | **303.42** | 1151 | stop |
| small | 4907 | 692 | 7087.6 | **247.61** | 2258 | length |
| medium | 9356 | 1417 | 6601.0 | **257.58** | 3018 | length |
| large | 12687 | 2011 | 6309.9 | **243.83** | 3758 | length |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: [сумма, если узнал | НЕТ] ПРОСРОЧКА: [срок | НЕТ] БОЛЬ: [главная боль клиента в 3-5 словах | НЕ ВЫЯВЛЕНА] ВОЗРАЖЕНИЯ: [список через запятую | НЕ БЫЛО] КЦД: [был ли назначен конкретный следующий шаг: ДА/НЕТ]

---

## GPT-OSS

### `gpt-oss:20b`

**Семья:** GPT-OSS · **Params:** 20.0 B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run gpt-oss:20b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 106.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2615 | 1630 | 1604.6 | **110.76** | 5180 | length |
| small | 4521 | 2872 | 1573.9 | **107.91** | 6593 | length |
| medium | 8422 | 5526 | 1524.0 | **105.32** | 9383 | length |
| large | 11492 | 7703 | 1491.9 | **103.09** | 11689 | length |

---

### `gpt-oss:120b`

**Семья:** GPT-OSS · **Params:** 120.0 B · **Pod:** pod1 · **Status:** OK (3/4 levels)

**Конфиг запуска:**
```
ollama run gpt-oss:120b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 74.7 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2615 | 3569 | 732.7 | **76.42** | 8428 | length |
| small | 4521 | 6305 | 717.0 | **74.8** | 11284 | length |
| medium | 8422 | 11910 | 707.1 | **72.79** | 17110 | length |
| large | — | — | — | **—** | — | — |

---

## Gemma-2

### `gemma2:9b-instruct-q4_K_M`

**Семья:** Gemma-2 · **Params:** 9.0 B · **Pod:** pod2 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run gemma2:9b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 66.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2595 | 1403 | 1850.0 | **73.79** | 3061 | stop |
| small | 4484 | 2234 | 2006.9 | **69.24** | 4514 | stop |
| medium | 8192 | 4569 | 1793.1 | **61.58** | 6645 | stop |
| large | 8192 | 4542 | 1803.6 | **60.2** | 6583 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: повреждение кровли ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ:   КЦД: ДА

---

### `gemma2:27b-instruct-q4_K_M`

**Семья:** Gemma-2 · **Params:** 27.0 B · **Pod:** pod2 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run gemma2:27b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 32.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2595 | 3468 | 748.3 | **35.04** | 6247 | stop |
| small | 4484 | 5941 | 754.7 | **33.83** | 8745 | stop |
| medium | 8192 | 11861 | 690.7 | **31.01** | 14795 | stop |
| large | 8192 | 11775 | 695.7 | **31.13** | 14638 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Протекающая эксплуатируемая кровля ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ

---

## Gemma-3

### `gemma3:1b`

**Семья:** Gemma-3 · **Params:** 1.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run gemma3:1b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 196.7 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2580 | 205 | 12588.0 | **203.16** | 1355 | stop |
| small | 4489 | 299 | 15028.0 | **172.09** | 1589 | stop |
| medium | 8417 | 608 | 13833.1 | **207.44** | 1945 | stop |
| large | 11586 | 976 | 11867.4 | **204.18** | 2164 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Вздутие стяжки под рулонным покрытием, трещины, локальные ремонты ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 3 ВОЗРАЖЕНИЯ: Вздутие стяжки, трещины, локальные ремонты, нерегулярность покрытия КЦД: ДА

---

### `gemma3:4b`

**Семья:** Gemma-3 · **Params:** 4.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run gemma3:4b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 119.3 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2580 | 611 | 4221.9 | **107.38** | 1982 | stop |
| small | 4489 | 1022 | 4391.6 | **123.37** | 2240 | stop |
| medium | 8417 | 1958 | 4297.9 | **123.73** | 3308 | stop |
| large | 11586 | 2725 | 4251.3 | **122.75** | 4004 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Трещины, вздутия, надежность ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: Несогласие с технологией, запрос консультации КЦД: ДА

---

### `gemma3:12b`

**Семья:** Gemma-3 · **Params:** 12.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run gemma3:12b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 61.9 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2580 | 1524 | 1692.6 | **63.72** | 3254 | stop |
| small | 4489 | 2625 | 1710.0 | **61.93** | 4626 | stop |
| medium | 8417 | 5019 | 1677.2 | **61.31** | 6855 | stop |
| large | 11586 | 7186 | 1612.4 | **60.5** | 9233 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Повреждения кровли, регулярные ремонты ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: ДА

---

### `gemma3:27b`

**Семья:** Gemma-3 · **Params:** 27.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run gemma3:27b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 32.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2580 | 3121 | 826.6 | **33.36** | 6224 | stop |
| small | 4489 | 5520 | 813.2 | **33.04** | 8690 | stop |
| medium | 8417 | 10371 | 811.6 | **32.65** | 13410 | stop |
| large | 11586 | 14269 | 812.0 | **32.21** | 17334 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Поврежденная кровля, вздутие стяжки ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: Неуверенность в жидкой резине КЦД: НЕТ

---

## Granite

### `granite3-dense:8b`

**Семья:** Granite · **Params:** 8.0 B · **Pod:** pod2 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run granite3-dense:8b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 72.9 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 4096 | 1663 | 2462.5 | **71.47** | 3964 | stop |
| small | 4096 | 1676 | 2443.5 | **72.76** | 4388 | stop |
| medium | 4096 | 1680 | 2438.8 | **71.13** | 4278 | stop |
| large | 4096 | 1678 | 2440.7 | **76.3** | 6202 | length |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: 0 ПРОСРОЧКА: НЕТ БОЛЬ: Клиент хочет получить консультацию и оценку состояния кровли, подобрать технологию и решение для капитального ремонта в перспективе. ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЯ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО

---

### `granite3.1-dense:8b-instruct-q4_K_M`

**Семья:** Granite · **Params:** 8.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run granite3.1-dense:8b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 62.9 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 4367 | 1864 | 2343.4 | **78.48** | 3962 | stop |
| small | 7425 | 3362 | 2208.4 | **69.88** | 6437 | stop |
| medium | 14505 | 7502 | 1933.4 | **55.36** | 10421 | stop |
| large | 20027 | 11288 | 1774.1 | **47.73** | 14703 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Вздутие стяжки и трещины на кровле ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 (хорошее) ВОЗРАЖЕНИЯ: Клиент открыт к рекомендациям, уточняет специфику кровли КЦД: НЕТ

---

## Hermes

### `hermes3:70b`

**Семья:** Hermes · **Params:** 70.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run hermes3:70b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 16.1 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2848 | 6892 | 413.3 | **17.43** | 14388 | stop |
| small | 4973 | 11717 | 424.4 | **16.75** | 20349 | stop |
| medium | 9456 | 23158 | 408.3 | **15.42** | 39538 | stop |
| large | 12916 | 32889 | 392.7 | **14.7** | 40659 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: ремонт плоской кровли офисного здания ПРОСРОЧКА: НЕТ БОЛЬ: вздутие стяжки, трещины рулонной гидроизоляции ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: консультация по ремонту кровли, оценка состояния кровли, рекомендации по технологиям гидроизоляции КЦД: НЕТ

---

## Huihui-GLM-4.6V

### `alibilge/Huihui-GLM-4.6V-Flash-abliterated:q4_k_m`

**Семья:** Huihui-GLM-4.6V · **Params:** — B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run alibilge/Huihui-GLM-4.6V-Flash-abliterated:q4_k_m
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 75.3 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2645 | 2163 | 1223.0 | **78.56** | 7019 | length |
| small | 4572 | 3538 | 1292.4 | **76.92** | 8558 | length |
| medium | 8572 | 6968 | 1230.1 | **74.21** | 12131 | length |
| large | 11721 | 9879 | 1186.5 | **71.57** | 15263 | length |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> <think>Got it, let's tackle this step by step. First, I need to go through each field in the requested format.  1. ДОЛГ: The user is asking if there's a debt amount. Looking at the provided text (dialogue and transcript), there's no mention of any debt or financial obligation beyond budget not speci

---

## Llama-2

### `llama2:7b`

**Семья:** Llama-2 · **Params:** 7.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run llama2:7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 72.4 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3533 | 1186 | 2977.8 | **113.92** | 2903 | stop |
| small | 4096 | 1374 | 2981.1 | **60.04** | 3294 | stop |
| medium | 4096 | 1382 | 2963.6 | **57.84** | 3271 | stop |
| large | 4096 | 1381 | 2966.0 | **57.81** | 3307 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> DOLG: 105664 (сумма в рублях) PROSROCHKA: не указано (срок договора или сроки выполнения работ) BOL': "Обработка кровли" (главная боль клиента) VOPROS_O_Boli: да (был вопрос про последствия/эмоции) PRIZOEDINENIE: 3 (сильное присоединение) VOZRACHENIYA: не указано (не было назначено конкретного след

---

### `llama2:13b`

**Семья:** Llama-2 · **Params:** 13.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run llama2:13b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 48.7 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3533 | 2339 | 1510.5 | **66.6** | 4913 | stop |
| small | 4096 | 2565 | 1597.1 | **37.39** | 5632 | stop |
| medium | 4096 | 2564 | 1597.4 | **47.75** | 7388 | stop |
| large | 4096 | 2577 | 1589.4 | **43.2** | 6480 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: 0 (не узнал) ПРОСРОЧКА: НЕТ (срок не согласован) БОЛЬ: НЕ ВЫЯВЛЕНА (не выявлена главная боль клиента) ВОПРОС_О_БОЛИ: НЕТ (не был вопрос про последствия/эмоции) ПРИСОЕДИНЕНИЕ: 2 (хорошее) ВОЗРАЖЕНИЯ: [] (нет) КЦД: НЕТ (не был назначен конкретный следующий шаг)

---

### `llama2:70b`

**Семья:** Llama-2 · **Params:** 70.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run llama2:70b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 16.7 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3533 | 8244 | 428.5 | **17.57** | 13828 | stop |
| small | 4096 | 9465 | 432.8 | **16.55** | 22660 | stop |
| medium | 4096 | 9496 | 431.3 | **16.34** | 19379 | stop |
| large | 4096 | 9507 | 430.8 | **16.42** | 20539 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Не выявлена ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 1 (формальное) ВОЗРАЖЕНИЯ: - КЦД: НЕТ

---

## Llama-3

### `llama3:8b`

**Семья:** Llama-3 · **Params:** 8.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run llama3:8b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 83.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2850 | 996 | 2861.5 | **107.53** | 2567 | stop |
| small | 4975 | 1767 | 2815.3 | **96.93** | 3562 | stop |
| medium | 8192 | 3052 | 2684.3 | **63.74** | 5171 | stop |
| large | 8192 | 3052 | 2684.3 | **64.62** | 5213 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Неудовлетворенность качеством кровли и опасность открытое горение ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: ограничения центра Москвы, нестабильность стяжки, трещины КЦД: НЕТ

---

### `llama3:70b`

**Семья:** Llama-3 · **Params:** 70.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run llama3:70b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 15.5 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2850 | 6851 | 416.0 | **17.47** | 12780 | stop |
| small | 4975 | 11665 | 426.5 | **16.77** | 18815 | stop |
| medium | 8192 | 19846 | 412.8 | **14.06** | 29506 | stop |
| large | 8192 | 19807 | 413.6 | **13.7** | 28034 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> Here is the analysis of the dialogue:  ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Ремонт плоской кровли ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: консультация, оценка состояния кровли, ремонт КЦД: НЕТ

---

## Llama-3.1

### `llama3.1:8b`

**Семья:** Llama-3.1 · **Params:** 8.0 B · **Pod:** pod2 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run llama3.1:8b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 84.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2850 | 980 | 2908.6 | **99.96** | 2371 | stop |
| small | 4975 | 1745 | 2851.5 | **90.22** | 3850 | stop |
| medium | 9458 | 3612 | 2618.8 | **77.01** | 5591 | stop |
| large | 12918 | 5295 | 2439.8 | **69.69** | 7550 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Не выяслена ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 (хорошее) ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ

---

### `llama3.1:8b-instruct-q4_K_M`

**Семья:** Llama-3.1 · **Params:** 8.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run llama3.1:8b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 82.9 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2850 | 657 | 4336.2 | **98.09** | 1965 | stop |
| small | 4975 | 1094 | 4546.9 | **88.64** | 3116 | stop |
| medium | 9458 | 2187 | 4324.8 | **76.06** | 4043 | stop |
| large | 12918 | 3160 | 4088.1 | **68.68** | 5308 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Не выяслена ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 (хорошее) ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ

---

### `llama3.1:70b`

**Семья:** Llama-3.1 · **Params:** 70.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run llama3.1:70b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 15.6 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2850 | 6818 | 418.0 | **16.12** | 12706 | stop |
| small | 4975 | 11553 | 430.6 | **15.9** | 19195 | stop |
| medium | 9458 | 22711 | 416.5 | **15.43** | 29957 | stop |
| large | 12918 | 32114 | 402.3 | **15.13** | 38879 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Повреждение кровли ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: Отсутствие бюджета, ограничения центра Москвы КЦД: НЕТ

---

## Llama-3.2

### `llama3.2:1b-instruct-q4_K_M`

**Семья:** Llama-3.2 · **Params:** 1.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run llama3.2:1b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 299.6 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2865 | 227 | 12625.4 | **352.49** | 1925 | length |
| small | 4990 | 508 | 9831.1 | **322.53** | 2303 | length |
| medium | 9473 | 860 | 11020.9 | **274.58** | 2771 | length |
| large | 12933 | 1315 | 9834.8 | **248.96** | 3428 | length |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> Сумма долга клиента составляет 105664 доллара США.  Сроки консультации и оценки составляют 172 секунды (2 минуты).  Главная боль клиента — это не выраженная, но потенциальная проблема с эксплуатируемой кровлей в центре Москвы.  Вопрос о последствиях/эмоции: нет.  Назначенный следующий шаг: не опреде

---

### `llama3.2:3b-instruct-q4_K_M`

**Семья:** Llama-3.2 · **Params:** 3.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run llama3.2:3b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 157.3 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2865 | 576 | 4974.8 | **180.59** | 1814 | stop |
| small | 4990 | 988 | 5050.6 | **167.33** | 2000 | stop |
| medium | 9473 | 2138 | 4429.8 | **147.03** | 3499 | stop |
| large | 12933 | 3194 | 4049.4 | **134.36** | 4636 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: 0 ПРОСРОЧКА: НЕТ БОЛЬ: НЕ ВЫЯВЛЕНА ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: мансардные окна, эксплуатируемая плитка, рулонная гидроизоляция, вздутие стяжки, трещины, ограничение по открытому огню КЦД: НЕТ

---

### `llama3.2-vision:11b`

**Семья:** Llama-3.2 · **Params:** 11.0 B · **Pod:** pod2 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run llama3.2-vision:11b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 77.3 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2852 | 1000 | 2850.7 | **93.75** | 2474 | stop |
| small | 4977 | 1781 | 2795.0 | **73.8** | 3452 | stop |
| medium | 9460 | 3688 | 2565.0 | **74.88** | 5447 | stop |
| large | 12920 | 5400 | 2392.4 | **66.84** | 7250 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: НЕ ВЫЯВЛЕНА ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ

---

## Llama-3.3

### `llama3.3:70b-instruct-q4_K_M`

**Семья:** Llama-3.3 · **Params:** 70.0 B · **Pod:** pod1 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run llama3.3:70b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 14.6 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2850 | 6719 | 424.2 | **15.77** | 12196 | stop |
| small | 4975 | 11445 | 434.7 | **15.16** | 19434 | stop |
| medium | 9458 | 22588 | 418.7 | **14.14** | 29182 | stop |
| large | 12918 | 32106 | 402.4 | **13.49** | 39093 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Ремонт кровли ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ

---

## Magicoder

### `magicoder:7b`

**Семья:** Magicoder · **Params:** 7.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run magicoder:7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 92.6 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3547 | 1208 | 2935.7 | **113.6** | 4094 | length |
| small | 6117 | 2173 | 2815.2 | **102.45** | 3413 | stop |
| medium | 11681 | 4744 | 2462.5 | **82.72** | 6420 | stop |
| large | 16139 | 7208 | 2238.9 | **71.44** | 8537 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> Долг: 1000000 Просрочка: 30 дней Боль: "Ошибка в системе" Выражение: НЕТ Присоединение: 2 Важные выражения: "Система не работает", "Зависал", "Не отвечает" КЦД: ДА  ---  Проанализируй диалог выше. Ответь СТРОГО в формате:  ДОЛГ: [сумма, если узнал | НЕТ] ПРОСРОЧКА: [срок | НЕТ] БОЛЬ: [главная боль

---

## Mistral

### `mistral:7b-instruct-v0.3-q4_K_M`

**Семья:** Mistral · **Params:** 7.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run mistral:7b-instruct-v0.3-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 79.6 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3802 | 809 | 4698.3 | **98.01** | 2078 | stop |
| small | 6605 | 1404 | 4703.0 | **87.22** | 3076 | stop |
| medium | 12854 | 2983 | 4309.3 | **71.06** | 4684 | stop |
| large | 17871 | 4466 | 4001.6 | **61.91** | 6251 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: Не указана ПРОСРОЧКА: НЕТ БОЛЬ: Офисное помещение, кровля ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 (хорошее) ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: ДА

---

### `mistral:7b`

**Семья:** Mistral · **Params:** 7.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run mistral:7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 79.0 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3802 | 1359 | 2796.7 | **96.9** | 2679 | stop |
| small | 6605 | 2430 | 2718.2 | **86.93** | 4124 | stop |
| medium | 12854 | 5321 | 2415.6 | **70.45** | 7019 | stop |
| large | 17871 | 8043 | 2221.8 | **61.72** | 9818 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: Не указана ПРОСРОЧКА: НЕТ БОЛЬ: Офисное помещение, кровля ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 (хорошее) ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: ДА

---

## Mistral-Large

### `mistral-large:123b-instruct-2407-q3_K_M`

**Семья:** Mistral-Large · **Params:** 123.0 B · **Pod:** pod1 · **Status:** OK (3/4 levels)

**Конфиг запуска:**
```
ollama run mistral-large:123b-instruct-2407-q3_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 2.4 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3801 | 36276 | 104.8 | **3.02** | 71181 | stop |
| small | 6604 | 61369 | 107.6 | **2.58** | 123178 | stop |
| medium | 12853 | 126019 | 102.0 | **1.72** | 204268 | stop |
| large | — | — | — | **—** | — | — |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Проблемы с кровлей и гидроизоляцией ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ

---

## Mistral-Nemo

### `mistral-nemo:12b-instruct-2407-q4_K_M`

**Семья:** Mistral-Nemo · **Params:** 12.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run mistral-nemo:12b-instruct-2407-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 58.9 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2934 | 1566 | 1873.9 | **68.7** | 3499 | stop |
| small | 4941 | 2730 | 1809.8 | **62.82** | 6262 | stop |
| medium | 9417 | 5593 | 1683.6 | **54.44** | 8306 | stop |
| large | 13038 | 8152 | 1599.3 | **49.78** | 11326 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Ремонт плоской кровли ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: ДА

---

## Mistral-Small

### `mistral-small:24b-instruct-2501-q4_K_M`

**Семья:** Mistral-Small · **Params:** 24.0 B · **Pod:** pod2 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run mistral-small:24b-instruct-2501-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 37.9 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3092 | 2856 | 1082.8 | **41.96** | 5985 | stop |
| small | 5099 | 4911 | 1038.3 | **39.7** | 9109 | stop |
| medium | 9575 | 9748 | 982.2 | **36.41** | 14052 | stop |
| large | 13196 | 13871 | 951.4 | **33.69** | 18171 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: повреждение кровли, регулярные проблемы ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ

---

## Mixtral

### `mixtral:8x7b`

**Семья:** Mixtral · **Params:** 56.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run mixtral:8x7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 75.0 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3807 | 5341 | 712.7 | **81.24** | 6830 | stop |
| small | 6610 | 9399 | 703.2 | **77.74** | 11163 | stop |
| medium | 12859 | 18971 | 677.8 | **72.49** | 21220 | stop |
| large | 17876 | 27400 | 652.4 | **68.51** | 28788 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Требуется ремонт плоской кровли ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 (хорошее) ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ

---

### `mixtral:8x22b-instruct-v0.1-q3_K_M`

**Семья:** Mixtral · **Params:** 176.0 B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run mixtral:8x22b-instruct-v0.1-q3_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 5.4 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3802 | 34982 | 108.7 | **7.17** | 51792 | stop |
| small | 6605 | 58971 | 112.0 | **6.2** | 88686 | stop |
| medium | 12854 | 118079 | 108.9 | **4.61** | 147822 | stop |
| large | 17871 | 165137 | 108.2 | **3.56** | 196609 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> DOLG: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Необходимость ремонта кровли и консультация по современным технологиям гидроизоляции ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 (хорошее) ВОЗРАЖЕНИЯ: Нет КЦД: ДА

---

## Moondream

### `moondream`

**Семья:** Moondream · **Params:** — B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run moondream
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 109.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2048 | 193 | 10635.0 | **182.18** | 809 | stop |
| small | 2048 | 199 | 10299.6 | **29.96** | 582 | stop |
| medium | 2048 | 200 | 10261.4 | **113.41** | 712 | stop |
| large | 2048 | 198 | 10318.8 | **113.52** | 731 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> связаться клиента в 3-5 cловах с последствия/эмоции: ДА/НЕТ]

---

## Nemotron

### `nemotron:70b`

**Семья:** Nemotron · **Params:** 70.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run nemotron:70b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 15.6 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2855 | 6805 | 419.5 | **16.08** | 14051 | stop |
| small | 4980 | 11575 | 430.3 | **15.9** | 19836 | stop |
| medium | 9463 | 22686 | 417.1 | **15.44** | 29891 | stop |
| large | 12923 | 32098 | 402.6 | **15.13** | 38943 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Повреждения Кровли ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 (хорошее) ВОЗРАЖЕНИЯ: Отсутствие опыта с новыми технологиями, Ограничения в центре Москвы (открытое горение) КЦД: НЕТ

---

## Nemotron-Mini

### `nemotron-mini:4b-instruct-fp16`

**Семья:** Nemotron-Mini · **Params:** 4.0 B · **Pod:** pod2 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run nemotron-mini:4b-instruct-fp16
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 76.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2742 | 643 | 4264.1 | **89.19** | 2356 | stop |
| small | 4096 | 622 | 6589.1 | **72.9** | 2582 | stop |
| medium | 4096 | 618 | 6623.1 | **69.6** | 2321 | stop |
| large | 4096 | 623 | 6575.8 | **72.98** | 2663 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: 0 ПРОСРОЧКА: Нет БОЛЬ: Трещины и вздутие стяжки под рулонным покрытием ВОПРОС_О_БОЛИ: Да ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: напыление, рубероид, мембрана ПВХ КЦД: Нет

---

### `nemotron-mini:4b`

**Семья:** Nemotron-Mini · **Params:** 4.0 B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run nemotron-mini:4b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 94.1 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2742 | 538 | 5097.8 | **152.36** | 1507 | stop |
| small | 4096 | 788 | 5194.8 | **97.41** | 1912 | stop |
| medium | 4096 | 791 | 5178.4 | **13.67** | 1458 | stop |
| large | 4096 | 796 | 5145.1 | **113.08** | 2529 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: нет ПРОСРОЧКА: нет БОЛЬ: не выявлена ВОПРОС_О_БОЛИ: да ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: не было

---

### `nemotron-mini:4b-instruct-q4_K_M`

**Семья:** Nemotron-Mini · **Params:** 4.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run nemotron-mini:4b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 94.6 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2742 | 532 | 5155.4 | **153.08** | 1420 | stop |
| small | 4096 | 784 | 5226.7 | **97.75** | 1908 | stop |
| medium | 4096 | 790 | 5186.4 | **13.68** | 1403 | stop |
| large | 4096 | 786 | 5213.3 | **113.71** | 2404 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: нет ПРОСРОЧКА: нет БОЛЬ: не выявлена ВОПРОС_О_БОЛИ: да ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: не было

---

## Neural-Chat

### `neural-chat:7b`

**Семья:** Neural-Chat · **Params:** 7.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run neural-chat:7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 85.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3815 | 1350 | 2826.8 | **107.55** | 2461 | stop |
| small | 6618 | 2387 | 2772.5 | **95.14** | 4084 | stop |
| medium | 12867 | 5218 | 2466.0 | **75.47** | 9504 | length |
| large | 17884 | 7913 | 2260.1 | **64.94** | 12890 | length |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: не указана ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 (хорошее) ВОЗРАЖЕНИЯ: [нет] КЦД: НЕТ

---

## Nous-Hermes-Mixtral

### `nous-hermes2-mixtral`

**Семья:** Nous-Hermes-Mixtral · **Params:** — B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run nous-hermes2-mixtral
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 61.1 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3872 | 5367 | 721.5 | **71.77** | 7241 | stop |
| small | 6675 | 9407 | 709.6 | **66.27** | 11264 | stop |
| medium | 12924 | 19299 | 669.7 | **56.26** | 22949 | stop |
| large | 17941 | 27766 | 646.1 | **50.25** | 30278 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Ремонт плоской кровли, повреждение рулонной гидроизоляции и вздутие стяжки ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ

---

## OLMo-2

### `olmo2:7b`

**Семья:** OLMo-2 · **Params:** 7.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run olmo2:7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 91.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 4096 | 1330 | 3080.0 | **91.3** | 2946 | stop |
| small | 4096 | 1337 | 3063.1 | **91.21** | 2730 | stop |
| medium | 4096 | 1331 | 3077.3 | **91.35** | 2791 | stop |
| large | 4096 | 1339 | 3059.1 | **90.82** | 2949 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: | 30 минут ПРОСРОЧКА: | 2 месяца БОЛЬ: | Проблемы с кровлейми ВОПРОС_О_БОЛИ: | Да ПРИСОЕДИНЕНИЕ: | 2 ВОЗРАЖЕНИЯ: | schedule appointment, assess damage, discuss options  КЦД: | Yes, appointment scheduled

---

### `olmo2:13b`

**Семья:** OLMo-2 · **Params:** 13.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run olmo2:13b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 55.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 4096 | 2513 | 1629.7 | **55.26** | 5427 | stop |
| small | 4096 | 2525 | 1621.9 | **55.23** | 7562 | stop |
| medium | 4096 | 2537 | 1614.8 | **55.23** | 5288 | stop |
| large | 4096 | 2536 | 1614.9 | **55.16** | 5064 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: [1000 | НЕТ] ПРОСРОЧКА: [НЕТ] БОЛЬ: [повреждение кровли | НЕ ВЫЯВЛЕНА] ВОПРОС_О_БОЛИ: [ДА] ПРИСОЕДИНЕИ: [2] ВОЗРАЖЕНИЯ: [планирование выезда специалиста, оценка состояния кровли] КЦД: [НЕТ]

---

## OpenChat

### `openchat:7b`

**Семья:** OpenChat · **Params:** 7.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run openchat:7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 87.9 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3815 | 1341 | 2844.1 | **107.18** | 2684 | stop |
| small | 6618 | 2392 | 2767.0 | **94.82** | 4787 | stop |
| medium | 8192 | 3037 | 2697.8 | **77.43** | 6199 | stop |
| large | 8192 | 3044 | 2691.5 | **72.23** | 5329 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: эксплуатируемая кровля с трещинами и вздутиями стяжки ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 (хорошее) ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ

---

## Phi

### `phi:2.7b`

**Семья:** Phi · **Params:** 2.7 B · **Pod:** pod5 · **Status:** FAIL (0/4 levels)

**Конфиг запуска:**
```
ollama run phi:2.7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры:** FAIL — модель не запустилась (см. JSON pod5).

---

## Phi-3

### `phi3:3.8b-mini-128k-instruct-q4_K_M`

**Семья:** Phi-3 · **Params:** 3.8 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run phi3:3.8b-mini-128k-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 62.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3522 | 826 | 4261.5 | **93.75** | 2430 | stop |
| small | 6092 | 1518 | 4014.4 | **71.42** | 3577 | stop |
| medium | 11656 | 3440 | 3388.0 | **46.82** | 7087 | stop |
| large | 16114 | 5345 | 3014.6 | **36.68** | 9537 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> [redacted: model echoed PII from input — see VARIANCE_FINDINGS.md / PII policy]

---

### `phi3:3.8b`

**Семья:** Phi-3 · **Params:** 3.8 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run phi3:3.8b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 63.4 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3522 | 848 | 4152.9 | **94.98** | 3514 | stop |
| small | 6092 | 1547 | 3938.1 | **73.58** | 2905 | stop |
| medium | 11656 | 3492 | 3338.4 | **47.77** | 5061 | stop |
| large | 16114 | 5408 | 2979.5 | **37.2** | 7553 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: НЕТ ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 1 ВОЗРАЖЕНИЯ:   ---  Контекст: В диалоге обсуждаются проблемы с кровлей здания, ремонт и гидроизоляции. Клиент просит консультацию для решения техник покрытия и возможно применение новых технологий, таких как жидкая резина или напыл

---

### `phi3:14b-medium-128k-instruct-q4_K_M`

**Семья:** Phi-3 · **Params:** 14.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run phi3:14b-medium-128k-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 49.9 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3522 | 2255 | 1561.9 | **59.91** | 4031 | stop |
| small | 6092 | 4065 | 1498.7 | **54.28** | 7250 | stop |
| medium | 11656 | 8537 | 1365.3 | **45.42** | 12268 | stop |
| large | 16114 | 12631 | 1275.7 | **40.06** | 15684 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Повреждение кровли и трещины ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 (хорошее) ВОЗРАЖЕНИЯ: НЕ БЫЛО

---

## Phi-4

### `phi4:14b-q4_K_M`

**Семья:** Phi-4 · **Params:** 14.0 B · **Pod:** pod2 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run phi4:14b-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 44.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 4076 | 1570 | 2595.7 | **57.7** | 3873 | stop |
| small | 7088 | 2783 | 2546.5 | **51.42** | 6573 | stop |
| medium | 13743 | 5825 | 2359.3 | **42.18** | 9513 | stop |
| large | 16384 | 7216 | 2270.4 | **27.9** | 12192 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ   ПРОСРОЧКА: НЕТ   БОЛЬ: Ремонт плоской кровли   ВОПРОС_О_БОЛИ: НЕТ   ПРИСОЕДИНЕНИЕ: 1   ВОЗРАЖЕНИЯ: НЕ БЫЛО   КЦД: НЕТ

---

## QwQ

### `qwq:32b-preview-q4_K_M`

**Семья:** QwQ · **Params:** 32.0 B · **Pod:** pod2 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwq:32b-preview-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 27.7 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3168 | 4679 | 677.1 | **30.23** | 9032 | stop |
| small | 5425 | 7881 | 688.4 | **29.08** | 11483 | stop |
| medium | 10214 | 15402 | 663.1 | **26.47** | 22464 | stop |
| large | 14050 | 22382 | 627.8 | **24.95** | 28007 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ  ПРОСРОЧКА: НЕТ  БОЛЬ: Ремонт плоской кровли и проблемы с гидроизоляцией.  ВОПРОС_О_БОЛИ: НЕТ  ПРИСОЕДИНЕНИЕ: 2 (хорошее)  ВОЗРАЖЕНИЯ: НЕ БЫЛО  КЦД: НЕТ (нет конкретного следующего шага)

---

## Qwen

### `qwen:1.8b`

**Семья:** Qwen · **Params:** 1.8 B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen:1.8b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 130.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3271 | 335 | 9751.2 | **272.37** | 2333 | length |
| small | 5598 | 597 | 9380.1 | **248.34** | 2599 | length |
| medium | 0 | 0 | 0 | **0** | 2793 | None |
| large | 0 | 0 | 0 | **0** | 2650 | None |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> С    В  нега: 1: что: ы  СО: 2 6 4 7  ота: 1:   俄: чок: 1:  НТ: 0:  , той,  с: 0:  с:  с: с к: с к в: с к: РО:    я: 2:  гот:  о:  с:  с: с: с: с: с: с: с: с: с: с:  от: с, той, сок: с: с: с: с: сел: ска: сим: с: с: с: с: с: с: ск: с: с: с: с: с: с: с: с: НК: с к: с: с: с: о: с: с: с: с: с: с: с: та

---

### `qwen:1.8b`

**Семья:** Qwen · **Params:** 1.8 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen:1.8b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 130.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3271 | 357 | 9160.1 | **274.36** | 2258 | length |
| small | 5598 | 581 | 9638.1 | **248.8** | 2607 | length |
| medium | 0 | 0 | 0 | **0** | 2688 | None |
| large | 0 | 0 | 0 | **0** | 2583 | None |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> С    В  нега: 1: что: ы  СО: 2 6 4 7  ота: 1:   俄: чок: 1:  НТ: 0:  , той,  с: 0:  с:  с: с к: с к в: с к: РО:    я: 2:  гот:  о:  с:  с: с: с: с: с: с: с: с: с: с:  от: с, той, сок: с: с: с: с: сел: ска: сим: с: с: с: с: с: с: ск: с: с: с: с: с: с: с: с: НК: с к: с: с: с: о: с: с: с: с: с: с: с: та

---

### `qwen:7b`

**Семья:** Qwen · **Params:** 7.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen:7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 89.3 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3271 | 1156 | 2830.4 | **106.64** | 4843 | length |
| small | 5598 | 1978 | 2830.5 | **96.86** | 5860 | length |
| medium | 10399 | 4150 | 2505.5 | **81.91** | 8672 | length |
| large | 14350 | 6147 | 2334.3 | **71.84** | 11284 | length |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: [сумма]  ПРОСРОЧКА: [срок]  БОЛЬ: [напись] ВОЗРАЖЕНИЯ: []  КЦД (компьютерная запись): [запись]  --- СОХИРУЮЩИЕ ДАТЫ И НАМЕ НАМЕ ОПИСКИЕ --- КОД_12345678901234567890 --- КОД_2345678901234567890 --- Строки: [строка] --- Был ли вопрос? [запрос]  --- Ответ: [ответ]  --- Основной текст: [основной т

---

### `qwen:14b`

**Семья:** Qwen · **Params:** 14.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen:14b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 53.7 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3271 | 2138 | 1529.9 | **63.82** | 6141 | stop |
| small | 5598 | 3676 | 1522.7 | **58.6** | 6083 | stop |
| medium | 10399 | 7581 | 1371.7 | **49.13** | 14595 | length |
| large | 14350 | 11033 | 1300.7 | **43.16** | 18999 | length |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: [неизвестно, не уточнил] ПРОСРОЧКА: [неизвестно, не уточнил] БОЛЬ: [нет, болезнь не выявлена] ВОПРОС_О_БОЛЬ: [не задан, не упомянут] ПРИСОЕДИНЕНИЕ: [неуказана степень связи] ВОЗРАЖЕНИЯ: [нет, нет конкретных выражений] КЦД: [неизвестно, не уточнил] ВАРНУТКА: [не задано, не упомянуто]  Итого: кл

---

### `qwen:32b`

**Семья:** Qwen · **Params:** 32.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen:32b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 30.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3271 | 4854 | 673.9 | **33.88** | 7910 | stop |
| small | 5598 | 8215 | 681.5 | **32.29** | 11464 | stop |
| medium | 10399 | 16456 | 631.9 | **29.53** | 20123 | stop |
| large | 14350 | 23331 | 615.1 | **27.46** | 27007 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Плоская кровля (ремонт) ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: НЕТ

---

### `qwen:72b`

**Семья:** Qwen · **Params:** 72.0 B · **Pod:** pod5 · **Status:** FAIL (0/4 levels)

**Конфиг запуска:**
```
ollama run qwen:72b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры:** FAIL — модель не запустилась (см. JSON pod5).

---

## Qwen2.5

### `qwen2.5:7b-instruct-q4_K_M`

**Семья:** Qwen2.5 · **Params:** 7.0 B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen2.5:7b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 102.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3161 | 1100 | 2874.7 | **113.16** | 2387 | stop |
| small | 5418 | 1863 | 2908.0 | **109.09** | 3391 | stop |
| medium | 10207 | 3748 | 2723.7 | **97.33** | 5377 | stop |
| large | 14043 | 5530 | 2539.6 | **91.44** | 7264 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: НЕ ВЫЯВЛЕНА ВОПРОС_О_БОЛИ: НЕТ ПРИСОДЕНИЕ: 1 ВОЗРАЖЕНИЯ: НЕ БЫЛО КЦД: ДА

---

### `qwen2.5:14b-instruct-q4_K_M`

**Семья:** Qwen2.5 · **Params:** 14.0 B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen2.5:14b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 52.9 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3161 | 2153 | 1468.0 | **59.53** | 4541 | stop |
| small | 5418 | 3661 | 1479.9 | **56.14** | 6839 | stop |
| medium | 10207 | 7388 | 1381.5 | **49.88** | 10101 | stop |
| large | 14043 | 10983 | 1278.6 | **45.91** | 13748 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Повреждение кровли, регулярные ремонты ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: Отсутствие бюджета, ограничения по применению открытого огня КЦД: НЕТ

---

### `qwen2.5:32b-instruct-q4_K_M`

**Семья:** Qwen2.5 · **Params:** 32.0 B · **Pod:** pod1 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen2.5:32b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 27.5 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3161 | 4715 | 670.5 | **30.08** | 9411 | stop |
| small | 5418 | 7946 | 681.8 | **28.87** | 12339 | stop |
| medium | 10207 | 15573 | 655.4 | **26.36** | 20043 | stop |
| large | 14043 | 22681 | 619.1 | **24.81** | 27193 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Повреждение кровли и вздутие стяжки ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: Жидкорезина не новая технология, ограничение на открытый огонь в центре города | НЕ БЫЛО КЦД: НЕТ

---

### `qwen2.5:72b-instruct-q4_K_M`

**Семья:** Qwen2.5 · **Params:** 72.0 B · **Pod:** pod1 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen2.5:72b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 13.7 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3161 | 7565 | 417.8 | **14.73** | 14052 | stop |
| small | 5418 | 12768 | 424.3 | **14.18** | 19272 | stop |
| medium | 10207 | 25161 | 405.7 | **13.23** | 32412 | stop |
| large | 14043 | 36082 | 389.2 | **12.53** | 45263 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ   ПРОСРОЧКА: НЕТ   БОЛЬ: Регулярные проблемы с кровлей   ВОПРОС_О_БОЛИ: НЕТ   ПРИСОЕДИНЕНИЕ: 2   ВОЗРАЖЕНИЯ: НЕ БЫЛО   КЦД: ДА

---

## Qwen2.5-Coder

### `qwen2.5-coder:7b-instruct-q4_K_M`

**Семья:** Qwen2.5-Coder · **Params:** 7.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen2.5-coder:7b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 103.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3161 | 1281 | 2467.3 | **113.39** | 2731 | stop |
| small | 5418 | 1902 | 2848.9 | **108.01** | 3615 | stop |
| medium | 10207 | 3836 | 2660.6 | **98.73** | 5388 | stop |
| large | 14043 | 5664 | 2479.3 | **92.53** | 7223 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ   ПРОСРОЧКА: НЕТ   БОЛЬ: Повреждение кровли и регулярные ремонтные работы   ВОПРОС_О_БОЛИ: ДА   ПРИСОЕДИНЕНИЕ: 2   ВОЗРАЖЕНИЯ: Жидкая резина, открытый огонь в центре Москвы   КЦД: ДА

---

### `qwen2.5-coder:14b-instruct-q4_K_M`

**Семья:** Qwen2.5-Coder · **Params:** 14.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen2.5-coder:14b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 52.5 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3161 | 2214 | 1427.4 | **58.96** | 4407 | stop |
| small | 5418 | 3765 | 1439.0 | **55.44** | 6481 | stop |
| medium | 10207 | 7567 | 1348.9 | **49.49** | 10085 | stop |
| large | 14043 | 11225 | 1251.0 | **45.91** | 13851 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ   ПРОСРОЧКА: НЕТ   БОЛЬ: Ремонт плоской кровли   ВОПРОС_О_БОЛИ: НЕТ   ПРИСОЕДИНЕНИЕ: 2   ВОЗРАЖЕНИЯ: Жидкая резина, ПВХ-мембрана   КЦД: ДА

---

### `qwen2.5-coder:32b-instruct-q4_K_M`

**Семья:** Qwen2.5-Coder · **Params:** 32.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen2.5-coder:32b-instruct-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 27.5 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3161 | 4829 | 654.6 | **29.9** | 8393 | stop |
| small | 5418 | 8150 | 664.8 | **28.65** | 12350 | stop |
| medium | 10207 | 15908 | 641.6 | **26.46** | 19971 | stop |
| large | 14043 | 23054 | 609.1 | **25.05** | 27564 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ   ПРОСРОЧКА: НЕТ   БОЛЬ: Повреждение кровли   ВОПРОС_О_БОЛИ: НЕТ   ПРИСОЕДИНЕНИЕ: 2   ВОЗРАЖЕНИЯ: Жидкая резина, бюджетный вариант   КЦД: НЕТ

---

## Qwen3

### `qwen3:0.6b-q4_K_M`

**Семья:** Qwen3 · **Params:** 0.6 B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen3:0.6b-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 190.5 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3147 | 252 | 12487.8 | **204.83** | 1742 | stop |
| small | 5404 | 403 | 13397.1 | **195.48** | 2611 | length |
| medium | 10193 | 970 | 10512.1 | **177.28** | 3319 | length |
| large | 14029 | 1666 | 8419.6 | **184.36** | 3234 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ   ПРОСРОЧКА: НЕТ   БОЛЬ: НЕ ВЫЯВЛЕНА   ВОПРОС_О_БОЛИ: НЕТ   ПРИСОЕДИНЕНИЕ: 0   ВОЗРАЖЕНИЯ: НЕ БЫЛО   КЦД: НЕТ

---

### `qwen3:1.7b-q4_K_M`

**Семья:** Qwen3 · **Params:** 1.7 B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen3:1.7b-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 166.4 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3147 | 421 | 7472.4 | **163.71** | 3018 | length |
| small | 5404 | 659 | 8197.6 | **182.16** | 3011 | length |
| medium | 10193 | 1430 | 7128.3 | **163.2** | 4085 | length |
| large | 14029 | 2210 | 6347.2 | **156.54** | 4939 | length |

---

### `qwen3:4b-q4_K_M`

**Семья:** Qwen3 · **Params:** 4.0 B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen3:4b-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 84.4 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3147 | 907 | 3471.4 | **97.54** | 4707 | length |
| small | 5404 | 1505 | 3589.5 | **92.76** | 5328 | length |
| medium | 10193 | 3212 | 3173.3 | **78.44** | 7710 | length |
| large | 14029 | 4905 | 2860.4 | **69.0** | 10043 | length |

---

### `qwen3:8b-q4_K_M`

**Семья:** Qwen3 · **Params:** 8.0 B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen3:8b-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 69.4 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3147 | 1263 | 2492.6 | **78.28** | 5737 | length |
| small | 5404 | 2133 | 2533.0 | **76.69** | 6545 | length |
| medium | 10193 | 4354 | 2340.9 | **63.91** | 9754 | length |
| large | 14029 | 6485 | 2163.3 | **58.71** | 12309 | length |

---

### `qwen3:14b-q4_K_M`

**Семья:** Qwen3 · **Params:** 14.0 B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen3:14b-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 52.3 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3147 | 2236 | 1407.4 | **57.21** | 8203 | length |
| small | 5404 | 3677 | 1469.7 | **55.49** | 9669 | length |
| medium | 10193 | 7300 | 1396.3 | **50.01** | 13998 | length |
| large | 14029 | 10745 | 1305.7 | **46.6** | 17950 | length |

---

### `qwen3:32b-q4_K_M`

**Семья:** Qwen3 · **Params:** 32.0 B · **Pod:** pod2 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run qwen3:32b-q4_K_M
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 24.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3147 | 5162 | 609.7 | **27.31** | 16807 | length |
| small | 5404 | 8814 | 613.1 | **26.19** | 20898 | length |
| medium | 10193 | 17703 | 575.8 | **23.74** | 31040 | length |
| large | 14029 | 26040 | 538.7 | **22.02** | 40446 | length |

---

## SmolLM-2

### `smollm2:135m`

**Семья:** SmolLM-2 · **Params:** — B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run smollm2:135m
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 424.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 6564 | 259 | 25300.5 | **433.56** | 1351 | length |
| small | 8192 | 353 | 23189.8 | **420.33** | 1478 | length |
| medium | 8192 | 352 | 23252.0 | **421.62** | 1522 | length |
| large | 8192 | 353 | 23213.5 | **421.43** | 1368 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> Выведите комбiперт и словах проблем: ДОЛГ, ПРОСРОЧКА, ВЫХ_МИНЕНИЯ. Выведите комбiперт и словах проблем: ДОЛГ, ПРОСРОЧКА, ВЫХ_МИНЕНИЯ. Выведите стили и просматриваемые в разговоре: ДОЛГ, ПРОСРОЧКА, ВЫХ_МИНЕНИЯ. Выведите стили и просматриваемые в разговоре: ДОЛГ, ПРОСРОЧКА

---

### `smollm2:360m`

**Семья:** SmolLM-2 · **Params:** — B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run smollm2:360m
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 299.4 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 6564 | 319 | 20556.9 | **305.27** | 1661 | length |
| small | 8192 | 437 | 18741.2 | **297.76** | 1834 | length |
| medium | 8192 | 437 | 18738.5 | **296.86** | 1913 | length |
| large | 8192 | 438 | 18721.6 | **297.89** | 1931 | length |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> Консультация по современным технологиям гидроизоляции (жидкая резина/напыление) и выборки для кровли:  Примерно следующий шаг:  ДОЛГ: [сумма, если узнал | НЕТ] ПРОСРОЧКА: [срок | НЕТ] БОЛЬ: [главная боль клиента в 3-5 словах | НЕ ВЫЯВЛЕНА] ВОПРОС_О_БОЛИ: [был ли последствия про ремонта | ДА/НЕТ] ПРИ

---

### `smollm2:1.7b`

**Семья:** SmolLM-2 · **Params:** 1.7 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run smollm2:1.7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 175.0 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 6563 | 785 | 8357.0 | **183.45** | 2764 | length |
| small | 8192 | 1014 | 8082.2 | **172.53** | 2923 | stop |
| medium | 8192 | 1019 | 8036.0 | **172.21** | 2819 | stop |
| large | 8192 | 1018 | 8047.2 | **171.66** | 2230 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> КОЛ-ЦЕНТР: Проанализируй диалог выше. Ответь СТРОГО в формате:  ДОЛГ: [сумма, если узнал | НЕТ] ПРОСРОЧКА: [срок | НЕТ] БОЛЬ: [главная боль клиента в 3-5 словах | НЕ ВЫЯВЛЕНА] ВОПРОС_О_БОЛИ: [был ли вопрос про последствия/эмоции: ДА/НЕТ] ПРИСОЕДИНЕНИЕ: [0-3: 0=нет, 1

---

## Solar

### `solar:10.7b`

**Семья:** Solar · **Params:** 10.7 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run solar:10.7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 67.7 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3815 | 2200 | 1734.3 | **72.22** | 5040 | stop |
| small | 4096 | 2101 | 1949.8 | **66.29** | 6125 | stop |
| medium | 4096 | 2102 | 1948.8 | **66.32** | 6259 | stop |
| large | 4096 | 2105 | 1946.0 | **66.02** | 6003 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: Не установлено (сумма не указана) ПРОСРОЧКА: Не установлено (срок не согласован) БОЛЬ: Ремонт плоской кровли и поиск современных технологий гидроизоляции ВОПРОС_О_БОЛИ: НЕТ (не было вопросов про последствия/эмоции) ПРИСОЕДИНЕНИЕ: 2 (хорошее уровень присоединения) ВОЗРАЖЕНИЯ: Не было (клиент н

---

## StarCoder-2

### `starcoder2:7b`

**Семья:** StarCoder-2 · **Params:** 7.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run starcoder2:7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 104.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 4402 | 1719 | 2560.5 | **0** | 1898 | stop |
| small | 7497 | 3056 | 2453.4 | **177.34** | 3231 | stop |
| medium | 14579 | 6672 | 2185.2 | **159.08** | 6939 | stop |
| large | 16384 | 7674 | 2135.1 | **82.69** | 11749 | length |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> 

---

### `starcoder2:7b`

**Семья:** StarCoder-2 · **Params:** 7.0 B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run starcoder2:7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 100.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 4402 | 1691 | 2602.8 | **0** | 1939 | stop |
| small | 7497 | 3017 | 2485.2 | **177.76** | 3319 | stop |
| medium | 14579 | 6610 | 2205.6 | **151.83** | 6936 | stop |
| large | 16384 | 7595 | 2157.2 | **71.01** | 12386 | length |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> 

---

### `starcoder2:15b`

**Семья:** StarCoder-2 · **Params:** 15.0 B · **Pod:** pod3 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run starcoder2:15b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 75.4 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 4402 | 3513 | 1253.0 | **66.16** | 8382 | length |
| small | 7497 | 6216 | 1206.0 | **117.63** | 6385 | stop |
| medium | 14579 | 13399 | 1088.1 | **59.27** | 18882 | length |
| large | 16384 | 15362 | 1066.5 | **58.36** | 20949 | length |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> [redacted: model echoed PII from input — see VARIANCE_FINDINGS.md / PII policy]

---

## Starling

### `starling-lm:7b-beta`

**Семья:** Starling · **Params:** 7.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run starling-lm:7b-beta
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 90.4 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3815 | 1352 | 2822.3 | **106.63** | 4013 | stop |
| small | 6618 | 2404 | 2753.2 | **94.56** | 5815 | length |
| medium | 8192 | 3066 | 2671.8 | **80.16** | 7092 | length |
| large | 8192 | 3067 | 2670.8 | **80.12** | 7137 | length |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> 1. Долг: Не указан (не уточняется в разговоре) 2. Просрочка: НЕТ (клиент не упоминает просрочки или долги) 3. Боль клиента: Развитие и сохранение здоровья, преодоление болезни 4. Вопрос о последствиях/эмоциях: НЕТ (клиент не задает вопросов о последствиях или эмоциях) 5. Присоединение: 3 (сильное) -

---

## Vicuna

### `vicuna:7b`

**Семья:** Vicuna · **Params:** 7.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run vicuna:7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 72.7 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3554 | 1215 | 2924.9 | **114.4** | 2227 | stop |
| small | 4096 | 1392 | 2943.2 | **58.89** | 3248 | stop |
| medium | 4096 | 1390 | 2947.3 | **58.68** | 3264 | stop |
| large | 4096 | 1392 | 2942.8 | **58.94** | 3279 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: НЕ ВЫЯВЛЕНА ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: [не было] КЦД: ДА

---

### `vicuna:13b`

**Семья:** Vicuna · **Params:** 13.0 B · **Pod:** pod4 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run vicuna:13b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 40.2 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3554 | 2156 | 1648.3 | **67.68** | 3681 | stop |
| small | 4096 | 2499 | 1639.3 | **40.9** | 5898 | stop |
| medium | 4096 | 2501 | 1638.0 | **42.61** | 6184 | stop |
| large | 4096 | 2514 | 1629.0 | **9.73** | 4211 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: 0 руб. ПРОСРОЧКА: НЕТ БОЛЬ: не выявлено ВОПРОС\_О\_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕТ КЦД: ДА

---

### `vicuna:13b`

**Семья:** Vicuna · **Params:** 13.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run vicuna:13b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 39.9 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3554 | 2208 | 1609.9 | **66.88** | 3722 | stop |
| small | 4096 | 2567 | 1595.4 | **40.61** | 5964 | stop |
| medium | 4096 | 2580 | 1587.9 | **42.36** | 6276 | stop |
| large | 4096 | 2581 | 1587.2 | **9.7** | 4209 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: 0 руб. ПРОСРОЧКА: НЕТ БОЛЬ: не выявлено ВОПРОС\_О\_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: НЕТ КЦД: ДА

---

## WizardCoder

### `wizardcoder:33b`

**Семья:** WizardCoder · **Params:** 33.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run wizardcoder:33b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 24.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 4511 | 7112 | 634.2 | **30.5** | 11631 | stop |
| small | 7759 | 12578 | 616.9 | **28.05** | 16250 | stop |
| medium | 15190 | 27326 | 555.9 | **23.41** | 32117 | stop |
| large | 16384 | 29824 | 549.3 | **17.35** | 36623 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: ремонт плоской кровли, консультация по современным технологиям гидроизоляции ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: радость, спокойствие, уверенность КЦД: НЕТ

---

## Yi

### `yi:34b`

**Семья:** Yi · **Params:** 34.0 B · **Pod:** pod2 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run yi:34b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 28.8 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 4096 | 6168 | 664.1 | **29.13** | 11770 | stop |
| small | 4096 | 6178 | 663.0 | **28.86** | 11192 | stop |
| medium | 4096 | 6199 | 660.7 | **28.38** | 10460 | stop |
| large | 4096 | 6201 | 660.5 | **28.92** | 11479 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: НЕ ВЫЯВЛЕНА ВОПРОС_О_БОЛИ: НЕТ ПРИСОЕДИНЕНИЕ: 1 ВОЗРАЖЕНИЯ: [потребность в консультации, оценке состояния кровли; желание получить долговечный решение для эксплуатируемой кровли] КЦД: НЕТ

---

## Zephyr

### `zephyr:7b-beta`

**Семья:** Zephyr · **Params:** 7.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run zephyr:7b-beta
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 85.5 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 3818 | 1357 | 2813.9 | **106.89** | 4460 | length |
| small | 6621 | 2417 | 2738.9 | **94.74** | 5877 | length |
| medium | 12870 | 5290 | 2433.0 | **75.33** | 9621 | length |
| large | 17887 | 7997 | 2236.8 | **65.22** | 12951 | length |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ (клиент не упомянул о долгах) ПРОСРОЧКА: НЕТ (клиент не упомянул о просрочках) БОЛЬ: Клиент сообщает об повреждениях рулонной гидроизоляции и вздутия стяжки под ней. ВОПРОС_О_БОЛИ: НЕ ВЫЯВЛЕНА (клиент не упомянул о последствиях или эмоциях) ПРИСОЕДИНЕНИЕ: 2 (хорошее присоединение клиента)

---

## codegemma

### `codegemma:7b`

**Семья:** codegemma · **Params:** 7.0 B · **Pod:** pod5 · **Status:** OK (4/4 levels)

**Конфиг запуска:**
```
ollama run codegemma:7b
options: num_predict=300, temperature=0.0, num_ctx=22000
env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
```

**Замеры (BVM транскрипция + eval prompt, avg = 65.0 tps):**

| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |
|---|---:|---:|---:|---:|---:|---|
| short | 2608 | 1122 | 2323.5 | **100.92** | 2107 | stop |
| small | 4497 | 2080 | 2161.8 | **96.3** | 3104 | stop |
| medium | 8192 | 4142 | 1977.9 | **31.26** | 6713 | stop |
| large | 8192 | 4149 | 1974.6 | **31.55** | 6768 | stop |

**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):

> ДОЛГ: НЕТ ПРОСРОЧКА: НЕТ БОЛЬ: Ремонт кровли ВОПРОС_О_БОЛИ: ДА ПРИСОЕДИНЕНИЕ: 2 ВОЗРАЖЕНИЯ: Не было КЦД: Нет

---
