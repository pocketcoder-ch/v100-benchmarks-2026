# Models queue — 40 моделей кандидатов на бенч

> Все в Q4_K_M (если не указано). Размер на диске = params × 0.55 + ~0.5 ГБ метаданных.
> KV cache (NUM_CTX=22000, NUM_PARALLEL=1, FA=1, KV q8) ≈ layers × 8 × 128 × 2 × 1 byte × 22000 = layers × 45 МБ.
> Bench занимает ~10-15 мин на модель (4 точки short/small/medium/large).

## Pod 1 (2× V100 NV1, 64 ГБ total) — крупные TP=2 модели

1. `qwen2.5:72b-instruct-q4_K_M` (47 ГБ) — Qwen2.5 72B dense **HERO**
2. `llama3.3:70b-instruct-q4_K_M` (42 ГБ) — Llama 3.3 70B dense
3. `deepseek-r1:70b-llama-distill-q4_K_M` (42 ГБ) — DeepSeek R1 reasoning distilled to Llama 70B
4. `qwen2.5:32b-instruct-q4_K_M` (19 ГБ) — single GPU, обычная Qwen 32B
5. `command-r-plus:104b-q4_K_S` (~58 ГБ) — Cohere R+ 104B (граничный)
6. `mistral-large:123b-instruct-2407-q3_K_M` (~52 ГБ) — Mistral Large 123B Q3
7. `gpt-oss:120b` (~64 ГБ) — OpenAI gpt-oss MoE 120B (extreme test)

## Pod 2 (2× V100 SYS PCIe, 64 ГБ total) — single GPU модели

> Без NVLink TP=2 на 70B+ слишком медленно. Лучше — 2 single-GPU модели параллельно (GPU0 + GPU1).

8. `qwq:32b-preview-q4_K_M` (~19 ГБ) — Qwen reasoning 32B
9. `gemma2:27b-instruct-q4_K_M` (~16 ГБ) — Gemma 2 27B
10. `qwen3:32b-q4_K_M` (~19 ГБ) — Qwen3 dense 32B (если есть тег)
11. `phi4:14b-q4_K_M` (~9 ГБ) — Microsoft Phi-4 14B
12. `gemma2:9b-instruct-q4_K_M` (~6 ГБ) — Gemma 2 9B
13. `mistral-small:24b-instruct-2501-q4_K_M` (~14 ГБ) — Mistral Small 24B (уже DONE)

## Pod 3 (2× V100 NV2, 64 ГБ total) — medium TP=2 + single

14. `qwen2.5-coder:32b-instruct-q4_K_M` (~19 ГБ) — Qwen Coder 32B **HERO** (уже DONE — повторить с реальными prompts)
15. `qwen2.5-coder:14b-instruct-q4_K_M` (~9 ГБ) — Qwen Coder 14B
16. `qwen2.5-coder:7b-instruct-q4_K_M` (~4.7 ГБ) — Qwen Coder 7B
17. `deepseek-coder-v2:16b-lite-instruct-q4_K_M` (~9 ГБ) — DeepSeek Coder V2 lite
18. `codestral:22b-v0.1-q4_K_M` (~13 ГБ) — Mistral Codestral 22B
19. `deepseek-r1:32b-q4_K_M` (~19 ГБ) — DS-R1 distill Qwen 32B reasoning
20. `deepseek-r1:14b-q4_K_M` (~9 ГБ) — DS-R1 distill Qwen 14B
21. `deepseek-r1:8b-q4_K_M` (~4.9 ГБ) — DS-R1 distill Llama 8B

## Pod 4 (2× V100 NV2, 64 ГБ total) — Qwen3.6/3.5/3 family + GPT-OSS

22. `qwen3:14b-q4_K_M` (~9 ГБ) — Qwen3 14B
23. `qwen3:8b-q4_K_M` (~5 ГБ) — Qwen3 8B
24. `qwen3:4b-q4_K_M` (~2.5 ГБ) — Qwen3 4B
25. `qwen3:1.7b-q4_K_M` (~1.1 ГБ) — Qwen3 1.7B
26. `qwen3:0.6b-q4_K_M` (~0.4 ГБ) — Qwen3 0.6B
27. `gpt-oss:20b` (~12 ГБ) — gpt-oss 20B MoE single GPU
28. `qwen2.5:14b-instruct-q4_K_M` (~9 ГБ) — Qwen2.5 14B
29. `qwen2.5:7b-instruct-q4_K_M` (~4.7 ГБ) — Qwen2.5 7B

## Pod 5 (2× V100 NV2, 64 ГБ total) — Llama / Mistral / others

30. `llama3.1:8b-instruct-q4_K_M` (~4.9 ГБ) — Llama 3.1 8B
31. `llama3.2:3b-instruct-q4_K_M` (~2 ГБ) — Llama 3.2 3B
32. `llama3.2:1b-instruct-q4_K_M` (~0.7 ГБ) — Llama 3.2 1B
33. `mistral:7b-instruct-v0.3-q4_K_M` (~4.4 ГБ) — Mistral 7B v0.3
34. `mistral-nemo:12b-instruct-2407-q4_K_M` (~7.5 ГБ) — Mistral Nemo 12B
35. `phi3:14b-medium-128k-instruct-q4_K_M` (~9 ГБ) — Phi-3 14B
36. `phi3:3.8b-mini-128k-instruct-q4_K_M` (~2.3 ГБ) — Phi-3 3.8B mini
37. `command-r:35b-08-2024-q4_K_M` (~20 ГБ) — Cohere Command-R 35B
38. `granite3.1-dense:8b-instruct-q4_K_M` (~5 ГБ) — IBM Granite 8B
39. `aya-expanse:32b-q4_K_M` (~19 ГБ) — Cohere Aya 32B multilingual
40. `nemotron-mini:4b-instruct-q4_K_M` (~2.7 ГБ) — NVIDIA Nemotron 4B

## VRAM budget на бенч (NUM_CTX=22000, NUM_PARALLEL=1, FA=1, KV q8)

KV cache на модель:
- 70-72B dense (80 layers, 8 kv_heads): 80 × 8 × 128 × 2 × 1 × 22000 = 3.6 ГБ → q8 ÷ 2 = 1.8 ГБ × 2 (K+V) = **3.6 ГБ**
- 32B (64 layers): 64 × 8 × 128 × 2 × 1 × 22000 = **2.9 ГБ**
- 14B (48 layers): 48 × 8 × 128 × 2 × 1 × 22000 = **2.2 ГБ**

С 64 ГБ total VRAM влезает: ~60 ГБ модель (104B Q4_K_S = 58 ГБ) + 3.6 ГБ KV + буфер = 62 ГБ ✅

## Что НЕ лезет (для статьи — записать как «попробовали, не влезло»)

- `llama3.1:405b-q4_K_M` (228 ГБ) — нужно 4× H100 или 8× V100
- `deepseek-v3:671b` (~340 ГБ) — не лезет ни в каком кванте
- `qwen2.5:235b-a22b` MoE — 130 ГБ Q4 → не лезет (5× V100 минимум)
- `dbrx:132b` (80 ГБ) — не лезет в 64 ГБ
- `mixtral:8x22b` (~80 ГБ) — не лезет

## Total: 40 моделей кандидатов

Распределение по подам — 7+6+8+8+11 моделей. На каждом поде последовательно (один пулл + один бенч за раз), параллельно между подами.

Estimated wall-time на ВСЁ:
- Pod 1 (7 моделей, в среднем 25 мин с pullом для крупных): ~3 часа
- Pod 2-5 (по 6-11 моделей, в среднем 12 мин): ~1.5-2 часа

Общий wall-time на 40 моделей: **~2-3 часа** параллельно. Disk overflow risk: после каждой модели делать `ollama rm <model>` чтоб не забить 200 ГБ.
