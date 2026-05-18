# Точные имена моделей — Ollama tag → HuggingFace repo

> Для статьи и таблиц ВАЖНО соблюдать точные оригинальные имена.
> Ollama часто упрощает «deepseek-r1:8b» когда на самом деле это «DeepSeek-R1-Distill-Llama-8B».

## DeepSeek-R1 (distilled)

R1 в Ollama lib — это **distilled** варианты, базовые модели — Llama и Qwen:

| Ollama tag | Полное имя (статья) | HuggingFace | Base |
|---|---|---|---|
| `deepseek-r1:1.5b` | DeepSeek-R1-Distill-Qwen-1.5B | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | Qwen2.5-Math-1.5B |
| `deepseek-r1:7b` | DeepSeek-R1-Distill-Qwen-7B | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | Qwen2.5-Math-7B |
| `deepseek-r1:8b` | DeepSeek-R1-Distill-Llama-8B | [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) | Llama-3.1-8B |
| `deepseek-r1:14b` | DeepSeek-R1-Distill-Qwen-14B | [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) | Qwen2.5-14B |
| `deepseek-r1:32b` | DeepSeek-R1-Distill-Qwen-32B | [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) | Qwen2.5-32B |
| `deepseek-r1:70b` | DeepSeek-R1-Distill-Llama-70B | [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) | Llama-3.3-70B |

**Важно:** «R1 настоящий» — это `deepseek-ai/DeepSeek-R1` (671B MoE), он в Ollama не запакован (слишком большой). Все ollama-теги ниже 671B = дистиллированные ученики.

## Llama family

| Ollama tag | Полное имя | HuggingFace |
|---|---|---|
| `llama2:7b` | Llama 2 7B Chat | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| `llama2:13b` | Llama 2 13B Chat | [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) |
| `llama2:70b` | Llama 2 70B Chat | [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) |
| `llama3:8b` | Meta Llama 3 8B Instruct | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| `llama3:70b` | Meta Llama 3 70B Instruct | [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) |
| `llama3.1:8b` | Meta Llama 3.1 8B Instruct | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| `llama3.2:1b` | Meta Llama 3.2 1B Instruct | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |
| `llama3.2:3b` | Meta Llama 3.2 3B Instruct | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| `llama3.3:70b` | Meta Llama 3.3 70B Instruct | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |

## Qwen family

| Ollama tag | Полное имя | HuggingFace |
|---|---|---|
| `qwen:1.8b/7b/14b/32b/72b` | Qwen 1.5 (legacy gen, Q1 2024) | [Qwen/Qwen1.5-*](https://huggingface.co/Qwen) |
| `qwen2.5:7b/14b/32b/72b` | Qwen 2.5 Instruct | [Qwen/Qwen2.5-*-Instruct](https://huggingface.co/Qwen) |
| `qwen2.5-coder:7b/14b/32b` | Qwen 2.5 Coder Instruct | [Qwen/Qwen2.5-Coder-*-Instruct](https://huggingface.co/Qwen) |
| `qwen3:0.6b…32b` | Qwen 3 (Q1 2025) | [Qwen/Qwen3-*](https://huggingface.co/Qwen) |
| `qwq:32b-preview` | QwQ-32B Preview (reasoning) | [Qwen/QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview) |

## Gemma family

| Ollama tag | Полное имя | HuggingFace |
|---|---|---|
| `gemma2:9b/27b` | Gemma 2 Instruct | [google/gemma-2-*-it](https://huggingface.co/google) |
| `gemma3:1b/4b/12b/27b` | Gemma 3 Instruct (март 2025) | [google/gemma-3-*-it](https://huggingface.co/google) |

## Phi family (Microsoft)

| Ollama tag | Полное имя | HuggingFace |
|---|---|---|
| `phi:2.7b` | Phi-2 | [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) |
| `phi3:3.8b-mini-128k` | Phi-3-mini-128k-Instruct | [microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) |
| `phi3:14b-medium-128k` | Phi-3-medium-128k-Instruct | [microsoft/Phi-3-medium-128k-instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct) |
| `phi4:14b` | Phi-4 | [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) |

## Mistral family

| Ollama tag | Полное имя | HuggingFace |
|---|---|---|
| `mistral:7b-instruct-v0.3` | Mistral 7B Instruct v0.3 | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| `mistral-nemo:12b-instruct-2407` | Mistral NeMo 12B Instruct | [mistralai/Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) |
| `mistral-small:24b-instruct-2501` | Mistral Small 3 (Jan 2025) | [mistralai/Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) |
| `mistral-large:123b-instruct-2407` | Mistral Large 2 (July 2024) | [mistralai/Mistral-Large-Instruct-2407](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) |
| `mixtral:8x7b` | Mixtral 8×7B Instruct | [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) |
| `codestral:22b-v0.1` | Codestral 22B | [mistralai/Codestral-22B-v0.1](https://huggingface.co/mistralai/Codestral-22B-v0.1) |

## Cohere

| Ollama tag | Полное имя | HuggingFace |
|---|---|---|
| `command-r:35b-08-2024` | Command-R 08-2024 | [CohereLabs/c4ai-command-r-08-2024](https://huggingface.co/CohereLabs/c4ai-command-r-08-2024) |
| `command-r-plus:104b` | Command R+ 104B | [CohereLabs/c4ai-command-r-plus](https://huggingface.co/CohereLabs/c4ai-command-r-plus) |
| `aya-expanse:32b` | Aya Expanse 32B | [CohereLabs/aya-expanse-32b](https://huggingface.co/CohereLabs/aya-expanse-32b) |

## Прочее

| Ollama tag | Полное имя | HuggingFace |
|---|---|---|
| `gpt-oss:20b` | OpenAI GPT-OSS 20B | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| `gpt-oss:120b` | OpenAI GPT-OSS 120B | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| `granite3.1-dense:8b` | IBM Granite 3.1 Dense 8B | [ibm-granite/granite-3.1-8b-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct) |
| `nemotron-mini:4b` | NVIDIA Nemotron Mini 4B | [nvidia/Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) |
| `deepseek-coder-v2:16b-lite` | DeepSeek-Coder-V2-Lite-Instruct | [deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) |
| `glm4:9b` | THUDM GLM-4 9B Chat | [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat) |
| `olmo2:7b/13b` | Allen AI OLMo 2 | [allenai/OLMo-2-1124-*-Instruct](https://huggingface.co/allenai) |
| `falcon3:7b/10b` | TII Falcon3 | [tiiuae/Falcon3-*-Instruct](https://huggingface.co/tiiuae) |
| `smollm2:135m/360m/1.7b` | HuggingFace SmolLM2 | [HuggingFaceTB/SmolLM2-*-Instruct](https://huggingface.co/HuggingFaceTB) |
| `dolphin3:8b` | Cognitive Computations Dolphin 3 | [cognitivecomputations/Dolphin3.0-Llama3.1-8B](https://huggingface.co/cognitivecomputations) |
| `hermes3:8b/70b` | Nous Research Hermes 3 | [NousResearch/Hermes-3-Llama-3.*](https://huggingface.co/NousResearch) |

---

## Правило для статьи

В таблицах оставлять **Ollama tag** в моноширинном (читателю проще копировать), но в заголовках разделов и кейсах писать **полное имя**. Например:

> **DeepSeek-R1-Distill-Qwen-32B** (`deepseek-r1:32b`)
> Distilled из Qwen2.5-32B по R1 reasoning traces. Q4_K_M = 19 GB, лезет на 2× V100 NVLink в TP=2 или single-GPU.
