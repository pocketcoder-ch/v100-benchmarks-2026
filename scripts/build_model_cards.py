"""build_model_cards.py — per-model карточки в Markdown.

Для каждой из 108+ моделей: метрики + конфиг + реальный response_preview.
Output: MODEL_CARDS.md (одним файлом + sections by family).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent / "pod_snapshots"
OUT = Path(__file__).parent / "MODEL_CARDS.md"

FAMILY_RX = [
    (r"^smollm2", "SmolLM-2"), (r"^llama3\.3", "Llama-3.3"), (r"^llama3\.2", "Llama-3.2"),
    (r"^llama3\.1", "Llama-3.1"), (r"^llama3", "Llama-3"), (r"^llama2", "Llama-2"),
    (r"^codellama", "CodeLlama"), (r"^qwen2\.5-coder", "Qwen2.5-Coder"),
    (r"^qwen2\.5", "Qwen2.5"), (r"^qwen3", "Qwen3"), (r"^qwen_", "Qwen-1.5"), (r"^qwen", "Qwen"),
    (r"^qwq", "QwQ"), (r"^deepseek-r1", "DeepSeek-R1"), (r"^deepseek-coder", "DeepSeek-Coder"),
    (r"^gemma3", "Gemma-3"), (r"^gemma2", "Gemma-2"), (r"^phi4", "Phi-4"), (r"^phi3", "Phi-3"),
    (r"^phi_", "Phi-2"), (r"^phi", "Phi"), (r"^mistral-large", "Mistral-Large"),
    (r"^mistral-small", "Mistral-Small"), (r"^mistral-nemo", "Mistral-Nemo"),
    (r"^mistral", "Mistral"), (r"^mixtral", "Mixtral"), (r"^codestral", "Codestral"),
    (r"^command-r-plus", "Command-R+"), (r"^command-r", "Command-R"),
    (r"^starcoder", "StarCoder-2"), (r"^wizardcoder", "WizardCoder"), (r"^magicoder", "Magicoder"),
    (r"^granite", "Granite"), (r"^olmo", "OLMo-2"), (r"^falcon", "Falcon-3"),
    (r"^gpt-oss", "GPT-OSS"), (r"^aya", "Aya"), (r"^nemotron-mini", "Nemotron-Mini"),
    (r"^nemotron", "Nemotron"), (r"^solar", "Solar"), (r"^vicuna", "Vicuna"),
    (r"^zephyr", "Zephyr"), (r"^nous-hermes2-mixtral", "Nous-Hermes-Mixtral"),
    (r"^nous-hermes", "Nous-Hermes"), (r"^hermes", "Hermes"), (r"^openchat", "OpenChat"),
    (r"^starling", "Starling"), (r"^neural-chat", "Neural-Chat"),
    (r"^dolphin-mixtral", "Dolphin-Mixtral"), (r"^dolphin3", "Dolphin-3"), (r"^dolphin", "Dolphin"),
    (r"^moondream", "Moondream"), (r"^glm-4\.7-flash", "GLM-4.7-Flash"),
    (r"^glm-ocr", "GLM-OCR"), (r"^glm-4\.7", "GLM-4.7"),
    (r"huihui|glm-4\.6v", "Huihui-GLM-4.6V"), (r"^glm4", "GLM-4"), (r"^glm", "GLM"),
    (r"^yi", "Yi"), (r"^exaone", "EXAONE"), (r"^orca", "Orca"), (r"^stablelm", "StableLM"),
]

def family_of(name):
    n = name.lower()
    for rx, fam in FAMILY_RX:
        if re.search(rx, n): return fam
    return name.split(":")[0].split("_")[0]

def params_of(name):
    n = name.lower()
    m = re.search(r"(\d+)x(\d+(?:\.\d+)?)\s*b", n)
    if m: return float(m.group(1)) * float(m.group(2))
    m = re.search(r"[:_-](\d+(?:\.\d+)?)\s*b", n)
    if m: return float(m.group(1))
    return 0.0

def collect():
    cards = []
    seen = set()
    for pod_dir in sorted(ROOT.iterdir()):
        if not pod_dir.is_dir() or not pod_dir.name.startswith("pod"): continue
        pod = pod_dir.name
        real = pod_dir / "ollama_bench_results" / "real"
        if not real.exists(): continue
        for jf in sorted(real.iterdir()):
            if jf.suffix != ".json": continue
            try: d = json.loads(jf.read_text())
            except Exception: continue
            if "model" not in d: continue
            model = d["model"]
            key = (model, pod)
            if key in seen: continue
            seen.add(key)
            cards.append({"d": d, "pod": pod, "model": model,
                          "family": family_of(model), "params_b": params_of(model)})
    return cards

def fmt_card(c):
    d = c["d"]; m = c["model"]; pod = c["pod"]
    results = d.get("results", [])
    by = {r.get("level"): r for r in results}
    decodes = [r.get("decode_tps") for r in results if r.get("decode_tps") is not None]
    avg = round(sum(decodes) / len(decodes), 1) if decodes else None
    n_ok = sum(1 for r in results if r.get("status") == "OK")
    opt = d.get("ollama_options_used", {})

    lines = []
    lines.append(f"### `{m}`")
    lines.append("")
    lines.append(f"**Семья:** {c['family']} · **Params:** {c['params_b'] or '—'} B · **Pod:** {pod} · "
                 f"**Status:** {'OK' if n_ok else 'FAIL'} ({n_ok}/4 levels)")
    lines.append("")
    lines.append("**Конфиг запуска:**")
    lines.append(f"```")
    lines.append(f"ollama run {m}")
    lines.append(f"options: num_predict={opt.get('num_predict', 300)}, "
                 f"temperature={opt.get('temperature', 0.0)}, num_ctx={opt.get('num_ctx', 22000)}")
    lines.append(f"env: OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0")
    lines.append(f"```")
    lines.append("")
    if avg is not None:
        lines.append(f"**Замеры (BVM транскрипция + eval prompt, avg = {avg} tps):**")
        lines.append("")
        lines.append("| level | input tok | TTFT ms | prefill tps | decode tps | wall ms | done |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for lvl in ("short", "small", "medium", "large"):
            r = by.get(lvl)
            if not r: continue
            lines.append(f"| {lvl} | {r.get('prompt_tokens', '—')} | {r.get('ttft_ms', '—')} | "
                         f"{r.get('prefill_tps', '—')} | **{r.get('decode_tps', '—')}** | "
                         f"{r.get('total_wall_ms', '—')} | {r.get('done_reason', '—')} |")
        lines.append("")
        # response previews
        previews = [(r.get("level"), r.get("response_preview")) for r in results if r.get("response_preview")]
        if previews:
            lines.append("**Пример ответа модели** (уровень `short`, BVM eval prompt «диагностика боли»):")
            lines.append("")
            for lvl, prev in previews[:1]:  # show only short to save space
                preview_clean = (prev or "").replace("\n", " ").strip()[:400]
                lines.append(f"> {preview_clean}")
                lines.append("")
    else:
        lines.append(f"**Замеры:** FAIL — модель не запустилась (см. JSON {pod}).")
        lines.append("")
    return "\n".join(lines)

def main():
    cards = collect()
    cards.sort(key=lambda c: (c["family"], c["params_b"]))
    by_family = {}
    for c in cards:
        by_family.setdefault(c["family"], []).append(c)

    out = []
    out.append("# Карточки моделей — V100 Ollama Real Bench")
    out.append("")
    out.append(f"**Дата:** 2026-05-15 · **Моделей:** {len(cards)} · **Pods:** 5× V100 32GB SXM2 NVLink (vast.ai)")
    out.append("")
    out.append("**Корпус:** реальные BVM транскрипции звонков колл-центра (4 уровня контекста ~3K/5K/10K/14K tok) + eval prompt «диагностика боли клиента».")
    out.append(f"**Стек:** Ollama 0.24.0, Q4_K_M (если не указано иное), N=1 (CV ≤ 0.10%, see VARIANCE_FINDINGS.md).")
    out.append(f"**Env vars:** `OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0`")
    out.append("")
    out.append("---")
    out.append("")
    out.append("## Содержание")
    out.append("")
    for fam in sorted(by_family.keys()):
        out.append(f"- [{fam}](#{fam.lower().replace('.','').replace('+','-plus').replace(' ', '-')}) — {len(by_family[fam])} моделей")
    out.append("")
    out.append("---")
    out.append("")
    for fam in sorted(by_family.keys()):
        anchor = fam.lower().replace('.','').replace('+','-plus').replace(' ', '-')
        out.append(f"## {fam}")
        out.append("")
        for c in by_family[fam]:
            out.append(fmt_card(c))
            out.append("---")
            out.append("")

    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT} ({len(cards)} cards, {sum(len(v) for v in by_family.values())} entries, {len(by_family)} families)")

if __name__ == "__main__":
    main()
