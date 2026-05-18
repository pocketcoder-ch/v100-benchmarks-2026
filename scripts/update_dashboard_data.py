"""update_dashboard_data.py — обновляет inline `const DATA = [...]` в dashboard.html
без затрагивания верстки/CSS/SVG/JS. Также пересохраняет data.json и detail.json.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent / "pod_snapshots"
DASH = Path(__file__).parent / "dashboard.html"
DATA_JSON = Path(__file__).parent / "data.json"
DETAIL_JSON = Path(__file__).parent / "detail.json"

FAMILY_RX = [
    (r"^smollm2", "SmolLM-2"),
    (r"^llama3\.3", "Llama-3.3"),
    (r"^llama3\.2", "Llama-3.2"),
    (r"^llama3\.1", "Llama-3.1"),
    (r"^llama3", "Llama-3"),
    (r"^llama2", "Llama-2"),
    (r"^codellama", "CodeLlama"),
    (r"^qwen2\.5-coder", "Qwen2.5-Coder"),
    (r"^qwen2\.5", "Qwen2.5"),
    (r"^qwen3", "Qwen3"),
    (r"^qwen_", "Qwen-1.5"),
    (r"^qwen", "Qwen"),
    (r"^qwq", "QwQ"),
    (r"^deepseek-r1", "DeepSeek-R1"),
    (r"^deepseek-coder", "DeepSeek-Coder"),
    (r"^gemma3", "Gemma-3"),
    (r"^gemma2", "Gemma-2"),
    (r"^phi4", "Phi-4"),
    (r"^phi3", "Phi-3"),
    (r"^phi_", "Phi-2"),
    (r"^phi", "Phi"),
    (r"^mistral-large", "Mistral-Large"),
    (r"^mistral-small", "Mistral-Small"),
    (r"^mistral-nemo", "Mistral-Nemo"),
    (r"^mistral", "Mistral"),
    (r"^mixtral", "Mixtral"),
    (r"^codestral", "Codestral"),
    (r"^command-r-plus", "Command-R+"),
    (r"^command-r", "Command-R"),
    (r"^starcoder", "StarCoder-2"),
    (r"^wizardcoder", "WizardCoder"),
    (r"^magicoder", "Magicoder"),
    (r"^granite", "Granite"),
    (r"^olmo", "OLMo-2"),
    (r"^falcon", "Falcon-3"),
    (r"^gpt-oss", "GPT-OSS"),
    (r"^aya", "Aya"),
    (r"^nemotron-mini", "Nemotron-Mini"),
    (r"^nemotron", "Nemotron"),
    (r"^solar", "Solar"),
    (r"^vicuna", "Vicuna"),
    (r"^zephyr", "Zephyr"),
    (r"^hermes", "Hermes"),
    (r"^nous-hermes2-mixtral", "Nous-Hermes-Mixtral"),
    (r"^nous-hermes", "Nous-Hermes"),
    (r"^openchat", "OpenChat"),
    (r"^starling", "Starling"),
    (r"^neural-chat", "Neural-Chat"),
    (r"^dolphin-mixtral", "Dolphin-Mixtral"),
    (r"^dolphin3", "Dolphin-3"),
    (r"^dolphin", "Dolphin"),
    (r"^moondream", "Moondream"),
    (r"^glm-4\.7-flash", "GLM-4.7-Flash"),
    (r"^glm-ocr", "GLM-OCR"),
    (r"^glm-4\.7", "GLM-4.7"),
    (r"^glm-4\.6v|huihui", "Huihui-GLM-4.6V"),
    (r"^glm4", "GLM-4"),
    (r"^glm", "GLM"),
    (r"^yi", "Yi"),
    (r"^exaone", "EXAONE"),
    (r"^orca", "Orca"),
    (r"^stablelm", "StableLM"),
    (r"^stable-code", "StableCode"),
    (r"^tinyllama", "TinyLlama"),
]

def family_of(name: str) -> str:
    n = name.lower()
    for rx, fam in FAMILY_RX:
        if re.search(rx, n):
            return fam
    return name.split(":")[0].split("_")[0]

def params_of(model_name: str) -> float:
    n = model_name.lower()
    m = re.search(r"(\d+)x(\d+(?:\.\d+)?)\s*b", n)
    if m:
        return float(m.group(1)) * float(m.group(2))
    m = re.search(r"[:_-](\d+(?:\.\d+)?)\s*b", n)
    if m:
        return float(m.group(1))
    m = re.search(r"^(\d+(?:\.\d+)?)\s*b", n)
    if m:
        return float(m.group(1))
    m = re.search(r"_(\d+(?:\.\d+)?)\.?json|_(\d+(?:\.\d+)?)$", n)
    return 0.0

def bucket_of(b: float) -> str:
    if b == 0: return "?"
    if b < 5: return "<5B"
    if b < 10: return "5-9B"
    if b < 20: return "10-19B"
    if b < 30: return "20-29B"
    if b < 50: return "30-49B"
    if b < 80: return "50-79B"
    return "80B+"

def collect():
    rows = []
    detail = []
    seen = set()
    for pod_dir in sorted(ROOT.iterdir()):
        if not pod_dir.is_dir() or not pod_dir.name.startswith("pod"):
            continue
        pod = pod_dir.name
        real = pod_dir / "ollama_bench_results" / "real"
        if not real.exists():
            continue
        for jf in sorted(real.iterdir()):
            if jf.suffix != ".json":
                continue
            try:
                with open(jf) as f:
                    d = json.load(f)
            except Exception:
                continue
            if "model" not in d:
                continue
            model = d["model"]
            key = (model, pod)
            if key in seen:
                continue
            seen.add(key)
            results = d.get("results", [])
            by = {r["level"]: r for r in results if "level" in r}
            params_b = params_of(model) or params_of(jf.stem)
            row = {
                "model": model,
                "pod": pod,
                "status": "OK" if any(r.get("status") == "OK" for r in results) else "FAIL",
                "short":  by.get("short",  {}).get("decode_tps"),
                "small":  by.get("small",  {}).get("decode_tps"),
                "medium": by.get("medium", {}).get("decode_tps"),
                "large":  by.get("large",  {}).get("decode_tps"),
                "ttft_short": by.get("short", {}).get("ttft_ms"),
                "ttft_large": by.get("large", {}).get("ttft_ms"),
                "tok_short": by.get("short", {}).get("prompt_tokens"),
                "tok_large": by.get("large", {}).get("prompt_tokens"),
                "prefill_short": by.get("short", {}).get("prefill_tps"),
                "prefill_large": by.get("large", {}).get("prefill_tps"),
                "wall_total_s": round(sum(r.get("total_wall_ms", 0) for r in results) / 1000, 3),
                "params_b": params_b,
                "bucket": bucket_of(params_b),
                "family": family_of(model),
            }
            decodes = [row[k] for k in ("short","small","medium","large") if row[k] is not None]
            row["avg"] = round(sum(decodes) / len(decodes), 4) if decodes else None
            rows.append(row)
            for r in results:
                detail.append({
                    "model": model, "pod": pod,
                    "level": r.get("level"),
                    "prompt_tokens": r.get("prompt_tokens"),
                    "output_tokens": r.get("output_tokens"),
                    "ttft_ms": r.get("ttft_ms"),
                    "prefill_tps": r.get("prefill_tps"),
                    "decode_tps": r.get("decode_tps"),
                    "total_wall_ms": r.get("total_wall_ms"),
                    "done_reason": r.get("done_reason"),
                    "status": r.get("status"),
                })
    return rows, detail

def main():
    rows, detail = collect()
    print(f"collected {len(rows)} models, {len(detail)} detail rows")

    # write standalone JSONs
    DATA_JSON.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    DETAIL_JSON.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {DATA_JSON} {len(rows)} rows")
    print(f"wrote {DETAIL_JSON} {len(detail)} rows")

    # inplace replace `const DATA = [...]` in dashboard.html (single line block)
    html = DASH.read_text(encoding="utf-8")
    new_data_line = "const DATA = " + json.dumps(rows, ensure_ascii=False) + ";"
    new_html, n = re.subn(r"^const DATA = \[.*?\];\s*$", new_data_line, html, count=1, flags=re.M | re.S)
    if n == 0:
        print("WARN: const DATA line not found, dashboard.html not updated")
    else:
        DASH.write_text(new_html, encoding="utf-8")
        print(f"updated {DASH} (DATA = {len(rows)} models)")

    # update header counts in HTML title/subtitle if present
    html2 = DASH.read_text(encoding="utf-8")
    html2 = re.sub(r"(<title>[^<]*?)(\d+)( models?</title>)", lambda m: f"{m.group(1)}{len(rows)}{m.group(3)}", html2)
    html2 = re.sub(r"(\b)\d+( моделей)", lambda m: f"{m.group(1)}{len(rows)}{m.group(2)}", html2)
    DASH.write_text(html2, encoding="utf-8")

if __name__ == "__main__":
    main()
