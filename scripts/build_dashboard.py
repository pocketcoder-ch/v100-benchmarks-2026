"""build_dashboard.py — stats_interactive.html: единая таблица LLM/image/video, белая тема.

Запуск: python3 build_dashboard.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent / "pod_snapshots"
OUT_HTML = Path(__file__).parent / "stats_interactive.html"

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
    (r"^flux", "FLUX"), (r"^sdxl|^stable-diffusion", "SDXL"), (r"^z[-_]image|zimage", "Z-Image"),
    (r"^sulphur", "Sulphur"), (r"^ltx", "LTX-Video"), (r"^wan", "Wan"),
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
    m = re.search(r"^(\d+(?:\.\d+)?)\s*b", n)
    if m: return float(m.group(1))
    return 0.0

def bucket_of(b):
    if b == 0: return "?"
    if b < 5: return "<5B"
    if b < 10: return "5-9B"
    if b < 20: return "10-19B"
    if b < 30: return "20-29B"
    if b < 50: return "30-49B"
    if b < 80: return "50-79B"
    return "80B+"

def collect_llm():
    rows, seen = [], set()
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
            results = d.get("results", [])
            by = {r["level"]: r for r in results if "level" in r}
            params_b = params_of(model) or params_of(jf.stem)
            decodes = [by.get(lvl, {}).get("decode_tps") for lvl in ("short","small","medium","large")]
            decodes_ok = [d for d in decodes if d is not None]
            avg = round(sum(decodes_ok) / len(decodes_ok), 1) if decodes_ok else None
            rows.append({
                "type": "LLM",
                "model": model, "family": family_of(model),
                "params_b": params_b, "bucket": bucket_of(params_b), "pod": pod,
                "status": "OK" if any(r.get("status") == "OK" for r in results) else "FAIL",
                "primary": avg, "primary_label": "tps avg" if avg else "",
                "short": decodes[0], "small": decodes[1], "medium": decodes[2], "large": decodes[3],
                "avg": avg,
                "ttft_short": by.get("short", {}).get("ttft_ms"),
                "ttft_large": by.get("large", {}).get("ttft_ms"),
                "prefill_short": by.get("short", {}).get("prefill_tps"),
                "prefill_large": by.get("large", {}).get("prefill_tps"),
                "tok_large": by.get("large", {}).get("prompt_tokens"),
                "wall_total_s": round(sum(r.get("total_wall_ms", 0) or 0 for r in results) / 1000, 1),
                "n_levels": sum(1 for r in results if r.get("status") == "OK"),
                "resolution": None, "steps": None, "wall_sec": None, "peak_vram_gb": None,
            })
    return rows

def collect_image():
    rows = []
    p1 = ROOT / "pod1"
    sources = ["diffusion_bench_results", "diffusion_v100_clean", "diffusion_bench_results_v3",
               "diffusion_bench_results_v2", "diffusion_bench_results_final", "zimage_out",
               "zimage_out_sdcpp"]
    raw = []
    for sub in sources:
        d = p1 / sub
        if not d.exists(): continue
        for f in sorted(d.iterdir()):
            if f.suffix not in (".jsonl", ".json"): continue
            try:
                if f.suffix == ".jsonl":
                    for line in f.read_text().splitlines():
                        if not line.strip(): continue
                        r = json.loads(line); r["_source"] = sub; raw.append(r)
                else:
                    obj = json.loads(f.read_text())
                    if isinstance(obj, list):
                        for r in obj: r["_source"] = sub; raw.append(r)
                    else:
                        obj["_source"] = sub; raw.append(obj)
            except Exception: continue
    for r in raw:
        wall = r.get("wall_sec") or r.get("wall_time") or 0
        if wall and wall < 0.5: continue  # phantom Ollama image gen (0.02s = didn't run)
        model = r.get("model", "?")
        rows.append({
            "type": "image",
            "model": model, "family": family_of(model),
            "params_b": 0, "bucket": "?", "pod": "pod1",
            "status": "OK" if wall and wall > 0.5 else ("FAIL" if "error" in r else "?"),
            "primary": round(wall, 1) if wall else None, "primary_label": "sec/img" if wall else "",
            "short": None, "small": None, "medium": None, "large": None, "avg": None,
            "ttft_short": None, "ttft_large": None, "prefill_short": None, "prefill_large": None,
            "tok_large": None,
            "wall_total_s": round(wall, 1) if wall else None, "n_levels": None,
            "resolution": r.get("resolution", "1024x1024"),
            "steps": r.get("steps"),
            "wall_sec": round(wall, 1) if wall else None,
            "peak_vram_gb": r.get("peak_vram_gb"),
            "prompt_id": r.get("prompt_id", ""),
            "stack": r.get("stack", ""),
        })
    return rows

def collect_video():
    """Wan2.2 (sd.cpp .mp4.avi) + CogVideoX (diffusers .mp4) results."""
    rows = []
    raw = []
    for podname, sub in [("pod1", "wan22_out"), ("pod5", "cogvideox_out")]:
        d = ROOT / podname / sub
        if not d.exists(): continue
        for f in sorted(d.iterdir()):
            if f.suffix not in (".jsonl", ".json"): continue
            try:
                if f.suffix == ".jsonl":
                    for line in f.read_text().splitlines():
                        if not line.strip(): continue
                        r = json.loads(line); r["_pod"] = podname; r["_source"] = sub; raw.append(r)
                else:
                    obj = json.loads(f.read_text())
                    if isinstance(obj, list):
                        for r in obj: r["_pod"] = podname; r["_source"] = sub; raw.append(r)
                    else:
                        obj["_pod"] = podname; obj["_source"] = sub; raw.append(obj)
            except Exception: continue
    for r in raw:
        wall = r.get("wall_sec") or 0
        model = r.get("model", "?")
        rows.append({
            "type": "video",
            "model": model, "family": family_of(model),
            "params_b": 0, "bucket": "?",
            "pod": r.get("_pod", "?"),
            "status": "OK" if wall and wall > 5 else ("FAIL" if "error" in r else "?"),
            "primary": round(wall, 1) if wall else None,
            "primary_label": "sec/video" if wall else "",
            "short": None, "small": None, "medium": None, "large": None, "avg": None,
            "ttft_short": None, "ttft_large": None, "prefill_short": None, "prefill_large": None,
            "tok_large": None,
            "wall_total_s": round(wall, 1) if wall else None, "n_levels": None,
            "resolution": r.get("resolution", ""),
            "steps": r.get("frames") or r.get("steps"),
            "wall_sec": round(wall, 1) if wall else None,
            "peak_vram_gb": r.get("peak_vram_gb"),
            "prompt_id": r.get("prompt_id", ""),
            "stack": r.get("stack", ""),
        })
    return rows

def annotate_runs(rows):
    """For each (model, type) group, add runs_count / runs_avg / runs_range to every row."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["type"])].append(r.get("primary"))
    for r in rows:
        vals = [v for v in groups[(r["model"], r["type"])] if v is not None]
        r["runs_count"] = len(groups[(r["model"], r["type"])])
        if len(vals) >= 2:
            avg = sum(vals) / len(vals)
            r["runs_avg"] = round(avg, 1)
            r["runs_range"] = f"{min(vals):.1f}..{max(vals):.1f}"
            spread = max(vals) - min(vals)
            r["runs_spread_pct"] = round(spread / avg * 100, 1) if avg else None
        elif len(vals) == 1:
            r["runs_avg"] = vals[0]; r["runs_range"] = ""; r["runs_spread_pct"] = 0.0
        else:
            r["runs_avg"] = None; r["runs_range"] = ""; r["runs_spread_pct"] = None
    return rows

def main():
    rows = collect_llm() + collect_image() + collect_video()
    rows = annotate_runs(rows)
    rows.sort(key=lambda r: (r["type"] != "LLM", r["primary"] is None, -(r["primary"] or 0)))
    print(f"total rows: {len(rows)} (LLM={sum(1 for r in rows if r['type']=='LLM')}, image={sum(1 for r in rows if r['type']=='image')})")

    BUCKETS = ["<5B", "5-9B", "10-19B", "20-29B", "30-49B", "50-79B", "80B+", "?"]
    families = sorted({r["family"] for r in rows})
    pods = sorted({r["pod"] for r in rows})

    data_json = json.dumps(rows, ensure_ascii=False)
    fam_opts = "".join(f'<option>{f}</option>' for f in families)
    pod_opts = "".join(f'<option>{p}</option>' for p in pods)
    bucket_pills = "".join(f'<span class="pill" data-bucket="{b}">{b}</span>' for b in BUCKETS)

    CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif; margin: 0; padding: 18px 24px; background: #f6f7f9; color: #1a1a1a; font-size: 13px; -webkit-font-smoothing: antialiased; }
h1 { font-size: 18px; margin: 0 0 4px; color: #0a0a0a; font-weight: 600; letter-spacing: -0.01em; }
.sub { font-size: 12px; color: #6e7681; margin-bottom: 14px; line-height: 1.5; }
.controls { background: #fff; padding: 12px 14px; border-radius: 10px; border: 1px solid #e2e4e8; margin-bottom: 12px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; position: sticky; top: 0; z-index: 50; }
.controls > div { display: flex; align-items: center; gap: 6px; font-size: 12px; }
.controls label { color: #6e7681; font-weight: 500; }
input[type=text], input[type=number], select { padding: 5px 9px; font-size: 12px; border: 1px solid #d0d3da; border-radius: 6px; background: #fff; color: #1a1a1a; font-family: inherit; transition: border-color 0.15s, box-shadow 0.15s; }
input:focus, select:focus { outline: none; border-color: #1f6feb; box-shadow: 0 0 0 3px rgba(31,111,235,0.1); }
input[type=number] { width: 64px; }
.search-input { width: 240px; }
.pills { display: flex; gap: 4px; flex-wrap: wrap; }
.pill { padding: 3px 10px; border: 1px solid #d0d3da; border-radius: 20px; cursor: pointer; background: #fff; font-size: 11px; user-select: none; transition: all 0.12s; color: #6e7681; }
.pill.active { background: #1f6feb; color: #fff; border-color: #1f6feb; font-weight: 600; }
.pill:hover:not(.active) { background: #f0f1f4; color: #1a1a1a; }
.pill-type { font-weight: 600; }
.pill-type[data-type="LLM"].active { background: #1a7f37; border-color: #1a7f37; }
.pill-type[data-type="image"].active { background: #bc4c00; border-color: #bc4c00; }
.pill-type[data-type="video"].active { background: #8250df; border-color: #8250df; }
.count-badge { margin-left: auto; padding: 5px 12px; background: rgba(31,111,235,0.1); color: #1f6feb; border-radius: 14px; font-size: 11px; font-weight: 600; font-variant-numeric: tabular-nums; }
.btn-reset { padding: 5px 12px; border: 1px solid #d0d3da; border-radius: 6px; cursor: pointer; background: #fff; color: #6e7681; font-size: 11px; font-weight: 500; transition: all 0.12s; font-family: inherit; }
.btn-reset:hover { background: #f0f1f4; color: #1a1a1a; }
.table-wrap { background: #fff; border-radius: 10px; border: 1px solid #e2e4e8; overflow: auto; max-height: calc(100vh - 200px); }
table { border-collapse: separate; border-spacing: 0; width: 100%; font-size: 12px; }
th, td { padding: 7px 10px; text-align: left; border-bottom: 1px solid #eef0f3; white-space: nowrap; }
thead th { background: #f0f1f4; cursor: pointer; user-select: none; position: sticky; top: 0; z-index: 10; font-weight: 600; font-size: 10.5px; color: #6e7681; text-transform: uppercase; letter-spacing: 0.04em; border-bottom: 2px solid #e2e4e8; transition: color 0.12s; }
thead th:hover { color: #0a0a0a; }
thead th.sorted { color: #1f6feb; }
thead th.sorted::after { content: " ↓"; }
thead th.sorted.asc::after { content: " ↑"; }
th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; font-feature-settings: "tnum"; }
.sticky-col { position: sticky; left: 0; background: #fff; z-index: 5; box-shadow: 1px 0 0 #eef0f3; }
thead .sticky-col { background: #f0f1f4; z-index: 15; }
tbody tr { transition: background 0.08s; }
tbody tr:hover td { background: #f5f7fa; }
tbody tr:hover td.sticky-col { background: #f5f7fa; }
.heat { position: relative; }
.heat::before { content: ""; position: absolute; inset: 0; opacity: var(--heat-opacity, 0); background: var(--heat-color, transparent); pointer-events: none; z-index: 0; }
.heat > span { position: relative; z-index: 1; }
.avg { font-weight: 700; }
.status-OK { color: #1a7f37; font-size: 11px; font-weight: 500; }
.status-FAIL { color: #cf222e; font-weight: 600; font-size: 11px; }
.muted { color: #8b949e; }
.model-cell { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11.5px; max-width: 340px; overflow: hidden; text-overflow: ellipsis; color: #0a0a0a; }
.family-tag { display: inline-block; padding: 1px 8px; border-radius: 10px; background: rgba(31,111,235,0.1); color: #1f6feb; font-size: 10.5px; font-weight: 500; }
.bucket-tag { display: inline-block; padding: 1px 7px; border-radius: 4px; background: #f6f7f9; border: 1px solid #e2e4e8; font-size: 10.5px; color: #6e7681; font-variant-numeric: tabular-nums; }
.pod-tag { font-size: 10.5px; color: #6e7681; font-family: ui-monospace, monospace; }
.type-LLM { display: inline-block; padding: 1px 7px; border-radius: 4px; background: rgba(26,127,55,0.12); color: #1a7f37; font-size: 10.5px; font-weight: 600; }
.type-image { display: inline-block; padding: 1px 7px; border-radius: 4px; background: rgba(188,76,0,0.12); color: #bc4c00; font-size: 10.5px; font-weight: 600; }
.type-video { display: inline-block; padding: 1px 7px; border-radius: 4px; background: rgba(130,80,223,0.12); color: #8250df; font-size: 10.5px; font-weight: 600; }
"""

    JS = """
const DATA = __DATA__;
const STATE = { sortKey: "primary", sortDir: -1, bucket: null, type: null };

function fmt(v, p) { if (v === null || v === undefined) return ""; if (typeof v === "number") return v.toFixed(p ?? 1); return v; }
function fmtInt(v) { return v === null || v === undefined ? "" : Math.round(v); }
function heat(v) {
  if (v === null || v === undefined) return "";
  const t = Math.min(v / 200, 1);
  let hue;
  if (t < 0.15) hue = 0 + (t / 0.15) * 30;
  else if (t < 0.40) hue = 30 + ((t - 0.15) / 0.25) * 80;
  else hue = 110 + ((t - 0.40) / 0.60) * 20;
  const op = 0.10 + t * 0.32;
  return `--heat-color: hsl(${hue}, 70%, 55%); --heat-opacity: ${op.toFixed(2)};`;
}

function apply() {
  const q = document.getElementById("fSearch").value.toLowerCase();
  const fam = document.getElementById("fFam").value;
  const pod = document.getElementById("fPod").value;
  const st = document.getElementById("fStatus").value;
  const mn = parseFloat(document.getElementById("fMin").value);
  const mx = parseFloat(document.getElementById("fMax").value);
  let cur = DATA.filter(r => {
    if (STATE.type && r.type !== STATE.type) return false;
    if (q && !r.model.toLowerCase().includes(q) && !r.family.toLowerCase().includes(q)) return false;
    if (fam && r.family !== fam) return false;
    if (pod && r.pod !== pod) return false;
    if (st && r.status !== st) return false;
    if (!isNaN(mn) && (r.params_b ?? 0) < mn) return false;
    if (!isNaN(mx) && (r.params_b ?? 0) > mx) return false;
    if (STATE.bucket && r.bucket !== STATE.bucket) return false;
    return true;
  });
  cur.sort((a, b) => {
    let av = a[STATE.sortKey], bv = b[STATE.sortKey];
    if (av == null) av = STATE.sortDir > 0 ? Infinity : -Infinity;
    if (bv == null) bv = STATE.sortDir > 0 ? Infinity : -Infinity;
    if (typeof av === "string") return av.localeCompare(bv) * STATE.sortDir;
    return (av - bv) * STATE.sortDir;
  });
  const tb = document.querySelector("#tbl tbody");
  tb.innerHTML = cur.map(r => `
    <tr>
      <td class="sticky-col model-cell" title="${r.model}">${r.model}</td>
      <td><span class="type-${r.type}">${r.type}</span></td>
      <td><span class="family-tag">${r.family}</span></td>
      <td class="num">${r.params_b || ""}</td>
      <td><span class="bucket-tag">${r.bucket}</span></td>
      <td><span class="pod-tag">${r.pod}</span></td>
      <td class="num heat avg" style="${heat(r.primary)}"><span>${fmt(r.primary)}</span></td>
      <td class="num muted">${r.primary_label || ""}</td>
      <td class="num muted">${r.runs_count > 1 ? r.runs_count + "×" : ""}</td>
      <td class="num muted">${r.runs_count > 1 ? fmt(r.runs_avg) : ""}</td>
      <td class="num muted" title="${r.runs_range || ''}">${r.runs_count > 1 && r.runs_spread_pct !== null ? r.runs_spread_pct.toFixed(1) + '%' : ''}</td>
      <td class="num heat" style="${heat(r.short)}"><span>${fmt(r.short)}</span></td>
      <td class="num heat" style="${heat(r.small)}"><span>${fmt(r.small)}</span></td>
      <td class="num heat" style="${heat(r.medium)}"><span>${fmt(r.medium)}</span></td>
      <td class="num heat" style="${heat(r.large)}"><span>${fmt(r.large)}</span></td>
      <td class="num muted">${fmtInt(r.ttft_short)}</td>
      <td class="num muted">${fmtInt(r.ttft_large)}</td>
      <td class="num muted">${fmtInt(r.prefill_short)}</td>
      <td class="num muted">${fmtInt(r.prefill_large)}</td>
      <td class="num muted">${fmtInt(r.tok_large)}</td>
      <td class="num muted">${r.resolution || ""}</td>
      <td class="num muted">${r.steps || ""}</td>
      <td class="num muted">${fmt(r.peak_vram_gb)}</td>
      <td class="num muted">${fmt(r.wall_total_s)}</td>
      <td class="num muted">${r.n_levels !== null ? (r.n_levels + "/4") : ""}</td>
      <td class="status-${r.status}">${r.status}</td>
    </tr>`).join("");
  document.querySelectorAll("thead th").forEach(th => {
    th.classList.remove("sorted", "asc");
    if (th.dataset.key === STATE.sortKey) { th.classList.add("sorted"); if (STATE.sortDir > 0) th.classList.add("asc"); }
  });
  document.getElementById("count").textContent = `показано ${cur.length} / ${DATA.length}`;
}

document.querySelectorAll("thead th").forEach(th => th.addEventListener("click", () => {
  const k = th.dataset.key; if (!k) return;
  if (STATE.sortKey === k) STATE.sortDir = -STATE.sortDir; else { STATE.sortKey = k; STATE.sortDir = -1; }
  apply();
}));
["fSearch","fFam","fPod","fStatus","fMin","fMax"].forEach(id => {
  document.getElementById(id).addEventListener("input", apply);
  document.getElementById(id).addEventListener("change", apply);
});
document.querySelectorAll("#typePills .pill").forEach(p => p.addEventListener("click", () => {
  const t = p.dataset.type;
  if (STATE.type === t) { STATE.type = null; p.classList.remove("active"); }
  else { STATE.type = t; document.querySelectorAll("#typePills .pill").forEach(x => x.classList.remove("active")); p.classList.add("active"); }
  apply();
}));
document.querySelectorAll("#bucketPills .pill").forEach(p => p.addEventListener("click", () => {
  const b = p.dataset.bucket;
  if (STATE.bucket === b) { STATE.bucket = null; p.classList.remove("active"); }
  else { STATE.bucket = b; document.querySelectorAll("#bucketPills .pill").forEach(x => x.classList.remove("active")); p.classList.add("active"); }
  apply();
}));
document.getElementById("btnReset").addEventListener("click", () => {
  ["fSearch","fFam","fPod","fStatus","fMin","fMax"].forEach(id => document.getElementById(id).value = "");
  STATE.bucket = null; STATE.type = null; STATE.sortKey = "primary"; STATE.sortDir = -1;
  document.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
  apply();
});
apply();
"""

    JS_FILLED = JS.replace("__DATA__", data_json)

    cnt_llm = sum(1 for r in rows if r["type"] == "LLM")
    cnt_img = sum(1 for r in rows if r["type"] == "image")
    cnt_vid = sum(1 for r in rows if r["type"] == "video")

    HTML = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>V100 Bench — {len(rows)} entries</title>
<style>{CSS}</style></head>
<body>
<h1>V100 Bench Dashboard · 2× V100 32GB SXM2 NVLink</h1>
<div class="sub">BVM транскрипции звонков · 4 уровня контекста · Ollama 0.24 (NUM_PARALLEL=1, FLASH_ATTENTION=1, KV_CACHE_TYPE=q8_0) · diffusion: sd.cpp + diffusers · N=1 (CV ≤ 0.10%) · <b>наведи на любой заголовок колонки чтобы увидеть что значит</b></div>
<div class="sub" style="margin-top: -6px;">Конфиги: <code>cogvideox_v100.py</code> · <code>run_wan22_sdcpp.sh</code> · <code>ollama_bench_real.py</code> · <code>build_dashboard.py</code> · полные команды в <code>HANDOFF.md</code></div>

<div class="controls">
  <div class="pills" id="typePills">
    <span class="pill pill-type" data-type="LLM">LLM ({cnt_llm})</span>
    <span class="pill pill-type" data-type="image">Image ({cnt_img})</span>
    <span class="pill pill-type" data-type="video">Video ({cnt_vid})</span>
  </div>
  <div><label>Поиск:</label> <input id="fSearch" type="text" class="search-input" placeholder="имя модели или семьи..."></div>
  <div><label>Семья:</label> <select id="fFam"><option value="">все ({len(families)})</option>{fam_opts}</select></div>
  <div><label>Pod:</label> <select id="fPod"><option value="">все</option>{pod_opts}</select></div>
  <div><label>B:</label> <input id="fMin" type="number" placeholder="мин"> – <input id="fMax" type="number" placeholder="макс"></div>
  <div><label>Status:</label> <select id="fStatus"><option value="">все</option><option>OK</option><option>FAIL</option></select></div>
  <div class="pills" id="bucketPills">{bucket_pills}</div>
  <button class="btn-reset" id="btnReset">сброс</button>
  <div class="count-badge" id="count"></div>
</div>

<div class="table-wrap"><table id="tbl"><thead><tr>
  <th class="sticky-col" data-key="model" title="Имя модели как в Ollama / sd.cpp / HuggingFace. Кликни для сортировки по алфавиту.">Модель</th>
  <th data-key="type" title="LLM = текст→текст (Ollama). image = текст→картинка (sd.cpp / diffusers). video = текст→видео.">Тип</th>
  <th data-key="family" title="Семья архитектуры (Qwen2.5, Llama-3.3, FLUX, Wan и т.д.). Используй для группировки.">Семья</th>
  <th data-key="params_b" class="num" title="Параметров в миллиардах (B). 7=7B Q4. MoE: 8x7b считается как 56. У diffusion часто 0 — не критично.">B</th>
  <th data-key="bucket" title="Размерная корзина: <5B / 5-9B / 10-19B / 20-29B / 30-49B / 50-79B / 80B+. Кликай pill сверху для фильтра.">Bucket</th>
  <th data-key="pod" title="На каком из 5 vast.ai V100 podов крутилось. pod1=NV1, pod2=SYS PCIe, pod3/4/5=NV2 (NVLink).">Pod</th>
  <th data-key="primary" class="num" title="Главная метрика для типа: LLM = avg decode tps (среднее по 4 уровням контекста). Image = sec/картинка. Video = sec/видео. Heat-map: красный=медленно, зелёный=быстро.">primary</th>
  <th data-key="primary_label" title="Единица primary колонки (tps avg / sec/img / sec/video).">unit</th>
  <th data-key="runs_count" class="num" title="Сколько раз эта модель прогналась (разные промпты или повторы). LLM = 1 запись = 4 уровня в одной строке. Image/Video = каждая строка = один промпт, runs_count = всего таких прогонов модели.">runs</th>
  <th data-key="runs_avg" class="num" title="Среднее значение primary по всем запускам этой модели. Если разброс <1% — модель стабильна, цифра достоверна.">avg(runs)</th>
  <th data-key="runs_spread_pct" class="num" title="Разброс между запусками = (max-min)/avg × 100%. <1% = стабильно как часы. >10% = выпадает один запуск, проверить причину.">spread %</th>
  <th data-key="short" class="num" title="LLM: decode tokens/sec на коротком промпте (~3K input tok, BVM транскрипция звонка). Heat-map: зелёный = быстро.">short</th>
  <th data-key="small" class="num" title="LLM: decode tps на small промпте (~5K tok). Сравни с short — деградация показывает чувствительность к контексту.">small</th>
  <th data-key="medium" class="num" title="LLM: decode tps на medium промпте (~10K tok).">medium</th>
  <th data-key="large" class="num" title="LLM: decode tps на large промпте (~14K tok, самый длинный звонок 49 мин). Падение vs short — главный индикатор «насколько модель грустит на длинном входе».">large</th>
  <th data-key="ttft_short" class="num" title="Time To First Token, ms — задержка до первого токена ответа. Включает prefill всего входа. Чем меньше тем отзывчивее.">TTFT s</th>
  <th data-key="ttft_large" class="num" title="TTFT на large (~14K tok). Часто 20-40 sec — модель долго переваривает контекст перед первым выводом. Важнее decode_tps для UX в чате.">TTFT l</th>
  <th data-key="prefill_short" class="num" title="Prefill throughput на short, tokens/sec. Сколько токенов входа модель переваривает в секунду. Высокий prefill = быстрая обработка длинных промптов.">pref s</th>
  <th data-key="prefill_large" class="num" title="Prefill throughput на large. Обычно ниже short из-за O(n²) attention.">pref l</th>
  <th data-key="tok_large" class="num" title="Сколько токенов в large промпте (BVM транскрипция 49-минутного звонка + eval prompt).">in tok</th>
  <th data-key="resolution" class="num" title="Image/Video: разрешение. Z-Image=1024×1024, FLUX=1024×1024, SDXL=1024×1024, Wan2.2=480×832, CogVideoX=720×480.">res</th>
  <th data-key="steps" class="num" title="Image: количество шагов denoising (Z-Image Turbo=8, FLUX=20). Video: количество кадров (Wan2.2=33, CogVideoX=49). Линейно влияет на wall time.">steps/frames</th>
  <th data-key="peak_vram_gb" class="num" title="Image/Video: пиковое потребление VRAM в GB на одной V100. Если >32 = не лезет в карту, нужен offload на CPU или вторая карта.">VRAM</th>
  <th data-key="wall_total_s" class="num" title="Суммарное wall time в секундах. LLM: сумма 4 запусков (short+small+medium+large). Image/Video: время одного запуска (одна картинка/видео).">wall s</th>
  <th data-key="n_levels" class="num" title="Сколько уровней контекста из 4 (short/small/medium/large) отработали без ошибок. 4/4 = идеально. Меньше = модель упала на длинных промптах.">lvl</th>
  <th data-key="status" title="OK = все запуски прошли. FAIL = модель упала: OOM (не хватило VRAM), NaN (overflow на Volta), 404 (нет в Ollama lib), PREWARM_FAIL (не лезет в GPU при загрузке).">status</th>
</tr></thead><tbody></tbody></table></div>

<script>{JS_FILLED}</script>
</body></html>"""

    OUT_HTML.write_text(HTML, encoding="utf-8")
    print(f"wrote {OUT_HTML} ({len(HTML)} bytes)")

if __name__ == "__main__":
    main()
