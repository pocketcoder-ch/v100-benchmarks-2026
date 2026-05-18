#!/usr/bin/env python3
"""
Bench Ollama на РЕАЛЬНЫХ BVM-транскрипциях звонков + prod eval prompt.

Входные файлы (на pod):
  /root/transcripts/short.txt   (~3K-3.5K tok — короткий разговор)
  /root/transcripts/small.txt   (~6K tok — средний)
  /root/transcripts/medium.txt  (~12K-13K tok — длинный)
  /root/transcripts/large.txt   (~18K-19K tok — очень длинный)
  /root/eval_prompt.txt         (~250 tok — BVM eval prompt #1: диагностика + боль)

ENV:
  MODEL              — обязательно (ollama model id, например "qwen2.5-coder:32b-instruct-q4_K_M")
  NUM_CTX            — default = 22000 (хватит для large+eval+output)
  MAX_TOKENS_OUT     — default 300 (структурированный ответ ~150-300 tok)
  OUT_DIR            — default /root/ollama_bench_results/real
  SKIP_LEVELS        — список через запятую (default пусто), напр. "large" если модель не лезет в 18K

Что замеряем:
  - prompt_tokens (real, от Ollama)
  - output_tokens
  - TTFT (prompt_eval_duration в ms)
  - prefill_tps
  - decode_tps (eval_count / eval_duration)
  - total wall time
  - done_reason
  - Ollama options (для документации конфигов)
"""
import json, time, urllib.request, os, sys, datetime

MODEL = os.environ.get("MODEL")
if not MODEL:
    print("ERROR: MODEL env required", file=sys.stderr); sys.exit(2)

NUM_CTX = int(os.environ.get("NUM_CTX", "22000"))
MAX_TOKENS_OUT = int(os.environ.get("MAX_TOKENS_OUT", "300"))
OUT_DIR = os.environ.get("OUT_DIR", "/root/ollama_bench_results/real")
SKIP_LEVELS = set(x.strip() for x in os.environ.get("SKIP_LEVELS", "").split(",") if x.strip())
TRANSCRIPTS_DIR = os.environ.get("TRANSCRIPTS_DIR", "/root/transcripts")
EVAL_PROMPT_FILE = os.environ.get("EVAL_PROMPT_FILE", "/root/eval_prompt.txt")
URL = "http://localhost:11434/api/generate"
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))

os.makedirs(OUT_DIR, exist_ok=True)
slug = MODEL.replace("/", "_").replace(":", "_")
OUT_FILE = f"{OUT_DIR}/{slug}.json"

# 1. Загрузить eval prompt
if not os.path.exists(EVAL_PROMPT_FILE):
    print(f"ERROR: {EVAL_PROMPT_FILE} not found", file=sys.stderr); sys.exit(3)
with open(EVAL_PROMPT_FILE, "r", encoding="utf-8") as f:
    EVAL_PROMPT = f.read().strip()

# 2. Загрузить транскрипции по уровням
LEVELS = ["short", "small", "medium", "large"]
transcripts = {}
for level in LEVELS:
    path = f"{TRANSCRIPTS_DIR}/{level}.txt"
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping {level}", file=sys.stderr)
        continue
    with open(path, "r", encoding="utf-8") as f:
        transcripts[level] = f.read()

if not transcripts:
    print("ERROR: no transcripts found", file=sys.stderr); sys.exit(4)

print(f"[setup] model={MODEL}", flush=True)
print(f"[setup] num_ctx={NUM_CTX} max_tokens_out={MAX_TOKENS_OUT} temperature={TEMPERATURE}", flush=True)
print(f"[setup] levels loaded: {list(transcripts.keys())}", flush=True)
print(f"[setup] skip levels: {SKIP_LEVELS}", flush=True)

def call(prompt, max_tokens):
    payload = {
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
        },
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type":"application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        body = json.loads(resp.read())
    body["_wall_ms"] = int((time.time()-t0)*1000)
    return body

# 3. Sweep по 4 уровням
run_meta = {
    "model": MODEL,
    "num_ctx": NUM_CTX,
    "max_tokens_out": MAX_TOKENS_OUT,
    "temperature": TEMPERATURE,
    "eval_prompt_file": EVAL_PROMPT_FILE,
    "eval_prompt_preview": EVAL_PROMPT[:200],
    "started_at": datetime.datetime.utcnow().isoformat() + "Z",
    "ollama_options_used": {
        "num_predict": MAX_TOKENS_OUT,
        "temperature": TEMPERATURE,
        "num_ctx": NUM_CTX,
    },
    "results": [],
}

for level in LEVELS:
    if level in SKIP_LEVELS:
        print(f"\n=== SKIP {level} (in SKIP_LEVELS) ===", flush=True)
        run_meta["results"].append({"level": level, "status": "SKIPPED"})
        continue
    if level not in transcripts:
        run_meta["results"].append({"level": level, "status": "MISSING_FILE"})
        continue

    transcript = transcripts[level]
    # Формат prod: ДИАЛОГ как context → EVAL_PROMPT как инструкция
    prompt = f"{transcript}\n\n---\n\n{EVAL_PROMPT}"

    print(f"\n=== level={level} (transcript chars={len(transcript)}, full prompt chars={len(prompt)}) ===", flush=True)
    try:
        d = call(prompt, max_tokens=MAX_TOKENS_OUT)
        pe = d.get("prompt_eval_count", 0)
        ev = d.get("eval_count", 0)
        pe_ms = d.get("prompt_eval_duration", 0) / 1e6
        ev_ms = d.get("eval_duration", 0) / 1e6
        decode_tps = ev / (ev_ms / 1000) if ev_ms else 0
        prefill_tps = pe / (pe_ms / 1000) if pe_ms else 0

        m = {
            "level": level,
            "prompt_tokens": pe,
            "output_tokens": ev,
            "ttft_ms": round(pe_ms),
            "prefill_tps": round(prefill_tps, 1),
            "decode_tps": round(decode_tps, 2),
            "total_wall_ms": d.get("_wall_ms"),
            "done_reason": d.get("done_reason"),
            "response_preview": (d.get("response") or "")[:300],
            "status": "OK",
        }
        print(json.dumps({k: v for k, v in m.items() if k != "response_preview"}, indent=2, ensure_ascii=False), flush=True)
        print(f"[response preview]\n{m['response_preview']}", flush=True)
        run_meta["results"].append(m)
    except Exception as e:
        m = {"level": level, "status": "FAIL", "err": str(e)[:400]}
        print(json.dumps(m, ensure_ascii=False), flush=True)
        run_meta["results"].append(m)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

run_meta["finished_at"] = datetime.datetime.utcnow().isoformat() + "Z"
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(run_meta, f, indent=2, ensure_ascii=False)

print(f"\n=== DONE ===\nresults → {OUT_FILE}", flush=True)
