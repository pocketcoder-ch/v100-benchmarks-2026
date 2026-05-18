#!/bin/bash
# run_queue.sh — последовательный прогон моделей на одном поде
#
# Использование:
#   ./run_queue.sh /root/models_queue_pod1.txt
#
# Каждая строка в queue файле: <model:tag>  [num_ctx=N]  [skip_levels=L1,L2]
# Пример:
#   qwen2.5:72b-instruct-q4_K_M  num_ctx=22000
#   deepseek-r1:70b-llama-distill-q4_K_M  num_ctx=22000  skip_levels=large
#   gpt-oss:20b  num_ctx=22000
#
# После каждой модели:
#   1) pull
#   2) prewarm (force load в GPU, verify offload N/N)
#   3) bench (ollama_bench_real.py)
#   4) save JSON в /root/ollama_bench_results/real/
#   5) ollama rm (освободить диск)
#
# Логи: /root/queue_runner.log (общий), /root/ollama_bench_results/real/<slug>.json (per-model)

set -u

QUEUE_FILE="${1:?usage: $0 <queue_file>}"
LOG="/root/queue_runner.log"
RESULTS_DIR="/root/ollama_bench_results/real"
mkdir -p "$RESULTS_DIR"

log() {
    echo "[$(date +'%H:%M:%S')] $*" | tee -a "$LOG"
}

# Verify ollama up
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    log "FATAL: ollama serve не отвечает на localhost:11434. Старт его сначала."
    exit 1
fi

# Verify бenchскрипт + транскрипты + eval prompt
for f in /root/ollama_bench_real.py /root/eval_prompt.txt \
         /root/transcripts/short.txt /root/transcripts/small.txt \
         /root/transcripts/medium.txt /root/transcripts/large.txt; do
    if [ ! -f "$f" ]; then
        log "FATAL: missing $f"
        exit 1
    fi
done

total=$(grep -cE '^[a-z0-9]' "$QUEUE_FILE")
idx=0

while IFS= read -r line; do
    # skip пустые и комменты
    line="${line%%#*}"  # обрезать комментарий
    line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [ -z "$line" ] && continue

    idx=$((idx+1))

    # парсинг строки
    MODEL=$(echo "$line" | awk '{print $1}')
    NUM_CTX=22000
    SKIP_LEVELS=""
    for kv in $(echo "$line" | awk '{for(i=2;i<=NF;i++)print $i}'); do
        case "$kv" in
            num_ctx=*) NUM_CTX="${kv#num_ctx=}" ;;
            skip_levels=*) SKIP_LEVELS="${kv#skip_levels=}" ;;
        esac
    done

    SLUG=$(echo "$MODEL" | tr ':/' '__')
    RESULT_JSON="$RESULTS_DIR/${SLUG}.json"

    log "===================="
    log "[$idx/$total] MODEL=$MODEL  NUM_CTX=$NUM_CTX  SKIP_LEVELS=${SKIP_LEVELS:-none}"
    log "===================="

    # Idempotency — пропустить если результат уже есть
    if [ -f "$RESULT_JSON" ]; then
        if python3 -c "import json,sys; d=json.load(open('$RESULT_JSON')); ok=sum(1 for r in d.get('results',[]) if r.get('status')=='OK'); sys.exit(0 if ok>=1 else 1)" 2>/dev/null; then
            log "  SKIP — уже есть валидный $RESULT_JSON"
            continue
        fi
    fi

    # 1. Pull
    log "  [pull]..."
    t0=$(date +%s)
    if ! ollama pull "$MODEL" 2>&1 | tail -3 | tee -a "$LOG"; then
        log "  FAIL: pull $MODEL"
        echo '{"model":"'"$MODEL"'","status":"PULL_FAIL"}' > "$RESULT_JSON"
        continue
    fi
    log "  pull done за $(($(date +%s) - t0))s"

    # 2. Prewarm — load в GPU, verify offload
    log "  [prewarm]..."
    t0=$(date +%s)
    PREWARM_RESP=$(curl -s --max-time 600 http://localhost:11434/api/generate -d "{
        \"model\":\"$MODEL\",
        \"prompt\":\"hi\",
        \"stream\":false,
        \"options\":{\"num_predict\":1,\"num_ctx\":$NUM_CTX}
    }" 2>&1)
    if ! echo "$PREWARM_RESP" | grep -q '"done":true'; then
        log "  FAIL: prewarm $MODEL"
        log "    resp head: $(echo "$PREWARM_RESP" | head -c 300)"
        echo '{"model":"'"$MODEL"'","status":"PREWARM_FAIL","resp":"'"$(echo "$PREWARM_RESP" | head -c 200 | tr -d '\"')"'"}' > "$RESULT_JSON"
        ollama rm "$MODEL" >/dev/null 2>&1 || true
        continue
    fi
    log "  prewarm done за $(($(date +%s) - t0))s"

    # Чекаем offload
    OFFLOAD=$(grep -E "offloaded.*layers" /root/ollama.log | tail -1 | sed 's/.*offloaded //')
    log "  offload: $OFFLOAD"

    # 3. Bench
    log "  [bench]..."
    t0=$(date +%s)
    MODEL="$MODEL" NUM_CTX="$NUM_CTX" SKIP_LEVELS="$SKIP_LEVELS" \
        OUT_DIR="$RESULTS_DIR" \
        python3 /root/ollama_bench_real.py 2>&1 | tee -a "$LOG" | tail -50
    BENCH_RC=${PIPESTATUS[0]}
    log "  bench rc=$BENCH_RC за $(($(date +%s) - t0))s"

    # 4. Освободить диск (опционально — оставляем pulled, потому что переключение туда-сюда дороже re-pull)
    # Если диск переполняется (>80%) — снести модель:
    DISK_USE=$(df /root | tail -1 | awk '{print $5}' | tr -d '%')
    if [ "$DISK_USE" -gt 80 ]; then
        log "  disk $DISK_USE% — удаляю $MODEL чтоб освободить место"
        ollama rm "$MODEL" >/dev/null 2>&1 || true
    fi

done < "$QUEUE_FILE"

log "===================="
log "QUEUE DONE. results: $RESULTS_DIR"
ls -la "$RESULTS_DIR"
