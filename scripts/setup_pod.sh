#!/bin/bash
# setup_pod.sh — единый bootstrap нового Vast.ai pod 2× V100 32GB PCIe под sweep-методологию.
# Запускается на pod после SSH. Идемпотентен — можно перезапустить после частичного отказа.
#
# Пререквизиты: на pod должен быть rsync /root/bench/ перед запуском
# (с ноута: rsync -avz benchmarks/ root@pod:/root/bench/)
#
# ENV:
#   HF_TOKEN  (required)
#   POD_ROLE  — "pod1" | "pod2" | "pod3" | "pod4" (определяет какие queue-файлы запускать)
#
# Что делает:
#   1. apt: python3.12-venv git curl rsync
#   2. .venv-base (vllm 0.18.1 + torch 2.10+cu128)
#   3. .venv-1cat (1Cat-vLLM 1.0.0 + flash_attn_v100 1.0.0)
#   4. HF login + HF_HUB_ENABLE_HF_TRANSFER
#   5. Проверка наличия bench-инфры
#   6. Запуск queue_runner(ов) под POD_ROLE

set -eu
set -o pipefail

: "${HF_TOKEN:?HF_TOKEN required (export HF_TOKEN=hf_...)}"
: "${POD_ROLE:?POD_ROLE required (pod1|pod2|pod3|pod4)}"

BENCH_ROOT="${BENCH_ROOT:-/root/bench}"

log() { echo "[setup][$POD_ROLE] $*"; }

log "=== preflight ==="
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv
nvidia-smi nvlink --status 2>/dev/null | head -10 || true
nvidia-smi topo -m 2>/dev/null | head -20 || true
df -h / | tail -1
free -g | grep Mem
python3.12 --version || { log "FATAL: python3.12 missing in container"; exit 1; }

# === apt ===
log "=== apt install ==="
apt-get update -qq
apt-get install -y -qq python3.12-venv python3.12-dev git curl rsync jq tmux

# === bench/ check ===
if [[ ! -f "$BENCH_ROOT/bench_common.py" ]]; then
    log "FATAL: $BENCH_ROOT/bench_common.py missing — rsync benchmarks/ to /root/bench/ first"
    exit 2
fi
chmod +x "$BENCH_ROOT/sweep_one_model.sh" "$BENCH_ROOT/queue_runner.sh" 2>/dev/null || true

# === venv-base ===
log "=== .venv-base (vllm 0.18.1 + torch+cu128) ==="
if [[ ! -x /root/.venv-base/bin/vllm ]]; then
    python3.12 -m venv /root/.venv-base
    /root/.venv-base/bin/pip install -q -U pip wheel
    /root/.venv-base/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
        vllm==0.18.1 hf_transfer httpx
fi
/root/.venv-base/bin/python -c "import torch, vllm; print('BASE:', torch.__version__, vllm.__version__, torch.cuda.get_device_capability())"

# === venv-1cat ===
log "=== .venv-1cat (1Cat fork v1.0.0) ==="
if [[ ! -x /root/.venv-1cat/bin/vllm ]]; then
    python3.12 -m venv /root/.venv-1cat
    /root/.venv-1cat/bin/pip install -q -U pip wheel
    /root/.venv-1cat/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.9.1
    /root/.venv-1cat/bin/pip install \
        https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/flash_attn_v100-1.0.0-cp312-cp312-linux_x86_64.whl
    /root/.venv-1cat/bin/pip install \
        https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/vllm-1.0.0-cp312-cp312-linux_x86_64.whl
    /root/.venv-1cat/bin/pip install -q hf_transfer httpx huggingface_hub
fi
/root/.venv-1cat/bin/python -c "import torch, vllm; print('1CAT:', torch.__version__, vllm.__version__, torch.cuda.get_device_capability())"

# === HF login ===
log "=== HF login ==="
/root/.venv-base/bin/huggingface-cli login --token "$HF_TOKEN" 2>&1 | tail -3
export HF_HUB_ENABLE_HF_TRANSFER=1

# === Запуск queue_runner ===
# Маппинг POD_ROLE → queue файлы
case "$POD_ROLE" in
    pod1)
        QUEUES=("trackA:0:pod1_trackA.csv" "trackB:1:pod1_trackB.csv")
        ;;
    pod2)
        QUEUES=("trackA:0:pod2_trackA.csv" "trackB:1:pod2_trackB.csv")
        ;;
    pod3)
        QUEUES=("single:0,1:pod3_single.csv")
        ;;
    pod4)
        QUEUES=("single:0,1:pod4_single.csv")
        ;;
    pod5)
        QUEUES=("single:0,1:pod5_single.csv")
        ;;
    *)
        log "FATAL: unknown POD_ROLE=$POD_ROLE"
        exit 2
        ;;
esac

log "=== starting queue_runner(s) in tmux ==="
# Используем tmux, чтобы переживало drop ssh и можно было attach для debug
if ! tmux has-session -t bench 2>/dev/null; then
    tmux new-session -d -s bench -x 200 -y 50
fi

for entry in "${QUEUES[@]}"; do
    IFS=':' read -r track cuda_dev queue_file <<< "$entry"
    full_queue="$BENCH_ROOT/queues/$queue_file"
    if [[ ! -f "$full_queue" ]]; then
        log "ERROR: queue file $full_queue missing"
        exit 2
    fi
    window_name="$track"
    # Создаём окно или используем существующее
    if ! tmux list-windows -t bench 2>/dev/null | grep -q "$window_name"; then
        tmux new-window -t bench -n "$window_name"
    fi
    # Стартовая команда runner-а
    cmd="cd $BENCH_ROOT && QUEUE_FILE=$full_queue TRACK=$track CUDA_DEV=$cuda_dev bash $BENCH_ROOT/queue_runner.sh 2>&1 | tee $BENCH_ROOT/queue_runner_${track}.log"
    tmux send-keys -t "bench:$window_name" "$cmd" C-m
    log "started runner $track (cuda=$cuda_dev, queue=$queue_file)"
done

log "=== READY ==="
log "tmux session 'bench' running. Attach: tmux attach -t bench"
log "Monitor: tail -F $BENCH_ROOT/queue_runner_*.log"
log "Progress: cat $BENCH_ROOT/progress_*.jsonl | jq -c '.'"
log "Status per model: ls $BENCH_ROOT/runs/*/sweep_status.txt | xargs grep -H ."
