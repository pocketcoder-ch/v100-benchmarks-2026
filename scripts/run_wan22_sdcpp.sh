#!/bin/bash
# Wan2.2 TI2V 5B text-to-video on V100 via sd.cpp + --type bf16 workaround
set -e
cd /root
mkdir -p /root/wan22_out
SD=/root/stable-diffusion.cpp/build/bin/sd-cli
DM=/root/wan_models/Wan2.2-TI2V-5B-Q8_0.gguf
VAE=/root/wan_models/Wan2.2_VAE.safetensors
T5=/root/wan_models/umt5_xxl.safetensors

declare -A PROMPTS=(
  ["office_winter"]="A cozy modern tech office with large windows, heavy winter snowfall outside, warm interior lighting, plants, laptops on desks, smooth slow camera pan"
  ["ai_agent"]="A glowing teal AI agent figure interacting with floating database tables and CRM cards, dark cyberpunk server room, smooth motion, cinematic"
  ["call_center"]="Call center operator at her desk, dual monitors showing animated analytics dashboards with waveforms, modern open-plan office, slow zoom"
)

RESULTS=/root/wan22_out/results.jsonl
> $RESULTS

for name in "${!PROMPTS[@]}"; do
  P="${PROMPTS[$name]}"
  OUT=/root/wan22_out/wan22_${name}.mp4
  echo "=== [$(date +%H:%M:%S)] $name ==="
  START=$(date +%s.%N)
  $SD -M vid_gen \
    --diffusion-model $DM --vae $VAE --t5xxl $T5 \
    -p "$P" --cfg-scale 6.0 --sampling-method euler \
    --type bf16 \
    -W 480 -H 832 \
    --video-frames 33 --flow-shift 3.0 \
    -s 42 -o $OUT 2>&1 | tail -15
  END=$(date +%s.%N)
  WALL=$(echo "$END - $START" | bc)
  SIZE=$(stat -c%s $OUT 2>/dev/null || echo 0)
  echo "{\"model\":\"Wan2.2-TI2V-5B-Q8\",\"stack\":\"sd.cpp bf16\",\"prompt_id\":\"$name\",\"resolution\":\"480x832\",\"frames\":33,\"wall_sec\":$WALL,\"mp4_size\":$SIZE,\"image\":\"wan22_${name}.mp4\"}" >> $RESULTS
  echo "[done] $name wall=${WALL}s mp4=${SIZE}b"
done
echo "DONE_ALL_WAN22"
