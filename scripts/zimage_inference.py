"""Z-Image-Turbo on 2× V100 32GB (Volta sm_70, no BF16).

Strategy (Gemini #4): split-device — text encoder (Qwen2.5-VL) on cuda:1 in FP32
(safe against NaN), DiT+VAE on cuda:0 in FP16. Embeddings cross via NVLink.
"""
import os
import time
import torch
from diffusers import ZImagePipeline

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
OUT_DIR = "/root/zimage_out"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPTS = {
    "call_center_operator": "A call center operator at her desk, dual monitors showing analytics dashboards with waveforms and KPI graphs, modern open-plan office, soft daylight, photorealistic, 8k",
    "speech_analytics_diagram": "Technical architecture diagram on a whiteboard, clean schematic of microphone -> ASR -> LLM -> CRM pipeline, isometric flat illustration",
    "ai_agent_database": "A glowing AI agent figure interacting with floating database tables and CRM cards, dark cyberpunk server room, neon teal accents, cinematic",
    "office_winter": "Cozy modern tech office with large windows showing heavy winter snowfall outside, warm interior lighting, plants, laptops on desks, photorealistic",
    "neon_city": "Neon-lit cyberpunk city at night, wet asphalt reflections, holographic billboards, anime-cinematic style, ultra detailed",
}

print(f"[load] pipeline {MODEL_ID} dtype=float16 (no BF16 on Volta)")
pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=False)

print("[place] text_encoder -> cuda:1 fp32 (Qwen2.5-VL needs full precision on Volta)")
pipe.text_encoder.to(device="cuda:1", dtype=torch.float32)

print("[place] transformer (DiT) + vae -> cuda:0 fp16")
pipe.transformer.to(device="cuda:0", dtype=torch.float16)
pipe.vae.to(device="cuda:0", dtype=torch.float16)

print("[mem] cuda:0", torch.cuda.memory_allocated(0)/1e9, "GiB | cuda:1", torch.cuda.memory_allocated(1)/1e9, "GiB")

import json
log_path = os.path.join(OUT_DIR, "results.jsonl")
with open(log_path, "a") as logf:
    for name, prompt in PROMPTS.items():
        out_png = os.path.join(OUT_DIR, f"zimage_{name}.png")
        if os.path.exists(out_png):
            print(f"[skip] {name} already exists")
            continue
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.reset_peak_memory_stats(1)
        t0 = time.time()
        gen = torch.Generator(device="cuda:0").manual_seed(42)
        with torch.inference_mode():
            try:
                img = pipe(
                    prompt=prompt,
                    height=1024, width=1024,
                    num_inference_steps=9,
                    guidance_scale=0.0,
                    generator=gen,
                ).images[0]
            except Exception as e:
                print(f"[FAIL] {name}: {e}")
                logf.write(json.dumps({"prompt_id": name, "error": str(e)[:200]}) + "\n")
                logf.flush()
                continue
        wall = time.time() - t0
        img.save(out_png)
        rec = {
            "prompt_id": name,
            "wall_sec": round(wall, 2),
            "peak_vram_gpu0_gb": round(torch.cuda.max_memory_allocated(0)/1e9, 2),
            "peak_vram_gpu1_gb": round(torch.cuda.max_memory_allocated(1)/1e9, 2),
            "steps": 9, "resolution": "1024x1024",
            "image": os.path.basename(out_png),
        }
        print(f"[ok] {name} wall={wall:.1f}s gpu0={rec['peak_vram_gpu0_gb']}G gpu1={rec['peak_vram_gpu1_gb']}G")
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

print("[done] all prompts processed")
