"""Z-Image-Turbo on 1× V100 32GB FP16 (single-card, no cross-device split).

Split-device трюк (Gemini #4) не работает: ZImagePipeline не маршрутизирует
embeddings cross-GPU автоматически (addmm crash device mismatch).
Решение проще: Z-Image-Turbo TE = Qwen2.5-1.5B (~3 GB FP16), DiT = 6B (~12 GB FP16),
VAE копейки. Влезает в 32 GB одной V100 с запасом.

Volta не имеет BF16 → принудительный torch.float16 для всего.
"""
import os, time, json, gc
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

print(f"[load] {MODEL_ID} all-fp16 single-card cuda:0")
pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=False)
pipe.to("cuda:0")
# Volta FP16 NaN ловушки: Qwen2.5-VL text encoder + VAE.
# Каст обоих в FP32 на той же карте. Память: TE 1.5B × 4 = 6 GB, VAE ~1 GB.
# Итого DiT 12 GB (FP16) + TE 6 GB (FP32) + VAE 1 GB (FP32) + активации ~5 GB ≈ 24 GB / 32 GB.
print("[fix] cast text_encoder -> fp32 (Qwen2.5-VL FP16 NaN on Volta)")
pipe.text_encoder.to(dtype=torch.float32)
print("[fix] cast vae -> fp32 (VAE FP16 NaN on Volta — known FLUX/SDXL pattern)")
pipe.vae.to(dtype=torch.float32)
print("[fix] cast transformer -> fp32 — V5: try DiT FP32 (suspect NaN in DiT FP16)")
pipe.transformer.to(dtype=torch.float32)
print(f"[mem] after load cuda:0 = {torch.cuda.memory_allocated(0)/1e9:.2f} GiB")

log_path = os.path.join(OUT_DIR, "results_v2.jsonl")
with open(log_path, "a") as logf:
    for name, prompt in PROMPTS.items():
        out_png = os.path.join(OUT_DIR, f"zimage_{name}.png")
        if os.path.exists(out_png):
            print(f"[skip] {name}")
            continue
        torch.cuda.reset_peak_memory_stats(0)
        t0 = time.time()
        gen = torch.Generator(device="cuda:0").manual_seed(42)
        try:
            with torch.inference_mode():
                img = pipe(
                    prompt=prompt,
                    height=512, width=512,
                    num_inference_steps=9,
                    guidance_scale=0.0,
                    generator=gen,
                ).images[0]
        except Exception as e:
            err = repr(e)[:300]
            print(f"[FAIL] {name}: {err}")
            logf.write(json.dumps({"prompt_id": name, "error": err}) + "\n")
            logf.flush()
            torch.cuda.empty_cache(); gc.collect()
            continue
        wall = time.time() - t0
        img.save(out_png)
        rec = {
            "model": "Z-Image-Turbo",
            "stack": "diffusers fp16 single-V100",
            "prompt_id": name,
            "wall_sec": round(wall, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated(0)/1e9, 2),
            "steps": 9, "resolution": "1024x1024",
            "image": os.path.basename(out_png),
        }
        print(f"[ok] {name} wall={wall:.1f}s peak={rec['peak_vram_gb']}G")
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

print("[done] all prompts processed")
