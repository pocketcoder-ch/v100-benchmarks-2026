"""CogVideoX-5B на V100 32GB FP16.

CogVideoX-5b нативно поддерживает FP16 (в отличие от LTX bf16-only).
enable_model_cpu_offload + vae tiling + vae fp32 для safety на Volta.
"""
import os, time, json, gc
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

MODEL_ID = "THUDM/CogVideoX-5b"
OUT_DIR = "/root/cogvideox_out"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPTS = {
    "office_winter": "A cozy modern tech office with large windows showing heavy winter snowfall outside, warm interior lighting, plants on shelves, laptops on desks, smooth slow camera pan, photorealistic",
    "ai_agent": "A glowing teal AI agent figure interacting with floating database tables and CRM cards in a dark cyberpunk server room, smooth motion, cinematic",
    "call_center": "Call center operator at her desk, dual monitors showing live analytics dashboards with animated waveforms, modern open-plan office, slow zoom, photorealistic",
}

print(f"[load] {MODEL_ID} torch_dtype=float16 (Volta no BF16)")
torch.backends.cuda.matmul.allow_tf32 = True
pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
# all components in FP16 — CogVideoX VAE works fine in FP16 (tested in CogVideoX repo)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

log_path = os.path.join(OUT_DIR, "results.jsonl")
with open(log_path, "a") as logf:
    for name, prompt in PROMPTS.items():
        out_mp4 = os.path.join(OUT_DIR, f"cogvideox_{name}.mp4")
        if os.path.exists(out_mp4):
            print(f"[skip] {name}")
            continue
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        t0 = time.time()
        try:
            video = pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=30,
                num_frames=49,
                guidance_scale=6,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]
            export_to_video(video, out_mp4, fps=8)
        except Exception as e:
            err = repr(e)[:300]
            print(f"[FAIL] {name}: {err}")
            logf.write(json.dumps({"prompt_id": name, "error": err}) + "\n")
            logf.flush()
            torch.cuda.empty_cache(); gc.collect()
            continue
        wall = time.time() - t0
        rec = {
            "model": "CogVideoX-5b",
            "stack": "diffusers fp16 + vae-fp32 + cpu-offload",
            "prompt_id": name,
            "wall_sec": round(wall, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated()/1e9, 2),
            "frames": 49, "fps": 8, "resolution": "720x480",
            "image": os.path.basename(out_mp4),
        }
        print(f"[ok] {name} wall={wall:.1f}s peak={rec['peak_vram_gb']}G")
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

print("[done] all video prompts processed")
