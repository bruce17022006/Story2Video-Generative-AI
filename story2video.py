# ============================================================
# STORY2VIDEO — fixed pipeline for Kaggle T4 (~15 GB VRAM)
# Audio -> Whisper -> Qwen scenes -> Stable Diffusion -> SVD -> MoviePy
# Install first (if needed):  pip install -q "moviepy<2" openai-whisper gradio accelerate
# ============================================================

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc, json, re, time
from pathlib import Path

import torch
import whisper
import gradio as gr
from PIL import Image, ImageOps
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from moviepy.editor import (
    VideoFileClip, AudioFileClip, concatenate_videoclips, vfx,
)

WORK = Path("/kaggle/working/story2video")
WORK.mkdir(parents=True, exist_ok=True)

QWEN_ID = "Qwen/Qwen2.5-3B-Instruct"
SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SVD_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
WHISPER_SIZE = "small"
NUM_SCENES = 4

DEFAULT_CHARACTER = (
    "7 year old boy, short dark brown hair, blue t-shirt, beige shorts, "
    "3d animated family film style"
)
NEGATIVE = "deformed, extra limbs, bad hands, text, watermark, blurry, low quality"

# SVD settings tried in order; if one runs out of memory the next (smaller) one is used.
SVD_LADDER = [
    dict(w=1024, h=576, frames=14),
    dict(w=768, h=432, frames=12),
    dict(w=576, h=320, frames=8),
]

OOM_ERRORS = (torch.cuda.OutOfMemoryError,)


# ---------------- memory helpers ----------------
def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ---------------- step 1: Whisper ----------------
def transcribe(audio_path):
    model = whisper.load_model(WHISPER_SIZE, device="cuda")
    try:
        text = model.transcribe(str(audio_path), fp16=True)["text"].strip()
    finally:
        del model
        cleanup_cuda()
    return text


# ---------------- step 2: Qwen scene plan + ONE fixed character sheet ----------------
def parse_json_object(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON found")
    return json.loads(text[start:end + 1])


def fallback_scenes(transcript):
    """If Qwen fails, split the transcript into equal parts so the app still works."""
    words = transcript.split()
    size = max(1, len(words) // NUM_SCENES)
    scenes = []
    for i in range(NUM_SCENES):
        chunk = " ".join(words[i * size:(i + 1) * size if i < NUM_SCENES - 1 else None])
        chunk = chunk or transcript
        scenes.append({"visual_action": chunk, "image_prompt": " ".join(chunk.split()[:25])})
    return scenes


def plan_scenes(transcript, user_character):
    """Returns (character_sheet, scenes). The character sheet is written ONCE and reused everywhere."""
    tok = AutoTokenizer.from_pretrained(QWEN_ID)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    character = user_character
    try:
        hint = f"The main character is: {user_character}\n" if user_character else ""
        messages = [
            {"role": "system", "content": (
                "You turn a story into a picture plan. First write ONE fixed character sheet for the "
                "main character (age, gender, hair, face, exact clothes with colors) in under 25 words. "
                f"Then split the story into exactly {NUM_SCENES} consecutive scenes. Do not invent events. "
                "In every image_prompt, do NOT describe the character's looks; refer to them only as "
                "'the boy' / 'the girl' etc., and describe only the action, place and mood. "
                'Return JSON only: {"character":"...","scenes":[{"visual_action":"...","image_prompt":"..."}]}'
            )},
            {"role": "user", "content": f"{hint}Story: {transcript[:3000]}\nimage_prompt: under 20 words."},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=700, do_sample=False,
                repetition_penalty=1.05, pad_token_id=tok.eos_token_id,
            )
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        data = parse_json_object(raw)
        scenes = [s for s in data.get("scenes", []) if isinstance(s, dict)]
        if len(scenes) < NUM_SCENES:
            raise ValueError("too few scenes")
        scenes = scenes[:NUM_SCENES]
        if not user_character:
            character = str(data.get("character", "")).strip()
    except Exception as e:
        print("Qwen planning failed, using fallback:", e)
        scenes = fallback_scenes(transcript)
    finally:
        del model, tok
        cleanup_cuda()

    character = " ".join((character or DEFAULT_CHARACTER).split()[:35])
    for i, s in enumerate(scenes):
        s["visual_action"] = str(s.get("visual_action") or s.get("image_prompt") or f"Scene {i+1}")
        s["image_prompt"] = " ".join(str(s.get("image_prompt") or s["visual_action"]).split()[:25])
    return character, scenes


# ---------------- step 3: Stable Diffusion with character consistency ----------------
def make_images(scenes, character, run_dir):
    """Scene 1 is the reference. Scenes 2+ are generated with IP-Adapter, which copies
    the look of scene 1 (face, hair, clothes) while the prompt changes the action/place."""
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_ID, torch_dtype=torch.float16, use_safetensors=True,
        safety_checker=None, requires_safety_checker=False,
    )
    use_ip = False
    try:
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        use_ip = True
    except Exception as e:
        print("IP-Adapter unavailable, using text + fixed seed only:", e)
    # IMPORTANT: attention slicing replaces the IP-Adapter attention layers and causes
    # "'tuple' object has no attribute 'shape'". 512x512 fits on a T4 without it,
    # so only use slicing when IP-Adapter is not loaded.
    if not use_ip:
        pipe.enable_attention_slicing()
    pipe = pipe.to("cuda")

    style = "3d animated family film style, cinematic light, detailed"
    blank = Image.new("RGB", (512, 512), (127, 127, 127))
    seed = 1234                      # SAME seed for every scene helps keep the look stable
    paths, reference = [], None
    try:
        for i, scene in enumerate(scenes, start=1):
            # Character sheet goes FIRST and is identical word-for-word in every scene.
            prompt = f"{character}, {scene['image_prompt']}, {style}"
            kwargs = {}
            if use_ip:
                if reference is None:
                    pipe.set_ip_adapter_scale(0.0)
                    kwargs["ip_adapter_image"] = blank
                else:
                    pipe.set_ip_adapter_scale(0.65)   # raise to 0.8 for stricter sameness
                    kwargs["ip_adapter_image"] = reference
            def render(extra):
                return pipe(
                    prompt=prompt, negative_prompt=NEGATIVE,
                    width=512, height=512, num_inference_steps=25, guidance_scale=7,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                    **extra,
                ).images[0]
            try:
                img = render(kwargs)
            except OOM_ERRORS:
                raise
            except Exception as e:
                if not use_ip:
                    raise
                # IP-Adapter failed at runtime: remove it and continue with text + seed only.
                print("IP-Adapter failed, continuing without it:", e)
                try:
                    pipe.unload_ip_adapter()
                except Exception:
                    pass
                use_ip, kwargs = False, {}
                pipe.enable_attention_slicing()
                img = render({})
            if reference is None:
                reference = img
            p = run_dir / f"scene_{i}.png"
            img.save(p)
            paths.append(p)
            cleanup_cuda()
    finally:
        del pipe
        cleanup_cuda()
    return paths


# ---------------- step 4: Stable Video Diffusion ----------------
def animate_images(image_paths, run_dir, progress):
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        SVD_ID, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.enable_model_cpu_offload()          # do NOT also call .to("cuda")
    pipe.unet.enable_forward_chunking()
    clip_paths = []
    try:
        for i, img_path in enumerate(image_paths, start=1):
            progress(0.40 + 0.35 * (i - 1) / len(image_paths), desc=f"Animating scene {i} of {len(image_paths)}")
            base = Image.open(img_path).convert("RGB")
            frames, last_err = None, None
            for cfg in SVD_LADDER:
                try:
                    image = ImageOps.fit(base, (cfg["w"], cfg["h"]), method=Image.Resampling.LANCZOS)
                    with torch.inference_mode():
                        frames = pipe(
                            image, width=cfg["w"], height=cfg["h"],
                            num_frames=cfg["frames"], decode_chunk_size=1,
                            num_inference_steps=20, motion_bucket_id=90,
                            noise_aug_strength=0.02,
                            generator=torch.Generator(device="cpu").manual_seed(2200 + i),
                        ).frames[0]
                    break
                except OOM_ERRORS as e:
                    last_err = e
                    cleanup_cuda()
            if frames is None:
                raise gr.Error(
                    "The GPU ran out of memory even at the smallest video size. "
                    "Restart the Kaggle session and run only this app."
                ) from last_err
            p = run_dir / f"scene_{i}.mp4"
            export_to_video(frames, str(p), fps=7)
            clip_paths.append(p)
            cleanup_cuda()
    finally:
        del pipe
        cleanup_cuda()
    return clip_paths


# ---------------- step 5: assemble video ----------------
def fit_to_duration(clip, target):
    """Slow the clip down a little (max 2x), then ping-pong loop it to fill `target` seconds."""
    clip = clip.fx(vfx.speedx, final_duration=min(target, clip.duration * 2))
    parts, total, flip = [clip], clip.duration, True
    while total < target:
        parts.append(clip.fx(vfx.time_mirror) if flip else clip)
        total += clip.duration
        flip = not flip
    return concatenate_videoclips(parts).subclip(0, target)


def assemble(clip_paths, audio_path, output_path):
    audio = AudioFileClip(str(audio_path))
    per_scene = audio.duration / len(clip_paths)
    sources = [VideoFileClip(str(p)) for p in clip_paths]
    final = None
    try:
        fitted = [fit_to_duration(c, per_scene) for c in sources]
        final = concatenate_videoclips(fitted, method="compose").set_audio(audio)
        final = final.set_duration(audio.duration)
        final.write_videofile(
            str(output_path), codec="libx264", audio_codec="aac",
            fps=24, preset="veryfast", threads=2, logger=None,
        )
    finally:
        if final is not None:
            final.close()
        for c in sources:
            c.close()
        audio.close()
    return audio.duration


# ---------------- main function ----------------
def make_video(audio_file, character, progress=gr.Progress()):
    if not audio_file:
        raise gr.Error("Please upload an audio file first.")
    if not torch.cuda.is_available():
        raise gr.Error("No GPU found. Turn on GPU T4 in the Kaggle settings.")
    user_character = (character or "").strip()

    # A fresh folder per run, so old images/clips are never reused by mistake.
    run_dir = WORK / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    src = Path(audio_file)
    audio_path = run_dir / ("narration" + (src.suffix or ".wav"))
    audio_path.write_bytes(src.read_bytes())

    try:
        progress(0.03, desc="Listening to your audio...")
        transcript = transcribe(audio_path)
        if not transcript:
            raise gr.Error("I couldn't hear any speech in that audio.")

        progress(0.12, desc="Planning the scenes...")
        character, scenes = plan_scenes(transcript, user_character)

        progress(0.20, desc="Drawing the pictures...")
        image_paths = make_images(scenes, character, run_dir)

        progress(0.40, desc="Bringing the pictures to life...")
        clip_paths = animate_images(image_paths, run_dir, progress)

        progress(0.80, desc="Putting the video together...")
        out = run_dir / "story_video.mp4"
        assemble(clip_paths, audio_path, out)
    except gr.Error:
        raise
    except OOM_ERRORS:
        cleanup_cuda()
        raise gr.Error("Ran out of GPU memory. Restart the session and try again.")
    except Exception as e:
        cleanup_cuda()
        raise gr.Error(f"Something went wrong: {e}")

    notes = f"What I heard:\n{transcript}\n\nCharacter used in every scene:\n{character}\n\n" + "\n".join(
        f"Scene {i+1}: {s['visual_action']}" for i, s in enumerate(scenes)
    )
    progress(1.0, desc="Done!")
    return str(out), [str(p) for p in image_paths], notes


# ---------------- simple interface ----------------
with gr.Blocks(title="Story to Video") as demo:
    gr.Markdown("## 🎬 Story to Video\nUpload a recording of a story and get a short animated video back.")

    audio_in = gr.Audio(label="Your story (audio file)", type="filepath", sources=["upload"])
    with gr.Accordion("Describe your main character (optional)", open=False):
        char_in = gr.Textbox(
            show_label=False, lines=2,
            placeholder="Leave empty and I'll work it out from the story. "
                        "Example: 7 year old boy, short brown hair, blue t-shirt, beige shorts",
        )
    go = gr.Button("Make my video", variant="primary")

    video_out = gr.Video(label="Your video")
    with gr.Accordion("See the scenes and the story text", open=False):
        gallery = gr.Gallery(label="Scenes", columns=4, height=220)
        notes_out = gr.Textbox(label="Story text", lines=10)

    go.click(make_video, inputs=[audio_in, char_in], outputs=[video_out, gallery, notes_out])

demo.queue(max_size=1).launch(share=True)
