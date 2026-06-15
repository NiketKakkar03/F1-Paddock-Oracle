import modal

MODEL_ID = "openbmb/MiniCPM-o-4_5"
MODEL_DIR = "/model-weights/minicpm-o-4_5"

app = modal.App("f1-paddock-oracle")

volume = modal.Volume.from_name("f1-model-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.51.0",
        "tokenizers==0.21.0",
        "accelerate>=0.30.0",
        "minicpmo-utils>=1.0.5",
        "sentencepiece",
        "soundfile",
        "scipy",
        "huggingface_hub",
        "kokoro>=0.9.4",
    )
    .pip_install(
        "click",
        "spacy",
    )
)


def _load_model():
    import os
    import shutil
    import torch
    from transformers import AutoModel, AutoTokenizer

    hf_token = os.environ["HF_TOKEN"]

    modules_cache = "/root/.cache/huggingface/modules"
    if os.path.exists(modules_cache):
        shutil.rmtree(modules_cache)

    from huggingface_hub import snapshot_download

    sentinel = os.path.join(MODEL_DIR, ".download_complete")
    if not os.path.exists(sentinel):
        if os.path.exists(MODEL_DIR):
            shutil.rmtree(MODEL_DIR)
        snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR, token=hf_token)
        open(sentinel, "w").close()
        volume.commit()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def _tts(text: str) -> bytes:
    import io
    import numpy as np
    import scipy.io.wavfile as wav_writer
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="b")  # "b" = British English
    samples = []
    for _, _, audio in pipeline(text, voice="bm_daniel", speed=1.1):
        samples.append(audio)

    if not samples:
        return b""

    audio_np = np.concatenate(samples)
    if audio_np.dtype != np.int16:
        audio_np = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    wav_writer.write(buf, 24000, audio_np)
    return buf.getvalue()


@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/model-weights": volume},
    scaledown_window=300,
)
def generate_commentary(prompt: str, warmup: bool = False) -> dict:
    model, tokenizer = _load_model()

    if warmup:
        return {}

    sys_msg = model.get_sys_prompt(mode="omni", language="en")
    msgs = [sys_msg, {"role": "user", "content": [prompt]}]

    text = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        max_new_tokens=512,
        use_tts_template=False,
        generate_audio=False,
        do_sample=True,
        temperature=0.7,
    )
    text = str(text)

    audio_bytes = _tts(text)

    return {"text": text, "audio_wav": audio_bytes}


@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/model-weights": volume},
    scaledown_window=300,
)
def persona_chat(system_prompt: str, user_message: str, warmup: bool = False) -> dict:
    model, tokenizer = _load_model()

    if warmup:
        return {}

    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    text = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        max_new_tokens=512,
        use_tts_template=False,
        generate_audio=False,
        do_sample=True,
        temperature=0.8,
    )
    text = str(text)

    audio_bytes = _tts(text)

    return {"text": text, "audio_wav": audio_bytes}


@app.local_entrypoint()
def smoke_test():
    prompt = (
        "LAP 47 of 57 at Monaco. Verstappen leads Hamilton by 6.2 seconds. "
        "Hamilton is on worn mediums, tyre age 28 laps. "
        "Generate a 2-sentence broadcast commentary update."
    )
    result = generate_commentary.remote(prompt=prompt, warmup=False)
    assert result["text"], "Smoke test failed: empty text"
    assert result["audio_wav"], "Smoke test failed: empty audio"
    print(f"OK — text ({len(result['text'])} chars), audio ({len(result['audio_wav'])} bytes)")
