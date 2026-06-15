import modal

from modal_backend.commentary import app

model_volume = modal.Volume.from_name("f1-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.51.0",
        "accelerate>=0.30.0",
        "huggingface-hub>=0.22.0",
        "sentencepiece",
    )
    .env({"HF_HOME": "/model-cache", "TRANSFORMERS_CACHE": "/model-cache"})
)

MODEL_ID = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"

SYSTEM_PROMPT = """You are a senior F1 strategist with deep knowledge of historical races.
The user will describe a real race and change one variable.
Your task:
1. Briefly acknowledge the actual race outcome (1 sentence)
2. Reason through how the changed variable affects pit windows, undercut/overcut risk, tire deg, and track position (3–5 sentences)
3. Narrate the alternate outcome with specific lap numbers and position changes
4. Produce a plausible alternate final top-5

Be specific. Reference real team strategies and driver tendencies."""


def _load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@app.function(
    image=image,
    gpu="A100",
    timeout=300,
    volumes={"/model-cache": model_volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
def reason_strategy(prompt: str, warmup: bool = False) -> dict:
    import os
    import torch

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    model, tokenizer = _load_model()

    if warmup:
        return {"status": "warm"}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    reasoning_chain = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return {"reasoning_chain": reasoning_chain}


@app.local_entrypoint()
def main():
    sample_prompt = (
        "Race: 2023 British Grand Prix\n"
        "Original outcome: Verstappen won from pole, Hamilton finished P4 after a late pit.\n"
        "User change: What if Hamilton had pitted 5 laps earlier on lap 30 for fresh mediums?"
    )

    print("Running warmup...")
    result = reason_strategy.remote(prompt="", warmup=True)
    print(f"Warmup result: {result}")

    print("\nRunning strategy inference...")
    result = reason_strategy.remote(prompt=sample_prompt, warmup=False)
    reasoning = result.get("reasoning_chain", "")
    assert reasoning, "reasoning_chain is empty — inference failed"
    print(f"\nReasoning chain:\n{reasoning}")
