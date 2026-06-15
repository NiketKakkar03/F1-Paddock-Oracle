"""Modal client — thin wrapper around Container 1 and Container 2 .remote() calls."""

import modal

generate_commentary = modal.Function.from_name("f1-paddock-oracle", "generate_commentary")
reason_strategy = modal.Function.from_name("f1-paddock-oracle", "reason_strategy")
_persona_chat = modal.Function.from_name("f1-paddock-oracle", "persona_chat")

_commentary_call_count = 0
_strategy_call_count = 0
_persona_call_count = 0

_LOADING_FIRST = "Connecting to the pit wall… (~20s first call)"
_LOADING_SUBSEQUENT = "On it — back in a few seconds"


def get_commentary_loading_message() -> str:
    return _LOADING_FIRST if _commentary_call_count == 0 else _LOADING_SUBSEQUENT


def get_strategy_loading_message() -> str:
    return _LOADING_FIRST if _strategy_call_count == 0 else _LOADING_SUBSEQUENT


def get_persona_loading_message() -> str:
    return _LOADING_FIRST if _persona_call_count == 0 else _LOADING_SUBSEQUENT


def call_generate_commentary(prompt: str, style: str = "", warmup: bool = False) -> dict:
    global _commentary_call_count
    full_prompt = f"[{style}] {prompt}" if style else prompt
    result = generate_commentary.remote(prompt=full_prompt, warmup=warmup)
    if not warmup:
        _commentary_call_count += 1
    return result


def call_reason_strategy(prompt: str, warmup: bool = False) -> dict:
    global _strategy_call_count
    result = reason_strategy.remote(prompt=prompt, warmup=warmup)
    if not warmup:
        _strategy_call_count += 1
    return result


def call_persona_chat(system_prompt: str, user_message: str, warmup: bool = False) -> dict:
    global _persona_call_count
    result = _persona_chat.remote(
        system_prompt=system_prompt,
        user_message=user_message,
        warmup=warmup,
    )
    if not warmup:
        _persona_call_count += 1
    return result


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe raw audio bytes to text using Cohere's transcription API.

    Args:
        audio_bytes: Raw audio bytes from gr.Audio component (WAV format).

    Returns:
        Transcribed text string. Returns empty string on failure.
    """
    import os
    import io
    import cohere

    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("COHERE_API_KEY environment variable not set.")

    client = cohere.Client(api_key)
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.wav"

    response = client.transcribe(file=audio_file)
    return response.text if hasattr(response, "text") else str(response)
