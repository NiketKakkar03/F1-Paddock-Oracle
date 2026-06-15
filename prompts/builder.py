"""Prompt Builder — stateless module that assembles ready-to-send prompt strings.

Each public function takes structured inputs and returns a single str.
All functions validate the result against the target model's context window.
"""

from pathlib import Path

import pandas as pd

from prompts.persona import get_system_prompt

_TEMPLATES_DIR = Path(__file__).parent

# Conservative token limits per target model (4 chars ≈ 1 token)
# MiniCPM-o 4.5 context: 32k tokens; Nemotron Nano context: 8k tokens
_CHARS_PER_TOKEN = 4
_MINICPM_TOKEN_LIMIT = 32_000
_NEMOTRON_TOKEN_LIMIT = 8_000

_ANTI_HALLUCINATION = (
    "IMPORTANT: Do not invent lap time values. "
    "Use only the lap data provided above. "
    "If a value is missing, acknowledge uncertainty rather than fabricating it."
)

_CUTOFF_BLOCK_TEMPLATE = (
    "\n---\n\n"
    "### Knowledge Cutoff Reminder\n\n"
    "Your knowledge of this driver ends at {cutoff}. "
    "Do not reference events, results, or team changes after that date. "
    "If the user asks about anything beyond your cutoff, deflect using one of your deflection phrases.\n"
)

# Map driver name → cutoff string (must match what the persona files state)
_DRIVER_CUTOFFS = {
    "verstappen": "end of 2023",
    "hamilton": "end of 2023",
    "norris": "end of 2023",
    "senna": "May 1994",
    "schumacher": "end of 2012",
}

_ACTIVE_DRIVERS = {"verstappen", "hamilton", "norris"}


def _load_template(filename: str) -> str:
    path = _TEMPLATES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8")


def _validate_length(prompt: str, token_limit: int, label: str) -> None:
    char_limit = token_limit * _CHARS_PER_TOKEN
    if len(prompt) > char_limit:
        tokens_approx = len(prompt) // _CHARS_PER_TOKEN
        raise ValueError(
            f"{label} prompt exceeds {token_limit}-token context window "
            f"(~{tokens_approx} tokens estimated). Reduce input size."
        )


def _lap_table_str(lap_df: pd.DataFrame) -> str:
    return lap_df.to_string(index=False)


def build_commentary_prompt(
    lap_df: pd.DataFrame,
    team_name: str,
    mode: str,
) -> str:
    """Build a commentary prompt for broadcast or radio mode.

    Args:
        lap_df: DataFrame from get_lap_window() (WINDOW_COLUMNS schema).
        team_name: Team name to include in the prompt (e.g. "Oracle Red Bull Racing").
        mode: Either "broadcast" or "radio".

    Returns:
        Fully-formed prompt string.

    Raises:
        ValueError: If mode is invalid or prompt exceeds context window.
    """
    if mode == "broadcast":
        template = _load_template("commentary_broadcast.txt")
    elif mode == "radio":
        template = _load_template("commentary_radio.txt")
    else:
        raise ValueError(f"Unknown commentary mode '{mode}'. Expected 'broadcast' or 'radio'.")

    prompt = template.format(
        team_name=team_name,
        lap_table=_lap_table_str(lap_df),
    )
    _validate_length(prompt, _MINICPM_TOKEN_LIMIT, f"Commentary ({mode})")
    return prompt


def build_strategy_prompt(
    lap_df: pd.DataFrame,
    what_if_variable: str,
) -> str:
    """Build a strategy what-if prompt for Nemotron Nano.

    Args:
        lap_df: DataFrame from get_lap_window() (WINDOW_COLUMNS schema).
        what_if_variable: User-supplied scenario change (e.g. "Hamilton pits 5 laps earlier").

    Returns:
        Fully-formed prompt string.

    Raises:
        ValueError: If prompt exceeds context window.
    """
    lap_table = _lap_table_str(lap_df)
    prompt = (
        f"### Lap Data (10-lap window)\n\n"
        f"{lap_table}\n\n"
        f"### What-If Variable\n\n"
        f"{what_if_variable.strip()}\n\n"
        f"### Instructions\n\n"
        f"Reason through how this change affects pit windows, undercut/overcut risk, "
        f"tyre degradation, and track position. Narrate the alternate outcome with "
        f"specific lap numbers and position changes. Produce a plausible alternate "
        f"final top-5.\n\n"
        f"{_ANTI_HALLUCINATION}"
    )
    _validate_length(prompt, _NEMOTRON_TOKEN_LIMIT, "Strategy")
    return prompt


def build_persona_prompt(
    driver: str,
    race_context: str | None = None,
) -> str:
    """Build a persona system prompt for the given driver.

    Args:
        driver: Driver name (case-insensitive). Must match a file in prompts/drivers/.
        race_context: Optional live race description injected for active drivers.

    Returns:
        Fully-formed system prompt string with cutoff block appended.

    Raises:
        FileNotFoundError: If no persona file exists for the driver.
        ValueError: If prompt exceeds context window.
    """
    key = driver.lower()
    prompt = get_system_prompt(key, race_context=race_context)

    cutoff = _DRIVER_CUTOFFS.get(key)
    if cutoff:
        prompt += _CUTOFF_BLOCK_TEMPLATE.format(cutoff=cutoff)

    _validate_length(prompt, _MINICPM_TOKEN_LIMIT, f"Persona ({driver})")
    return prompt
