"""Loads driver persona files and builds system prompts for the persona chat mode."""

from pathlib import Path

_DRIVERS_DIR = Path(__file__).parent / "drivers"

# Drivers whose era overlaps with post-2023 race data
_ACTIVE_DRIVERS = {"verstappen", "hamilton", "norris"}


def _driver_file(name: str) -> Path:
    return _DRIVERS_DIR / f"{name.lower()}.txt"


def get_system_prompt(driver: str, race_context: str | None = None) -> str:
    """Return a complete system prompt string for the given driver.

    Args:
        driver: Driver name (case-insensitive). Must match a file in prompts/drivers/.
        race_context: Optional race description injected for active drivers only.
                      Silently ignored for historical drivers (Senna, Schumacher).

    Raises:
        FileNotFoundError: If no persona file exists for the given driver name.
    """
    key = driver.lower()
    path = _driver_file(key)

    if not path.exists():
        raise FileNotFoundError(
            f"No persona file found for driver '{driver}'. "
            f"Expected file: {path}. "
            f"Available drivers: {', '.join(_available_drivers())}."
        )

    persona_text = path.read_text(encoding="utf-8")

    if race_context and key in _ACTIVE_DRIVERS:
        race_block = (
            "\n---\n\n"
            "### Current Race Context\n\n"
            f"{race_context.strip()}\n\n"
            "Draw on this race context when answering questions about current strategy, "
            "tire behavior, gaps, or session conditions. Do not contradict information "
            "provided here.\n"
        )
        return persona_text + race_block

    return persona_text


def _available_drivers() -> list[str]:
    return sorted(p.stem for p in _DRIVERS_DIR.glob("*.txt"))
