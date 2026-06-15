"""
Race Data Module — loads pre-cached Parquet for a given race and returns a
10-lap window around a pivot lap for two specified drivers.

Not imported during data ingestion. Safe to import in the Gradio Space runtime.
"""

from pathlib import Path

import pandas as pd
import yaml

_CACHE_DIR = Path(__file__).parent / "cache"
_YAML_PATH = Path(__file__).parent / "curated_races.yaml"

# Columns returned to callers (ordered)
WINDOW_COLUMNS = [
    "lap_number",
    "driver_code",
    "position",
    "gap_to_leader_s",
    "compound",
    "tyre_life",
    "lap_time_s",
    "sc_active",
]


def _race_key(season: int, round_num: int) -> str:
    """Look up race name from YAML and return the stable filename key."""
    with open(_YAML_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for race in config["races"]:
        if race["season"] == season and race["round"] == round_num:
            name = race["name"].replace(" ", "_")
            return f"{season}_{name}"

    raise ValueError(
        f"Race (season={season}, round={round_num}) not found in curated_races.yaml. "
        f"Available races: "
        + ", ".join(
            f"{r['season']} R{r['round']} {r['name']}" for r in config["races"]
        )
    )


def get_race_window(
    season: int,
    round_num: int,
    pivot_lap: int,
) -> pd.DataFrame:
    """Return a 10-lap window of lap data for all drivers around a pivot lap.

    Args:
        season: Championship year (e.g. 2023).
        round_num: Round number matching curated_races.yaml.
        pivot_lap: The lap to centre the window on.

    Returns:
        DataFrame with columns: lap_number, driver_code, position,
        gap_to_leader_s, compound, tyre_life, lap_time_s, sc_active.
        Rows for all drivers within [pivot-5, pivot+5], truncated at race
        boundaries. Sorted by lap_number, then position.

    Raises:
        FileNotFoundError: If the Parquet file for the race doesn't exist.
        ValueError: If race not in YAML or file is corrupted/unreadable.
    """
    key = _race_key(season, round_num)
    parquet_path = _CACHE_DIR / f"{key}_laps.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No cached data for {season} R{round_num}. "
            f"Expected file: {parquet_path}. "
            f"Run data/fetch_races.py to generate it."
        )

    try:
        laps = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read Parquet file {parquet_path}: {exc}"
        ) from exc

    min_lap = int(laps["lap_number"].min())
    max_lap = int(laps["lap_number"].max())
    lap_lo = max(min_lap, pivot_lap - 5)
    lap_hi = min(max_lap, pivot_lap + 5)

    mask = laps["lap_number"].between(lap_lo, lap_hi)
    window = laps.loc[mask, WINDOW_COLUMNS].copy()
    window.sort_values(["lap_number", "position"], inplace=True, ignore_index=True)
    return window


def get_lap_window(
    season: int,
    round_num: int,
    pivot_lap: int,
    driver_a: str,
    driver_b: str,
) -> pd.DataFrame:
    """Return a 10-lap window of lap data for two drivers around a pivot lap.

    Args:
        season: Championship year (e.g. 2023).
        round_num: Round number matching curated_races.yaml.
        pivot_lap: The lap to centre the window on.
        driver_a: 3-letter driver code (e.g. "VER").
        driver_b: 3-letter driver code (e.g. "HAM").

    Returns:
        DataFrame with columns: lap_number, driver_code, position,
        gap_to_leader_s, compound, tyre_life, lap_time_s, sc_active.
        Rows for both drivers within [pivot-5, pivot+5], truncated at race
        boundaries. Sorted by lap_number, then driver_code.

    Raises:
        FileNotFoundError: If the Parquet file for the race doesn't exist.
        ValueError: If race not in YAML, driver codes not found, or file is
                    corrupted/unreadable.
    """
    key = _race_key(season, round_num)
    parquet_path = _CACHE_DIR / f"{key}_laps.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No cached data for {season} R{round_num}. "
            f"Expected file: {parquet_path}. "
            f"Run data/fetch_races.py to generate it."
        )

    try:
        laps = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read Parquet file {parquet_path}: {exc}"
        ) from exc

    # Validate driver codes
    available = set(laps["driver_code"].unique())
    for code in (driver_a, driver_b):
        if code not in available:
            raise ValueError(
                f"Driver '{code}' not found in {season} R{round_num} data. "
                f"Available drivers: {', '.join(sorted(available))}."
            )

    # Compute window bounds, clamped to actual race laps
    min_lap = int(laps["lap_number"].min())
    max_lap = int(laps["lap_number"].max())
    lap_lo = max(min_lap, pivot_lap - 5)
    lap_hi = min(max_lap, pivot_lap + 5)

    mask = (
        laps["driver_code"].isin({driver_a, driver_b})
        & laps["lap_number"].between(lap_lo, lap_hi)
    )
    window = laps.loc[mask, WINDOW_COLUMNS].copy()
    window.sort_values(["lap_number", "driver_code"], inplace=True, ignore_index=True)
    return window
