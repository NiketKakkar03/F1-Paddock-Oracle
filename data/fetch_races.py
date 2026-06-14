"""
Offline ingestion script — run once before deployment.
Reads curated_races.yaml, fetches FastF1 data for each race,
and writes Parquet files to data/cache/.

Not imported by app.py. FastF1 is a dev-only dependency.

Usage:
    python data/fetch_races.py
    python data/fetch_races.py --dry-run   # validate YAML only, no fetching
"""

import argparse
import os
from pathlib import Path

import fastf1
import pandas as pd
import yaml

from data.schemas import LAPS_COLUMNS, WEATHER_COLUMNS

CACHE_DIR = Path("data/cache")
YAML_PATH = Path("data/curated_races.yaml")

fastf1.Cache.enable_cache(str(CACHE_DIR / ".fastf1_cache"))


def load_race_list() -> list[dict]:
    with open(YAML_PATH) as f:
        config = yaml.safe_load(f)
    return config["races"]


def race_key(season: int, name: str) -> str:
    """Stable filename key: e.g. '2023_British_GP'"""
    return f"{season}_{name.replace(' ', '_')}"


def derive_sc_active(session) -> pd.Series:
    """Derive a per-lap boolean SC flag from session track status events."""
    try:
        track_status = session.track_status
        sc_laps = set()
        for _, row in track_status.iterrows():
            # Status "4" = safety car, "5" = virtual safety car, "6" = red flag
            if str(row.get("Status", "")) in {"4", "5"}:
                lap_time = row.get("Time")
                if lap_time is not None:
                    lap_num = session.laps[
                        session.laps["Time"] >= lap_time
                    ]["LapNumber"].min()
                    if pd.notna(lap_num):
                        sc_laps.add(int(lap_num))
        return sc_laps
    except Exception:
        return set()


def fetch_race(season: int, round_num: int, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    session = fastf1.get_session(season, round_num, "R")
    session.load(laps=True, weather=True, telemetry=False, messages=False)

    sc_lap_numbers = derive_sc_active(session)

    raw_laps = session.laps

    # Compute gap to leader per lap
    leader_times = (
        raw_laps[raw_laps["Position"] == 1]
        .set_index("LapNumber")["Time"]
    )

    def gap_to_leader(row):
        leader_time = leader_times.get(row["LapNumber"])
        if leader_time is None or pd.isna(row["Time"]) or pd.isna(leader_time):
            return float("nan")
        delta = (row["Time"] - leader_time).total_seconds()
        return max(delta, 0.0)

    laps = pd.DataFrame()
    laps["lap_number"] = raw_laps["LapNumber"].astype("int32")
    laps["driver_code"] = raw_laps["Driver"].astype("str")
    laps["team"] = raw_laps["Team"].astype("str")
    laps["lap_time_s"] = raw_laps["LapTime"].dt.total_seconds()
    laps["compound"] = raw_laps["Compound"].fillna("UNKNOWN").astype("str")
    laps["tyre_life"] = raw_laps["TyreLife"].fillna(0).astype("int32")
    laps["position"] = raw_laps["Position"].fillna(0).astype("int32")
    laps["gap_to_leader_s"] = raw_laps.apply(gap_to_leader, axis=1)
    laps["pit_in"] = raw_laps["PitInTime"].notna()
    laps["pit_out"] = raw_laps["PitOutTime"].notna()
    laps["sc_active"] = raw_laps["LapNumber"].isin(sc_lap_numbers)

    laps = laps[LAPS_COLUMNS]

    # Weather: sample once per lap using session weather_data
    weather_raw = session.weather_data
    total_laps = int(raw_laps["LapNumber"].max())
    session_duration = session.laps["Time"].max()

    weather_rows = []
    for lap_num in range(1, total_laps + 1):
        lap_rows = raw_laps[raw_laps["LapNumber"] == lap_num]
        if lap_rows.empty:
            continue
        lap_end_time = lap_rows["Time"].max()
        # Take weather snapshot at lap end time
        w = weather_raw[weather_raw["Time"] <= lap_end_time]
        if w.empty:
            w = weather_raw.iloc[[0]]
        else:
            w = w.iloc[[-1]]
        weather_rows.append({
            "lap_number": lap_num,
            "air_temp_c": float(w["AirTemp"].iloc[0]),
            "track_temp_c": float(w["TrackTemp"].iloc[0]),
            "rainfall": bool(w["Rainfall"].iloc[0]),
            "humidity": float(w["Humidity"].iloc[0]),
            "wind_speed_ms": float(w["WindSpeed"].iloc[0]),
        })

    weather = pd.DataFrame(weather_rows, columns=WEATHER_COLUMNS)

    return laps, weather


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  Saved {path} ({len(df)} rows)")


def main(dry_run: bool = False) -> None:
    races = load_race_list()
    print(f"Found {len(races)} races in {YAML_PATH}")

    if dry_run:
        for race in races:
            print(f"  [DRY RUN] {race['season']} {race['name']} (round {race['round']})")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for race in races:
        season = race["season"]
        round_num = race["round"]
        name = race["name"]
        key = race_key(season, name)

        laps_path = CACHE_DIR / f"{key}_laps.parquet"
        weather_path = CACHE_DIR / f"{key}_weather.parquet"

        if laps_path.exists() and weather_path.exists():
            print(f"Skipping {season} {name} — already cached")
            continue

        print(f"Fetching {season} {name} (round {round_num})...")
        try:
            laps, weather = fetch_race(season, round_num, name)
            save_parquet(laps, laps_path)
            save_parquet(weather, weather_path)
        except Exception as e:
            print(f"  ERROR fetching {season} {name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate YAML only, no fetching")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
