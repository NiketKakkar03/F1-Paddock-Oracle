"""
Generate synthetic Parquet fixtures for race_data unit tests.
Run once: python tests/fixtures/make_fixtures.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

OUT_DIR = Path(__file__).parent
LAPS_COLUMNS = [
    "lap_number", "driver_code", "team", "lap_time_s", "compound",
    "tyre_life", "position", "gap_to_leader_s", "pit_in", "pit_out", "sc_active",
]

DRIVERS = ["VER", "HAM", "NOR", "PER", "ALO"]
TOTAL_LAPS = 52


def _make_laps() -> pd.DataFrame:
    rows = []
    for lap in range(1, TOTAL_LAPS + 1):
        for i, drv in enumerate(DRIVERS):
            rows.append({
                "lap_number": lap,
                "driver_code": drv,
                "team": "RBR" if drv in ("VER", "PER") else "MER",
                "lap_time_s": 90.0 + i * 0.3 + np.random.uniform(-0.1, 0.1),
                "compound": "MEDIUM" if lap <= 25 else "HARD",
                "tyre_life": lap if lap <= 25 else lap - 25,
                "position": i + 1,
                "gap_to_leader_s": 0.0 if i == 0 else i * 2.5,
                "pit_in": lap == 26 and i == 0,
                "pit_out": lap == 27 and i == 0,
                "sc_active": lap in (10, 11, 12),
            })
    df = pd.DataFrame(rows, columns=LAPS_COLUMNS)
    df["lap_number"] = df["lap_number"].astype("int32")
    df["tyre_life"] = df["tyre_life"].astype("int32")
    df["position"] = df["position"].astype("int32")
    return df


if __name__ == "__main__":
    laps = _make_laps()
    path = OUT_DIR / "2023_British_GP_laps.parquet"
    laps.to_parquet(path, index=False)
    print(f"Written {len(laps)} rows to {path}")
