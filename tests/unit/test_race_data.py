"""Unit tests for the Race Data Module (Issue #3)."""

import io
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import data.race_data as rdm

FIXTURES = Path(__file__).parent.parent / "fixtures"

# Minimal YAML covering one race that matches the fixture filename
_YAML_CONTENT = b"""
races:
  - season: 2023
    round: 10
    name: "British GP"
    circuit: "Silverstone"
    modes: [what_if, demo_anchor]
"""


@pytest.fixture(autouse=True)
def patch_paths(monkeypatch, tmp_path):
    """Redirect module to use fixture Parquet and inline YAML."""
    yaml_file = tmp_path / "curated_races.yaml"
    yaml_file.write_bytes(_YAML_CONTENT)
    monkeypatch.setattr(rdm, "_CACHE_DIR", FIXTURES)
    monkeypatch.setattr(rdm, "_YAML_PATH", yaml_file)


# ---------------------------------------------------------------------------
# Window correctness
# ---------------------------------------------------------------------------

class TestWindowBounds:
    def test_mid_race_pivot_returns_11_laps(self):
        df = rdm.get_lap_window(2023, 10, 30, "VER", "HAM")
        laps = sorted(df["lap_number"].unique())
        assert laps == list(range(25, 36))

    def test_mid_race_only_two_drivers(self):
        df = rdm.get_lap_window(2023, 10, 30, "VER", "HAM")
        assert set(df["driver_code"].unique()) == {"VER", "HAM"}

    def test_pivot_near_start_truncates_at_lap_1(self):
        df = rdm.get_lap_window(2023, 10, 3, "VER", "HAM")
        assert df["lap_number"].min() == 1
        assert df["lap_number"].max() == 8

    def test_pivot_near_end_truncates_at_final_lap(self):
        # Fixture has 52 laps; pivot 50 => window [45, 52]
        df = rdm.get_lap_window(2023, 10, 50, "VER", "HAM")
        assert df["lap_number"].min() == 45
        assert df["lap_number"].max() == 52

    def test_result_sorted_by_lap_then_driver(self):
        df = rdm.get_lap_window(2023, 10, 20, "VER", "HAM")
        expected = df.sort_values(["lap_number", "driver_code"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

class TestColumns:
    def test_returns_exactly_required_columns(self):
        df = rdm.get_lap_window(2023, 10, 20, "VER", "HAM")
        assert list(df.columns) == rdm.WINDOW_COLUMNS

    def test_sc_active_is_boolean(self):
        df = rdm.get_lap_window(2023, 10, 10, "VER", "HAM")
        assert df["sc_active"].dtype == bool

    def test_sc_active_true_on_sc_laps(self):
        df = rdm.get_lap_window(2023, 10, 10, "VER", "HAM")
        sc_laps = set(df.loc[df["sc_active"], "lap_number"].unique())
        # Fixture sets sc_active on laps 10, 11, 12
        assert {10, 11, 12}.issubset(sc_laps)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrors:
    def test_wrong_driver_code_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            rdm.get_lap_window(2023, 10, 20, "VER", "ZZZ")
        assert "ZZZ" in str(exc_info.value)
        assert "Available drivers" in str(exc_info.value)

    def test_missing_parquet_raises_file_not_found(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rdm, "_CACHE_DIR", tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            rdm.get_lap_window(2023, 10, 20, "VER", "HAM")
        assert "fetch_races.py" in str(exc_info.value)
        assert "2023" in str(exc_info.value)

    def test_corrupted_parquet_raises_value_error(self, monkeypatch, tmp_path):
        bad_file = tmp_path / "2023_British_GP_laps.parquet"
        bad_file.write_bytes(b"this is not a parquet file")
        monkeypatch.setattr(rdm, "_CACHE_DIR", tmp_path)
        with pytest.raises(ValueError) as exc_info:
            rdm.get_lap_window(2023, 10, 20, "VER", "HAM")
        assert "Failed to read" in str(exc_info.value)

    def test_unknown_race_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            rdm.get_lap_window(2021, 1, 20, "VER", "HAM")
        assert "not found in curated_races.yaml" in str(exc_info.value)
