"""Tests for the What-If tab logic in app.py."""

import math
import types
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── helpers imported directly (no Gradio or Modal needed) ──────────────────
from app import (
    _build_timing_table,
    _build_strategy_prompt,
    _run_what_if,
    _PIVOT_LAP_MIN,
    _PIVOT_LAP_MAX,
)


def _make_window(n_rows=3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lap_number": [30, 30, 31],
            "driver_code": ["VER", "HAM", "VER"],
            "position": [1, 2, 1],
            "gap_to_leader_s": [float("nan"), 3.412, float("nan")],
            "compound": ["MEDIUM", "HARD", "MEDIUM"],
            "tyre_life": [10, 14, 11],
            "lap_time_s": [88.1, 88.9, 88.0],
            "sc_active": [False, False, False],
        }
    )


class TestBuildTimingTable(unittest.TestCase):
    def test_header_row_present(self):
        table = _build_timing_table(_make_window())
        assert "LAP" in table
        assert "DRV" in table
        assert "CMPD" in table

    def test_leader_label(self):
        table = _build_timing_table(_make_window())
        assert "LEADER" in table

    def test_gap_formatted(self):
        table = _build_timing_table(_make_window())
        assert "3.412" in table

    def test_empty_dataframe(self):
        result = _build_timing_table(pd.DataFrame())
        assert "No data" in result


class TestBuildStrategyPrompt(unittest.TestCase):
    def _race(self):
        return {
            "name": "British Grand Prix",
            "season": 2023,
            "circuit": "Silverstone",
            "top_finishers": "Verstappen, Hamilton, Alonso",
        }

    def test_fields_injected(self):
        table = "LAP  DRV  POS  GAP(s)  CMPD  AGE"
        prompt = _build_strategy_prompt(self._race(), 35, "What if HAM pitted earlier?", table)
        assert "British Grand Prix" in prompt
        assert "2023" in prompt
        assert "Silverstone" in prompt
        assert "35" in prompt
        assert "What if HAM pitted earlier?" in prompt
        assert table in prompt


class TestRunWhatIf(unittest.TestCase):
    """Test the generator function _run_what_if without hitting Modal."""

    def _collect(self, gen):
        return list(gen)

    def test_empty_whatif_returns_prompt(self):
        race = {"season": 2023, "round": 5, "name": "X", "circuit": "Y"}
        results = self._collect(_run_what_if(race, 30, "   "))
        left, right = results[0]
        assert "Enter" in left

    def test_pivot_below_min_shows_warning(self):
        race = {"season": 2023, "round": 5, "name": "X", "circuit": "Y"}
        results = self._collect(_run_what_if(race, _PIVOT_LAP_MIN - 1, "some scenario"))
        left, right = results[0]
        assert "outside the recommended range" in left
        assert right == ""

    def test_pivot_above_max_shows_warning(self):
        race = {"season": 2023, "round": 5, "name": "X", "circuit": "Y"}
        results = self._collect(_run_what_if(race, _PIVOT_LAP_MAX + 1, "some scenario"))
        left, right = results[0]
        assert "outside the recommended range" in left
        assert right == ""

    def test_pivot_at_boundary_passes_validation(self):
        race = {"season": 2023, "round": 5, "name": "X", "circuit": "Y"}
        window_df = _make_window()
        with patch("app.get_race_window", return_value=window_df), \
             patch("app.call_reason_strategy", return_value={"reasoning_chain": "Step 1. Step 2."}):
            results = self._collect(_run_what_if(race, _PIVOT_LAP_MIN, "HAM earlier pit?"))
        # last yield should contain reasoning in right panel
        final_left, final_right = results[-1]
        assert "Step 1" in final_right

    def test_file_not_found_surfaces_error(self):
        race = {"season": 2023, "round": 5, "name": "X", "circuit": "Y"}
        with patch("app.get_race_window", side_effect=FileNotFoundError("No cache")):
            results = self._collect(_run_what_if(race, 30, "some scenario"))
        # first yield is loading, second is error
        left, right = results[-1]
        assert "No cache" in left

    def test_timing_table_in_left_panel_on_success(self):
        race = {"season": 2023, "round": 5, "name": "X", "circuit": "Y"}
        window_df = _make_window()
        with patch("app.get_race_window", return_value=window_df), \
             patch("app.call_reason_strategy", return_value={"reasoning_chain": "Analysis done."}):
            results = self._collect(_run_what_if(race, 30, "What if VER pitted?"))
        final_left, final_right = results[-1]
        assert "LAP" in final_left
        assert "Analysis done." in final_right

    def test_no_audio_component(self):
        """Right panel is plain text — no audio output from _run_what_if."""
        race = {"season": 2023, "round": 5, "name": "X", "circuit": "Y"}
        window_df = _make_window()
        with patch("app.get_race_window", return_value=window_df), \
             patch("app.call_reason_strategy", return_value={"reasoning_chain": "Reasoning."}):
            results = self._collect(_run_what_if(race, 30, "What if?"))
        for left, right in results:
            # right panel is always a string, never bytes/audio
            assert isinstance(right, str)


if __name__ == "__main__":
    unittest.main()
