"""Unit tests for the Prompt Builder module (Issue #7)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prompts.builder import (
    build_commentary_prompt,
    build_persona_prompt,
    build_strategy_prompt,
    _ANTI_HALLUCINATION,
    _MINICPM_TOKEN_LIMIT,
    _NEMOTRON_TOKEN_LIMIT,
    _CHARS_PER_TOKEN,
)
from data.race_data import WINDOW_COLUMNS


# ---------------------------------------------------------------------------
# Fixture: minimal 10-lap × 2-driver DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture()
def lap_df() -> pd.DataFrame:
    rows = []
    for lap in range(41, 51):  # laps 41-50, pivot=45
        for driver, pos, gap in [("VER", 1, 0.0), ("HAM", 2, 6.2)]:
            rows.append({
                "lap_number": lap,
                "driver_code": driver,
                "position": pos,
                "gap_to_leader_s": gap,
                "compound": "MEDIUM",
                "tyre_life": lap - 30,
                "lap_time_s": 90.1 + (lap * 0.01),
                "sc_active": False,
            })
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


# ---------------------------------------------------------------------------
# Commentary — broadcast mode
# ---------------------------------------------------------------------------

class TestBroadcastMode:
    def test_returns_string(self, lap_df):
        result = build_commentary_prompt(lap_df, "Oracle Red Bull Racing", "broadcast")
        assert isinstance(result, str)

    def test_team_name_interpolated(self, lap_df):
        result = build_commentary_prompt(lap_df, "Oracle Red Bull Racing", "broadcast")
        assert "Oracle Red Bull Racing" in result

    def test_lap_data_interpolated(self, lap_df):
        result = build_commentary_prompt(lap_df, "Oracle Red Bull Racing", "broadcast")
        assert "VER" in result
        assert "HAM" in result

    def test_broadcast_tone_marker_present(self, lap_df):
        result = build_commentary_prompt(lap_df, "Oracle Red Bull Racing", "broadcast")
        # Template contains "television broadcast" or "broadcast commentator"
        assert "broadcast" in result.lower()

    def test_within_minicpm_context_window(self, lap_df):
        result = build_commentary_prompt(lap_df, "Oracle Red Bull Racing", "broadcast")
        assert len(result) <= _MINICPM_TOKEN_LIMIT * _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Commentary — radio mode
# ---------------------------------------------------------------------------

class TestRadioMode:
    def test_returns_string(self, lap_df):
        result = build_commentary_prompt(lap_df, "Mercedes-AMG Petronas", "radio")
        assert isinstance(result, str)

    def test_team_name_interpolated(self, lap_df):
        result = build_commentary_prompt(lap_df, "Mercedes-AMG Petronas", "radio")
        assert "Mercedes-AMG Petronas" in result

    def test_radio_tone_marker_present(self, lap_df):
        result = build_commentary_prompt(lap_df, "Mercedes-AMG Petronas", "radio")
        # Template contains "race engineer" or "team radio"
        assert "radio" in result.lower() or "race engineer" in result.lower()

    def test_within_minicpm_context_window(self, lap_df):
        result = build_commentary_prompt(lap_df, "Mercedes-AMG Petronas", "radio")
        assert len(result) <= _MINICPM_TOKEN_LIMIT * _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Broadcast vs radio are structurally different
# ---------------------------------------------------------------------------

class TestBroadcastVsRadioStructure:
    def test_prompts_are_different(self, lap_df):
        broadcast = build_commentary_prompt(lap_df, "Oracle Red Bull Racing", "broadcast")
        radio = build_commentary_prompt(lap_df, "Oracle Red Bull Racing", "radio")
        assert broadcast != radio

    def test_different_instruction_sections(self, lap_df):
        broadcast = build_commentary_prompt(lap_df, "Oracle Red Bull Racing", "broadcast")
        radio = build_commentary_prompt(lap_df, "Oracle Red Bull Racing", "radio")
        # Broadcast instructs TV-style full names; radio uses terse fragments
        assert "television" in broadcast.lower() or "global audience" in broadcast.lower()
        assert "terse" in radio.lower() or "clipped" in radio.lower() or "race engineer" in radio.lower()

    def test_invalid_mode_raises(self, lap_df):
        with pytest.raises(ValueError, match="Unknown commentary mode"):
            build_commentary_prompt(lap_df, "Red Bull", "freestyle")


# ---------------------------------------------------------------------------
# Strategy mode
# ---------------------------------------------------------------------------

class TestStrategyMode:
    WHAT_IF = "What if Hamilton had pitted 5 laps earlier on lap 30 for fresh mediums?"

    def test_returns_string(self, lap_df):
        result = build_strategy_prompt(lap_df, self.WHAT_IF)
        assert isinstance(result, str)

    def test_lap_data_interpolated(self, lap_df):
        result = build_strategy_prompt(lap_df, self.WHAT_IF)
        assert "VER" in result
        assert "HAM" in result

    def test_what_if_variable_interpolated(self, lap_df):
        result = build_strategy_prompt(lap_df, self.WHAT_IF)
        assert self.WHAT_IF.strip() in result

    def test_anti_hallucination_instruction_present(self, lap_df):
        result = build_strategy_prompt(lap_df, self.WHAT_IF)
        assert _ANTI_HALLUCINATION in result

    def test_anti_hallucination_references_invented_values(self, lap_df):
        result = build_strategy_prompt(lap_df, self.WHAT_IF)
        assert "invent" in result.lower() or "fabricat" in result.lower()

    def test_within_nemotron_context_window(self, lap_df):
        result = build_strategy_prompt(lap_df, self.WHAT_IF)
        assert len(result) <= _NEMOTRON_TOKEN_LIMIT * _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Persona mode — cutoff block
# ---------------------------------------------------------------------------

class TestPersonaMode:
    def test_returns_string_for_active_driver(self):
        result = build_persona_prompt("verstappen")
        assert isinstance(result, str)
        assert len(result) > 100

    def test_cutoff_block_appended_for_active_driver(self):
        for driver in ["verstappen", "hamilton", "norris"]:
            result = build_persona_prompt(driver)
            assert "Knowledge Cutoff" in result, f"{driver}: missing cutoff block"
            assert "end of 2023" in result, f"{driver}: wrong cutoff year"

    def test_cutoff_block_appended_for_historical_drivers(self):
        result_senna = build_persona_prompt("senna")
        assert "Knowledge Cutoff" in result_senna
        assert "1994" in result_senna

        result_schumi = build_persona_prompt("schumacher")
        assert "Knowledge Cutoff" in result_schumi
        assert "2012" in result_schumi

    def test_race_context_injected_for_active_driver(self):
        ctx = "2023 Dutch GP, Lap 42. Verstappen leads by 4.8s."
        result = build_persona_prompt("verstappen", race_context=ctx)
        assert ctx in result
        assert "Current Race Context" in result

    def test_race_context_absent_for_historical_driver(self):
        ctx = "2023 Dutch GP, Lap 42."
        result = build_persona_prompt("senna", race_context=ctx)
        assert ctx not in result
        assert "Current Race Context" not in result

        result_s = build_persona_prompt("schumacher", race_context=ctx)
        assert ctx not in result_s

    def test_cutoff_block_comes_after_persona_text(self):
        result = build_persona_prompt("verstappen")
        persona_end = result.index("Knowledge Cutoff")
        # There must be substantial persona content before the cutoff block
        assert persona_end > 200

    def test_missing_driver_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            build_persona_prompt("alonso")

    def test_within_minicpm_context_window(self):
        result = build_persona_prompt("verstappen")
        assert len(result) <= _MINICPM_TOKEN_LIMIT * _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Prompt length validation
# ---------------------------------------------------------------------------

class TestPromptLengthValidation:
    def test_oversized_strategy_prompt_raises(self):
        # Build a DataFrame with enough rows to exceed Nemotron's limit
        rows = []
        for lap in range(1, 5001):
            rows.append({
                "lap_number": lap,
                "driver_code": "VER",
                "position": 1,
                "gap_to_leader_s": 0.0,
                "compound": "HARD",
                "tyre_life": lap,
                "lap_time_s": 90.0,
                "sc_active": False,
            })
        huge_df = pd.DataFrame(rows, columns=WINDOW_COLUMNS)
        with pytest.raises(ValueError, match="context window"):
            build_strategy_prompt(huge_df, "any change")

    def test_max_10lap_2driver_strategy_within_bounds(self, lap_df):
        # The standard 10-lap × 2-driver window must not trip the limit
        result = build_strategy_prompt(lap_df, "Hamilton pits 5 laps earlier.")
        assert len(result) <= _NEMOTRON_TOKEN_LIMIT * _CHARS_PER_TOKEN
