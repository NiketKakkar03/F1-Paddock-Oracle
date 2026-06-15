"""Unit tests for the Commentary tab handler (Issue #10)."""

import sys
import types
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def lap_df():
    rows = []
    for lap in range(15, 26):
        for driver, pos, team, gap in [
            ("VER", 1, "Red Bull Racing", 0.0),
            ("NOR", 2, "McLaren", 4.5),
        ]:
            rows.append({
                "lap_number": lap,
                "driver_code": driver,
                "team": team,
                "position": pos,
                "gap_to_leader_s": gap,
                "compound": "MEDIUM",
                "tyre_life": lap - 10,
                "lap_time_s": 90.1,
                "sc_active": False,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def race():
    return {"season": 2024, "round": 21, "name": "Brazilian GP", "circuit": "Interlagos", "modes": ["commentary"]}


# ---------------------------------------------------------------------------
# _top_two_drivers
# ---------------------------------------------------------------------------

class TestTopTwoDrivers:
    def test_returns_p1_p2_at_pivot_lap(self, lap_df):
        with patch("app._CACHE_DIR") as mock_cache, \
             patch("app._race_key", return_value="2024_Brazilian_GP"), \
             patch("pandas.read_parquet", return_value=lap_df):
            from app import _top_two_drivers
            driver_a, driver_b, team_name = _top_two_drivers(2024, 21, 20)
        assert driver_a == "VER"
        assert driver_b == "NOR"
        assert team_name == "Red Bull Racing"

    def test_falls_back_to_last_lap_when_pivot_missing(self, lap_df):
        # pivot_lap=99 doesn't exist — should fall back to lap 25
        with patch("app._race_key", return_value="2024_Brazilian_GP"), \
             patch("pandas.read_parquet", return_value=lap_df):
            from app import _top_two_drivers
            driver_a, driver_b, team_name = _top_two_drivers(2024, 21, 99)
        assert driver_a == "VER"
        assert driver_b == "NOR"


# ---------------------------------------------------------------------------
# _generate_commentary — mocked Modal call
# ---------------------------------------------------------------------------

class TestGenerateCommentary:
    def _make_wav_bytes(self):
        """Minimal valid 16-bit WAV bytes (44-byte header + 2 bytes of silence)."""
        import struct
        data = b"\x00\x00"
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + len(data), b"WAVE",
            b"fmt ", 16, 1, 1, 24000, 48000, 2, 16,
            b"data", len(data),
        )
        return header + data

    def test_broadcast_returns_audio_path_and_text(self, race, lap_df):
        audio_bytes = self._make_wav_bytes()
        modal_result = {"text": "Verstappen extends his lead.", "audio_wav": audio_bytes}

        with patch("app._top_two_drivers", return_value=("VER", "NOR", "Red Bull Racing")), \
             patch("app.get_lap_window", return_value=lap_df), \
             patch("app.build_commentary_prompt", return_value="[prompt]"), \
             patch("app.call_generate_commentary", return_value=modal_result):
            from app import _generate_commentary
            audio_path, text = _generate_commentary(race, 20, "Broadcast")

        assert text == "Verstappen extends his lead."
        assert audio_path is not None
        assert audio_path.endswith(".wav")

    def test_radio_passes_mode_radio_to_prompt_builder(self, race, lap_df):
        modal_result = {"text": "Box this lap.", "audio_wav": b""}

        with patch("app._top_two_drivers", return_value=("VER", "NOR", "Red Bull Racing")), \
             patch("app.get_lap_window", return_value=lap_df), \
             patch("app.build_commentary_prompt", return_value="[prompt]") as mock_build, \
             patch("app.call_generate_commentary", return_value=modal_result):
            from app import _generate_commentary
            _generate_commentary(race, 20, "Radio")

        mock_build.assert_called_once_with(lap_df, "Red Bull Racing", "radio")

    def test_broadcast_passes_mode_broadcast_to_prompt_builder(self, race, lap_df):
        modal_result = {"text": "Hamilton closes in.", "audio_wav": b""}

        with patch("app._top_two_drivers", return_value=("VER", "HAM", "Red Bull Racing")), \
             patch("app.get_lap_window", return_value=lap_df), \
             patch("app.build_commentary_prompt", return_value="[prompt]") as mock_build, \
             patch("app.call_generate_commentary", return_value=modal_result):
            from app import _generate_commentary
            _generate_commentary(race, 20, "Broadcast")

        mock_build.assert_called_once_with(lap_df, "Red Bull Racing", "broadcast")

    def test_no_audio_bytes_returns_none_path(self, race, lap_df):
        modal_result = {"text": "VER leads.", "audio_wav": b""}

        with patch("app._top_two_drivers", return_value=("VER", "NOR", "Red Bull Racing")), \
             patch("app.get_lap_window", return_value=lap_df), \
             patch("app.build_commentary_prompt", return_value="[prompt]"), \
             patch("app.call_generate_commentary", return_value=modal_result):
            from app import _generate_commentary
            audio_path, text = _generate_commentary(race, 20, "Broadcast")

        assert audio_path is None
        assert text == "VER leads."


# ---------------------------------------------------------------------------
# Two-tier loading copy
# ---------------------------------------------------------------------------

class TestLoadingMessages:
    def test_first_call_returns_pit_wall_message(self):
        from modal_backend.client import get_commentary_loading_message
        import modal_backend.client as client_mod
        original = client_mod._commentary_call_count
        try:
            client_mod._commentary_call_count = 0
            msg = get_commentary_loading_message()
            assert "pit wall" in msg.lower()
        finally:
            client_mod._commentary_call_count = original

    def test_subsequent_call_returns_shorter_message(self):
        from modal_backend.client import get_commentary_loading_message
        import modal_backend.client as client_mod
        original = client_mod._commentary_call_count
        try:
            client_mod._commentary_call_count = 1
            msg = get_commentary_loading_message()
            assert msg != "Connecting to the pit wall… (~20s first call)"
        finally:
            client_mod._commentary_call_count = original
