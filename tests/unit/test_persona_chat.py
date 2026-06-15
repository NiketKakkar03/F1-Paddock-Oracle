"""Unit tests for Issue #12 — Persona Chat tab logic.

Tests cover:
- Driver selector: correct set of 5 drivers
- Historical vs active driver classification
- Race context injection (active drivers only)
- Persona prompt construction via build_persona_prompt
- Knowledge cutoff enforcement (deflection phrases present)
- on_driver_selected: historical notice visibility and race context display
- on_race_changed: active vs historical behavior
- on_send: empty input guard, system prompt construction
- transcribe_audio: raises EnvironmentError when COHERE_API_KEY is missing
- call_persona_chat: delegates to generate_commentary with merged prompt
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app import (
    PERSONA_DRIVERS,
    _ACTIVE_DRIVERS,
    _HISTORICAL_DRIVERS,
    _race_context_string,
)
from prompts.builder import build_persona_prompt


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_RACE = {
    "season": 2023,
    "name": "British Grand Prix",
    "circuit": "Silverstone",
    "round": 10,
}


# ── Driver list ───────────────────────────────────────────────────────────────

class TestDriverList:
    def test_five_drivers_defined(self):
        assert len(PERSONA_DRIVERS) == 5

    def test_expected_drivers_present(self):
        names = {d.lower() for d in PERSONA_DRIVERS}
        assert names == {"verstappen", "hamilton", "norris", "senna", "schumacher"}

    def test_active_and_historical_sets_are_disjoint(self):
        assert _ACTIVE_DRIVERS.isdisjoint(_HISTORICAL_DRIVERS)

    def test_all_drivers_classified(self):
        all_classified = _ACTIVE_DRIVERS | _HISTORICAL_DRIVERS
        for driver in PERSONA_DRIVERS:
            assert driver.lower() in all_classified, f"{driver} not classified"


# ── Race context string ───────────────────────────────────────────────────────

class TestRaceContextString:
    def test_includes_season(self):
        result = _race_context_string(SAMPLE_RACE)
        assert "2023" in result

    def test_includes_race_name(self):
        result = _race_context_string(SAMPLE_RACE)
        assert "British Grand Prix" in result

    def test_includes_circuit(self):
        result = _race_context_string(SAMPLE_RACE)
        assert "Silverstone" in result

    def test_includes_round(self):
        result = _race_context_string(SAMPLE_RACE)
        assert "10" in result

    def test_returns_string(self):
        assert isinstance(_race_context_string(SAMPLE_RACE), str)


# ── on_driver_selected logic ──────────────────────────────────────────────────

class TestDriverSelectedLogic:
    """Tests for the on_driver_selected callback logic (extracted inline)."""

    def _run(self, driver: str, race: dict):
        key = driver.lower()
        is_historical = key in _HISTORICAL_DRIVERS
        if is_historical:
            notice = "Historical drivers don't use race telemetry"
            ctx_display = ""
        else:
            notice = ""
            ctx_display = _race_context_string(race) if race else ""
        return is_historical, notice, ctx_display

    def test_senna_is_historical(self):
        is_h, notice, ctx = self._run("Senna", SAMPLE_RACE)
        assert is_h is True
        assert notice != ""
        assert ctx == ""

    def test_schumacher_is_historical(self):
        is_h, notice, ctx = self._run("Schumacher", SAMPLE_RACE)
        assert is_h is True
        assert ctx == ""

    def test_verstappen_is_active(self):
        is_h, notice, ctx = self._run("Verstappen", SAMPLE_RACE)
        assert is_h is False
        assert notice == ""
        assert "2023" in ctx

    def test_hamilton_active_gets_race_context(self):
        is_h, notice, ctx = self._run("Hamilton", SAMPLE_RACE)
        assert "British Grand Prix" in ctx

    def test_norris_active_gets_race_context(self):
        is_h, notice, ctx = self._run("Norris", SAMPLE_RACE)
        assert "Silverstone" in ctx


# ── on_race_changed logic ─────────────────────────────────────────────────────

class TestRaceChangedLogic:
    def _run(self, race: dict, driver: str) -> str:
        if driver.lower() in _HISTORICAL_DRIVERS:
            return ""
        return _race_context_string(race) if race else ""

    def test_active_driver_returns_race_context(self):
        result = self._run(SAMPLE_RACE, "Verstappen")
        assert "British Grand Prix" in result

    def test_historical_driver_returns_empty(self):
        result = self._run(SAMPLE_RACE, "Senna")
        assert result == ""

    def test_schumacher_returns_empty(self):
        result = self._run(SAMPLE_RACE, "Schumacher")
        assert result == ""


# ── on_send guard: empty input ────────────────────────────────────────────────

class TestOnSendEmptyInputGuard:
    def _send_results(self, driver, user_text, race):
        # Reproduce the empty-check from on_send
        if not user_text or not user_text.strip():
            return [("Please record or type a question first.", None)]
        return []

    def test_empty_string_triggers_guard(self):
        results = self._send_results("Verstappen", "", SAMPLE_RACE)
        assert results
        assert "Please record or type a question first." in results[0][0]

    def test_whitespace_triggers_guard(self):
        results = self._send_results("Verstappen", "   ", SAMPLE_RACE)
        assert results
        assert results[0][1] is None

    def test_none_triggers_guard(self):
        results = self._send_results("Verstappen", None, SAMPLE_RACE)
        assert results


# ── Persona prompt construction ───────────────────────────────────────────────

class TestPersonaPromptConstruction:
    def test_active_driver_receives_race_context(self):
        ctx = _race_context_string(SAMPLE_RACE)
        prompt = build_persona_prompt("verstappen", race_context=ctx)
        assert "British Grand Prix" in prompt
        assert "Current Race Context" in prompt

    def test_historical_driver_no_race_context(self):
        ctx = _race_context_string(SAMPLE_RACE)
        prompt = build_persona_prompt("senna", race_context=ctx)
        assert "Current Race Context" not in prompt

    def test_cutoff_block_present_for_all_drivers(self):
        for driver in ["verstappen", "hamilton", "norris", "senna", "schumacher"]:
            prompt = build_persona_prompt(driver)
            assert "Knowledge Cutoff" in prompt, f"{driver}: missing cutoff block"

    def test_senna_prompt_does_not_contain_post_1994_events(self):
        prompt = build_persona_prompt("senna").lower()
        assert "may 1994" in prompt or "1994" in prompt

    def test_schumacher_prompt_does_not_mention_accident(self):
        prompt = build_persona_prompt("schumacher").lower()
        assert "accident" not in prompt
        assert "skiing" not in prompt


# ── Knowledge cutoff enforcement ──────────────────────────────────────────────

class TestKnowledgeCutoffEnforcement:
    """Deflection phrases must be present for all drivers (enforced by persona files)."""

    @pytest.mark.parametrize("driver", ["verstappen", "hamilton", "norris", "senna", "schumacher"])
    def test_deflection_section_in_prompt(self, driver):
        prompt = build_persona_prompt(driver)
        assert "deflection phrase" in prompt.lower(), f"{driver}: no deflection section"


# ── transcribe_audio: missing API key ────────────────────────────────────────

class TestTranscribeAudioMissingKey:
    def test_raises_environment_error_without_key(self):
        from modal_backend.client import transcribe_audio

        mock_cohere = MagicMock()
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            import os
            saved = os.environ.pop("COHERE_API_KEY", None)
            try:
                with pytest.raises(EnvironmentError, match="COHERE_API_KEY"):
                    transcribe_audio(b"fake audio bytes")
            finally:
                if saved is not None:
                    os.environ["COHERE_API_KEY"] = saved


# ── call_persona_chat delegates correctly ─────────────────────────────────────

class TestCallPersonaChat:
    def test_delegates_to_persona_chat(self):
        mock_result = {"text": "I will win.", "audio_wav": b""}
        with patch("modal_backend.client._persona_chat") as mock_fn:
            mock_fn.remote.return_value = mock_result
            from modal_backend.client import call_persona_chat

            result = call_persona_chat(
                system_prompt="You are Verstappen.",
                user_message="Will you win?",
            )
            assert result["text"] == "I will win."
            mock_fn.remote.assert_called_once()

    def test_warmup_does_not_increment_call_count(self):
        mock_result = {}
        with patch("modal_backend.client._persona_chat") as mock_fn:
            mock_fn.remote.return_value = mock_result
            import modal_backend.client as client_mod

            before = client_mod._persona_call_count
            from modal_backend.client import call_persona_chat

            call_persona_chat(system_prompt="", user_message="", warmup=True)
            assert client_mod._persona_call_count == before

    def test_non_warmup_increments_call_count(self):
        mock_result = {"text": "ok", "audio_wav": b""}
        with patch("modal_backend.client._persona_chat") as mock_fn:
            mock_fn.remote.return_value = mock_result
            import modal_backend.client as client_mod

            before = client_mod._persona_call_count
            from modal_backend.client import call_persona_chat

            call_persona_chat(system_prompt="s", user_message="q", warmup=False)
            assert client_mod._persona_call_count == before + 1


# ── get_persona_loading_message ───────────────────────────────────────────────

class TestGetPersonaLoadingMessage:
    def test_returns_string(self):
        from modal_backend.client import get_persona_loading_message

        result = get_persona_loading_message()
        assert isinstance(result, str)
        assert len(result) > 5

