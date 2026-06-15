"""Unit tests for the persona module (Issue #6)."""

import pytest
from pathlib import Path

# Allow imports from the project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prompts.persona import get_system_prompt, _driver_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_driver_file(name: str) -> str:
    path = _driver_file(name)
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Knowledge cutoff assertions
# ---------------------------------------------------------------------------

class TestCutoffYear:
    def test_verstappen_cutoff_present(self):
        text = _read_driver_file("verstappen")
        assert "end of 2023" in text.lower() or "2023" in text

    def test_hamilton_cutoff_present(self):
        text = _read_driver_file("hamilton")
        assert "end of 2023" in text.lower() or "2023" in text

    def test_norris_cutoff_present(self):
        text = _read_driver_file("norris")
        assert "end of 2023" in text.lower() or "2023" in text

    def test_schumacher_cutoff_is_2012(self):
        text = _read_driver_file("schumacher")
        assert "2012" in text
        assert "end of 2012" in text.lower()

    def test_senna_cutoff_is_may_1994(self):
        text = _read_driver_file("senna")
        assert "1994" in text
        assert "may 1994" in text.lower()


# ---------------------------------------------------------------------------
# Schumacher content guard — no forbidden words/years
# ---------------------------------------------------------------------------

class TestSchumacherContentGuard:
    def setup_method(self):
        self.text = _read_driver_file("schumacher").lower()

    def test_no_word_accident(self):
        assert "accident" not in self.text

    def test_no_word_skiing(self):
        assert "skiing" not in self.text

    def test_no_year_2013(self):
        assert "2013" not in self.text

    def test_no_years_after_2012(self):
        for year in range(2013, 2030):
            assert str(year) not in self.text, f"Forbidden year {year} found in schumacher.txt"


# ---------------------------------------------------------------------------
# Deflection phrases — non-empty and count
# ---------------------------------------------------------------------------

class TestDeflectionPhrases:
    @pytest.mark.parametrize("driver", ["verstappen", "hamilton", "norris", "senna", "schumacher"])
    def test_deflection_section_present(self, driver):
        text = _read_driver_file(driver)
        assert "deflection phrase" in text.lower(), f"{driver}: missing deflection phrases section"

    @pytest.mark.parametrize("driver", ["verstappen", "hamilton", "norris", "senna", "schumacher"])
    def test_at_least_five_deflection_phrases(self, driver):
        text = _read_driver_file(driver)
        # Deflection phrases are numbered list items 1-5
        count = sum(1 for line in text.splitlines() if line.strip() and line.strip()[0].isdigit() and ". " in line[:4])
        assert count >= 5, f"{driver}: expected at least 5 numbered deflection phrases, found {count}"

    @pytest.mark.parametrize("driver", ["verstappen", "hamilton", "norris", "senna", "schumacher"])
    def test_deflection_phrases_non_empty(self, driver):
        text = _read_driver_file(driver)
        lines = [l.strip() for l in text.splitlines() if l.strip() and l.strip()[0].isdigit() and ". " in l.strip()[:4]]
        for line in lines:
            # Strip leading "1. " etc. and check there's real content
            content = line.split(". ", 1)[-1].strip()
            assert len(content) > 10, f"{driver}: deflection phrase too short: {repr(line)}"


# ---------------------------------------------------------------------------
# Personality distinctiveness — spot-check key register words
# ---------------------------------------------------------------------------

class TestPersonalityRegister:
    def test_verstappen_blunt_register(self):
        text = _read_driver_file("verstappen").lower()
        assert "blunt" in text or "direct" in text

    def test_hamilton_diplomatic_register(self):
        text = _read_driver_file("hamilton").lower()
        assert "diplomatic" in text or "articulate" in text or "considered" in text

    def test_norris_self_deprecating_register(self):
        text = _read_driver_file("norris").lower()
        assert "self-deprecating" in text or "self deprecating" in text


# ---------------------------------------------------------------------------
# Persona module: get_system_prompt
# ---------------------------------------------------------------------------

class TestGetSystemPrompt:
    def test_returns_string_for_all_drivers(self):
        for driver in ["verstappen", "hamilton", "norris", "senna", "schumacher"]:
            result = get_system_prompt(driver)
            assert isinstance(result, str)
            assert len(result) > 100

    def test_case_insensitive_driver_name(self):
        lower = get_system_prompt("verstappen")
        upper = get_system_prompt("Verstappen")
        mixed = get_system_prompt("VERSTAPPEN")
        assert lower == upper == mixed

    def test_missing_driver_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            get_system_prompt("alonso")
        assert "alonso" in str(exc_info.value).lower()
        assert "available drivers" in str(exc_info.value).lower()

    def test_missing_driver_error_is_descriptive(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            get_system_prompt("grosjean")
        msg = str(exc_info.value)
        # Must name the bad driver and hint at valid ones
        assert "grosjean" in msg.lower()
        assert any(d in msg for d in ["verstappen", "hamilton", "norris"])


# ---------------------------------------------------------------------------
# Race context injection
# ---------------------------------------------------------------------------

class TestRaceContextInjection:
    RACE = "2023 Dutch Grand Prix, Lap 42 of 72. Verstappen leads by 4.8s on hard tires, age 31."

    def test_race_context_injected_for_verstappen(self):
        result = get_system_prompt("verstappen", race_context=self.RACE)
        assert self.RACE in result
        assert "Current Race Context" in result

    def test_race_context_injected_for_hamilton(self):
        result = get_system_prompt("hamilton", race_context=self.RACE)
        assert self.RACE in result

    def test_race_context_injected_for_norris(self):
        result = get_system_prompt("norris", race_context=self.RACE)
        assert self.RACE in result

    def test_race_context_silently_absent_for_senna(self):
        result = get_system_prompt("senna", race_context=self.RACE)
        # No error raised, race context not injected
        assert self.RACE not in result
        assert "Current Race Context" not in result

    def test_race_context_silently_absent_for_schumacher(self):
        result = get_system_prompt("schumacher", race_context=self.RACE)
        assert self.RACE not in result
        assert "Current Race Context" not in result

    def test_none_race_context_active_driver_no_injection(self):
        result = get_system_prompt("verstappen", race_context=None)
        assert "Current Race Context" not in result

    def test_empty_race_context_treated_as_absent(self):
        result = get_system_prompt("verstappen", race_context="")
        assert "Current Race Context" not in result
