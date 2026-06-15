"""Smoke tests for the Modal client (Issue #8).

MANUAL-ONLY — not run in CI.

Run with:
    python -m pytest tests/smoke/test_modal_client.py -v

WARNING: Each invocation hits live Modal containers and consumes Modal credits.
Do not add this to CI or run repeatedly without need.
"""

import time

from modal_backend.client import call_generate_commentary, call_reason_strategy

COMMENTARY_PROMPT = (
    "LAP 47 of 57 at Monaco. Verstappen leads Hamilton by 6.2 seconds. "
    "Hamilton is on worn mediums, tyre age 28 laps. "
    "Generate a 2-sentence broadcast commentary update."
)

STRATEGY_PROMPT = (
    "Race: 2023 British Grand Prix\n"
    "Original outcome: Verstappen won from pole, Hamilton finished P4 after a late pit.\n"
    "User change: What if Hamilton had pitted 5 laps earlier on lap 30 for fresh mediums?"
)


def test_container1_reachable_and_non_empty():
    """Container 1 (MiniCPM-o) returns a non-empty text response."""
    result = call_generate_commentary(prompt=COMMENTARY_PROMPT)
    assert isinstance(result, dict), "Expected dict response from Container 1"
    assert result.get("text"), "Container 1 returned empty text"


def test_container2_reachable_and_non_empty():
    """Container 2 (Nemotron Nano) returns a non-empty reasoning chain."""
    result = call_reason_strategy(prompt=STRATEGY_PROMPT)
    assert isinstance(result, dict), "Expected dict response from Container 2"
    assert result.get("reasoning_chain"), "Container 2 returned empty reasoning_chain"


def test_warmup_returns_under_2_seconds():
    """Warmup flag (both containers already warm) completes in under 2 seconds."""
    start = time.monotonic()
    call_generate_commentary(prompt="", warmup=True)
    call_reason_strategy(prompt="", warmup=True)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"Warmup took {elapsed:.2f}s — containers may be cold"
