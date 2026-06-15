import io
import tempfile
from pathlib import Path

import gradio as gr
import yaml

from data.race_data import get_lap_window, get_race_window, _CACHE_DIR, _race_key
from modal_backend.client import (
    call_generate_commentary,
    call_persona_chat,
    call_reason_strategy,
    get_commentary_loading_message,
    get_persona_loading_message,
    get_strategy_loading_message,
    transcribe_audio,
)
from prompts.builder import build_commentary_prompt, build_persona_prompt

RACES_PATH = Path(__file__).parent / "data" / "curated_races.yaml"

_HISTORICAL_DRIVERS = {"senna", "schumacher"}
_ACTIVE_DRIVERS = {"verstappen", "hamilton", "norris"}
PERSONA_DRIVERS = ["Verstappen", "Hamilton", "Norris", "Senna", "Schumacher"]

F1_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
    --f1-bg: #0f0f0f;
    --f1-panel: #171717;
    --f1-panel-soft: #202020;
    --f1-text: #f4f4f4;
    --f1-muted: #a8a8a8;
    --f1-accent: #e8002d;
}

.gradio-container {
    background: var(--f1-bg) !important;
    color: var(--f1-text) !important;
}

#f1-shell {
    max-width: 1180px;
    margin: 0 auto;
}

#race-topbar {
    background: var(--f1-panel);
    border: 1px solid #2b2b2b;
    border-left: 4px solid var(--f1-accent);
    padding: 16px;
}

#race-topbar label,
#race-topbar span,
#race-topbar input,
#race-topbar button,
#race-topbar select {
    font-family: "JetBrains Mono", monospace !important;
}

.timing-data,
.stub-panel textarea,
.stub-panel input {
    font-family: "JetBrains Mono", monospace !important;
}

button.primary,
.selected,
[aria-selected="true"] {
    border-color: var(--f1-accent) !important;
}

.tabs {
    background: var(--f1-bg) !important;
}

.stub-panel {
    background: var(--f1-panel-soft);
    border: 1px solid #2b2b2b;
    padding: 16px;
}

#historical-notice {
    background: #1a1a1a;
    border: 1px dashed #444;
    border-radius: 4px;
    padding: 8px 12px;
    color: var(--f1-muted);
    font-family: "JetBrains Mono", monospace;
    font-size: 0.8rem;
}

.persona-chat-output {
    background: var(--f1-panel-soft);
    border: 1px solid #2b2b2b;
    border-radius: 4px;
    font-family: "JetBrains Mono", monospace;
    min-height: 120px;
}
"""


def load_curated_races() -> list[dict]:
    with open(RACES_PATH, encoding="utf-8") as race_file:
        return yaml.safe_load(race_file)["races"]


def race_label(race: dict) -> str:
    return f"{race['season']} {race['name']} - {race['circuit']}"


def race_choice_value(race: dict) -> str:
    return f"{race['season']}:{race['round']}"


def selected_race_from_value(selected_value: str, races: list[dict]) -> dict:
    for race in races:
        if race_choice_value(race) == selected_value:
            return race
    return races[0]


def _top_two_drivers(season: int, round_num: int, pivot_lap: int) -> tuple[str, str, str]:
    import pandas as pd

    key = _race_key(season, round_num)
    parquet_path = _CACHE_DIR / f"{key}_laps.parquet"
    laps = pd.read_parquet(parquet_path)

    at_pivot = laps[laps["lap_number"] == pivot_lap].sort_values("position")
    if len(at_pivot) < 2:
        last_lap = int(laps["lap_number"].max())
        at_pivot = laps[laps["lap_number"] == last_lap].sort_values("position")

    driver_a = at_pivot.iloc[0]["driver_code"]
    driver_b = at_pivot.iloc[1]["driver_code"]
    team_name = at_pivot.iloc[0]["team"]
    return driver_a, driver_b, team_name


def _generate_commentary(
    race: dict,
    pivot_lap: int,
    style: str,
) -> tuple[str | None, str]:
    season = race["season"]
    round_num = race["round"]

    driver_a, driver_b, team_name = _top_two_drivers(season, round_num, int(pivot_lap))
    lap_df = get_lap_window(season, round_num, int(pivot_lap), driver_a, driver_b)

    mode = "broadcast" if style == "Broadcast" else "radio"
    prompt = build_commentary_prompt(lap_df, team_name, mode)

    result = call_generate_commentary(prompt, style=mode)

    text = result.get("text", "")
    audio_bytes: bytes = result.get("audio_wav", b"")

    audio_path = None
    if audio_bytes:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.flush()
        audio_path = tmp.name

    return audio_path, text


def build_commentary_tab(race_state: gr.State) -> None:
    with gr.Tab("Commentary (TV)"):
        style_toggle = gr.Radio(
            choices=["Broadcast", "Radio"],
            value="Broadcast",
            label="Commentary style",
            interactive=True,
        )
        lap_slider = gr.Slider(
            minimum=1,
            maximum=80,
            value=20,
            step=1,
            label="Pivot lap",
            interactive=True,
        )
        generate_btn = gr.Button("Generate commentary", variant="primary")
        loading_status = gr.Textbox(
            label="Status",
            value="",
            interactive=False,
            visible=False,
        )
        commentary_audio = gr.Audio(
            label="Commentary audio",
            type="filepath",
            interactive=False,
        )
        commentary_text = gr.Textbox(
            label="Commentary text",
            interactive=False,
            lines=4,
            elem_classes="timing-data",
        )

        def on_generate(race, pivot_lap, style):
            loading = get_commentary_loading_message()
            yield gr.update(value=loading, visible=True), None, ""
            audio_path, text = _generate_commentary(race, pivot_lap, style)
            yield gr.update(value="Done.", visible=True), audio_path, text

        generate_btn.click(
            fn=on_generate,
            inputs=[race_state, lap_slider, style_toggle],
            outputs=[loading_status, commentary_audio, commentary_text],
        )


_PIVOT_LAP_MIN = 15
_PIVOT_LAP_MAX = 45


def _build_timing_table(race_window_df) -> str:
    if race_window_df.empty:
        return "No data available."
    uniq_laps = sorted(race_window_df["lap_number"].unique())
    mid_lap = uniq_laps[len(uniq_laps) // 2]
    snap = race_window_df[race_window_df["lap_number"] == mid_lap].copy().sort_values("position")
    lines = [f"{'LAP':<5} {'DRV':<6} {'POS':<5} {'GAP(s)':>8} {'CMPD':<10} {'AGE':>5}"]
    lines.append("-" * 44)
    for _, row in snap.iterrows():
        import math
        gap_val = row["gap_to_leader_s"]
        gap = "LEADER" if (math.isnan(gap_val) or gap_val == 0) else f"+{gap_val:.3f}"
        lines.append(
            f"{int(mid_lap):<5} {row['driver_code']:<6} {int(row['position']):<5} {gap:>8} {str(row['compound']):<10} {int(row['tyre_life']):>5}"
        )
    return "\n".join(lines)


def _build_strategy_prompt(race: dict, pivot_lap: int, scenario: str, timing_table: str) -> str:
    return (
        f"Race: {race['name']} ({race['season']}) at {race['circuit']}\n"
        f"Pivot lap: {pivot_lap}\n\n"
        f"### Race Snapshot\n\n{timing_table}\n\n"
        f"### What-If Scenario\n\n{scenario}\n\n"
        f"### Instructions\n\n"
        f"Reason through how this change affects pit windows, undercut/overcut risk, "
        f"tyre degradation, and track position. Narrate the alternate outcome with "
        f"specific lap numbers and position changes. Produce a plausible alternate final top-5."
    )


def _run_what_if(race: dict, pivot_lap: int, what_if_text: str):
    if not what_if_text or not what_if_text.strip():
        yield "Enter a what-if scenario first.", ""
        return

    if pivot_lap < _PIVOT_LAP_MIN or pivot_lap > _PIVOT_LAP_MAX:
        yield (
            f"Pivot lap {int(pivot_lap)} is outside the recommended range ({_PIVOT_LAP_MIN}–{_PIVOT_LAP_MAX}). "
            f"Adjust the lap slider and try again.",
            "",
        )
        return

    try:
        race_window = get_race_window(race["season"], race["round"], int(pivot_lap))
    except FileNotFoundError as exc:
        yield f"[Data not found: {exc}]", ""
        return

    timing_str = _build_timing_table(race_window)
    prompt = _build_strategy_prompt(race, pivot_lap, what_if_text.strip(), timing_str)

    yield "Connecting to the pit wall…", timing_str

    result = call_reason_strategy(prompt)
    reasoning = result.get("reasoning_chain", "")
    yield timing_str, reasoning


def build_what_if_tab(race_state: gr.State) -> None:
    with gr.Tab("What-If"):
        lap_slider = gr.Slider(
            minimum=1,
            maximum=80,
            value=30,
            step=1,
            label="Pivot lap  (works best for strategy changes in laps 15–45)",
            interactive=True,
        )
        lap_warning = gr.Markdown(value="", visible=False)
        whatif_input = gr.Textbox(
            label="Change one variable (e.g. 'Hamilton pits 5 laps earlier on fresh mediums')",
            placeholder="Describe your what-if scenario...",
            lines=2,
            interactive=True,
        )
        generate_btn = gr.Button("Generate", variant="primary")
        loading_status = gr.Textbox(
            label="Status",
            value="",
            interactive=False,
            visible=False,
        )
        with gr.Row():
            timing_table = gr.Textbox(
                label="Actual race snapshot",
                interactive=False,
                lines=15,
                elem_classes="timing-data",
                scale=1,
            )
            reasoning_output = gr.Textbox(
                label="Nemotron reasoning",
                interactive=False,
                lines=15,
                elem_classes="timing-data",
                scale=1,
            )

        def on_lap_change(pivot_lap):
            if pivot_lap < 15 or pivot_lap > 45:
                return gr.update(
                    value=f"> Warning: lap {int(pivot_lap)} is outside the recommended 15–45 window. Strategy reasoning may be less reliable.",
                    visible=True,
                )
            return gr.update(value="", visible=False)

        lap_slider.change(fn=on_lap_change, inputs=[lap_slider], outputs=[lap_warning])

        def on_generate(race, pivot_lap, what_if_text):
            loading = get_strategy_loading_message()
            first = True
            for left, right in _run_what_if(race, pivot_lap, what_if_text):
                status_val = loading if first else "Done."
                yield gr.update(value=status_val, visible=True), left, right
                first = False

        generate_btn.click(
            fn=on_generate,
            inputs=[race_state, lap_slider, whatif_input],
            outputs=[loading_status, timing_table, reasoning_output],
        )


def _race_context_string(race: dict) -> str:
    return (
        f"{race['season']} {race['name']} at {race['circuit']}. "
        f"Round {race['round']} of the season."
    )


def build_persona_chat_tab(race_state: gr.State) -> None:
    with gr.Tab("Persona Chat"):
        driver_selector = gr.Radio(
            choices=PERSONA_DRIVERS,
            value="Verstappen",
            label="Select driver",
            elem_id="driver-selector",
        )

        historical_notice = gr.Markdown(
            value="",
            elem_id="historical-notice",
            visible=False,
        )

        race_context_display = gr.Textbox(
            label="Race context (seeded into prompt for active drivers)",
            interactive=False,
            elem_id="persona-race-context",
            lines=1,
        )

        mic_input = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="Record your question",
            elem_id="persona-mic",
        )

        transcription_box = gr.Textbox(
            label="Transcription - edit before sending",
            placeholder="Record audio above or type directly...",
            lines=3,
            interactive=True,
            elem_id="persona-transcription",
        )

        with gr.Row():
            send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        chat_output = gr.Textbox(
            label="Driver reply",
            interactive=False,
            lines=5,
            elem_classes="persona-chat-output",
        )

        tts_output = gr.Audio(
            label="Voiced reply",
            type="numpy",
            interactive=False,
            autoplay=True,
            elem_id="persona-tts-output",
        )

        def on_driver_selected(driver, race):
            key = driver.lower()
            is_historical = key in _HISTORICAL_DRIVERS
            if is_historical:
                notice = (
                    "> **Historical drivers don't use race telemetry** - "
                    "Senna and Schumacher prompts are not seeded with current race data."
                )
                ctx_display = ""
            else:
                notice = ""
                ctx_display = _race_context_string(race) if race else ""
            return gr.update(value=notice, visible=is_historical), ctx_display

        driver_selector.change(
            fn=on_driver_selected,
            inputs=[driver_selector, race_state],
            outputs=[historical_notice, race_context_display],
        )

        def on_race_changed(race, driver):
            if driver.lower() in _HISTORICAL_DRIVERS:
                return ""
            return _race_context_string(race) if race else ""

        race_state.change(
            fn=on_race_changed,
            inputs=[race_state, driver_selector],
            outputs=[race_context_display],
        )

        def on_audio_recorded(audio_data):
            if audio_data is None:
                return ""
            import numpy as np
            import scipy.io.wavfile as wav_writer

            sample_rate, audio_array = audio_data
            if audio_array.dtype != np.int16:
                audio_array = (audio_array * 32767).clip(-32768, 32767).astype(np.int16)

            buf = io.BytesIO()
            wav_writer.write(buf, sample_rate, audio_array)
            try:
                return transcribe_audio(buf.getvalue())
            except Exception as exc:
                return f"[Transcription failed: {exc}]"

        mic_input.stop_recording(
            fn=on_audio_recorded,
            inputs=[mic_input],
            outputs=[transcription_box],
        )

        def on_send(driver, user_text, race):
            if not user_text or not user_text.strip():
                yield "Please record or type a question first.", None
                return

            key = driver.lower()
            ctx = _race_context_string(race) if key in _ACTIVE_DRIVERS and race else None

            try:
                system_prompt = build_persona_prompt(key, race_context=ctx)
            except FileNotFoundError as exc:
                yield f"[Persona error: {exc}]", None
                return

            yield get_persona_loading_message(), None

            try:
                result = call_persona_chat(
                    system_prompt=system_prompt,
                    user_message=user_text.strip(),
                )
            except Exception as exc:
                yield f"[Modal call failed: {exc}]", None
                return

            reply_text = result.get("text", "")
            audio_bytes = result.get("audio_wav", b"")

            audio_numpy = None
            if audio_bytes:
                import numpy as np
                import scipy.io.wavfile as wav_reader

                buf = io.BytesIO(audio_bytes)
                sample_rate, audio_array = wav_reader.read(buf)
                audio_numpy = (sample_rate, audio_array)

            yield reply_text, audio_numpy

        send_event = send_btn.click(
            fn=on_send,
            inputs=[driver_selector, transcription_box, race_state],
            outputs=[chat_output, tts_output],
        )

        clear_btn.click(
            fn=lambda: ("", None, ""),
            outputs=[transcription_box, tts_output, chat_output],
            cancels=[send_event],
        )


def build_app() -> gr.Blocks:
    races = load_curated_races()
    choices = [(race_label(race), race_choice_value(race)) for race in races]
    initial_value = race_choice_value(races[0])

    with gr.Blocks(css=F1_CSS, title="F1 Paddock Oracle") as app:
        with gr.Column(elem_id="f1-shell"):
            race_state = gr.State(races[0])

            with gr.Row(elem_id="race-topbar"):
                race_dropdown = gr.Dropdown(
                    label="15 hand-picked races",
                    choices=choices,
                    value=initial_value,
                    interactive=True,
                    scale=3,
                )
                lap_range = gr.Slider(
                    minimum=1,
                    maximum=80,
                    value=1,
                    step=1,
                    label="Lap range",
                    interactive=True,
                    scale=2,
                )

            race_dropdown.change(
                fn=lambda selected: selected_race_from_value(selected, races),
                inputs=race_dropdown,
                outputs=race_state,
            )

            with gr.Tabs():
                build_commentary_tab(race_state)
                build_what_if_tab(race_state)
                build_persona_chat_tab(race_state)

    return app


if __name__ == "__main__":
    build_app().launch()
