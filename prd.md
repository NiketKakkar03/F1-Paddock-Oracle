# PRD: F1 Paddock Oracle — Full Build Specification

## MVP vs Stretch Goals

Everything in this PRD is intended for implementation. This table clarifies sequencing — MVP items are the minimum for a working submission; stretch items elevate the demo and badge coverage.

### MVP (must ship)

| Area | Item |
|------|------|
| Infrastructure | Two Modal containers (MiniCPM-o + Nemotron Nano), Cohere hosted ASR |
| Infrastructure | Pre-cached Parquet for all 15 curated races via `curated_races.yaml` |
| Infrastructure | Two-tier loading copy + keep-warm scheduled function for judging window |
| Mode 1 | Broadcast commentary generation with TTS audio output |
| Mode 1 | Team Radio commentary toggle (separate prompt template) |
| Mode 2 | Strategy What-If split-panel with grounded 10-lap FastF1 context |
| Mode 2 | Pivot lap gate (laps 15–45) with UI note |
| Mode 3 | All five driver personas with knowledge cutoff blocks and deflection phrases |
| Mode 3 | Full voice I/O — Cohere Transcribe input + MiniCPM-o TTS output |
| Mode 3 | Confirm-before-send transcription display |
| UI | Shared `gr.State` race selector across all three tabs |
| UI | Historical driver greyed-out tooltip for race context bar |
| UI | Custom F1 dark CSS theme (`#0f0f0f`, `#e8002d`, JetBrains Mono) |
| Tests | Unit tests for Race Data Module, Prompt Builder, Persona Module |
| Tests | Manual smoke test script for both Modal containers |

### Stretch Goals (ship today if time allows)

| Area | Item | Badge Impact |
|------|------|-------------|
| Mode 1 | Compare button — parallel Broadcast + Radio side-by-side render | Best Demo |
| Mode 2 | TTS narration of strategy output via second MiniCPM-o call | Best Demo |
| Fine-tuning | Fine-tune Nemotron Nano on F1 strategy transcripts using Modal training jobs | NVIDIA + Best Use of Modal |
| Fine-tuning | Fine-tune MiniCPM5-1B on driver interview transcripts for sharper persona voice | Tiny Titan + Best Demo |
| UI | "Compare" screenshot-ready two-column layout for social post | Best Demo |
| Demo | Subtitle the 90s demo video and post to X with #BuildSmall | Best Demo |

### Fine-Tuning Note
Modal is the platform for running fine-tuning jobs (leveraging the $250 credits). Codex handles the implementation code for the training scripts. Fine-tuning Nemotron Nano on F1 strategy data directly strengthens the NVIDIA Nemotron hardware prize case while also demonstrating Modal beyond inference-only use. Fine-tuning MiniCPM5-1B on driver interview transcripts sharpens persona authenticity and supports the Tiny Titan badge (≤4B active parameters, fine-tuned on domain data).

---

## Problem Statement

Formula 1 is one of the most data-rich sports in the world, but that data is locked behind technical barriers — lap timing CSVs, telemetry APIs, and dense strategy analysis that requires deep domain knowledge to interpret. Casual fans and engaged enthusiasts alike have no accessible, interactive way to experience what that data *feels* like: the drama of a race unfolding lap by lap, the strategic tension of a pit window, or the personality of a driver they admire. Existing F1 apps are either pure data dashboards (for engineers) or passive video content (for viewers). Nothing puts the fan *inside* the race.

## Solution

F1 Paddock Oracle is a voice-enabled, AI-powered Gradio app that transforms historical F1 race data into three interactive entertainment experiences — all powered by small open-source models (≤32B parameters each) running on Modal serverless GPU infrastructure:

- **Race Commentary** (Mode 1): Select a race and generate lap-by-lap commentary in either broadcast or team radio style, with audio playback via MiniCPM-o's native TTS.
- **Strategy What-If Theatre** (Mode 2): Pick a real race, change one strategic variable, and watch Nemotron Nano reason through the alternate timeline using grounded FastF1 lap data — displayed in a split panel against the actual outcome.
- **Driver Persona Chat** (Mode 3): Converse with AI simulations of five iconic F1 drivers via voice or text, each with era-appropriate knowledge boundaries and pre-written deflection phrases for out-of-scope questions.

A shared race selector persists across all three modes as global state, enabling a natural fan journey: pick a race, watch the commentary, rewrite history, then ask the driver about it.

## User Stories

### Global / Cross-Mode
1. As an F1 fan, I want to select a race from a curated dropdown of 15 hand-picked races, so that I can immediately explore that race across all three app modes without waiting for live data fetching.
2. As an F1 fan, I want the selected race to persist as I switch between Commentary, What-If, and Persona Chat tabs, so that I can explore the same race from multiple angles without re-selecting it each time.
3. As a first-time user, I want to see a loading message like "Connecting to the pit wall… (~20s first call)" when the app is warming up, so that I understand the delay is expected and don't assume the app is broken.
4. As a returning user, I want subsequent inference calls to show a shorter "On it — back in a few seconds" message, so that I can see the system is now warm and responsive.
5. As a user on any device, I want the microphone permission dialog to only appear when I navigate to the Persona Chat tab and click the mic button, so that I am not interrupted by a browser permission prompt on page load.

### Mode 1 — Race Commentary
6. As an F1 fan, I want to generate lap-by-lap broadcast commentary for a selected race, so that I can relive the drama of the race in a narrative, emotional register.
7. As an F1 fan, I want to switch to Team Radio style commentary for the same race data, so that I can hear the same lap sequence as a terse, tactical race engineer communication.
8. As a demo viewer, I want to see Broadcast and Team Radio outputs side-by-side via a Compare button, so that I can immediately appreciate the contrast between the two registers from one inference.
9. As an F1 fan, I want to select a lap range for the commentary, so that I can focus on a specific stint or dramatic sequence rather than the full race.
10. As a user, I want the commentary audio to play back after generation, so that I can hear MiniCPM-o TTS output without manually clicking play.
11. As a user, I want the commentary text to also render as readable output alongside the audio, so that I can read or listen depending on my preference.

### Mode 2 — Strategy What-If Theatre
12. As an F1 strategist fan, I want to specify a pivot lap and a free-text what-if variable for any curated race, so that I can explore alternate strategic outcomes grounded in real race data.
13. As an F1 fan, I want to see the actual race outcome displayed on the left panel of a split layout, so that I can compare the real result against the AI-generated alternate timeline side by side.
14. As an F1 fan, I want the alternate timeline reasoning steps displayed on the right panel, so that I can follow Nemotron Nano's strategic logic lap by lap.
15. As a credibility-conscious user, I want the alternate timeline reasoning to reference the actual lap times, tire compounds, and gap data from the real race, so that the what-if scenario does not contradict verifiable facts.
16. As a user, I want the pivot lap input to be gated with a note explaining it works best for laps 15–45, so that I do not enter a whole-race what-if that exceeds the system's reliable reasoning scope.
17. As an F1 fan, I want the actual outcome on the left panel to show position, driver, and gap data in a timing-tower table format, so that the real race result is legible at a glance.
18. As a user, I want Strategy What-If output to be text-only with no audio, so that the reading experience of examining the alternate timeline is deliberate and focused.

### Mode 3 — Driver Persona Chat
19. As an F1 fan, I want to select from five driver personas (Verstappen, Hamilton, Norris, Senna, Schumacher) and chat with them in character, so that I can experience an immersive conversation with a driver I admire.
20. As a user, I want to speak my question via microphone and have it transcribed by Cohere Transcribe, so that I can interact with the driver persona in a natural conversational way.
21. As a user, I want to see my transcribed speech displayed in a text box before I send it, so that I can catch ASR errors on F1 terminology before they reach the model.
22. As a user, I want to receive a voiced reply from the driver persona via MiniCPM-o TTS, so that the conversation feels immersive and character-driven.
23. As an F1 fan talking to Verstappen, I want him to deflect questions about events after 2023 with a blunt, in-character response, so that the persona feels consistent and does not hallucinate recent race results.
24. As an F1 fan talking to Hamilton, I want him to deflect post-2023 questions diplomatically and redirect to something he knows, so that the persona feels authentic to his known communication style.
25. As an F1 fan talking to Norris, I want him to deflect post-2023 questions with self-deprecating humor, so that the persona captures his known personality.
26. As an F1 fan talking to Senna, I want the persona to only reference events up to May 1994, so that the conversation is grounded in his actual career without anachronisms.
27. As an F1 fan talking to Schumacher, I want the persona to only reference his career up to the end of 2012, with no references to post-retirement events, so that the conversation respects personal boundaries and stays in-era.
28. As a user, I want the driver persona to be optionally seeded with context about the currently selected race (if applicable to the driver's era), so that I can ask the driver about a specific race I have been exploring in other modes.
29. As a user selecting a historical driver (Senna or Schumacher), I want the race context selector to visually grey out with a tooltip explaining that historical drivers do not use race telemetry, so that I understand the architecture without reading documentation.
30. As a user, I want each driver to have a distinct visual card showing their number and team livery colors, so that the persona selection feels characterful and themed.

### Data and Infrastructure
31. As a developer, I want race data to be pre-cached as Parquet files on HF Datasets from a curated list defined in a single YAML config, so that there is no live FastF1 fetching during demo or judging.
32. As a developer, I want the curated race list to be the single source of truth for both data ingestion and the UI race dropdown, so that adding or removing a race requires changing only one file.
33. As a developer, I want the data ingestion scripts to be separate from the running app and not included in the app's dependency graph, so that the Gradio Space does not need FastF1 installed at runtime.
34. As a developer running the keep-warm script before judging, I want Modal containers to respond to warmup pings that skip model inference, so that I can hold containers warm at near-zero GPU cost.
35. As a developer, I want the keep-warm scheduler to be a manually-activated Modal scheduled function rather than a persistent background service, so that it does not silently drain credits after the judging window closes.

## Implementation Decisions

### Architecture
- Two Modal serverless GPU containers: Container 1 runs MiniCPM-o 4.5 (8B) for commentary text+TTS and persona chat; Container 2 runs Nemotron 3 Nano (3B active MoE) for strategy reasoning only. Cohere Transcribe is called as a hosted API, not a self-hosted container.
- The Gradio Space runs CPU-only on HF Free tier. All heavy inference is delegated to Modal via `.remote()` calls.
- Pre-cached Parquet files on HF Datasets serve all race data. No live FastF1 fetching at runtime.

### Race Data Module
- Accepts a race identifier (season + round) and a pivot lap number; returns a structured 10-lap window (±5 laps from pivot) for the two relevant drivers only.
- Window truncates cleanly at race start (lap 1) and race end.
- Output is a formatted data table (not prose) with columns: lap number, driver code, position, gap to leader, compound, tyre age, lap time.
- Missing or corrupted Parquet files raise a descriptive error, not a silent empty DataFrame.
- Safety car flag is derived from session events and included as a boolean column.
- `curated_races.yaml` is the authoritative source for valid race identifiers.

### Prompt Builder Module
- Stateless function: takes structured race data + mode + parameters, returns a prompt string.
- Commentary mode: accepts style parameter (`broadcast` or `radio`) and selects the corresponding prompt template file. The two templates have structurally different rules, not just tone variations.
- Strategy mode: interpolates the 10-lap structured table, the what-if variable, and an explicit anti-hallucination instruction ("Do not invent lap time values").
- Persona mode: loads the driver persona file and appends the knowledge cutoff block. Optionally injects race context if provided and applicable to the driver's era.
- Prompt length is validated against the target model's context window before dispatch.

### Persona Module
- Each driver has a dedicated persona file containing: career overview, personality traits, famous quotes, response style, knowledge cutoff block, and pre-written deflection phrases.
- Knowledge cutoffs: Verstappen/Hamilton/Norris → end of 2023 season; Senna → May 1994; Schumacher → end of 2012.
- Schumacher's persona file contains zero references to the 2013 skiing accident or any post-2012 events.
- Deflection phrases (5 per active driver) are pre-written in the persona file and used by the prompt builder rather than generated by the model.
- Race context injection is silently ignored for historical drivers (Senna, Schumacher) without raising an error.

### Modal Client Module
- Thin wrapper around Modal `.remote()` calls for both containers.
- Accepts a `warmup: bool` flag; when `True`, returns immediately after model load without running inference (used by keep-warm scheduler).
- Keep-warm scheduler is a separate Modal scheduled function, deployed manually before the judging window and stopped manually after.

### UI Layout
- Global race selector (`gr.State`) in a persistent top bar above all tabs. Single linked Season+Race dropdown populated from `curated_races.yaml`. Lap range slider as second global input.
- Three tabs: Commentary (📺), What-If (🔀), Persona Chat (🎙️). Each tab's `build_*` function receives the shared race state.
- Commentary tab: lap range selector, Broadcast/Radio toggle (`gr.Radio`), Generate button, optional Compare button as stretch goal (parallel calls to same container).
- What-If tab: pivot lap number input (gated to laps 15–45 for MVP), free-text what-if variable, Generate button. Split-panel output: actual outcome left, reasoning right.
- Persona Chat tab: driver selector with livery-colored cards, `gr.Audio(sources=["microphone"])` scoped to this tab only, transcription display textbox, confirm-before-send flow, chat output with TTS audio.
- Historical driver selected → race context bar greys out with tooltip: "Historical drivers don't use race telemetry."
- Two-tier loading copy: first call shows "🔧 Connecting to the pit wall… (~20s first call)"; subsequent calls show "⚙️ On it — back in a few seconds."

### Voice I/O Scope
- Commentary: TTS output only (MiniCPM-o). No microphone input.
- What-If: Text-only in both directions.
- Persona Chat: Full voice I/O — Cohere Transcribe for input, MiniCPM-o TTS for output.

### Curated Race List
- 15 races across 2021–2025 defined in `data/curated_races.yaml`.
- Each entry tagged with applicable modes (`commentary`, `what_if`, `demo_anchor`).
- The README advertises "15 hand-picked races" — the "2018–2025" season range claim is dropped entirely.

## Testing Decisions

### What Makes a Good Test
Tests cover external behavior, not implementation details. A good test asserts what comes *out* of a module given a specific *input* — not how the module internally achieves that output. Tests should catch the silent failures that produce wrong-but-plausible outputs (a bad lap window, a malformed prompt, a wrong cutoff year) rather than exceptions that would surface anyway.

### Unit Tests — Race Data Module
- Correct 10-lap window extracted around a mid-race pivot lap
- Window truncates correctly at race start (pivot lap 3 → laps 1–8 only)
- Window truncates correctly at race end (pivot lap 50 in a 52-lap race → laps 45–52)
- Only the two specified drivers are extracted, not all 20
- SC flag correctly derived from session events and present as boolean
- Missing Parquet file raises a descriptive error (not silent empty DataFrame)
- Corrupted Parquet raises a descriptive error

### Unit Tests — Prompt Builder Module
- All template variables correctly interpolated (race name, pivot lap, driver codes, lap times, compounds, tyre ages)
- Knowledge cutoff block appended for active driver prompts
- Knowledge cutoff block not appended for historical driver prompts (they have natural cutoffs)
- Broadcast and Radio templates produce structurally different output (not just different preamble)
- Anti-hallucination instruction present in strategy prompt output
- Prompt length stays within context window for maximum 10-lap × 2-driver input

### Unit Tests — Persona Module
- Correct cutoff year applied per driver (Verstappen/Hamilton/Norris: 2023; Senna: 1994; Schumacher: 2012)
- Schumacher persona produces a prompt with zero occurrences of "accident", "skiing", "2013", or any year after 2012
- All five drivers have non-empty deflection phrase lists
- Race context correctly injected when race is provided and driver era is compatible
- Race context silently absent when historical driver selected (no error raised)
- Persona file missing raises a descriptive error

### Smoke Tests — Modal Client (manual, pre-judging only)
- Commentary container reachable and returns non-empty text output
- Strategy container reachable and returns non-empty reasoning chain
- Warmup flag causes both containers to return in under 2 seconds (no inference performed)
- Run manually the morning judging opens; not included in CI (costs Modal credits, requires live network)

### Not Tested
- Gradio UI layer — verified by manually running the app and clicking through all three modes
- Keep-warm scheduler — verified by checking Modal dashboard for scheduled run execution
- Data ingestion scripts — run once offline; validated by inspecting uploaded Parquet files on HF Datasets

## Out of Scope

- **TTS narration for Strategy What-If** — Nemotron's text output is not piped to MiniCPM-o for audio. Flagged as v2 in the README roadmap.
- **Voice input for Commentary and What-If modes** — voice input is scoped to Persona Chat only.
- **Live race data** — no real-time FastF1 fetching during app runtime. All data is pre-cached.
- **Races outside the curated 15** — the "2018–2025 full season" claim is dropped. Uncached races are not supported.
- **Pivot laps outside the 15–45 range** — whole-race what-ifs are gated in the UI.
- **Additional driver personas** — the five personas (Verstappen, Hamilton, Norris, Senna, Schumacher) are the full MVP set.
- **Compare button parallel rendering** — listed as a stretch goal for the demo video; not required for submission.
- **Automated CI for smoke tests** — smoke tests are manual-only to avoid unintentional Modal credit spend.
- **Ergast API fallback** — dropped in favor of pre-cached Parquet only; no live API calls at runtime.

## Further Notes

### Hackathon Badge Targeting
- **Best Demo** — Toggle contrast moment (Broadcast vs. Radio) and Persona voice exchange are the two showcase sequences for the demo video. Save the Senna voice moment for the final 30 seconds.
- **Tiny Titan** — MiniCPM-o 4.5 at 8B satisfies the parameter cap; confirm active parameter count against the model card before submission.
- **Off Brand** — Custom F1 dark CSS theme (`#0f0f0f` background, `#e8002d` accent, JetBrains Mono for timing data) must be visually striking enough to distinguish from stock Gradio.
- **Best Use of Modal** — Two-container architecture, warmup flag design, and keep-warm scheduler should all be called out explicitly in the README.
- **NVIDIA Nemotron** — Nemotron 3 Nano usage and the strategy reasoning role should be noted in the README for hardware prize eligibility.

### Pre-Judging Operations Checklist
1. Deploy keep-warm scheduled function 1 hour before judging window opens
2. Run smoke tests manually against both containers
3. Confirm loading copy renders on first call
4. Set calendar reminder to stop keep-warm after judging closes
5. Monitor Modal credit spend (~$0.10/day expected during window)

### Historical Driver Persona Design Note
Senna and Schumacher personas are grounded entirely in injected career knowledge — no FastF1 telemetry is available for pre-2018 races. This is stated explicitly in the README as an architectural fact, not a limitation.
