# F1 Paddock Oracle

> *An AI race strategist, commentator, and what-if machine — all under 32B parameters.*

An AI-powered Gradio app that turns Formula 1 data into generative entertainment: live-style race commentary, alternate strategy timelines, and driver persona conversations.

---

## Modes

| Mode | Description |
|------|-------------|
| 📺 Race Commentary | Generate lap-by-lap commentary in Broadcast or Team Radio style, with TTS audio output |
| 🔀 Strategy What-If | Change one variable from a real race and watch the alternate timeline reason out — grounded in real lap data |
| 🎙️ Driver Persona Chat | Converse with Verstappen, Hamilton, Norris, Senna, or Schumacher via voice or text |

---

## Races

The app includes **15 hand-picked races** across the 2021–2025 seasons, pre-cached for instant use during demos and judging. Each race was selected for the richness of its strategic or narrative content.

Historical driver personas (Senna, Schumacher) are grounded in injected career knowledge — FastF1 telemetry covers 2018 onwards only. This is an architectural fact, not a limitation.

---

## Models

| Container | Model | Role |
|-----------|-------|------|
| Modal Container 1 | MiniCPM-o 4.5 (8B) | Commentary text + TTS, Persona chat |
| Modal Container 2 | Nemotron 3 Nano (3B active MoE) | Strategy what-if reasoning |
| Cohere hosted API | Cohere Transcribe | Voice input (Persona Chat) |

All models individually satisfy the ≤32B parameter cap.

---

## Stack

- **Frontend:** Gradio 5 on Hugging Face Space (CPU-only)
- **Inference:** Modal serverless GPU functions
- **Data:** Pre-cached Parquet files on HF Datasets (`f1-race-logs/`)
- **Voice in:** Cohere Transcribe API
- **Voice out:** MiniCPM-o native TTS

---

## Development Setup

```bash
pip install -r requirements-dev.txt   # includes fastf1, for data ingestion only
python data/fetch_races.py            # fetch and cache all 15 races
HF_TOKEN=hf_... python data/push_to_hub.py --repo your-username/f1-race-logs
```

FastF1 is a **dev-only dependency** — it is not installed in the Gradio Space runtime.

---

## Hackathon

Built for the [Hugging Face Build Small](https://huggingface.co/build-small) hackathon — Thousand Token Wood (Whimsical / Creative) track.

Targeting: Best Demo · Tiny Titan · Off Brand · Best Use of Modal · NVIDIA Nemotron
