# asr-overlay

An overlay for **OBS Browser Source** that listens to the microphone in real time and generates live subtitles. Everything runs locally, CPU-only.

The long-term goal is a full voice assistant overlay: **mic → Whisper STT → (optional) Audio8 TTS → subtitles + spoken playback**, all with no cloud services.

## Current state

- [x] HTTP server (axum) serving an overlay page for OBS
- [x] Push-to-talk / live microphone capture (CPAL)
- [x] Local transcription in Spanish (whisper-rs / whisper.cpp, CPU)
- [x] Subtitles rendered and synced on the overlay page
- [ ] VAD (continuous listening, no push-to-talk)
- [ ] Audio8 TTS so the overlay can speak back
- [ ] Voice registration from the microphone (Audio8 encoder)

See [ROADMAP.md](./ROADMAP.md) for the full implementation plan.

## Architecture

```text
                    ┌───────────────┐
Micrófono ─────────▶│ CPAL          │
                    │ PCM f32       │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Resample      │
                    │ mono f32      │
                    │ → 16 kHz      │
                    │ (rubato)      │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Whisper STT   │
                    │ whisper-rs    │
                    │ CPU · es      │
                    └───────┬───────┘
                            │ texto
                            ▼
                    ┌───────────────┐
                    │ axum server   │
                    │ subtítulos    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ OBS Browser   │
                    │ Source        │
                    └───────────────┘
```

Pipeline modules:

- **Mic** — `cpal` input stream; the audio callback only copies samples into a channel. Inference never runs in the callback.
- **Resample** — `rubato` converts to mono `f32` @ 16 kHz for Whisper.
- **STT** — `whisper-rs` transcribes to Spanish text locally on CPU.
- **Server** — axum serves the overlay page and pushes caption updates to OBS.
- **Overlay** — a plain HTML/CSS/JS page rendered in OBS showing the live caption text.

## Commands

| Command | Description |
|---------|-------------|
| `cargo run` | Start the caption server |
| `cargo check` | Type check |
| `cargo clippy -- -D warnings` | Lint Rust codebase |
| `cargo test` | Run unit and integration tests |
| `cargo build --release` | Build production binary |

## HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Overlay page (OBS Browser Source) |
| GET | `/ws` | WebSocket caption stream + control commands |
| GET | `/status` | `{ running, listening, has_text }` |
| POST | `/start` | Start microphone capture / listening |
| POST | `/stop` | Stop listening |
| GET | `/settings` | Get current config JSON |
| POST | `/settings` | Update config JSON |

## OBS Integration

1. Run the app (`cargo run`)
2. In OBS, add a **Browser Source**
3. Set URL to `http://localhost:8080/`
4. Set width/height to match your stream layout
5. Check "Shutdown source when not visible" for optimal performance

## Requirements

- Rust toolchain (1.70+)
- `ggml-small.bin` (or `ggml-base.bin`) from whisper.cpp placed in `models/whisper/`
- Linux audio deps via ALSA/PulseAudio: `libasound2-dev pkg-config`