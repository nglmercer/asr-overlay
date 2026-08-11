# asr-overlay

An overlay for **OBS Browser Source** that listens to the microphone in real time
and generates live subtitles. Everything runs locally, CPU-only.

The long-term goal is a full voice assistant overlay:
**mic → Whisper STT → (optional) Audio8 TTS → subtitles + spoken playback**,
with no cloud services, no Python runtime and no GPU.

## Quick start

```bash
./scripts/download-models.sh whisper small   # ~466 MiB, required
cargo run
```

Then point an OBS Browser Source at `http://localhost:8080/`.

Without a model file the app starts, serves the overlay and shows the tray icon,
but the pipeline thread exits immediately with
`failed to open 'models/whisper/ggml-small.bin'`. Run the download script.

## Current state

- [x] HTTP server (axum) serving an overlay page for OBS
- [x] Live microphone capture (CPAL) with start/stop from tray, HTTP and WebSocket
- [x] RMS-energy VAD — continuous listening, no push-to-talk
- [x] Local multilingual transcription (whisper-rs / whisper.cpp, CPU)
- [x] Subtitles pushed over WebSocket + rendered as a transparent PNG (cairo)
- [ ] Audio8 TTS so the overlay can speak back — scaffolding only, see below
- [ ] Voice registration from the microphone (Audio8 codec encoder)

`src/tts/` currently contains the manifest parser, the sampler and typed stubs.
Every ONNX session wrapper returns `not yet implemented`; no synthesis happens
yet. See [ROADMAP.md](./ROADMAP.md).

## Languages

Whisper is multilingual. Set the language with `--language <code>`, or
`--language auto` to detect it per utterance:

```bash
cargo run -- --language auto
cargo run -- --language en
```

The default is `es`. Audio8 TTS covers 11 languages — Cantonese, Chinese, Dutch,
English, French, German, Italian, Japanese, Korean, Polish and Spanish — so those
11 are the set where STT and TTS will both work end to end.

## Architecture

```text
                    ┌───────────────┐
Microphone ────────▶│ CPAL          │
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
                    │ VAD (RMS)     │
                    │ utterance     │
                    │ segmentation  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Whisper STT   │
                    │ whisper-rs    │
                    │ CPU           │
                    └───────┬───────┘
                            │ text
                            ▼
                    ┌───────────────┐
                    │ axum server   │
                    │ captions      │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ OBS Browser   │
                    │ Source        │
                    └───────────────┘
```

Pipeline modules:

- **Mic** (`audio.rs`) — `cpal` input stream; the audio callback only copies
  samples into a channel. Inference never runs in the callback.
- **Resample** (`audio.rs`) — `rubato` converts to mono `f32` @ 16 kHz.
- **VAD + STT** (`caption.rs`, `stt.rs`) — RMS gate splits utterances, then
  `whisper-rs` transcribes locally on CPU.
- **TTS** (`tts/`) — Audio8 DualAR ONNX INT4. Not wired up yet.
- **Server** (`server.rs`) — axum serves the overlay and pushes caption updates.
- **Render** (`render.rs`) — cairo draws the caption to a transparent PNG for
  sources that cannot run JavaScript.
- **Overlay** (`static/`) — plain HTML/CSS/JS rendered in OBS.

## Commands

| Command | Description |
|---------|-------------|
| `./scripts/download-models.sh whisper` | Download the Whisper ggml model |
| `./scripts/download-models.sh audio8` | Download Audio8 TTS ONNX INT4 |
| `cargo run` | Start the caption server |
| `cargo check` | Type check |
| `cargo clippy -- -D warnings` | Lint Rust codebase |
| `cargo test` | Run unit and integration tests |
| `cargo build --release` | Build production binary |

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model <path>` | `models/whisper/ggml-small.bin` | Whisper ggml model |
| `--language <code>` | `es` | Language code, or `auto` |
| `--port <n>` | `8080` | HTTP port |
| `--no-auto-start` | off | Do not start listening on launch |

Everything else (VAD thresholds, thread count) lives in
`~/.config/asr-overlay/config.json`.

## HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Overlay page (OBS Browser Source) |
| GET | `/ws` | WebSocket caption stream + control commands |
| GET | `/caption.png` | Latest caption as a transparent PNG |
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
- A Whisper ggml model in `models/whisper/` (`./scripts/download-models.sh whisper`)
- Linux audio deps: `libasound2-dev pkg-config`
- GTK3 + libayatana-appindicator (tray) and cairo dev headers (PNG captions)
