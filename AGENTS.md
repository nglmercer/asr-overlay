# asr-overlay

Local, CPU-only voice pipeline served over HTTP for OBS Browser Source:
**mic → Whisper STT → subtitles → (planned) Audio8 TTS playback**.
No cloud, no Python runtime, no GPU.

## Architecture

- **Audio**: `cpal` input stream; the callback only forwards samples to a
  `crossbeam-channel`. Inference never runs in the audio callback.
- **Resample**: `rubato` → mono `f32` @ 16 kHz for Whisper.
- **VAD**: RMS-energy gate in `caption.rs` (threshold / silence / min / max ms
  from `AsrConfig`).
- **STT**: `whisper-rs` (whisper.cpp), CPU.
- **TTS**: `src/tts/` — Audio8 DualAR ONNX INT4. Scaffolding only; every session
  wrapper currently returns "not yet implemented". See ROADMAP.md.
- **Server**: `axum` serves the overlay page, a WebSocket caption stream, and a
  cairo-rendered PNG caption.
- **Tray**: `tray-icon` + GTK.
- **Config**: JSON at `~/.config/asr-overlay/config.json`.

## Commands

| Command | Description |
|---------|-------------|
| `./scripts/download-models.sh whisper` | Fetch the Whisper ggml model (required) |
| `cargo run` | Start the caption server |
| `cargo check` | Type check |
| `cargo clippy -- -D warnings` | Lint |
| `cargo test` | Unit + integration tests |
| `cargo build --release` | Production binary |

## HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Overlay page (OBS Browser Source) |
| GET | `/ws` | WebSocket caption stream + control commands |
| GET | `/caption.png` | Latest caption as a transparent PNG |
| GET | `/status` | `{ running, listening, has_text }` |
| POST | `/start` | Start listening |
| POST | `/stop` | Stop listening |
| GET/POST | `/settings` | Read / update config JSON |

## Rules

- Keep the dependency set minimal. Before adding a crate, check whether an
  existing one covers it (`cpal` also does playback; `ort` 2.x takes raw slices,
  so `ndarray` is not needed).
- Never hardcode Audio8 shapes or token ids — read `runtime_manifest.json`.
- Don't run inference on the cpal callback thread.

## Requirements

- Rust 1.70+
- ALSA/PulseAudio headers: `libasound2-dev pkg-config`
- GTK3 + libayatana-appindicator for the tray
- cairo dev headers for caption rendering
