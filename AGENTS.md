# asr-overlay

Local, CPU-only voice pipeline served over HTTP for OBS Browser Source:
**mic → ARK-ASR → subtitles → (planned) Audio8 TTS playback**.
No cloud, no Python runtime, no GPU.

## Architecture

- **Audio**: `cpal` input stream; the callback only forwards samples to a
  `crossbeam-channel`. Inference never runs in the audio callback.
- **Resample**: `audio.rs` windowed-sinc → mono `f32` @ 16 kHz. No `rubato`:
  its `FftFixedOut`, sized per call, returned silence for long buffers.
- **VAD**: RMS-energy gate in `caption.rs` (threshold / silence / min / max ms
  from `AsrConfig`).
- **Mel**: `asr/mel.rs` — Whisper log-mel frontend (128 bins, n_fft 400,
  hop 160) on `rustfft`.
- **ASR**: `asr/model.rs` — `Audio8/ark-asr-0.6b-int8-onnx` on `ort`. Four
  graphs: whisper encoder → MLP adapter → token embedding → Qwen2 decoder with
  an external KV cache. 19 languages, auto-detected. Ported from the official
  `infer_ark_audio_onnx.py`.
- **Hub**: `hub.rs` — auto-download to `~/.cache/asr-overlay/models/`,
  overridable with `ASR_OVERLAY_CACHE`.
- **TTS**: `src/tts/` — Audio8 DualAR ONNX INT4. Scaffolding only; every session
  wrapper currently returns "not yet implemented".
- **Server**: `axum` serves the overlay page, a WebSocket caption stream, and a
  cairo-rendered PNG caption.
- **Tray**: `tray-icon` + GTK.
- **Config**: JSON at `~/.config/asr-overlay/config.json`; edited from the
  `/config` page, which previews caption styling live via `static/style.js`
  (shared with the overlay so the preview cannot drift).

## Commands

| Command | Description |
|---------|-------------|
| `cargo run` | Start the caption server (auto-downloads the model) |
| `cargo run --bin asr-tool -- inspect` | Print each ONNX graph's inputs/outputs |
| `cargo run --release --bin asr-tool -- transcribe x.wav` | Transcribe a file |
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
| GET | `/config` | Settings page with live preview |
| GET | `/devices` | Available microphones as `{ id, label, default }` |
| POST | `/style` | Live style preview (not persisted) |
| GET | `/status` | `{ running, listening, has_caption }` |
| POST | `/start` | Start listening |
| POST | `/stop` | Stop listening |
| GET/POST | `/settings` | Read / update config JSON |
| GET | `/config` | Settings page with live preview |
| GET | `/devices` | Available microphones as `{ id, label, default }` |
| POST | `/style` | Live style preview (not persisted) |

## Rules

- Keep the dependency set minimal. Before adding a crate, check whether an
  existing one covers it (`cpal` also does playback; `ort` 2.x takes raw slices,
  so `ndarray` is not needed).
- Never hardcode model shapes or token ids — read `runtime_manifest.json` and
  `llm_kv_cpu_fp32_int8.json`. Verify with `asr-tool inspect` before writing a
  wrapper.
- Don't run inference on the cpal callback thread.

## Requirements

- Rust 1.70+
- ~1.7 GiB disk for the ASR model (fetched on first run)
- ALSA/PulseAudio headers: `libasound2-dev pkg-config`
- GTK3 + libayatana-appindicator for the tray
- cairo dev headers for caption rendering
