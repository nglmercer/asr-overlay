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
  from `AsrConfig`), with pre-roll, tail and a half-threshold hysteresis to keep
  speech continuous. Never strip the quiet parts out of an utterance: the model
  answers spliced audio with fluent nonsense, sometimes in the wrong language.
- **Padding**: `transcribe` pads clips shorter than `min_audio_ms` (8 s) with
  silence, and the padding must reach the decoder as `<|audio|>` tokens.
  Measured: below ~6 s, or with the padding tokens trimmed, the decoder answers
  in the wrong language.
- **Streaming**: `transcribe_streaming` reports the text after every token;
  `run_pipeline` pushes those as `partial: true` captions, then the final one.
- **Mel**: `asr/mel.rs` — Whisper log-mel frontend (128 bins, n_fft 400,
  hop 160) on `rustfft`.
- **ASR**: `asr/model.rs` — `Audio8/ark-asr-0.6b-int8-onnx` on `ort`. Four
  graphs: whisper encoder → MLP adapter → token embedding → Qwen2 decoder with
  an external KV cache. 19 languages, auto-detected. Ported from the official
  `infer_ark_audio_onnx.py`.
- **Providers**: `asr/provider.rs` (`UtteranceAsr` / `StreamingAsr` /
  `AsrEngine`, built by `asr::create_engine`). `ark.rs` wraps the local model
  (lazy load, keeps `min_audio_ms` + token streaming); `openai_compatible.rs`
  posts WAV to `{base_url}/audio/transcriptions`; `gemini.rs` / `voxtral.rs`
  stream 16 kHz PCM over WebSocket with reconnect. Utterance providers run the
  VAD pipeline in `caption::run_pipeline`, streaming ones the continuous
  `caption::run_streaming_pipeline`. API keys resolve from `api_key_env` at
  startup and never touch `config.json` or `/settings` output.
- **Hub**: `hub.rs` — auto-download to `~/.cache/asr-overlay/models/`,
  overridable with `ASR_OVERLAY_CACHE`.
- **TTS**: `src/tts/` — Audio8 DualAR ONNX INT4. Scaffolding only; every session
  wrapper currently returns "not yet implemented".
- **Server**: `axum` serves the overlay page, a WebSocket caption stream, and a
  cairo-rendered PNG caption.
- **Tray**: `tray-icon` + GTK.
- **Config**: JSON at `~/.config/asr-overlay/config.json`; edited from the
  `/config` page, which previews caption styling live via
  `frontend/src/shared/style.ts` (shared with the overlay so the preview
  cannot drift).
- **Frontend**: Vue 3 + TypeScript in `frontend/` (Vite multi-page build:
  overlay + settings). `build.rs` runs `npm ci` + `npm run build` and embeds
  `frontend/dist/`; output names are stable (`index.js`, `config.js`,
  `style.js`, `vendor.js`) to match the server routes. Set
  `ASR_OVERLAY_SKIP_FRONTEND_BUILD=1` to embed a prebuilt `dist/` instead.
  `npm run typecheck` covers `.ts` via native `tsc`; SFC internals await
  vue-tsc TypeScript 7 support.

## Commands

| Command | Description |
|---------|-------------|
| `cargo run` | Start the caption server (auto-downloads the model) |
| `cargo run --bin asr-tool -- inspect` | Print each ONNX graph's inputs/outputs |
| `cargo run --release --bin asr-tool -- transcribe x.wav` | Transcribe a file |
| `cargo run --release --bin asr-tool -- vad x.wav [gain]` | Segment through the live VAD, then transcribe each utterance |
| `cargo check` | Type check |
| `cargo clippy -- -D warnings` | Lint |
| `cargo test` | Unit + integration tests |
| `cargo build --release` | Production binary |
| `npm run dev` (in `frontend/`) | Vite dev server, API proxied to `cargo run` |
| `npm run typecheck` (in `frontend/`) | Typecheck `.ts` with native `tsc` |

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
- Node.js v20.19+ or v22.12+ with npm (the build compiles the Vue frontend)
- ~1.7 GiB disk for the ASR model (fetched on first run)
- ALSA/PulseAudio headers: `libasound2-dev pkg-config`
- GTK3 + libayatana-appindicator for the tray
- cairo dev headers for caption rendering
