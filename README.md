# asr-overlay

An overlay for **OBS Browser Source** that listens to the microphone and generates
live subtitles. **Phonon-2** is the default provider: a 164 MB English-only model
served locally by Fermion's CPU runtime. The Rust overlay sends VAD-segmented
16 kHz WAV audio to that server; no GPU or cloud service is required.

Multilingual **Audio8 ARK-ASR 0.6B** (INT8 ONNX, in-process CPU inference) remains
available in `/config`. Audio8 TTS playback is planned.

## Quick start (Phonon-2, CPU only)

Install the upstream server in a separate Python environment. On Linux:

```bash
python3 -m venv .venv-phonon
source .venv-phonon/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fermion-research safetensors soundfile scipy zstandard
fermion serve phonon-2 --port 8000
```

In another terminal, start the overlay:

```bash
cargo run
```

The server downloads and caches Phonon-2 on first use. Keep it running while
listening. Point OBS Browser Source at `http://localhost:8080/` and open
`http://localhost:8080/config` to pick a microphone and style the captions.
The default Phonon base URL is `http://127.0.0.1:8000/v1`, model `phonon-2`,
with no API key. The overlay checks the server's model list before listening.
A stopped server produces a setup error; it does not fall back to another model.
Choose English or Auto-detect for Phonon; other language hints are rejected.

### Runtime support

The [Phonon-2 model repository](https://huggingface.co/FermionResearch/Phonon-2)
publishes packed weights, **no ONNX graphs**. The supported CPU engine uses
PyTorch on Linux x86-64 (AVX2) and Arm64 (NEON), Windows x86-64, and macOS;
Apple silicon also supports MLX. See the upstream
[CPU guide](https://github.com/fermionresearch/phonon/blob/main/docs/cpu.md) and
[server API](https://github.com/fermionresearch/phonon/blob/main/docs/server.md).
The integration uses the supported HTTP API rather than attempting an
unverified ONNX conversion. Fermion's published CPU container is another
installation option. Model weights are CC-BY-4.0, derived from NVIDIA Parakeet;
the upstream runtime code is Apache-2.0.

### Use ARK instead

Select **Local ARK-ASR** in `/config` for 19-language recognition without a
separate Python server. Its ~1.7 GiB ONNX model downloads automatically to
`~/.cache/asr-overlay/models/`. Explicitly saved provider selections are kept;
new configurations and configurations without a provider use Phonon-2.

Set `ASR_OVERLAY_CACHE` to put the cache somewhere else.

## Current state

- [x] Phonon-2 local CPU server provider (default, English only)
- [x] HTTP server (axum) serving an overlay page for OBS
- [x] Live microphone capture (CPAL) with start/stop from tray, HTTP and WebSocket
- [x] RMS-energy VAD — continuous listening, no push-to-talk
- [x] **ARK-ASR 0.6B INT8 ONNX** transcription, CPU-only, ported from the
      official runtime
- [x] Automatic model download with an on-disk cache
- [x] Subtitles pushed over WebSocket + rendered as a transparent PNG (cairo)
- [ ] Audio8 TTS so the overlay can speak back — scaffolding only
- [ ] Voice registration from the microphone (Audio8 codec encoder)

## Speech recognition

[`Audio8/ark-asr-0.6b-int8-onnx`](https://huggingface.co/Audio8/ark-asr-0.6b-int8-onnx)
is an audio LLM, not a classic encoder-decoder ASR. Four ONNX graphs run in
sequence:

| Graph | Shape | Role |
|-------|-------|------|
| `audio_encoder_whisper_int8.onnx` | `[1,128,T]` → `[1,T',1280]` | Whisper-style encoder (32 layers, d_model 1280) |
| `audio_encoder_adapter_int8.onnx` | `[1,T'/4,5120]` → `[1,N,896]` | MLP adapter into the LLM's embedding space |
| `embedding_fp32.onnx` | ids → `[1,L,896]` | token embedding table |
| `llm_kv_cpu_fp32_int8.onnx` | → `[1,1,163958]` | Qwen2 decoder, 24 layers, external KV cache |

The audio is encoded, merged 4 frames at a time, projected to the decoder's
896-dim space, and spliced into the prompt in place of `<|audio|>` placeholder
tokens. The decoder then greedily generates the transcript.

**Languages** (19, auto-detected — there is no language flag): Chinese,
English, German, Japanese, French, Korean, Spanish, Polish, Italian, Romanian,
Hungarian, Czech, Dutch, Finnish, Croatian, Slovak, Slovene, Estonian,
Lithuanian.

### Performance

Measured on this machine (12 threads, release build) for a short utterance
padded to 8 s:

```
encoder 1.7 s · prefill 0.45 s (110 tok) · decode 90 ms/token
first word on screen 2.1 s after the utterance closes, full line at 3.3 s
```

The ONNX audio encoder is ~85% of that; the Rust log-mel frontend is 1 ms and
the adapter 7 ms, so there is nothing to win on our side of the graph. Three
things move the number:

- **Threads.** The encoder scales well: 12 threads instead of 5 cuts a short
  utterance from ~3.9 s to ~2.9 s. `threads` defaults to every core, capped
  at 12.
- **`min_audio_ms`.** Encoder cost is linear in clip length, so each second of
  padding costs ~0.25 s. 8 s is the shortest padding that keeps the model in
  the right language (see below).
- **Streaming.** Captions are pushed token by token as they decode, so the
  first words appear about a second before the line is finished.

Short utterances are padded to `min_audio_ms` (8 s by default) before they
reach the encoder. That padding is not optional: on a 2 s clip this model stops
listening to the audio and falls back on its own priors — the Spanish "El alma
del Japón" comes back as "El南国的日本", and the same clip padded to 8 s
transcribes correctly. 6 s still drifts, 12 s is no better than 8 s. The
padding has to reach the decoder as `<|audio|>` tokens too: encoding 8 s but
prompting with only the real speech tokens saves 270 ms of prefill and brings
the wrong-language drift straight back.

## Captions are text, not an image

The overlay renders the caption as real DOM text (`#caption .caption-text`),
so it is stylable with plain CSS and never a video frame. The settings page at
`/config` edits font, colour, background, position and free-form custom CSS,
with a live preview that uses the very same stylesheet generator as the
overlay. Saved styles push to connected overlays over the WebSocket, so OBS
updates without reloading the Browser Source.

`/caption.png` still exists for sources that cannot run JavaScript.

## Architecture

```text
                    ┌───────────────┐
Microphone ────────▶│ CPAL          │
                    │ PCM f32       │
                    └───────┬───────┘
                            ▼
                    ┌───────────────┐
                    │ Resample      │
                    │ mono → 16 kHz │
                    │ windowed sinc │
                    └───────┬───────┘
                            ▼
                    ┌───────────────┐
                    │ VAD (RMS)     │
                    │ utterance     │
                    │ segmentation  │
                    └───────┬───────┘
                            ▼
                    ┌───────────────┐
                    │ log-mel 128   │
                    │ (rustfft)     │
                    └───────┬───────┘
                            ▼
                    ┌───────────────┐
                    │ ARK-ASR 0.6B  │
                    │ encoder →     │
                    │ adapter → LLM │
                    │ ONNX · CPU    │
                    └───────┬───────┘
                            │ text
                            ▼
                    ┌───────────────┐
                    │ axum server   │
                    │ captions      │
                    └───────┬───────┘
                            ▼
                    ┌───────────────┐
                    │ OBS Browser   │
                    │ Source        │
                    └───────────────┘
```

Modules:

- **Mic** (`audio.rs`) — `cpal` input stream; the callback only copies samples
  into a channel. Inference never runs in the callback.
- **Resample** (`audio.rs`) — windowed-sinc conversion to 16 kHz mono.
- **VAD** (`caption.rs`) — RMS gate splitting utterances. It keeps 300 ms of
  pre-roll, the pauses inside an utterance, and 300 ms of tail, and needs only
  half the threshold to stay in speech. Handing the model clipped, spliced
  fragments makes it hallucinate — confidently, and often in another language.
- **Mel** (`asr/mel.rs`) — Whisper feature extractor ported to Rust (`rustfft`).
- **ASR** (`asr/model.rs`) — the four ONNX graphs and the decode loop.
- **Hub** (`hub.rs`) — model auto-download with an on-disk cache.
- **TTS** (`tts/`) — Audio8 DualAR ONNX INT4. Not wired up yet.
- **Server** (`server.rs`) — axum serves the overlay and pushes captions.
- **Render** (`render.rs`) — cairo draws the caption to a transparent PNG for
  consumers that cannot run JavaScript. The overlay page does not use it.

## Commands

| Command | Description |
|---------|-------------|
| `cargo run` | Start the caption server (downloads the model if needed) |
| `cargo run --bin asr-tool -- inspect` | Print every ONNX graph's inputs/outputs |
| `cargo run --release --bin asr-tool -- transcribe x.wav` | Transcribe a file |
| `cargo run --release --bin asr-tool -- vad x.wav [gain]` | Segment a file through the live VAD and transcribe each utterance |
| `cargo run --release --bin asr-tool -- micpath x.wav` | Compare direct vs. resampled-through-48 kHz transcription |
| `cargo check` | Type check |
| `cargo clippy -- -D warnings` | Lint |
| `cargo test` | Unit and integration tests |
| `cargo build --release` | Production binary |

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--language <code>` | `auto` | Hint only — ARK-ASR detects the language itself |
| `--max-new-tokens <n>` | `256` | Decoder budget per utterance |
| `--port <n>` | `8080` | HTTP port |
| `--no-auto-start` | off | Do not start listening on launch |

Microphone, VAD thresholds, threads and caption styling all live in
`~/.config/asr-overlay/config.json` — edit them from `/config`.

`input_device` (also `--input-device`) holds an id from `/devices`, not a
display name: `pw:<node>` for a PipeWire/PulseAudio source, or a cpal device id
such as `alsa:pipewire`. Empty means "system default". On PipeWire the sound
server usually holds the sound card, so its sources are the ones that can
actually be captured — the picker only lists devices that open. A device that
is gone or busy falls back to the default rather than refusing to listen.

## Transcription providers

`phonon` (local Phonon-2 CPU server, English only) is the default;
`ark` runs multilingual ARK-ASR in-process. The `/config` page also offers
`openai_compatible` (`POST {base_url}/audio/transcriptions`, for OpenAI,
self-hosted Whisper and Mistral batch Voxtral) and realtime streaming via
`voxtral_realtime`, `gemini_live_transcribe` and `gemini_live_translate`.
Streaming providers receive continuous 16 kHz PCM and emit partial/final
captions — no VAD segmentation, no `min_audio_ms` padding. API keys are never
stored in `config.json`: set `api_key_env` to the variable holding the key
(e.g. `GEMINI_API_KEY`) and it is resolved server-side when listening starts.

## HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Overlay page (OBS Browser Source) |
| GET | `/ws` | WebSocket caption stream + control commands |
| GET | `/caption.png` | Latest caption as a transparent PNG |
| GET | `/config` | Settings page with live preview |
| GET | `/devices` | Available microphones |
| POST | `/style` | Live style preview (not persisted) |
| GET | `/status` | `{ running, listening, has_caption }` |
| POST | `/start` | Start listening |
| POST | `/stop` | Stop listening |
| GET | `/settings` | Get current config JSON |
| POST | `/settings` | Update config JSON |

## OBS Integration

1. Run the app (`cargo run`)
2. In OBS, add a **Browser Source**
3. Set URL to `http://localhost:8080/`
4. Set width/height to match your stream layout
5. Check "Shutdown source when not visible" for optimal performance

## Frontend development

The overlay and settings pages are a Vue 3 + TypeScript app in `frontend/`,
built with Vite and embedded into the binary — `cargo build` runs the npm
build automatically. For UI work, run the backend once (`cargo run`) and then,
in `frontend/`:

```bash
npm install
npm run dev      # dev server with hot reload, API proxied to :8080
npm run typecheck
```

## Requirements

- For the default provider: a separately running Fermion Phonon-2 CPU server
  (Python/PyTorch, or the upstream CPU container)
- Rust toolchain (1.70+)
- Node.js v20.19+ or v22.12+ with npm (builds the frontend)
- 164 MB Phonon-2 model download plus upstream CPU runtime dependencies;
  ~1.7 GiB model disk if using ARK (downloaded on first run)
- Linux audio deps: `libasound2-dev pkg-config`
- GTK3 + libayatana-appindicator (tray) and cairo dev headers (PNG captions)

ONNX Runtime binaries are fetched automatically by the `ort` crate at build
time — nothing to install by hand.
