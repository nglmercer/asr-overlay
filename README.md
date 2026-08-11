# asr-overlay

An overlay for **OBS Browser Source** that listens to the microphone in real
time and generates live subtitles. Everything runs locally on CPU, using
**Audio8 ARK-ASR 0.6B** (INT8 ONNX) for speech recognition.

The long-term goal is a full voice assistant overlay:
**mic → ARK-ASR → subtitles → Audio8 TTS → spoken playback**, with no cloud
services, no Python runtime and no GPU.

## Quick start

```bash
cargo run
```

That's it. On first run the ASR model (~1.7 GiB) is downloaded automatically
into `~/.cache/asr-overlay/models/` and reused from then on — the same way
`transformers` caches models. Then point an OBS Browser Source at
`http://localhost:8080/`.

Set `ASR_OVERLAY_CACHE` to put the cache somewhere else.

## Current state

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

Measured on this machine with the 11 s `jfk.wav` sample, release build,
5 threads:

```
encode 3.9 s · prefill 0.8 s (147 tok) · decode 2.4 s (27 tok, ~87 ms/tok)
total 7.5 s for 11 s of audio — 1.4x realtime
```

The encoder dominates and its cost scales with clip length, so short VAD
utterances (1–3 s) are far cheaper than this 11 s sample. Note that 1.4x
realtime is **slower than whisper-small** on the same CPU — the accuracy and
the 19-language coverage are what you get for it.

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
                    │ (rubato)      │
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
- **VAD** (`caption.rs`) — RMS gate splitting utterances.
- **Mel** (`asr/mel.rs`) — Whisper feature extractor ported to Rust (`rustfft`).
- **ASR** (`asr/model.rs`) — the four ONNX graphs and the decode loop.
- **Hub** (`hub.rs`) — model auto-download with an on-disk cache.
- **TTS** (`tts/`) — Audio8 DualAR ONNX INT4. Not wired up yet.
- **Server** (`server.rs`) — axum serves the overlay and pushes captions.
- **Render** (`render.rs`) — cairo draws the caption to a transparent PNG.

## Commands

| Command | Description |
|---------|-------------|
| `cargo run` | Start the caption server (downloads the model if needed) |
| `cargo run --bin asr-tool -- inspect` | Print every ONNX graph's inputs/outputs |
| `cargo run --release --bin asr-tool -- transcribe x.wav` | Transcribe a file |
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

VAD thresholds and thread count live in `~/.config/asr-overlay/config.json`.

## HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Overlay page (OBS Browser Source) |
| GET | `/ws` | WebSocket caption stream + control commands |
| GET | `/caption.png` | Latest caption as a transparent PNG |
| GET | `/status` | `{ running, listening, has_text }` |
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

## Requirements

- Rust toolchain (1.70+)
- ~1.7 GiB of disk for the ASR model (downloaded on first run)
- Linux audio deps: `libasound2-dev pkg-config`
- GTK3 + libayatana-appindicator (tray) and cairo dev headers (PNG captions)

ONNX Runtime binaries are fetched automatically by the `ort` crate at build
time — nothing to install by hand.
