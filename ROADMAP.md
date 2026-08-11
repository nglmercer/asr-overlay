# Roadmap

Implementar asr-overlay como asistente de voz local (CPU-only) integrando **Audio8 TTS** (ONNX INT4) y **Whisper STT**.

## Objetivo

1. Capturar micrófono con `cpal`.
2. Transcribir español localmente con `whisper-rs` (CPU).
3. Sintetizar la respuesta con **Audio8** ejecutado en Rust vía `ort` (ONNX Runtime CPU).
4. Reproducir a 44.1 kHz y mostrar la transcripción como subtítulo en OBS.
5. Registrar la voz del usuario usando el encoder ONNX incluido.

Nada de PyTorch, Python runtime, GPU ni servicios externos.

## Modelo de referencia

- Modelo: `Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4` (Hugging Face)
- Runtime Python oficial: `Audio8-AI/Audio8_TTS/onnx_runtime`
- Archivos: `slow_ar_int4.onnx(.data)`, `fast_ar_int4.onnx(.data)`, `codec_decoder_fp16.onnx(.data)`, `runtime_manifest.json`, `tokenizer/tokenizer.json`, `registration/codec_encoder_fp16.onnx(.data)`

Descarga:

```bash
python3 -m pip install -U "huggingface_hub[cli]"
hf download Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4 --local-dir models/audio8
```

No separar los `.onnx` de sus `.onnx.data`. Modelos online ~572 MiB; con encoder ~968 MiB.

## Arquitectura de Audio8

Es **DualAR autoregresivo**, no un Transformer clásico (`session.run(input_ids)` no basta).

1. **Slow AR** → genera un token semántico por frame (tiene KV cache de 24 capas, `[1,2,2048,64]` FP16).
2. **Fast AR** → condicionado por el hidden state del Slow AR, genera los 10 codebooks por frame (4 capas, cache `[1,2,10,64]`).
3. **Codec decoder** → recibe `codes[1,10,T]` int64 y devuelve waveform `float32` mono @ 44100 Hz.

Generación por frame:

```text
Slow AR → semantic → frame[0] = semantic - semantic_begin_id
Fast AR pos 1..9 → frame[1..9]
frame final = [cb0..cb9]
feed [semantic, cb0..cb9] al Slow AR
```

## Fases

### FASE 1 — Cimentación
- `cargo project`, deps: `ort` 2.x, `ndarray`, `half`, `tokenizers`, `serde`, `rand`/`rand_chacha`, `cpal`, `rubato`, `rodio`, `whisper-rs`, `hound`, `crossbeam-channel`, `clap`.
- CPAL detecta micro y acepta config default.

### FASE 2 — Captura de audio
- Grabar 5 s → guardar `microphone.wav`.

### FASE 3 — STT (Whisper)
- `whisper-rs` + `ggml-small.bin`, Spanish, mono f32 16 kHz.
- Mic → resample → Whisper → texto. **No conectar Audio8 hasta que esto funcione perfecto.**

### FASE 4 — Abrir los modelos ONNX
- Abrir `codec_decoder`, `slow_ar`, `fast_ar`; **inspeccionar e imprimir** nombres/dtypes/shapes de inputs y outputs (fuente de verdad = el ONNX y el runtime oficial).

### FASE 5 — Manifest parser
- Leer `runtime_manifest.json` en runtime (max_seq_len, num_layers, num_codebooks, head/dims, semantic/end ids, codec rate/hop, stream frames). No hardcodear salvo tests.

### FASE 6 — Tokenizer + PromptBuilder
- Portar `prompt.py` exactamente.
- Prompt con system (referencia de voz + semantic codes), user (texto objetivo), assistant (`<|voice|>`).
- Tensor `[1, num_codebooks+1, seq]`; fila 0 = tokens texto/semánticos, filas 1..10 = codebooks de referencia.

### FASE 7 — Sampler
- temperature=0.7, top_p=0.9, top_k=50, seed=42 (+ anti-repetition del runtime).
- Samplear solo sobre `semantic_begin..semantic_end` + `im_end_id`.
- Tests deterministas con seed fijo.

### FASE 8 — Slow AR
- KV caches por capa: key/value `[1, n_local_heads, max_seq_len, head_dim]` FP16.
- `slow_ar.step(codes, input_pos, caches)` → logits, hidden; actualizar caches con deltas.

### FASE 9 — Fast AR
- Caches independientes `[1, fast_n_local_heads, num_codebooks, fast_head_dim]`, 4 capas.

### FASE 10 — Generación
- Portar `iter_codes()` línea por línea con los pasos de la sección Arquitectura.

### FASE 11 — Decoder
- `codes[1,10,T]` int64 → `codec_decoder_fp16.onnx` → waveform f32.

### FASE 12 — TTS texto → output.wav
- `hound` para guardar WAV de debugging. Validar contra seed/temperatura de Python (checkpoints: ids tokenizer, shape prompt, primeros logits, primer semantic, primer frame, primeros N frames, shape/size waveform).

### FASE 13 — Playback
- `rodio::buffer::SamplesBuffer::new(1, 44100, samples)`. Flag `--output` para WAV.

### FASE 14 — MVP asistente
```text
ENTER → grabar 5 s → resample 16k → Whisper → imprimir texto
     → Audio8 sintetiza → reproducir → volver al inicio
```

### FASE 15 — Integración OBS subtítulos
- Servir overlay con el texto transcrito + WebSocket para actualización en tiempo real.

### FASE 16 — Registro de voz
- `codec_encoder_fp16.onnx`: grabar 5–10 s, mono 44.1 kHz, pad a múltiplo de 2048, `audio[1,1,samples]` → `codes [1,10,T]`.
- Guardar perfil `voices/<name>/{meta.json, codes.bin}` (reference_text exacto + codes).

### FASE 17 — VAD
- Escuchar continuo: detectar voz → buffer → fin de habla (~500–800 ms) → Whisper. Posible usar `WhisperVadContext` de whisper-rs.

### FASE 18 — Streaming TTS
- Decodificar ventanas de `chunk_frames=12`, `context_frames=128`, `guard_frames=1`, hop 2048; empezar playback mientras se genera.

## Threading

- Thread 1: audio callback → channel/ring buffer (nunca inferencia en callback).
- Thread 2: STT.
- Thread 3: TTS (pausar STT mientras Audio8 sintetiza para no transcribir la propia voz).
- No compartir sesiones ONNX mutables sin abstracción segura.

## Tests requeridos

`tests/tokenizer.rs`, `tests/prompt.rs`, `tests/sampler.rs`, `tests/slow_ar.rs`, `tests/fast_ar.rs`, `tests/decoder.rs`, `tests/integration_tts.rs`.

Comparar contra el runtime Python oficial (seed=42, T=0.7, p=0.9, k=50): ids, prompt, logits, semantic, frame, waveform.

## Reglas

- No inventar shapes: imprimir inputs/outputs ONNX antes de cada wrapper.
- Portar Python línea por línea; refactorizar solo tras tests equivalentes.
- No optimizar ni hacer streaming hasta que `synthesize(text)` produzca un WAV válido.
- Usar 5 threads ONNX (default del runtime oficial).