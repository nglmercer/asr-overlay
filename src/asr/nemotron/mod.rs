//! Nemotron 3.5 ASR streaming 0.6B, INT4 ONNX, CPU-only.
//!
//! A cache-aware FastConformer-RNNT, so nothing like the audio LLM in
//! `asr/model.rs`: there is no prompt, no token budget and no minimum clip
//! length. Audio arrives in fixed 560 ms chunks, the encoder carries its
//! attention and convolution state across them, and a small LSTM prediction
//! network plus a joint network emit tokens greedily.
//!
//! Three graphs (`genai_config.json` names them):
//!
//! 1. `encoder.onnx` — log-mel `[1,128,T]` + caches → `[1,1024,T/8]`
//! 2. `decoder.onnx` — previous token + LSTM state → `[1,640,1]`
//! 3. `joint.onnx` — encoder frame + decoder state → vocab logits

pub mod mel;
pub mod model;

pub use model::{NemotronAsr, Vocab};
