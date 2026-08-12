//! Speech recognition, CPU-only, with a choice of ONNX backend.
//!
//! Two models are wired up so their CPU cost can be compared on the same
//! audio, the same machine and the same VAD:
//!
//! - [`Backend::Nemotron`] — `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`,
//!   a cache-aware FastConformer-RNNT. Consumes 560 ms chunks and keeps its
//!   encoder state between them, so there is no padding and no re-encoding.
//!   This is the default backend.
//! - [`Backend::Ark`] — `Audio8/ark-asr-0.6b-int8-onnx`, an audio LLM
//!   (whisper encoder → adapter → Qwen2 decoder). Transcribes a whole
//!   utterance at once and needs several seconds of audio to stay in the right
//!   language.
//!
//! `asr-tool bench` runs both over the same file and prints the real-time
//! factors.

pub mod mel;
pub mod model;
pub mod nemotron;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::hub::{ensure_repo, human_bytes, is_cached, RepoSpec};

pub use mel::MelExtractor;
pub use model::ArkAsr;
pub use nemotron::NemotronAsr;

/// The ARK-ASR repo and the files needed to run it.
///
/// `embedding_fp32.onnx` is a tiny graph whose weights live in the sibling
/// `.data` file — both must be present, and ONNX Runtime resolves the pair by
/// filename, so they have to sit in the same directory.
pub const ARK_ASR: RepoSpec = RepoSpec {
    repo: "Audio8/ark-asr-0.6b-int8-onnx",
    revision: "main",
    files: &[
        "runtime_manifest.json",
        "llm_kv_cpu_fp32_int8.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "audio_encoder_adapter_int8.onnx",
        "embedding_fp32.onnx",
        "embedding_fp32.data",
        "llm_kv_cpu_fp32_int8.onnx",
        "audio_encoder_whisper_int8.onnx",
    ],
};

/// The Nemotron 3.5 streaming repo. Each `.onnx` graph keeps its weights in a
/// sibling `.onnx.data` file, so the pairs must be fetched together.
///
/// `silero_vad.onnx` ships in the repo but is not used: segmentation is the
/// energy VAD in `caption.rs`, shared with the other backend so a benchmark
/// compares models rather than segmenters.
pub const NEMOTRON_ASR: RepoSpec = RepoSpec {
    repo: "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4",
    revision: "main",
    files: &[
        "genai_config.json",
        "audio_processor_config.json",
        "tokenizer.json",
        "vocab.txt",
        "encoder.onnx",
        "encoder.onnx.data",
        "decoder.onnx",
        "decoder.onnx.data",
        "joint.onnx",
        "joint.onnx.data",
    ],
};

/// Which ASR model the pipeline runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// Audio8 ARK-ASR 0.6B INT8 — the original backend.
    Ark,
    /// NVIDIA Nemotron 3.5 streaming 0.6B INT4 — the default.
    #[default]
    Nemotron,
}

impl Backend {
    /// Parse a config value; unknown names fall back to the default so a typo
    /// in `config.json` cannot leave the overlay with no model at all.
    pub fn parse(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "nemotron" | "nemotron-3.5" | "nemo" => Backend::Nemotron,
            "ark" | "ark-asr" | "audio8" => Backend::Ark,
            _ => Backend::Nemotron,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Ark => "ark",
            Backend::Nemotron => "nemotron",
        }
    }

    pub fn spec(self) -> &'static RepoSpec {
        match self {
            Backend::Ark => &ARK_ASR,
            Backend::Nemotron => &NEMOTRON_ASR,
        }
    }

    /// Approximate download size, for the "this will take a while" log line.
    fn download_size(self) -> &'static str {
        match self {
            Backend::Ark => "~1.7 GiB",
            Backend::Nemotron => "~790 MiB",
        }
    }

    /// Every backend, in benchmark order.
    pub fn all() -> [Backend; 2] {
        [Backend::Ark, Backend::Nemotron]
    }
}

/// What a caption source has to provide, whatever model is behind it.
///
/// The callback is `&mut dyn FnMut` rather than a generic so the pipeline can
/// hold a `Box<dyn SpeechModel>` and swap backends at runtime.
pub trait SpeechModel: Send {
    fn backend(&self) -> Backend;

    /// Transcribe mono `f32` audio at 16 kHz, reporting the text so far as it
    /// becomes available.
    fn transcribe_streaming(
        &mut self,
        samples: &[f32],
        on_partial: &mut dyn FnMut(&str),
    ) -> Result<String>;

    /// Transcribe without partials.
    fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        self.transcribe_streaming(samples, &mut |_| {})
    }

    /// Shortest clip handed to the model, in ms. Only the audio-LLM backend
    /// pads; a streaming encoder has no minimum.
    fn set_min_audio_ms(&mut self, _ms: u64) {}

    /// Decoder instruction. Audio-LLM backends only.
    fn set_instruction(&mut self, _instruction: &str) {}

    /// Language hint: a locale like `es-US`, or `auto`. Nemotron conditions its
    /// encoder on this; ARK-ASR detects the language itself and ignores it.
    fn set_language(&mut self, _language: &str) -> Result<()> {
        Ok(())
    }
}

/// Download the model for `backend` if it is not cached yet, then load it.
///
/// The download happens once; later runs resolve straight from the cache,
/// the same way `transformers` handles model files.
pub fn load(
    backend: Backend,
    threads: usize,
    max_new_tokens: usize,
) -> Result<Box<dyn SpeechModel>> {
    let dir = ensure_backend(backend)?;
    Ok(match backend {
        Backend::Ark => Box::new(ArkAsr::load(&dir, threads, max_new_tokens)?),
        Backend::Nemotron => Box::new(NemotronAsr::load(&dir, threads)?),
    })
}

/// Fetch a backend's files if needed and return its local directory.
pub fn ensure_backend(backend: Backend) -> Result<std::path::PathBuf> {
    let spec = backend.spec();
    if !is_cached(spec) {
        log::info!(
            "ASR model not cached — downloading {} ({}, one time)",
            spec.repo,
            backend.download_size()
        );
    }
    ensure_repo(spec, |file, done, total| match total {
        Some(total) => log::info!(
            "  {file}: {} / {} ({:.0}%)",
            human_bytes(done),
            human_bytes(total),
            100.0 * done as f64 / total as f64
        ),
        None => log::info!("  {file}: {}", human_bytes(done)),
    })
    .map_err(|e| anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_lists_the_embedding_data_sidecar() {
        // Loading embedding_fp32.onnx without its .data file fails at runtime,
        // so the pair must always be fetched together.
        assert!(ARK_ASR.files.contains(&"embedding_fp32.onnx"));
        assert!(ARK_ASR.files.contains(&"embedding_fp32.data"));
    }

    #[test]
    fn spec_lists_every_graph_from_the_manifest() {
        for file in [
            "audio_encoder_whisper_int8.onnx",
            "audio_encoder_adapter_int8.onnx",
            "embedding_fp32.onnx",
            "llm_kv_cpu_fp32_int8.onnx",
            "tokenizer.json",
            "runtime_manifest.json",
        ] {
            assert!(ARK_ASR.files.contains(&file), "missing {file}");
        }
    }

    #[test]
    fn every_nemotron_graph_brings_its_weights() {
        // The INT4 export keeps all weights external; a graph without its
        // `.data` sibling loads and then fails on the first run.
        for graph in ["encoder.onnx", "decoder.onnx", "joint.onnx"] {
            assert!(NEMOTRON_ASR.files.contains(&graph));
            let data = format!("{graph}.data");
            assert!(
                NEMOTRON_ASR.files.iter().any(|f| *f == data),
                "missing {data}"
            );
        }
    }

    #[test]
    fn backend_names_round_trip() {
        for backend in Backend::all() {
            assert_eq!(Backend::parse(backend.as_str()), backend);
        }
        // Unknown names fall back rather than failing to start.
        assert_eq!(Backend::parse("whisper"), Backend::Nemotron);
        assert_eq!(Backend::parse("NEMOTRON"), Backend::Nemotron);
        assert_eq!(Backend::parse("ark"), Backend::Ark);
    }
}
