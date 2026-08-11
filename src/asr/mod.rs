//! Speech recognition: Audio8 ARK-ASR 0.6B, INT8 ONNX, CPU-only.

pub mod mel;
pub mod model;

use anyhow::Result;

use crate::hub::{ensure_repo, human_bytes, is_cached, RepoSpec};

pub use mel::MelExtractor;
pub use model::ArkAsr;

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

/// Download the ASR model if it is not cached yet, then load it.
///
/// The download happens once; later runs resolve straight from the cache,
/// the same way `transformers` handles model files.
pub fn load(threads: usize, max_new_tokens: usize) -> Result<ArkAsr> {
    if !is_cached(&ARK_ASR) {
        log::info!(
            "ASR model not cached — downloading {} (~1.7 GiB, one time)",
            ARK_ASR.repo
        );
    }
    let dir = ensure_repo(&ARK_ASR, |file, done, total| match total {
        Some(total) => log::info!(
            "  {file}: {} / {} ({:.0}%)",
            human_bytes(done),
            human_bytes(total),
            100.0 * done as f64 / total as f64
        ),
        None => log::info!("  {file}: {}", human_bytes(done)),
    })?;
    ArkAsr::load(&dir, threads, max_new_tokens)
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
}
