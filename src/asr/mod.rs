//! Speech recognition, behind one of several providers.
//!
//! - `phonon`: local English-only Phonon-2 CPU server over HTTP (default).
//! - `ark`: local Audio8 ARK-ASR 0.6B, INT8 ONNX, CPU-only.
//! - `openai_compatible`: `POST {base_url}/audio/transcriptions` (OpenAI,
//!   self-hosted Whisper, Mistral batch Voxtral).
//! - `voxtral_realtime`, `gemini_live_transcribe`, `gemini_live_translate`:
//!   streaming providers over WebSocket.
//!
//! [`create_engine`] builds the configured backend; the controller then runs
//! either the utterance pipeline (VAD → transcribe → caption) or the
//! streaming pipeline (continuous PCM → partial/final captions).

pub mod ark;
pub mod gemini;
pub mod mel;
pub mod model;
pub mod openai_compatible;
pub mod phonon;
pub mod provider;
pub mod voxtral;

use anyhow::Result;

use crate::config::{AsrConfig, ProviderKind};
use crate::hub::{ensure_repo, human_bytes, is_cached, RepoSpec};

pub use ark::ArkEngine;
pub use gemini::{GeminiLiveAsr, GeminiMode};
pub use mel::MelExtractor;
pub use model::ArkAsr;
pub use openai_compatible::OpenAiCompatibleAsr;
pub use provider::{AsrEngine, StreamingAsr, TranscriptEvent, UtteranceAsr};
pub use voxtral::VoxtralRealtimeAsr;

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

/// Build the transcription backend selected in `config`.
///
/// Construction is cheap and offline for every provider: the ARK model loads
/// lazily on [`UtteranceAsr::prepare`], and API keys resolve from the
/// environment without any I/O. Streaming providers fail here when their
/// required key is missing, so misconfiguration surfaces at startup instead
/// of mid-stream.
pub fn create_engine(config: &AsrConfig) -> Result<AsrEngine> {
    let provider = &config.provider;
    match provider.kind {
        ProviderKind::Phonon => Ok(AsrEngine::Utterance(Box::new(phonon::PhononAsr::new(
            &provider.base_url,
            &provider.model,
            provider.api_key(),
        )))),
        ProviderKind::Ark => {
            log::info!("ASR provider: local ARK-ASR");
            Ok(AsrEngine::Utterance(Box::new(ArkEngine::new(
                config.threads,
                config.max_new_tokens,
                config.min_audio_ms,
            ))))
        }
        ProviderKind::OpenAiCompatible => {
            let asr =
                OpenAiCompatibleAsr::new(&provider.base_url, &provider.model, provider.api_key());
            log::info!(
                "ASR provider: OpenAI-compatible at {} (model {})",
                asr.base_url(),
                asr.model()
            );
            Ok(AsrEngine::Utterance(Box::new(asr)))
        }
        ProviderKind::VoxtralRealtime => {
            let api_key = provider.require_api_key()?;
            let language = effective_language(&config.language);
            log::info!("ASR provider: Voxtral realtime");
            Ok(AsrEngine::Streaming(Box::new(VoxtralRealtimeAsr::new(
                api_key,
                &provider.model,
                language,
            ))))
        }
        ProviderKind::GeminiLiveTranscribe => {
            let api_key = provider.require_api_key()?;
            log::info!("ASR provider: Gemini Live transcription");
            Ok(AsrEngine::Streaming(Box::new(GeminiLiveAsr::new(
                api_key,
                &provider.model,
                GeminiMode::Transcribe,
                "",
            ))))
        }
        ProviderKind::GeminiLiveTranslate => {
            let api_key = provider.require_api_key()?;
            log::info!(
                "ASR provider: Gemini Live translation → {}",
                if provider.target_language.trim().is_empty() {
                    "en"
                } else {
                    provider.target_language.trim()
                }
            );
            Ok(AsrEngine::Streaming(Box::new(GeminiLiveAsr::new(
                api_key,
                &provider.model,
                GeminiMode::Translate,
                &provider.target_language,
            ))))
        }
    }
}

/// The language hint sent to providers, or `None` for auto-detect.
pub fn effective_language(language: &str) -> Option<&str> {
    let language = language.trim();
    if language.is_empty() || language.eq_ignore_ascii_case("auto") {
        None
    } else {
        Some(language)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderConfig;

    fn config_with(kind: ProviderKind) -> AsrConfig {
        AsrConfig {
            provider: ProviderConfig {
                kind,
                base_url: "http://localhost:8000/v1".to_string(),
                model: "test-model".to_string(),
                api_key_env: "ASR_OVERLAY_TEST_FACTORY_KEY".to_string(),
                target_language: "es".to_string(),
                streaming_delay_ms: 100,
            },
            ..AsrConfig::default()
        }
    }

    #[test]
    fn factory_selects_utterance_providers() {
        for kind in [
            ProviderKind::Phonon,
            ProviderKind::Ark,
            ProviderKind::OpenAiCompatible,
        ] {
            match create_engine(&config_with(kind)) {
                Ok(AsrEngine::Utterance(_)) => {}
                other => panic!("{kind:?} must build an utterance engine: {}", other.is_ok()),
            }
        }
    }

    #[test]
    fn factory_selects_streaming_providers() {
        std::env::set_var("ASR_OVERLAY_TEST_FACTORY_KEY", "test-key");
        for kind in [
            ProviderKind::VoxtralRealtime,
            ProviderKind::GeminiLiveTranscribe,
            ProviderKind::GeminiLiveTranslate,
        ] {
            match create_engine(&config_with(kind)) {
                Ok(AsrEngine::Streaming(_)) => {}
                other => panic!("{kind:?} must build a streaming engine: {}", other.is_ok()),
            }
        }
        std::env::remove_var("ASR_OVERLAY_TEST_FACTORY_KEY");
    }

    #[test]
    fn streaming_providers_require_an_api_key() {
        std::env::remove_var("ASR_OVERLAY_TEST_FACTORY_MISSING");
        let mut config = config_with(ProviderKind::GeminiLiveTranscribe);
        config.provider.api_key_env = "ASR_OVERLAY_TEST_FACTORY_MISSING".to_string();
        assert!(create_engine(&config).is_err());

        // …while OpenAI-compatible endpoints may have none.
        let mut config = config_with(ProviderKind::OpenAiCompatible);
        config.provider.api_key_env = String::new();
        assert!(matches!(
            create_engine(&config),
            Ok(AsrEngine::Utterance(_))
        ));
    }

    #[test]
    fn default_factory_needs_no_model_on_disk() {
        // Must not download or load anything: selection is offline.
        match create_engine(&AsrConfig::default()) {
            Ok(AsrEngine::Utterance(_)) => {}
            _ => panic!("default config must build an utterance engine"),
        }
    }

    #[test]
    fn effective_language_maps_auto_and_blank_to_none() {
        assert_eq!(effective_language("auto"), None);
        assert_eq!(effective_language("AUTO"), None);
        assert_eq!(effective_language(""), None);
        assert_eq!(effective_language("  "), None);
        assert_eq!(effective_language("en"), Some("en"));
        assert_eq!(effective_language(" es "), Some("es"));
    }

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
