use serde::{Deserialize, Serialize};

/// How the overlay draws captions. Every field maps to CSS on the overlay
/// page, so the caption stays real DOM text — stylable and selectable — rather
/// than a rendered image.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OverlayStyle {
    pub font_family: String,
    pub font_size: u32,
    pub font_weight: u32,
    /// CSS colour for the text (any CSS colour syntax).
    pub color: String,
    /// Background behind the caption block; use `transparent` for none.
    pub background: String,
    pub padding: u32,
    pub radius: u32,
    /// Full CSS `text-shadow` value — the usual way to keep white text legible
    /// over a bright scene.
    pub text_shadow: String,
    /// `bottom` | `center` | `top`
    pub position: String,
    /// `left` | `center` | `right`
    pub align: String,
    /// Caption block width as a percentage of the source.
    pub max_width: u32,
    /// Distance from the chosen edge, in px.
    pub margin: u32,
    pub uppercase: bool,
    /// Fade-in duration in ms.
    pub fade_ms: u32,
    /// Appended verbatim to the overlay stylesheet, so anything not covered by
    /// the fields above can still be styled.
    pub custom_css: String,
}

impl Default for OverlayStyle {
    fn default() -> Self {
        Self {
            font_family: "-apple-system, \"Segoe UI\", Roboto, sans-serif".to_string(),
            font_size: 28,
            font_weight: 600,
            color: "#ffffff".to_string(),
            background: "transparent".to_string(),
            padding: 8,
            radius: 8,
            text_shadow: "0 2px 6px rgba(0,0,0,0.9), 0 0 2px #000".to_string(),
            position: "bottom".to_string(),
            align: "center".to_string(),
            max_width: 90,
            margin: 40,
            uppercase: false,
            fade_ms: 200,
            custom_css: String::new(),
        }
    }
}
use std::fs;
use std::path::PathBuf;

/// Which transcription backend the pipeline uses.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    /// Local Phonon-2 CPU server (English only).
    #[default]
    Phonon,
    /// Local multilingual ARK-ASR ONNX model.
    Ark,
    /// `POST {base_url}/audio/transcriptions` (OpenAI, self-hosted Whisper,
    /// Mistral batch Voxtral, …).
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
    /// Mistral realtime transcription over WebSocket.
    VoxtralRealtime,
    /// Gemini Live streaming transcription.
    GeminiLiveTranscribe,
    /// Gemini Live streaming translation.
    GeminiLiveTranslate,
}

/// Settings for the selected [`ProviderKind`]. Only the fields a provider
/// needs are shown in the UI; the rest are ignored.
///
/// API keys are never stored here — only the *name* of the environment
/// variable holding the key, resolved server-side at pipeline start.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProviderConfig {
    #[serde(default)]
    pub kind: ProviderKind,

    /// OpenAI-compatible servers: the API root, e.g. `http://localhost:8000/v1`.
    #[serde(default)]
    pub base_url: String,

    /// Model id sent to the provider, e.g. `whisper-large-v3`.
    #[serde(default)]
    pub model: String,

    /// Environment variable holding the API key, e.g. `MISTRAL_API_KEY`.
    /// Empty means the endpoint needs no authentication.
    #[serde(default)]
    pub api_key_env: String,

    /// Gemini Live translation target, e.g. `es`.
    #[serde(default)]
    pub target_language: String,

    /// Streaming providers: how much audio (in ms) is buffered into each
    /// chunk sent to the provider. ~100 ms matches what realtime APIs expect.
    #[serde(default = "default_streaming_delay")]
    pub streaming_delay_ms: u32,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            kind: ProviderKind::Phonon,
            base_url: String::new(),
            model: String::new(),
            api_key_env: String::new(),
            target_language: String::new(),
            streaming_delay_ms: default_streaming_delay(),
        }
    }
}

impl ProviderConfig {
    /// True for the providers that stream audio continuously instead of
    /// transcribing VAD-segmented utterances.
    pub fn is_streaming(&self) -> bool {
        matches!(
            self.kind,
            ProviderKind::VoxtralRealtime
                | ProviderKind::GeminiLiveTranscribe
                | ProviderKind::GeminiLiveTranslate
        )
    }

    /// Resolve the API key from [`Self::api_key_env`]. `None` when no
    /// variable is configured or it is unset/empty — valid for endpoints
    /// without authentication.
    pub fn api_key(&self) -> Option<String> {
        let name = self.api_key_env.trim();
        if name.is_empty() {
            return None;
        }
        std::env::var(name).ok().filter(|v| !v.is_empty())
    }

    /// Like [`Self::api_key`], but an error when no key resolves — for the
    /// providers that cannot work without one.
    pub fn require_api_key(&self) -> anyhow::Result<String> {
        let name = self.api_key_env.trim();
        if name.is_empty() {
            return Err(anyhow::anyhow!(
                "Provider {:?} needs an API key: set api_key_env to the variable holding it",
                self.kind
            ));
        }
        std::env::var(name)
            .ok()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Environment variable {name} is not set or empty"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    /// Hint only — ARK-ASR detects the spoken language itself. Kept for the
    /// overlay UI and for future TTS voice selection.
    #[serde(default = "default_language")]
    pub language: String,
    /// Maximum tokens the ASR decoder may emit per utterance.
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
    #[serde(default = "default_auto_start")]
    pub auto_start: bool,
    /// RMS energy threshold above which the pipeline treats a chunk as speech.
    #[serde(default = "default_vad_threshold")]
    pub vad_threshold: f32,
    /// Silence (in ms) required after speech before an utterance is closed.
    #[serde(default = "default_vad_silence_ms")]
    pub vad_silence_ms: u64,
    /// Minimum utterance length (in ms) to be worth transcribing.
    #[serde(default = "default_vad_min_ms")]
    pub vad_min_ms: u64,
    /// Maximum utterance length (in ms) — forces segmentation of long speech.
    #[serde(default = "default_vad_max_ms")]
    pub vad_max_ms: u64,
    /// Shortest clip handed to the model (in ms); shorter utterances are padded
    /// with silence. Below a few seconds the decoder leans on its own priors
    /// instead of the audio and answers in the wrong language, so this is the
    /// main accuracy/latency dial: higher is more accurate and slower.
    #[serde(default = "default_min_audio_ms")]
    pub min_audio_ms: u64,
    /// Input device name, or empty for the system default.
    #[serde(default)]
    pub input_device: String,
    /// Number of ONNX threads used during inference.
    #[serde(default = "default_threads")]
    pub threads: usize,
    /// Caption appearance on the overlay page.
    #[serde(default)]
    pub style: OverlayStyle,
    /// Transcription backend. Defaults to local ARK-ASR.
    #[serde(default)]
    pub provider: ProviderConfig,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            port: default_port(),
            language: default_language(),
            max_new_tokens: default_max_new_tokens(),
            auto_start: default_auto_start(),
            vad_threshold: default_vad_threshold(),
            vad_silence_ms: default_vad_silence_ms(),
            vad_min_ms: default_vad_min_ms(),
            vad_max_ms: default_vad_max_ms(),
            min_audio_ms: default_min_audio_ms(),
            input_device: String::new(),
            threads: default_threads(),
            style: OverlayStyle::default(),
            provider: ProviderConfig::default(),
        }
    }
}

fn default_streaming_delay() -> u32 {
    100
}

fn default_min_audio_ms() -> u64 {
    8_000
}

fn default_port() -> u16 {
    8080
}

fn default_language() -> String {
    "auto".to_string()
}

fn default_max_new_tokens() -> usize {
    256
}

fn default_auto_start() -> bool {
    true
}

fn default_vad_threshold() -> f32 {
    0.005
}

fn default_vad_silence_ms() -> u64 {
    600
}

fn default_vad_min_ms() -> u64 {
    300
}

fn default_vad_max_ms() -> u64 {
    30_000
}

/// Every core the machine has, capped at 12.
///
/// The ONNX encoder is the bulk of the latency and scales well with threads:
/// on a 12-thread CPU, 12 threads transcribe a short utterance in ~2.9 s where
/// 5 threads take ~3.9 s. Past ~12 the returns flatten and the overlay starves
/// whatever else the machine is doing (OBS, a game, an encoder).
fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(12))
        .unwrap_or(5)
}

fn config_dir() -> PathBuf {
    let base = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
    let dir = base.join("asr-overlay");
    let _ = fs::create_dir_all(&dir);
    dir
}

fn config_path() -> PathBuf {
    config_dir().join("config.json")
}

pub fn load() -> AsrConfig {
    load_from(config_path())
}

pub fn save(config: &AsrConfig) {
    save_to(config, config_path())
}

fn load_from(path: PathBuf) -> AsrConfig {
    match fs::read_to_string(&path) {
        Ok(json) => serde_json::from_str(&json).unwrap_or_default(),
        Err(_) => AsrConfig::default(),
    }
}

fn save_to(config: &AsrConfig, path: PathBuf) {
    if let Ok(json) = serde_json::to_string_pretty(config) {
        if let Err(e) = fs::write(&path, json) {
            log::warn!("Failed to save config to {path:?}: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AsrConfig::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.language, "auto");
        assert_eq!(config.max_new_tokens, 256);
        assert!(config.auto_start);
        assert_eq!(config.vad_threshold, 0.005);
        assert_eq!(config.vad_silence_ms, 600);
        assert_eq!(config.vad_min_ms, 300);
        assert_eq!(config.vad_max_ms, 30_000);
        assert_eq!(config.threads, default_threads());
    }

    #[test]
    fn test_config_serialize_deserialize() {
        let config = AsrConfig {
            port: 9090,
            language: "en".to_string(),
            max_new_tokens: 128,
            auto_start: false,
            vad_threshold: 0.01,
            vad_silence_ms: 800,
            vad_min_ms: 500,
            vad_max_ms: 20_000,
            min_audio_ms: 6_000,
            input_device: String::new(),
            threads: 4,
            style: OverlayStyle::default(),
            provider: ProviderConfig::default(),
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: AsrConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, config);
    }

    #[test]
    fn test_config_deserialize_partial() {
        let json = r#"{"port": 9090}"#;
        let config: AsrConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.port, 9090);
        assert_eq!(config.language, "auto");
        assert_eq!(config.threads, default_threads());
        assert_eq!(config.min_audio_ms, 8_000);
    }

    #[test]
    fn test_config_deserialize_empty() {
        let json = "{}";
        let config: AsrConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config, AsrConfig::default());
    }

    #[test]
    fn configs_without_provider_default_to_phonon() {
        // A config written before providers existed has no "provider" key.
        let json = r#"{
            "port": 9090, "language": "en", "max_new_tokens": 128,
            "vad_threshold": 0.01, "threads": 4, "min_audio_ms": 6000
        }"#;
        let config: AsrConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.provider.kind, ProviderKind::Phonon);
        assert_eq!(config.provider, ProviderConfig::default());

        // So does an empty provider object.
        let config: AsrConfig = serde_json::from_str(r#"{"provider": {}}"#).unwrap();
        assert_eq!(config.provider.kind, ProviderKind::Phonon);
        assert_eq!(config.provider.streaming_delay_ms, 100);
    }

    #[test]
    fn saved_ark_selection_is_preserved() {
        let config: AsrConfig = serde_json::from_str(r#"{"provider":{"kind":"ark"}}"#).unwrap();
        assert_eq!(config.provider.kind, ProviderKind::Ark);
    }

    #[test]
    fn all_provider_kinds_roundtrip() {
        for kind in [
            ProviderKind::Phonon,
            ProviderKind::Ark,
            ProviderKind::OpenAiCompatible,
            ProviderKind::VoxtralRealtime,
            ProviderKind::GeminiLiveTranscribe,
            ProviderKind::GeminiLiveTranslate,
        ] {
            let config = ProviderConfig {
                kind,
                base_url: "http://localhost:8000/v1".to_string(),
                model: "test-model".to_string(),
                api_key_env: "TEST_API_KEY".to_string(),
                target_language: "es".to_string(),
                streaming_delay_ms: 120,
            };
            let json = serde_json::to_string(&config).unwrap();
            let parsed: ProviderConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, config, "roundtrip failed for {kind:?}");
        }
    }

    #[test]
    fn provider_kinds_use_snake_case() {
        assert_eq!(
            serde_json::to_value(ProviderKind::OpenAiCompatible).unwrap(),
            serde_json::Value::String("openai_compatible".to_string())
        );
        assert_eq!(
            serde_json::from_str::<ProviderKind>("\"gemini_live_translate\"").unwrap(),
            ProviderKind::GeminiLiveTranslate
        );
        assert_eq!(
            serde_json::from_str::<ProviderKind>("\"voxtral_realtime\"").unwrap(),
            ProviderKind::VoxtralRealtime
        );
    }

    #[test]
    fn example_configs_from_the_spec_parse() {
        let openai: AsrConfig = serde_json::from_str(
            r#"{"provider": {"kind": "openai_compatible",
                "base_url": "http://localhost:8000/v1",
                "model": "whisper-large-v3", "api_key_env": ""}}"#,
        )
        .unwrap();
        assert_eq!(openai.provider.kind, ProviderKind::OpenAiCompatible);
        assert_eq!(openai.provider.base_url, "http://localhost:8000/v1");

        let mistral: AsrConfig = serde_json::from_str(
            r#"{"provider": {"kind": "openai_compatible",
                "base_url": "https://api.mistral.ai/v1",
                "model": "voxtral-mini-latest", "api_key_env": "MISTRAL_API_KEY"}}"#,
        )
        .unwrap();
        assert_eq!(mistral.provider.api_key_env, "MISTRAL_API_KEY");

        let gemini: AsrConfig = serde_json::from_str(
            r#"{"provider": {"kind": "gemini_live_translate",
                "model": "gemini-3.5-live-translate-preview",
                "api_key_env": "GEMINI_API_KEY", "target_language": "es"}}"#,
        )
        .unwrap();
        assert_eq!(gemini.provider.kind, ProviderKind::GeminiLiveTranslate);
        assert_eq!(gemini.provider.target_language, "es");
    }

    #[test]
    fn api_keys_are_never_serialized() {
        // The secret lives in the environment; the config only names it.
        std::env::set_var("ASR_OVERLAY_TEST_SECRET_KEY", "sk-super-secret-value-12345");
        let config = AsrConfig {
            provider: ProviderConfig {
                kind: ProviderKind::GeminiLiveTranscribe,
                api_key_env: "ASR_OVERLAY_TEST_SECRET_KEY".to_string(),
                ..ProviderConfig::default()
            },
            ..AsrConfig::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(
            !json.contains("sk-super-secret-value-12345"),
            "secret leaked into serialized config: {json}"
        );
        assert!(json.contains("ASR_OVERLAY_TEST_SECRET_KEY"));
        // Resolution still finds the value server-side.
        assert_eq!(
            config.provider.api_key().as_deref(),
            Some("sk-super-secret-value-12345")
        );
        std::env::remove_var("ASR_OVERLAY_TEST_SECRET_KEY");
    }

    #[test]
    fn api_key_resolution_rules() {
        let none = ProviderConfig::default();
        assert_eq!(none.api_key(), None);
        assert!(none.require_api_key().is_err());

        let missing = ProviderConfig {
            api_key_env: "ASR_OVERLAY_TEST_DEFINITELY_UNSET".to_string(),
            ..ProviderConfig::default()
        };
        std::env::remove_var("ASR_OVERLAY_TEST_DEFINITELY_UNSET");
        assert_eq!(missing.api_key(), None);
        assert!(missing.require_api_key().is_err());
    }

    #[test]
    fn streaming_classification() {
        for kind in [
            ProviderKind::VoxtralRealtime,
            ProviderKind::GeminiLiveTranscribe,
            ProviderKind::GeminiLiveTranslate,
        ] {
            assert!(
                ProviderConfig {
                    kind,
                    ..ProviderConfig::default()
                }
                .is_streaming(),
                "{kind:?} must be streaming"
            );
        }
        for kind in [
            ProviderKind::Phonon,
            ProviderKind::Ark,
            ProviderKind::OpenAiCompatible,
        ] {
            assert!(
                !ProviderConfig {
                    kind,
                    ..ProviderConfig::default()
                }
                .is_streaming(),
                "{kind:?} must be utterance-based"
            );
        }
    }

    #[test]
    fn test_save_and_load() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("asr-overlay-test-config-{}", std::process::id()));

        let config = AsrConfig {
            port: 7777,
            ..AsrConfig::default()
        };

        save_to(&config, path.clone());
        let loaded = load_from(path.clone());
        assert_eq!(loaded.port, 7777);

        let _ = std::fs::remove_file(path);
    }
}
