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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    /// Which ASR model runs: `nemotron` (cache-aware streaming RNNT, the
    /// default) or `ark` (audio LLM, whole utterances). See `asr-tool bench`.
    #[serde(default = "default_asr_backend")]
    pub asr_backend: String,
    /// Spoken language. ARK-ASR detects it itself and ignores this; Nemotron
    /// conditions its encoder on it, and takes either `auto` or a locale tag
    /// such as `es-US`, `en-US` or `pt-BR`.
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
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            port: default_port(),
            asr_backend: default_asr_backend(),
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
        }
    }
}

fn default_min_audio_ms() -> u64 {
    8_000
}

fn default_port() -> u16 {
    8080
}

/// Nemotron is the default: the streaming encoder starts transcribing within
/// a couple of chunks and never needs the padding the audio LLM relies on.
fn default_asr_backend() -> String {
    "nemotron".to_string()
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
        assert_eq!(config.asr_backend, "nemotron");
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
            asr_backend: "nemotron".to_string(),
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
        assert_eq!(config.asr_backend, "nemotron");
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
