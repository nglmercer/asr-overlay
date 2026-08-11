use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

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
    /// Number of ONNX/whisper threads used during inference.
    #[serde(default = "default_threads")]
    pub threads: usize,
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
            threads: default_threads(),
        }
    }
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

fn default_threads() -> usize {
    5
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
        assert_eq!(config.threads, 5);
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
            threads: 4,
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
        assert_eq!(config.threads, 5);
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
