use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

/// Parsed `runtime_manifest.json` — the source of truth for Audio8's
/// architecture. Values are read at runtime, never hardcoded.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Manifest {
    pub max_seq_len: usize,

    pub num_layers: usize,
    pub num_fast_layers: usize,
    pub num_codebooks: usize,

    pub n_local_heads: usize,
    pub fast_n_local_heads: usize,

    pub head_dim: usize,
    pub fast_head_dim: usize,

    pub codebook_size: usize,

    pub semantic_begin_id: i64,
    pub semantic_end_id: i64,
    pub im_end_id: i64,

    pub codec_sample_rate: u32,
    pub codec_hop_length: usize,

    pub stream_context_frames: usize,
    pub stream_guard_frames: usize,
}

impl Manifest {
    /// Load and parse a manifest from a JSON file.
    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read manifest {path}"))?;
        Self::from_json(&json)
    }

    /// Parse a manifest from a JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| anyhow!("Invalid manifest JSON: {e}"))
    }

    /// Slow AR KV cache shape: `[1, n_local_heads, max_seq_len, head_dim]`.
    pub fn slow_cache_shape(&self) -> [usize; 4] {
        [
            1,
            self.n_local_heads,
            self.max_seq_len,
            self.head_dim,
        ]
    }

    /// Fast AR KV cache shape: `[1, fast_n_local_heads, num_codebooks, fast_head_dim]`.
    pub fn fast_cache_shape(&self) -> [usize; 4] {
        [
            1,
            self.fast_n_local_heads,
            self.num_codebooks,
            self.fast_head_dim,
        ]
    }

    /// Number of slow AR KV tensors (key + value per layer).
    pub fn slow_kv_tensors(&self) -> usize {
        self.num_layers * 2
    }

    /// Number of fast AR KV tensors (key + value per layer).
    pub fn fast_kv_tensors(&self) -> usize {
        self.num_fast_layers * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "max_seq_len": 2048,
        "num_layers": 24,
        "num_fast_layers": 4,
        "num_codebooks": 10,
        "n_local_heads": 2,
        "fast_n_local_heads": 2,
        "head_dim": 64,
        "fast_head_dim": 64,
        "codebook_size": 4096,
        "semantic_begin_id": 151678,
        "semantic_end_id": 155773,
        "im_end_id": 151645,
        "codec_sample_rate": 44100,
        "codec_hop_length": 2048,
        "stream_context_frames": 128,
        "stream_guard_frames": 1
    }"#;

    #[test]
    fn parses_current_manifest() {
        let m = Manifest::from_json(SAMPLE).unwrap();
        assert_eq!(m.max_seq_len, 2048);
        assert_eq!(m.num_layers, 24);
        assert_eq!(m.num_fast_layers, 4);
        assert_eq!(m.num_codebooks, 10);
        assert_eq!(m.codebook_size, 4096);
        assert_eq!(m.semantic_begin_id, 151678);
        assert_eq!(m.semantic_end_id, 155773);
        assert_eq!(m.im_end_id, 151645);
        assert_eq!(m.codec_sample_rate, 44100);
    }

    #[test]
    fn cache_shapes_match_runtime() {
        let m = Manifest::from_json(SAMPLE).unwrap();
        assert_eq!(m.slow_cache_shape(), [1, 2, 2048, 64]);
        assert_eq!(m.fast_cache_shape(), [1, 2, 10, 64]);
        assert_eq!(m.slow_kv_tensors(), 48);
        assert_eq!(m.fast_kv_tensors(), 8);
    }

    #[test]
    fn rejects_invalid_json() {
        assert!(Manifest::from_json("{nope").is_err());
        assert!(Manifest::from_json("{}").is_err());
    }
}
