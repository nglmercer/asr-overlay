use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

/// Parsed `runtime_manifest.json` — the source of truth for Audio8's
/// architecture. Values are read at runtime, never hardcoded.
///
/// Unknown fields are ignored, so a newer manifest revision still loads.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Manifest {
    pub model_family: String,
    /// `"float16"` for the preview build — KV caches and hidden states are FP16.
    pub activation_dtype: String,

    /// Layout of the slow AR logits vector. `"semantic_then_eos"` means index
    /// `0..codebook_size` are semantic codes and the final index is EOS.
    pub slow_logits_layout: String,
    /// Length of the slow AR logits vector (`codebook_size + 1`).
    pub slow_logits_size: usize,
    /// `"valid_prefix"`: attention reads only the filled prefix of the cache.
    pub kv_attention_layout: String,

    pub max_seq_len: usize,

    pub num_layers: usize,
    pub num_fast_layers: usize,
    pub num_codebooks: usize,

    pub n_local_heads: usize,
    pub fast_n_local_heads: usize,

    pub head_dim: usize,
    pub fast_head_dim: usize,
    pub fast_dim: usize,

    pub vocab_size: usize,
    pub codebook_size: usize,

    pub semantic_begin_id: i64,
    pub semantic_end_id: i64,
    pub im_end_id: i64,
    pub eos_token_id: i64,
    pub pad_token_id: i64,

    pub codec_sample_rate: u32,
    pub codec_hop_length: usize,
    pub codec_frame_size: usize,
    pub sample_rate: u32,

    pub stream_context_frames: usize,
    pub stream_guard_frames: usize,
}

/// What a sampled slow AR logit index means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlowToken {
    /// A semantic code, already relative to `semantic_begin_id` — i.e. exactly
    /// what goes into `frame[0]`.
    Semantic(i64),
    /// End of utterance.
    Eos,
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
        let m: Self = serde_json::from_str(json).map_err(|e| anyhow!("Invalid manifest JSON: {e}"))?;
        m.validate()?;
        Ok(m)
    }

    /// Reject manifests whose layouts this implementation does not handle,
    /// instead of silently generating noise.
    fn validate(&self) -> Result<()> {
        if self.slow_logits_layout != "semantic_then_eos" {
            return Err(anyhow!(
                "Unsupported slow_logits_layout {:?}; only \"semantic_then_eos\" is implemented",
                self.slow_logits_layout
            ));
        }
        if self.kv_attention_layout != "valid_prefix" {
            return Err(anyhow!(
                "Unsupported kv_attention_layout {:?}; only \"valid_prefix\" is implemented",
                self.kv_attention_layout
            ));
        }
        if self.slow_logits_size != self.codebook_size + 1 {
            return Err(anyhow!(
                "slow_logits_size {} != codebook_size + 1 ({})",
                self.slow_logits_size,
                self.codebook_size + 1
            ));
        }
        Ok(())
    }

    /// Interpret an index sampled from the slow AR logits.
    ///
    /// The slow AR emits only `codebook_size` semantic logits plus EOS — it does
    /// NOT emit the full `vocab_size`. So the sampled index *is* the semantic
    /// code; there is no `token - semantic_begin_id` subtraction to do.
    pub fn decode_slow_index(&self, index: usize) -> Result<SlowToken> {
        if index < self.codebook_size {
            Ok(SlowToken::Semantic(index as i64))
        } else if index == self.codebook_size {
            Ok(SlowToken::Eos)
        } else {
            Err(anyhow!(
                "Slow AR index {index} out of range (slow_logits_size = {})",
                self.slow_logits_size
            ))
        }
    }

    /// The absolute token id fed back into the slow AR for a semantic code.
    pub fn semantic_token_id(&self, semantic: i64) -> i64 {
        self.semantic_begin_id + semantic
    }

    /// Slow AR KV cache shape: `[1, n_local_heads, max_seq_len, head_dim]`.
    pub fn slow_cache_shape(&self) -> [usize; 4] {
        [1, self.n_local_heads, self.max_seq_len, self.head_dim]
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

    /// Samples produced per generated frame.
    pub fn samples_per_frame(&self) -> usize {
        self.codec_hop_length
    }
}

/// Verbatim copy of `runtime_manifest.json` from
/// `Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4`, shared by the TTS unit tests.
#[cfg(test)]
pub(crate) const SAMPLE_MANIFEST: &str = r#"{
        "model_family": "audio8_tts",
        "activation_dtype": "float16",
        "slow_logits_layout": "semantic_then_eos",
        "slow_logits_size": 4097,
        "kv_attention_layout": "valid_prefix",
        "max_seq_len": 2048,
        "num_layers": 24,
        "num_fast_layers": 4,
        "num_codebooks": 10,
        "n_local_heads": 2,
        "fast_n_local_heads": 2,
        "head_dim": 64,
        "fast_head_dim": 64,
        "fast_dim": 896,
        "vocab_size": 155776,
        "codebook_size": 4096,
        "semantic_begin_id": 151678,
        "semantic_end_id": 155773,
        "eos_token_id": 151645,
        "pad_token_id": 151643,
        "codec_sample_rate": 44100,
        "codec_frame_size": 2048,
        "sample_rate": 44100,
        "codec_hop_length": 2048,
        "stream_context_frames": 128,
        "stream_guard_frames": 1,
        "decoder_provider": "cpu",
        "default_codec_precision": "fp16",
        "available_codec_precisions": ["fp16"],
        "codec_models": { "fp16": "codec_decoder_fp16.onnx" },
        "im_end_id": 151645,
        "model_fingerprint": "62dcff0adf6c2535b3260467a7c1d482b556da57266c96a444518b76e140d2c3",
        "default_precision": "int4",
        "available_precisions": ["int4"]
    }"#;

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = SAMPLE_MANIFEST;

    #[test]
    fn parses_real_manifest_and_ignores_unknown_fields() {
        let m = Manifest::from_json(SAMPLE).unwrap();
        assert_eq!(m.model_family, "audio8_tts");
        assert_eq!(m.max_seq_len, 2048);
        assert_eq!(m.num_layers, 24);
        assert_eq!(m.num_fast_layers, 4);
        assert_eq!(m.num_codebooks, 10);
        assert_eq!(m.codebook_size, 4096);
        assert_eq!(m.slow_logits_size, 4097);
        assert_eq!(m.vocab_size, 155776);
        assert_eq!(m.semantic_begin_id, 151678);
        assert_eq!(m.semantic_end_id, 155773);
        assert_eq!(m.im_end_id, m.eos_token_id);
        assert_eq!(m.codec_sample_rate, 44100);
        assert_eq!(m.samples_per_frame(), 2048);
    }

    #[test]
    fn semantic_id_range_matches_codebook_size() {
        let m = Manifest::from_json(SAMPLE).unwrap();
        // 151678 + 4096 - 1 == 155773
        assert_eq!(
            m.semantic_token_id(m.codebook_size as i64 - 1),
            m.semantic_end_id
        );
    }

    #[test]
    fn slow_index_maps_to_semantic_or_eos() {
        let m = Manifest::from_json(SAMPLE).unwrap();
        assert_eq!(m.decode_slow_index(0).unwrap(), SlowToken::Semantic(0));
        assert_eq!(m.decode_slow_index(4095).unwrap(), SlowToken::Semantic(4095));
        assert_eq!(m.decode_slow_index(4096).unwrap(), SlowToken::Eos);
        assert!(m.decode_slow_index(4097).is_err());
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
    fn rejects_unsupported_layouts() {
        let bad = SAMPLE.replace("semantic_then_eos", "eos_then_semantic");
        assert!(Manifest::from_json(&bad).is_err());
        let bad = SAMPLE.replace("valid_prefix", "full_window");
        assert!(Manifest::from_json(&bad).is_err());
    }

    #[test]
    fn rejects_invalid_json() {
        assert!(Manifest::from_json("{nope").is_err());
        assert!(Manifest::from_json("{}").is_err());
    }
}
