use anyhow::{anyhow, Result};

use super::manifest::Manifest;

/// Result of the prompt builder: the [num_codebooks + 1, seq] token matrix and
/// the flattened semantic token list for the reference.
#[derive(Debug, Clone)]
pub struct PromptTensor {
    /// `[num_codebooks + 1, seq]`, first row is text/semantic tokens.
    pub tokens: Vec<i64>,
    pub seq_len: usize,
    pub num_rows: usize,
}

/// Builds the Audio8 prompt from the tokenizer.
///
/// NOTE: the tokenizer (`tokenizer.json`) and the reference voice codes are
/// required. This module currently only validates the shape math; actual
/// tokenization is wired up in FASE 6 of the roadmap.
pub fn build_prompt(
    _manifest: &Manifest,
    _reference_text: &str,
    _reference_codes: &[Vec<i64>],
    _target_text: &str,
) -> Result<PromptTensor> {
    Err(anyhow!(
        "Prompt builder not yet implemented: requires tokenizer.json (FASE 6)"
    ))
}

/// Number of rows in the prompt tensor = num_codebooks + 1.
pub fn prompt_rows(manifest: &Manifest) -> usize {
    manifest.num_codebooks + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_is_codebooks_plus_one() {
        let json = r#"{
            "max_seq_len": 2048, "num_layers": 24, "num_fast_layers": 4,
            "num_codebooks": 10, "n_local_heads": 2, "fast_n_local_heads": 2,
            "head_dim": 64, "fast_head_dim": 64, "codebook_size": 4096,
            "semantic_begin_id": 151678, "semantic_end_id": 155773,
            "im_end_id": 151645, "codec_sample_rate": 44100,
            "codec_hop_length": 2048, "stream_context_frames": 128,
            "stream_guard_frames": 1
        }"#;
        let m = Manifest::from_json(json).unwrap();
        assert_eq!(prompt_rows(&m), 11);
    }
}
