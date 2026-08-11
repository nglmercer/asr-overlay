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
        let m = Manifest::from_json(crate::tts::manifest::SAMPLE_MANIFEST).unwrap();
        assert_eq!(prompt_rows(&m), 11);
    }
}
