//! Local ARK-ASR provider: the existing ONNX implementation behind the
//! [`UtteranceAsr`](crate::asr::provider::UtteranceAsr) interface.
//!
//! The model loads lazily — construction only remembers the settings, and the
//! ~1.7 GiB download plus graph load happens in [`prepare`](crate::asr::provider::UtteranceAsr::prepare)
//! (or on the first [`transcribe`](crate::asr::provider::UtteranceAsr::transcribe)).
//! That keeps engine construction cheap and offline so provider selection stays
//! testable without the model cached.
//!
//! ARK is the only provider that pads short clips to `min_audio_ms`: below a
//! few seconds its decoder leans on its own priors and answers in the wrong
//! language. Network providers receive audio as-is.

use anyhow::Result;

use super::model::ArkAsr;
use super::provider::UtteranceAsr;

/// [`UtteranceAsr`] over the local ARK-ASR model.
pub struct ArkEngine {
    threads: usize,
    max_new_tokens: usize,
    min_audio_ms: u64,
    asr: Option<ArkAsr>,
}

impl ArkEngine {
    pub fn new(threads: usize, max_new_tokens: usize, min_audio_ms: u64) -> Self {
        Self {
            threads,
            max_new_tokens,
            min_audio_ms,
            asr: None,
        }
    }

    /// The padding window this engine hands to the model.
    pub fn min_audio_ms(&self) -> u64 {
        self.min_audio_ms
    }

    fn ensure_loaded(&mut self) -> Result<&mut ArkAsr> {
        if self.asr.is_none() {
            let mut asr = super::load(self.threads, self.max_new_tokens)?;
            asr.set_min_audio_ms(self.min_audio_ms);
            self.asr = Some(asr);
        }
        // `set_min_audio_ms` is re-applied in case the value changed.
        let asr = self.asr.as_mut().expect("just loaded");
        asr.set_min_audio_ms(self.min_audio_ms);
        Ok(asr)
    }
}

impl UtteranceAsr for ArkEngine {
    fn prepare(&mut self) -> Result<()> {
        self.ensure_loaded().map(|_| ())
    }

    fn transcribe(&mut self, pcm_16k: &[f32], _language: Option<&str>) -> Result<String> {
        // ARK-ASR auto-detects its 19 languages; the hint is ignored.
        self.ensure_loaded()?.transcribe(pcm_16k)
    }

    fn transcribe_with_partials(
        &mut self,
        pcm_16k: &[f32],
        _language: Option<&str>,
        on_partial: &mut dyn FnMut(&str),
    ) -> Result<String> {
        // Decoding is ~90 ms per token, so stream words to the overlay while
        // the rest is still decoding instead of showing nothing for seconds.
        self.ensure_loaded()?
            .transcribe_streaming(pcm_16k, on_partial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AsrConfig;

    #[test]
    fn ark_engine_remembers_its_settings_without_loading() {
        // Construction must not touch the model cache: this test runs with no
        // model downloaded and no network.
        let engine = ArkEngine::new(4, 128, 6_000);
        assert_eq!(engine.min_audio_ms(), 6_000);
    }

    #[test]
    fn ark_keeps_the_configured_padding() {
        let config = AsrConfig {
            min_audio_ms: 5_000,
            ..AsrConfig::default()
        };
        let engine = ArkEngine::new(config.threads, config.max_new_tokens, config.min_audio_ms);
        assert_eq!(engine.min_audio_ms(), 5_000);
    }
}
