use anyhow::{anyhow, Result};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
};

/// CPU-only Whisper STT wrapper. Expects mono `f32` audio at 16 kHz.
pub struct WhisperStt {
    ctx: WhisperContext,
    language: String,
    threads: i32,
}

impl WhisperStt {
    /// Load the ggml model and prepare a state factory.
    pub fn load(model_path: &str, language: &str, threads: usize) -> Result<Self> {
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu = std::env::var("WHISPER_USE_GPU")
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        let ctx = WhisperContext::new_with_params(model_path, ctx_params)
            .map_err(|e| anyhow!("Failed to load Whisper model {model_path}: {e}"))?;
        log::info!("Loaded Whisper model: {model_path}");
        Ok(Self {
            ctx,
            language: language.to_string(),
            threads: threads as i32,
        })
    }

    /// Transcribe mono f32 audio at 16 kHz and return the joined text.
    pub fn transcribe(&mut self, samples_16k: &[f32]) -> Result<String> {
        if samples_16k.is_empty() {
            return Ok(String::new());
        }

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| anyhow!("Failed to create Whisper state: {e}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(&self.language));
        params.set_n_threads(self.threads);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state
            .full(params, samples_16k)
            .map_err(|e| anyhow!("Whisper inference failed: {e}"))?;

        let mut text = String::new();
        for segment in state.as_iter() {
            text.push_str(segment.to_str()?);
        }
        Ok(text.trim().to_string())
    }
}
