//! Transcription backend abstraction.
//!
//! Two shapes of provider:
//!
//! - [`UtteranceAsr`] transcribes one VAD-segmented utterance at a time
//!   (local ARK-ASR, OpenAI-compatible HTTP).
//! - [`StreamingAsr`] consumes a continuous PCM stream and emits partial and
//!   final transcripts as they arrive (Gemini Live, Voxtral realtime).
//!
//! [`AsrEngine`] is the factory product the controller dispatches on. Provider
//! construction is cheap and offline: model downloads and network dials happen
//! in [`UtteranceAsr::prepare`] / [`StreamingAsr::run`], never in the factory.

/// One transcript update from a streaming provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptEvent {
    /// An in-progress hypothesis; replaces the previous partial.
    Partial(String),
    /// A finished segment; safe to append to a transcript log.
    Final(String),
}

/// Transcribes complete utterances: mono `f32` audio at 16 kHz in, text out.
pub trait UtteranceAsr: Send {
    fn transcribe(&mut self, pcm_16k: &[f32], language: Option<&str>) -> anyhow::Result<String>;

    /// Transcribe, reporting partial text through `on_partial` when the
    /// provider produces any. The default implementation reports nothing
    /// until the final text is ready; ARK-ASR overrides this to stream
    /// tokens as they decode so the overlay stays live.
    fn transcribe_with_partials(
        &mut self,
        pcm_16k: &[f32],
        language: Option<&str>,
        on_partial: &mut dyn FnMut(&str),
    ) -> anyhow::Result<String> {
        let _ = on_partial;
        self.transcribe(pcm_16k, language)
    }

    /// Load models / validate connectivity before the first utterance.
    /// Called once when listening starts; the default is a no-op for
    /// providers that need no warm-up.
    fn prepare(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Transcribes a continuous stream: mono `i16` PCM chunks at 16 kHz in,
/// [`TranscriptEvent`]s out. Runs until `stop_rx` fires or `audio_rx` closes.
///
/// Implementations own their event loop (including any async runtime and
/// reconnect policy) and must return promptly once stopped.
pub trait StreamingAsr: Send {
    fn run(
        self: Box<Self>,
        audio_rx: crossbeam_channel::Receiver<Vec<i16>>,
        event_tx: crossbeam_channel::Sender<TranscriptEvent>,
        stop_rx: crossbeam_channel::Receiver<()>,
    ) -> anyhow::Result<()>;
}

/// Either side of the provider split, as built by
/// [`crate::asr::create_engine`].
pub enum AsrEngine {
    Utterance(Box<dyn UtteranceAsr>),
    Streaming(Box<dyn StreamingAsr>),
}

/// True when the pipeline was asked to stop: the stop signal arrived or the
/// stop sender is gone. Polled from provider loops that multiplex their own
/// async I/O alongside the stop channel.
pub fn should_stop(stop_rx: &crossbeam_channel::Receiver<()>) -> bool {
    !matches!(
        stop_rx.try_recv(),
        Err(crossbeam_channel::TryRecvError::Empty)
    )
}

/// `f32` in `[-1, 1]` to `i16` PCM, matching the resampler output range.
pub fn f32_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * 32767.0) as i16
}

/// Fold one transcript chunk into the in-progress turn buffer.
///
/// Some servers re-send the whole turn with every update, others send deltas:
/// a chunk that extends the buffer replaces it, anything else appends. Either
/// way the buffer ends up holding the full text so far.
pub fn push_transcript_chunk(buffer: &mut String, chunk: &str) {
    if chunk.len() >= buffer.len() && chunk.starts_with(buffer.as_str()) {
        buffer.clear();
        buffer.push_str(chunk);
    } else {
        buffer.push_str(chunk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct BatchOnly;

    impl UtteranceAsr for BatchOnly {
        fn transcribe(
            &mut self,
            _pcm_16k: &[f32],
            _language: Option<&str>,
        ) -> anyhow::Result<String> {
            Ok("hello".to_string())
        }
    }

    #[test]
    fn utterance_default_reports_no_partials() {
        let mut asr = BatchOnly;
        let mut partials = 0;
        let text = asr
            .transcribe_with_partials(&[0.0; 16], None, &mut |_| partials += 1)
            .unwrap();
        assert_eq!(text, "hello");
        assert_eq!(partials, 0);
        assert!(asr.prepare().is_ok());
    }

    #[test]
    fn engine_holds_either_side() {
        let _: AsrEngine = AsrEngine::Utterance(Box::new(BatchOnly));
    }

    #[test]
    fn chunk_folding_handles_cumulative_and_delta_servers() {
        let mut buf = String::new();
        push_transcript_chunk(&mut buf, "hola");
        assert_eq!(buf, "hola");
        // Cumulative re-send replaces.
        push_transcript_chunk(&mut buf, "hola mundo");
        assert_eq!(buf, "hola mundo");
        // Delta appends.
        push_transcript_chunk(&mut buf, " adiós");
        assert_eq!(buf, "hola mundo adiós");
        // Exact repeat is a no-op, not a doubling.
        push_transcript_chunk(&mut buf, "hola mundo adiós");
        assert_eq!(buf, "hola mundo adiós");
    }

    #[test]
    fn f32_to_i16_clamps_and_scales() {
        assert_eq!(f32_to_i16(0.0), 0);
        assert_eq!(f32_to_i16(1.0), 32767);
        assert_eq!(f32_to_i16(-1.0), -32767);
        assert_eq!(f32_to_i16(2.0), 32767);
        assert_eq!(f32_to_i16(-2.0), -32767);
    }
}
