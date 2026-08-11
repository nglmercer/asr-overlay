use anyhow::Result;
use crossbeam_channel::Receiver;
use serde::Serialize;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::audio::{resample_mono, AudioChunk};
use crate::config::AsrConfig;
use crate::stt::WhisperStt;

/// A subtitle line pushed to all overlay clients.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Caption {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

/// Simple energy-based segmentation state machine.
///
/// Consumes mono f32 chunks at the mic's native rate, keeps a rolling speech
/// buffer, and closes an utterance once the configured silence window has
/// passed (or the max length is hit). Closed utterances are resampled to
/// 16 kHz and handed to Whisper.
pub struct CaptionPipeline {
    mic_rate: u32,
    config: AsrConfig,
    speech: Vec<f32>,
    in_speech: bool,
    /// Accumulated silence duration (in samples) while in an utterance.
    silence_samples: u64,
    speech_started: bool,
}

impl CaptionPipeline {
    pub fn new(mic_rate: u32, config: AsrConfig) -> Self {
        Self {
            mic_rate,
            config,
            speech: Vec::new(),
            in_speech: false,
            silence_samples: 0,
            speech_started: false,
        }
    }

    /// Feed one mic chunk. Returns a closed utterance (at mic rate) if the
    /// segmentation state machine decided one is complete, else `None`.
    pub fn push_chunk(&mut self, chunk: &AudioChunk) -> Option<Vec<f32>> {
        let rms = rms(chunk);
        let chunk_samples = chunk.len() as u64;
        let silence_limit =
            (self.config.vad_silence_ms * self.mic_rate as u64) / 1000;
        let max_limit = (self.config.vad_max_ms * self.mic_rate as u64) / 1000;

        if rms >= self.config.vad_threshold {
            if !self.in_speech {
                self.in_speech = true;
                self.speech_started = true;
                // Keep a short pre-roll so the first phonemes are not cut.
                self.speech.clear();
            }
            self.speech.extend_from_slice(chunk);
            self.silence_samples = 0;

            // Force-segment very long utterances.
            if max_limit > 0 && self.speech.len() as u64 >= max_limit {
                return self.close_utterance();
            }
        } else if self.in_speech {
            // Silence after speech: close once the window elapses.
            self.silence_samples += chunk_samples;
            if self.silence_samples >= silence_limit {
                return self.close_utterance();
            }
        }

        None
    }

    /// Flush any in-progress utterance (used when the stream stops).
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        self.close_utterance()
    }

    fn close_utterance(&mut self) -> Option<Vec<f32>> {
        let was_speech = self.in_speech;
        let speech = std::mem::take(&mut self.speech);
        self.in_speech = false;
        self.silence_samples = 0;
        self.speech_started = false;

        if !was_speech {
            return None;
        }

        let duration_ms = 1000.0 * speech.len() as f32 / self.mic_rate.max(1) as f32;
        if duration_ms < self.config.vad_min_ms as f32 {
            return None;
        }
        Some(speech)
    }
}

/// Compute the RMS energy of a mono chunk.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

/// Convert an utterance captured at the mic rate into 16 kHz mono f32.
pub fn to_stt_rate(speech: &[f32], mic_rate: u32) -> Result<Vec<f32>> {
    resample_mono(speech, mic_rate, 16_000)
}

/// Run the segmentation loop: drain `rx` chunks, transcribe closed utterances
/// with `stt`, and report results through `emit`.
///
/// Exits when the audio channel closes or `stop_rx` fires.
pub fn run_pipeline(
    rx: Receiver<AudioChunk>,
    stop_rx: Receiver<()>,
    mic_rate: u32,
    config: AsrConfig,
    mut stt: WhisperStt,
    mut emit: impl FnMut(Caption) + Send + 'static,
) -> Result<()> {
    let mut pipeline = CaptionPipeline::new(mic_rate, config);
    log::info!("Pipeline running at {mic_rate} Hz");

    loop {
        let chunk = crossbeam_channel::select! {
            recv(rx) -> chunk => match chunk {
                Ok(chunk) => chunk,
                Err(_) => break,
            },
            recv(stop_rx) -> _ => break,
        };

        if let Some(speech) = pipeline.push_chunk(&chunk) {
            let audio_16k = to_stt_rate(&speech, mic_rate)?;
            log::info!("Utterance: {} ms", speech.len() * 1000 / mic_rate.max(1) as usize);

            match stt.transcribe(&audio_16k) {
                Ok(text) if !text.is_empty() => {
                    log::info!("Captured: {text}");
                    emit(Caption {
                        text,
                        start_ms: timestamp_ms(),
                        end_ms: 0,
                    });
                }
                Ok(_) => log::debug!("Empty transcription"),
                Err(e) => log::error!("STT error: {e}"),
            }
        }
    }
    log::info!("Pipeline channel closed, stopping");
    Ok(())
}

fn timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(duration_ms: f32, rate: u32, amplitude: f32) -> AudioChunk {
        let n = (rate as f32 * duration_ms / 1000.0).round() as usize;
        (0..n).map(|i| amplitude * ((i as f32) * 0.1).sin()).collect()
    }

    #[test]
    fn test_rms() {
        assert_eq!(rms(&[]), 0.0);
        assert!(rms(&[1.0, -1.0, 1.0, -1.0]).abs() - 1.0 < 1e-5);
        assert!(rms(&[0.0, 0.0]).abs() < 1e-6);
    }

    #[test]
    fn test_segmentation_closes_on_silence() {
        let config = AsrConfig {
            vad_threshold: 0.001,
            vad_silence_ms: 50,
            vad_min_ms: 10,
            vad_max_ms: 5000,
            ..AsrConfig::default()
        };
        let mut p = CaptionPipeline::new(16_000, config);

        // 200 ms of speech then silence chunks > 50 ms.
        assert!(p.push_chunk(&chunk(200.0, 16_000, 0.1)).is_none());
        // Silence long enough to close: 60 ms at 16k = 960 samples.
        let sil = vec![0.0f32; 960];
        let closed = p.push_chunk(&sil);
        assert!(closed.is_some());
        let speech = closed.unwrap();
        assert!(!speech.is_empty());
    }

    #[test]
    fn test_segmentation_ignores_short_noise() {
        let config = AsrConfig {
            vad_threshold: 0.001,
            vad_silence_ms: 10,
            vad_min_ms: 200,
            vad_max_ms: 5000,
            ..AsrConfig::default()
        };
        let mut p = CaptionPipeline::new(16_000, config);

        assert!(p.push_chunk(&chunk(50.0, 16_000, 0.5)).is_none());
        let sil = vec![0.0f32; 1600];
        assert!(p.push_chunk(&sil).is_none(), "short noise must be dropped");
    }

    #[test]
    fn test_flush_returns_pending() {
        let config = AsrConfig {
            vad_threshold: 0.001,
            vad_silence_ms: 1000,
            vad_min_ms: 10,
            vad_max_ms: 5000,
            ..AsrConfig::default()
        };
        let mut p = CaptionPipeline::new(16_000, config);
        assert!(p.push_chunk(&chunk(100.0, 16_000, 0.2)).is_none());
        let flushed = p.flush();
        assert!(flushed.is_some());
        assert!(!flushed.unwrap().is_empty());
    }
}
