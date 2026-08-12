use anyhow::Result;
use crossbeam_channel::Receiver;
use serde::Serialize;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::audio::{resample_mono, AudioChunk};
use crate::config::AsrConfig;
use crate::asr::SpeechModel;

/// A subtitle line pushed to all overlay clients.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Caption {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    /// True while the utterance is still decoding. Consumers that only want
    /// finished lines (a transcript log, say) can skip these; the overlay
    /// shows them so words appear as they are recognized.
    #[serde(default)]
    pub partial: bool,
}

/// Audio kept from before the threshold is crossed, so an utterance starts on
/// the attack of the first word instead of part-way through it.
const PREROLL_MS: u64 = 300;

/// Silence kept at the end of an utterance, for the same reason.
const TAIL_MS: u64 = 300;

/// Once speech has started, it takes half the energy to keep it going.
/// Unvoiced consonants (s, f, t) sit well below the level of a vowel, and
/// without this hysteresis every one of them looks like the end of a sentence.
const CONTINUE_RATIO: f32 = 0.5;

/// Simple energy-based segmentation state machine.
///
/// Consumes mono f32 chunks at the mic's native rate, keeps a rolling speech
/// buffer, and closes an utterance once the configured silence window has
/// passed (or the max length is hit). Closed utterances are resampled to
/// 16 kHz and handed to the model.
///
/// The buffer keeps the pauses *inside* an utterance: an audio LLM transcribes
/// continuous speech, and splicing the quiet parts out leaves it with clipped,
/// glued-together words that it answers with confident nonsense — often in the
/// wrong language.
pub struct CaptionPipeline {
    mic_rate: u32,
    config: AsrConfig,
    speech: Vec<f32>,
    /// Rolling buffer of the audio just before speech was detected.
    preroll: std::collections::VecDeque<f32>,
    in_speech: bool,
    /// Trailing silence (in samples) at the end of the current buffer.
    silence_samples: u64,
    /// How much of the buffer was above the threshold: the utterance is judged
    /// on this, not on its total length, which now includes the pauses.
    voiced_samples: u64,
}

impl CaptionPipeline {
    pub fn new(mic_rate: u32, config: AsrConfig) -> Self {
        Self {
            mic_rate,
            config,
            speech: Vec::new(),
            preroll: std::collections::VecDeque::new(),
            in_speech: false,
            silence_samples: 0,
            voiced_samples: 0,
        }
    }

    /// Samples in `ms` milliseconds at the mic's rate.
    fn samples_for(&self, ms: u64) -> u64 {
        ms * self.mic_rate as u64 / 1000
    }

    /// Feed one mic chunk. Returns a closed utterance (at mic rate) if the
    /// segmentation state machine decided one is complete, else `None`.
    pub fn push_chunk(&mut self, chunk: &AudioChunk) -> Option<Vec<f32>> {
        let rms = rms(chunk);
        let chunk_samples = chunk.len() as u64;
        let silence_limit = self.samples_for(self.config.vad_silence_ms);
        let max_limit = self.samples_for(self.config.vad_max_ms);

        let threshold = if self.in_speech {
            self.config.vad_threshold * CONTINUE_RATIO
        } else {
            self.config.vad_threshold
        };
        let voiced = rms >= threshold;

        if !self.in_speech {
            if !voiced {
                // Remember the last PREROLL_MS of quiet audio.
                self.preroll.extend(chunk.iter().copied());
                let keep = self.samples_for(PREROLL_MS) as usize;
                while self.preroll.len() > keep {
                    self.preroll.pop_front();
                }
                return None;
            }
            self.in_speech = true;
            self.speech.clear();
            self.speech.extend(self.preroll.drain(..));
            self.voiced_samples = 0;
            self.silence_samples = 0;
        }

        // In speech: every chunk is kept, silent or not.
        self.speech.extend_from_slice(chunk);
        if voiced {
            self.voiced_samples += chunk_samples;
            self.silence_samples = 0;
        } else {
            self.silence_samples += chunk_samples;
            if self.silence_samples >= silence_limit {
                return self.close_utterance();
            }
        }

        // Force-segment very long utterances.
        if max_limit > 0 && self.speech.len() as u64 >= max_limit {
            return self.close_utterance();
        }

        None
    }

    /// Flush any in-progress utterance (used when the stream stops).
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        self.close_utterance()
    }

    fn close_utterance(&mut self) -> Option<Vec<f32>> {
        let was_speech = self.in_speech;
        let mut speech = std::mem::take(&mut self.speech);
        let silence = self.silence_samples;
        let voiced = self.voiced_samples;
        self.in_speech = false;
        self.silence_samples = 0;
        self.voiced_samples = 0;
        self.preroll.clear();

        if !was_speech {
            return None;
        }

        // Keep a little of the closing silence; drop the rest.
        let tail = self.samples_for(TAIL_MS);
        if silence > tail {
            let trimmed = speech.len().saturating_sub((silence - tail) as usize);
            speech.truncate(trimmed);
        }

        let voiced_ms = 1000 * voiced / self.mic_rate.max(1) as u64;
        if voiced_ms < self.config.vad_min_ms {
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
/// with `asr`, and report results through `emit`.
///
/// Exits when the audio channel closes or `stop_rx` fires.
pub fn run_pipeline(
    rx: Receiver<AudioChunk>,
    stop_rx: Receiver<()>,
    mic_rate: u32,
    config: AsrConfig,
    mut asr: Box<dyn SpeechModel>,
    mut emit: impl FnMut(Caption) + Send + 'static,
) -> Result<()> {
    let mut pipeline = CaptionPipeline::new(mic_rate, config);
    // Chunk size is fixed by the audio backend; remember it to size the backlog.
    let mut backlog_chunk_samples;
    log::info!("Pipeline running at {mic_rate} Hz");

    loop {
        let chunk = crossbeam_channel::select! {
            recv(rx) -> chunk => match chunk {
                Ok(chunk) => chunk,
                Err(_) => break,
            },
            recv(stop_rx) -> _ => break,
        };

        backlog_chunk_samples = chunk.len();

        if let Some(speech) = pipeline.push_chunk(&chunk) {
            let audio_16k = to_stt_rate(&speech, mic_rate)?;

            // Report the level that actually reaches the model. Silence here
            // means the capture or resample path is broken, not the model — an
            // all-zero buffer transcribes to "" with no error at all.
            let rms = (audio_16k.iter().map(|v| v * v).sum::<f32>()
                / audio_16k.len().max(1) as f32)
                .sqrt();
            log::info!(
                "Utterance: {} ms, {} samples @16k, rms {rms:.4}",
                speech.len() * 1000 / mic_rate.max(1) as usize,
                audio_16k.len()
            );
            if rms <= f32::EPSILON {
                log::warn!("Utterance is silent after resampling — check the input device");
            }

            let started = timestamp_ms();
            let mut showed_partial = false;
            let streamed = asr.transcribe_streaming(&audio_16k, &mut |partial: &str| {
                if !partial.is_empty() {
                    showed_partial = true;
                    emit(Caption {
                        text: partial.to_string(),
                        start_ms: started,
                        end_ms: 0,
                        partial: true,
                    });
                }
            });

            match streamed {
                Ok(text) if !text.is_empty() => {
                    log::info!("Captured: {text}");
                    emit(Caption {
                        text,
                        start_ms: started,
                        end_ms: timestamp_ms(),
                        partial: false,
                    });
                }
                Ok(_) => {
                    log::info!("No speech recognized in this utterance");
                    // Partial text is on screen; clear it rather than leaving
                    // a line the model just decided against.
                    if showed_partial {
                        emit(Caption {
                            text: String::new(),
                            start_ms: started,
                            end_ms: timestamp_ms(),
                            partial: false,
                        });
                    }
                }
                Err(e) => log::error!("ASR error: {e}"),
            }

            drop_backlog(&rx, mic_rate, backlog_chunk_samples);
        }
    }
    log::info!("Pipeline channel closed, stopping");
    Ok(())
}

/// Audio allowed to queue up behind the model before it is thrown away.
const MAX_BACKLOG_MS: usize = 10_000;

/// Drop queued microphone audio when transcription has fallen too far behind.
///
/// Transcription is slower than real time on a busy CPU, and the capture
/// channel is unbounded, so a long stretch of speech would otherwise push the
/// captions further and further into the past with no way to recover. A live
/// overlay is better off skipping audio than showing what was said a minute
/// ago.
fn drop_backlog(rx: &Receiver<AudioChunk>, mic_rate: u32, chunk_samples: usize) {
    if chunk_samples == 0 || mic_rate == 0 {
        return;
    }
    let max_chunks = (MAX_BACKLOG_MS * mic_rate as usize / 1000) / chunk_samples;
    if rx.len() <= max_chunks.max(1) {
        return;
    }

    let mut dropped = 0usize;
    while let Ok(chunk) = rx.try_recv() {
        dropped += chunk.len();
    }
    log::warn!(
        "Transcription fell behind: dropped {:.1}s of audio to catch up",
        dropped as f32 / mic_rate as f32
    );
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

    /// The bug this replaced: quiet chunks inside an utterance were thrown
    /// away, so a sentence with a pause in it reached the model as two
    /// half-words spliced together — which it answered with fluent nonsense,
    /// sometimes in another language entirely.
    #[test]
    fn pauses_inside_an_utterance_are_kept() {
        let config = AsrConfig {
            vad_threshold: 0.01,
            vad_silence_ms: 600,
            vad_min_ms: 100,
            vad_max_ms: 30_000,
            ..AsrConfig::default()
        };
        let mut p = CaptionPipeline::new(16_000, config);

        // 200 ms speech, 300 ms pause (under the 600 ms window), 200 ms speech.
        assert!(p.push_chunk(&chunk(200.0, 16_000, 0.5)).is_none());
        assert!(p.push_chunk(&vec![0.0; 4800]).is_none());
        assert!(p.push_chunk(&chunk(200.0, 16_000, 0.5)).is_none());
        // Now close it with a full silence window.
        let closed = p.push_chunk(&vec![0.0; 9600]).expect("utterance");

        // 200 speech + the 300 ms pause + 200 speech + 300 ms of kept tail.
        // (Nothing preceded the first word here, so there is no pre-roll.)
        let ms = closed.len() * 1000 / 16_000;
        assert!((990..=1010).contains(&ms), "kept {ms} ms");
    }

    #[test]
    fn speech_starts_with_the_preroll() {
        let config = AsrConfig {
            vad_threshold: 0.01,
            vad_silence_ms: 100,
            vad_min_ms: 10,
            vad_max_ms: 30_000,
            ..AsrConfig::default()
        };
        let mut p = CaptionPipeline::new(16_000, config);

        // A second of near-silence before the first word.
        for _ in 0..10 {
            assert!(p.push_chunk(&vec![0.0; 1600]).is_none());
        }
        assert!(p.push_chunk(&chunk(100.0, 16_000, 0.5)).is_none());
        let closed = p.push_chunk(&vec![0.0; 3200]).expect("utterance");

        // Only PREROLL_MS of the quiet run is kept, not all of it.
        let ms = closed.len() * 1000 / 16_000;
        assert!((590..=610).contains(&ms), "kept {ms} ms");
    }

    /// Quiet consonants must not end a sentence: once speech is running, half
    /// the threshold keeps it running.
    #[test]
    fn quiet_speech_does_not_close_the_utterance() {
        let config = AsrConfig {
            vad_threshold: 0.1,
            vad_silence_ms: 200,
            vad_min_ms: 10,
            vad_max_ms: 30_000,
            ..AsrConfig::default()
        };
        let mut p = CaptionPipeline::new(16_000, config);

        assert!(p.push_chunk(&chunk(200.0, 16_000, 0.5)).is_none());
        // 300 ms at 60% of the start threshold: above the continue threshold,
        // so it must not close even though the silence window is only 200 ms.
        for _ in 0..3 {
            assert!(p.push_chunk(&chunk(100.0, 16_000, 0.06 * 1.5)).is_none());
        }
        assert!(p.push_chunk(&vec![0.0; 4800]).is_some());
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
