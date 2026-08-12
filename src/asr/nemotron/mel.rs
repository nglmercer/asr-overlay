//! NeMo `AudioToMelSpectrogramPreprocessor`, the frontend Nemotron 3.5 was
//! trained with. Different from the Whisper frontend in `asr/mel.rs` in every
//! parameter that matters, so it gets its own extractor:
//!
//! | | Whisper (ARK) | NeMo (Nemotron) |
//! |---|---|---|
//! | `n_fft` | 400 | 512 |
//! | window | periodic Hann, 400 | symmetric Hann, 400, zero-padded to 512 |
//! | pre-emphasis | none | 0.97 |
//! | log | `log10`, clamped to 8 decades, rescaled | `ln(x + 2^-24)`, raw |
//!
//! Values come from `audio_processor_config.json` / `genai_config.json` in the
//! model repo.

use rustfft::{num_complex::Complex32, FftPlanner};

use crate::asr::mel::{reflect_pad, slaney_filters};

pub const SAMPLE_RATE: u32 = 16_000;
pub const N_FFT: usize = 512;
pub const WIN_LENGTH: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
pub const PREEMPHASIS: f32 = 0.97;

/// `log_eps` from `genai_config.json` — 2^-24, NeMo's default
/// `log_zero_guard_value`. `audio_processor_config.json` in the same repo
/// prints 1e-10 instead; the genai value is the one the reference runtime
/// (onnxruntime-genai) actually uses, and it is what the exported model was
/// calibrated against.
pub const LOG_EPS: f32 = 5.960_464_5e-8;

/// Reusable NeMo feature extractor. The filterbank and the FFT plan are the
/// expensive part, so keep one alive for the whole session.
pub struct NemoMel {
    filters: Vec<Vec<f32>>,
    /// Symmetric Hann of `WIN_LENGTH`, already zero-padded to `N_FFT` the way
    /// `torch.stft` pads a short window: centred, `(N_FFT - WIN_LENGTH) / 2`
    /// zeros on each side.
    window: Vec<f32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl Default for NemoMel {
    fn default() -> Self {
        Self::new()
    }
}

impl NemoMel {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        Self {
            filters: slaney_filters(N_FFT, N_MELS, SAMPLE_RATE, 0.0, 8_000.0),
            window: padded_hann(),
            fft: planner.plan_fft_forward(N_FFT),
        }
    }

    /// Frames NeMo reports for `n_samples`: `floor(n / hop) + 1` with
    /// `center=True` (`FilterbankFeatures.get_seq_len`).
    pub fn num_frames(n_samples: usize) -> usize {
        n_samples / HOP_LENGTH + 1
    }

    /// Log-mel features for mono 16 kHz audio, as `[N_MELS * frames]` in
    /// `[mel][frame]` row-major order — the layout the encoder's `audio_signal`
    /// input expects as `[1, 128, frames]`.
    pub fn features(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        if samples.is_empty() {
            return (Vec::new(), 0);
        }
        let frames = Self::num_frames(samples.len());

        // Pre-emphasis first, over the raw signal: y[0] = x[0],
        // y[t] = x[t] - 0.97 * x[t-1].
        let mut emphasized = Vec::with_capacity(samples.len());
        emphasized.push(samples[0]);
        for pair in samples.windows(2) {
            emphasized.push(pair[1] - PREEMPHASIS * pair[0]);
        }

        // center=True: reflect-pad by n_fft / 2 so frame `t` is centred on
        // sample `t * hop`.
        let pad = N_FFT / 2;
        let padded = reflect_pad(&emphasized, pad);

        let mut mel = vec![0.0f32; N_MELS * frames];
        let mut buffer = vec![Complex32::new(0.0, 0.0); N_FFT];
        let mut power = vec![0.0f32; N_FFT / 2 + 1];

        for frame in 0..frames {
            let start = frame * HOP_LENGTH;
            for (i, slot) in buffer.iter_mut().enumerate() {
                let sample = padded.get(start + i).copied().unwrap_or(0.0);
                *slot = Complex32::new(sample * self.window[i], 0.0);
            }
            self.fft.process(&mut buffer);
            for (bin, slot) in power.iter_mut().enumerate() {
                // mag_power = 2.0 → |X|^2.
                *slot = buffer[bin].norm_sqr();
            }
            for (bin, weights) in self.filters.iter().enumerate() {
                let energy = power[bin];
                if energy == 0.0 {
                    continue;
                }
                for (m, &weight) in weights.iter().enumerate() {
                    if weight != 0.0 {
                        mel[m * frames + frame] += weight * energy;
                    }
                }
            }
        }

        // log_zero_guard_type = "add": ln(x + eps), and no normalization
        // ("normalize": "NA"), so the encoder sees the raw log-mel range.
        for value in mel.iter_mut() {
            *value = (*value + LOG_EPS).ln();
        }

        (mel, frames)
    }
}

/// Symmetric (`periodic=False`) Hann of `WIN_LENGTH`, centred inside an
/// `N_FFT`-long buffer of zeros — exactly what `torch.stft` does when
/// `win_length < n_fft`.
fn padded_hann() -> Vec<f32> {
    let mut window = vec![0.0f32; N_FFT];
    let offset = (N_FFT - WIN_LENGTH) / 2;
    for i in 0..WIN_LENGTH {
        let x = std::f64::consts::PI * i as f64 / (WIN_LENGTH - 1) as f64;
        window[offset + i] = x.sin().powi(2) as f32;
    }
    window
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_is_symmetric_and_centred() {
        let window = padded_hann();
        assert_eq!(window.len(), N_FFT);
        let offset = (N_FFT - WIN_LENGTH) / 2;
        // Zero padding on both sides.
        assert!(window[..offset].iter().all(|&w| w == 0.0));
        assert!(window[offset + WIN_LENGTH..].iter().all(|&w| w == 0.0));
        // Symmetric, not periodic: both ends of the window itself are zero.
        assert!(window[offset].abs() < 1e-6);
        assert!(window[offset + WIN_LENGTH - 1].abs() < 1e-6);
        assert!((window[offset + WIN_LENGTH / 2] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn frame_count_matches_nemo() {
        // 560 ms chunk → 56 hops, plus the centre frame at t = 0.
        assert_eq!(NemoMel::num_frames(8_960), 57);
        assert_eq!(NemoMel::num_frames(16_000), 101);
    }

    #[test]
    fn features_have_the_expected_shape_and_are_finite() {
        let mel = NemoMel::new();
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 16_000.0).sin() * 0.5)
            .collect();
        let (features, frames) = mel.features(&samples);
        assert_eq!(frames, 101);
        assert_eq!(features.len(), N_MELS * frames);
        assert!(features.iter().all(|v| v.is_finite()));

        // A 440 Hz tone belongs in a low mel band.
        let band = |m: usize| features[m * frames..(m + 1) * frames].iter().sum::<f32>();
        assert!(band(10) > band(120));
    }

    #[test]
    fn silence_sits_at_the_log_floor() {
        let mel = NemoMel::new();
        let (features, frames) = mel.features(&vec![0.0; 16_000]);
        assert_eq!(frames, 101);
        // ln(2^-24) ≈ -16.6: no normalization, so silence is not zero.
        assert!(features.iter().all(|&v| (v - LOG_EPS.ln()).abs() < 1e-3));
    }

    #[test]
    fn empty_audio_yields_no_frames() {
        let (features, frames) = NemoMel::new().features(&[]);
        assert_eq!(frames, 0);
        assert!(features.is_empty());
    }
}
