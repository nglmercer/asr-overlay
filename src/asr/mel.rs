//! Whisper-style log-mel frontend, ported from HuggingFace's
//! `WhisperFeatureExtractor` + `transformers.audio_utils`.
//!
//! Parameters come from ARK-ASR's `preprocessor_config.json`:
//! `feature_size=128`, `n_fft=400`, `hop_length=160`, `sampling_rate=16000`.

use rustfft::{num_complex::Complex32, FftPlanner};

pub const SAMPLE_RATE: u32 = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
/// 30 s at 16 kHz — the encoder's maximum input.
pub const MAX_SAMPLES: usize = 480_000;

/// Slaney mel scale (`mel_scale="slaney"` in `mel_filter_bank`).
fn hz_to_mel(freq: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP; // 15.0
    let logstep = 6.4f64.ln() / 27.0;
    if freq >= MIN_LOG_HZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() / logstep
    } else {
        freq / F_SP
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    } else {
        F_SP * mel
    }
}

/// Triangular mel filterbank with Slaney area normalization.
/// Returns `[num_freq_bins][N_MELS]`, matching
/// `mel_filter_bank(201, 128, 0.0, 8000.0, 16000, norm="slaney", mel_scale="slaney")`.
pub fn mel_filters() -> Vec<Vec<f32>> {
    let num_freq_bins = N_FFT / 2 + 1; // 201
    let nyquist = SAMPLE_RATE as f64 / 2.0;

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(nyquist);
    // N_MELS + 2 evenly spaced points on the mel axis.
    let filter_freqs: Vec<f64> = (0..N_MELS + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f64 / (N_MELS + 1) as f64;
            mel_to_hz(mel)
        })
        .collect();

    let fft_freqs: Vec<f64> = (0..num_freq_bins)
        .map(|i| nyquist * i as f64 / (num_freq_bins - 1) as f64)
        .collect();

    let filter_diff: Vec<f64> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();

    let mut bank = vec![vec![0.0f32; N_MELS]; num_freq_bins];
    for (bin, &fft_freq) in fft_freqs.iter().enumerate() {
        for mel in 0..N_MELS {
            // slopes[j] = filter_freqs[j] - fft_freq
            let down = -(filter_freqs[mel] - fft_freq) / filter_diff[mel];
            let up = (filter_freqs[mel + 2] - fft_freq) / filter_diff[mel + 1];
            let value = down.min(up).max(0.0);
            // Slaney normalization: 2 / (f[i+2] - f[i])
            let enorm = 2.0 / (filter_freqs[mel + 2] - filter_freqs[mel]);
            bank[bin][mel] = (value * enorm) as f32;
        }
    }
    bank
}

/// Periodic Hann window of length `N_FFT`, as used by
/// `window_function(n_fft, "hann")`.
fn hann_window() -> Vec<f32> {
    (0..N_FFT)
        .map(|i| {
            let x = std::f64::consts::PI * i as f64 / N_FFT as f64;
            (x.sin().powi(2)) as f32
        })
        .collect()
}

/// Reusable feature extractor. Building the filterbank and FFT plan is the
/// expensive part, so keep one instance alive across utterances.
pub struct MelExtractor {
    filters: Vec<Vec<f32>>,
    window: Vec<f32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl Default for MelExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl MelExtractor {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        Self {
            filters: mel_filters(),
            window: hann_window(),
            fft: planner.plan_fft_forward(N_FFT),
        }
    }

    /// Number of mel frames the extractor will emit for `n_samples`.
    /// Matches `_calculate_audio_token_counts_per_sample`: `len // hop_length`.
    pub fn num_frames(n_samples: usize) -> usize {
        n_samples.min(MAX_SAMPLES) / HOP_LENGTH
    }

    /// Number of `<|audio|>` placeholder tokens for a given sample count.
    /// Ported from `ArkasrProcessor.calculate_audio_token_count`:
    /// `max(((mel_frames + 1) / 2) / merge_factor, 1)`.
    pub fn num_audio_tokens(n_samples: usize, merge_factor: usize) -> usize {
        let mel_frames = Self::num_frames(n_samples);
        let downsampled = mel_frames.div_ceil(2);
        (downsampled / merge_factor.max(1)).max(1)
    }

    /// Compute log-mel features for mono 16 kHz audio.
    ///
    /// Returns `[N_MELS * frames]` in row-major `[mel][frame]` order — the
    /// layout the encoder's `audios` input expects as `[1, 128, frames]`.
    pub fn features(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let samples = &samples[..samples.len().min(MAX_SAMPLES)];
        let frames = Self::num_frames(samples.len());
        if frames == 0 {
            return (Vec::new(), 0);
        }

        // center=True with reflect padding: n_fft/2 on each side.
        let pad = N_FFT / 2;
        let padded = reflect_pad(samples, pad);

        let mut power = vec![0.0f32; (N_FFT / 2 + 1) * frames];
        let mut buffer = vec![Complex32::new(0.0, 0.0); N_FFT];

        for frame in 0..frames {
            let start = frame * HOP_LENGTH;
            for (i, slot) in buffer.iter_mut().enumerate() {
                let sample = padded.get(start + i).copied().unwrap_or(0.0);
                *slot = Complex32::new(sample * self.window[i], 0.0);
            }
            self.fft.process(&mut buffer);
            for bin in 0..N_FFT / 2 + 1 {
                // power = |X|^2
                power[bin * frames + frame] = buffer[bin].norm_sqr();
            }
        }

        // mel projection: [128, frames] = filters^T @ power
        let mut mel = vec![0.0f32; N_MELS * frames];
        for (bin, weights) in self.filters.iter().enumerate() {
            for (m, &weight) in weights.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                let src = &power[bin * frames..(bin + 1) * frames];
                let dst = &mut mel[m * frames..(m + 1) * frames];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += weight * s;
                }
            }
        }

        // log10, clamp to a 8-decade dynamic range, then scale to ~[-1, 1].
        let mut max_log = f32::MIN;
        for value in mel.iter_mut() {
            *value = value.max(1e-10).log10();
            if *value > max_log {
                max_log = *value;
            }
        }
        let floor = max_log - 8.0;
        for value in mel.iter_mut() {
            *value = (value.max(floor) + 4.0) / 4.0;
        }

        (mel, frames)
    }
}

/// Symmetric reflect padding (`np.pad(mode="reflect")`).
fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    if samples.is_empty() {
        return vec![0.0; pad * 2];
    }
    let mut out = Vec::with_capacity(samples.len() + 2 * pad);
    for i in (1..=pad).rev() {
        out.push(samples[i.min(samples.len() - 1)]);
    }
    out.extend_from_slice(samples);
    let last = samples.len() - 1;
    for i in 1..=pad {
        let idx = last.saturating_sub(i);
        out.push(samples[idx]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_scale_roundtrips() {
        for hz in [0.0, 100.0, 999.9, 1000.0, 4000.0, 8000.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((back - hz).abs() < 1e-6, "{hz} → {back}");
        }
    }

    #[test]
    fn mel_breakpoint_is_continuous() {
        // The slaney scale switches from linear to log at 1000 Hz.
        let below = hz_to_mel(999.999);
        let above = hz_to_mel(1000.001);
        assert!((above - below).abs() < 1e-4);
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn filterbank_has_the_documented_shape() {
        let filters = mel_filters();
        assert_eq!(filters.len(), N_FFT / 2 + 1);
        assert_eq!(filters[0].len(), N_MELS);
        // Every mel filter must have some energy.
        for mel in 0..N_MELS {
            let sum: f32 = filters.iter().map(|bin| bin[mel]).sum();
            assert!(sum > 0.0, "mel filter {mel} is empty");
        }
        // No negative weights.
        assert!(filters.iter().all(|bin| bin.iter().all(|&w| w >= 0.0)));
    }

    #[test]
    fn hann_window_is_periodic() {
        let w = hann_window();
        assert_eq!(w.len(), N_FFT);
        assert!(w[0].abs() < 1e-6);
        // Periodic (not symmetric): the last sample is not zero.
        assert!(w[N_FFT - 1] > 0.0);
        // Peak at the centre.
        assert!((w[N_FFT / 2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn frame_and_token_counts_match_the_processor() {
        // 1 s of audio → 100 mel frames → (100+1)/2 = 50 → 50/4 = 12 tokens.
        assert_eq!(MelExtractor::num_frames(16_000), 100);
        assert_eq!(MelExtractor::num_audio_tokens(16_000, 4), 12);
        // Very short audio still gets at least one token.
        assert_eq!(MelExtractor::num_audio_tokens(100, 4), 1);
        // Longer than 30 s is truncated to the encoder window.
        assert_eq!(MelExtractor::num_frames(16_000 * 60), 3000);
    }

    #[test]
    fn features_have_the_expected_shape_and_range() {
        let extractor = MelExtractor::new();
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 16_000.0).sin() * 0.5)
            .collect();
        let (mel, frames) = extractor.features(&samples);
        assert_eq!(frames, 100);
        assert_eq!(mel.len(), N_MELS * frames);
        assert!(mel.iter().all(|v| v.is_finite()));
        // The floor is `max - 8` decades and everything is divided by 4, so the
        // spread is exactly 2.0 regardless of how loud the input is.
        let max = mel.iter().cloned().fold(f32::MIN, f32::max);
        let min = mel.iter().cloned().fold(f32::MAX, f32::min);
        assert!((max - min - 2.0).abs() < 1e-5, "span was {}", max - min);
        // A 440 Hz tone must put its peak in a low mel band, not a high one.
        let band_energy = |m: usize| mel[m * frames..(m + 1) * frames].iter().sum::<f32>();
        assert!(band_energy(10) > band_energy(120));
    }

    #[test]
    fn empty_audio_yields_no_frames() {
        let extractor = MelExtractor::new();
        let (mel, frames) = extractor.features(&[]);
        assert_eq!(frames, 0);
        assert!(mel.is_empty());
    }

    #[test]
    fn reflect_pad_mirrors_without_repeating_the_edge() {
        let out = reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2);
        assert_eq!(out, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }
}
