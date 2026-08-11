use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, SizedSample, StreamConfig};

/// A mono `f32` audio chunk delivered off the audio callback thread.
pub type AudioChunk = Vec<f32>;

/// Captures the default input device and forwards mono `f32` chunks on a
/// crossbeam channel. Never runs inference in the callback: it only converts
/// and copies samples, then sends them out.
pub struct Microphone {
    pub sample_rate: u32,
    pub rx: crossbeam_channel::Receiver<AudioChunk>,
    #[allow(dead_code)]
    stream: cpal::Stream,
}

/// Open the default input device and start streaming.
///
/// `chunk_samples` controls how many mono samples each channel message
/// carries (a sensible default is one CPAL buffer, whatever the device chose).
pub fn open_microphone() -> Result<Microphone> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("No default input device found"))?;

    let device_name = device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "unknown".into());
    log::info!("Input device: {device_name}");

    let default_config = device.default_input_config()?;
    let sample_rate = default_config.sample_rate();
    let channels = default_config.channels();
    let sample_format = default_config.sample_format();
    log::info!("Input config: {sample_rate} Hz, {channels} ch, {sample_format:?}");

    let stream_config: StreamConfig = default_config.into();
    let (tx, rx) = crossbeam_channel::unbounded::<AudioChunk>();

    let stream = match sample_format {
        SampleFormat::F32 => build_stream::<f32>(&device, stream_config, tx, channels)?,
        SampleFormat::I16 => build_stream::<i16>(&device, stream_config, tx, channels)?,
        SampleFormat::U16 => build_stream::<u16>(&device, stream_config, tx, channels)?,
        SampleFormat::I32 => build_stream::<i32>(&device, stream_config, tx, channels)?,
        SampleFormat::U32 => build_stream::<u32>(&device, stream_config, tx, channels)?,
        _ => {
            return Err(anyhow!(
                "Unsupported input sample format: {sample_format:?}"
            ))
        }
    };

    stream.play()?;
    Ok(Microphone {
        sample_rate,
        rx,
        stream,
    })
}

fn build_stream<T>(
    device: &cpal::Device,
    config: StreamConfig,
    tx: crossbeam_channel::Sender<AudioChunk>,
    channels: u16,
) -> Result<cpal::Stream>
where
    T: SizedSample + 'static,
    T::Float: Into<f32>,
{
    let err_fn = |err| log::error!("Audio stream error: {err}");
    let stream = device.build_input_stream::<T, _, _>(
        config,
        move |data: &[T], _| {
            let mono = to_mono_f32(data, channels);
            let _ = tx.send(mono);
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

/// Average an interleaved sample buffer down to mono `f32`.
fn to_mono_f32<T>(data: &[T], channels: u16) -> AudioChunk
where
    T: Sample + Copy,
    T::Float: Into<f32>,
{
    let channels = channels.max(1) as usize;
    let mut out = Vec::with_capacity(data.len() / channels);
    for frame in data.chunks(channels) {
        let sum: f32 = frame
            .iter()
            .map(|&s| s.to_float_sample().into())
            .sum::<f32>();
        out.push(sum / channels as f32);
    }
    out
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let pi_x = std::f64::consts::PI * x;
        pi_x.sin() / pi_x
    }
}

/// Blackman window over `[-half, half]`, zero outside.
fn blackman(distance: f64, half: f64) -> f64 {
    if distance.abs() > half {
        return 0.0;
    }
    let t = (distance + half) / (2.0 * half);
    0.42 - 0.5 * (std::f64::consts::TAU * t).cos() + 0.08 * (2.0 * std::f64::consts::TAU * t).cos()
}

/// Convert a mono `f32` buffer from one sample rate to another.
///
/// Windowed-sinc interpolation with an anti-aliasing cutoff at the lower of the
/// two Nyquist limits. Coefficients are normalized per output sample, so the
/// signal level is preserved exactly and there is no warm-up delay — a
/// whole-buffer conversion produces a whole buffer of audio.
pub fn resample_mono(input: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == 0 || to_rate == 0 {
        return Err(anyhow!("Invalid sample rate: {from_rate} -> {to_rate}"));
    }
    if from_rate == to_rate {
        return Ok(input.to_vec());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    if out_len == 0 {
        return Ok(Vec::new());
    }

    // Cutoff relative to the input sample rate. When downsampling this sits at
    // the *output* Nyquist so nothing aliases back into the band.
    let cutoff = 0.5 * ratio.min(1.0);
    // Downsampling stretches the kernel, so widen it to keep the same quality.
    let half = (16.0 / ratio.min(1.0)).ceil() as isize;

    let last = input.len() as isize - 1;
    let mut output = Vec::with_capacity(out_len);

    for j in 0..out_len {
        // Position of this output sample, in input-sample units.
        let center = j as f64 / ratio;
        let base = center.floor() as isize;

        let mut acc = 0.0f64;
        let mut norm = 0.0f64;
        for i in (base - half)..=(base + half + 1) {
            let distance = center - i as f64;
            let window = blackman(distance, half as f64);
            if window == 0.0 {
                continue;
            }
            let coefficient = window * sinc(2.0 * cutoff * distance);
            norm += coefficient;
            // Clamp to the edges rather than reading zeros past the end, which
            // would fade the first and last samples toward silence.
            let index = i.clamp(0, last) as usize;
            acc += coefficient * input[index] as f64;
        }

        output.push(if norm.abs() > 1e-12 {
            (acc / norm) as f32
        } else {
            0.0
        });
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_mono_f32_avg() {
        let data: Vec<f32> = vec![1.0, 3.0, 0.5, 0.5];
        let mono = to_mono_f32(&data, 2);
        assert_eq!(mono, vec![2.0, 0.5]);
    }

    #[test]
    fn test_to_mono_f32_i16() {
        let data: Vec<i16> = vec![0, 32767, -32768, 0];
        let mono = to_mono_f32(&data, 2);
        assert!((mono[0] - 0.5).abs() < 0.001);
        assert!((mono[1] + 0.5).abs() < 0.001);
    }

    #[test]
    fn test_resample_same_rate() {
        let input = vec![0.0, 0.1, 0.2];
        let out = resample_mono(&input, 44100, 44100).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn test_resample_downsample_length() {
        // 48000 Hz -> 16000 Hz = 3x fewer samples.
        let input: Vec<f32> = (0..4800).map(|i| (i as f32 / 10.0).sin()).collect();
        let out = resample_mono(&input, 48000, 16000).unwrap();
        assert!(out.len() >= 1500 && out.len() <= 1601, "got {}", out.len());
    }

    /// Length alone says nothing about whether the audio survived. A 440 Hz
    /// tone must still be a 440 Hz tone, at the same level, after 48k -> 16k.
    #[test]
    fn test_resample_preserves_the_signal() {
        let rate = 48_000;
        let freq = 440.0_f32;
        let input: Vec<f32> = (0..rate)
            .map(|i| (i as f32 * freq * std::f32::consts::TAU / rate as f32).sin() * 0.5)
            .collect();

        let out = resample_mono(&input, rate as u32, 16_000).unwrap();
        assert!(out.len() >= 15_000, "too short: {}", out.len());

        let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let in_rms = rms(&input);
        let out_rms = rms(&out[100..out.len() - 100]);
        assert!(
            (out_rms - in_rms).abs() < 0.05,
            "level changed: {in_rms} -> {out_rms}"
        );

        // Count zero crossings to recover the frequency: 440 Hz over ~1 s is
        // ~880 crossings regardless of sample rate.
        let crossings = out
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count();
        let seconds = out.len() as f32 / 16_000.0;
        let measured = crossings as f32 / 2.0 / seconds;
        assert!(
            (measured - freq).abs() < 20.0,
            "frequency changed: {measured} Hz"
        );
    }

    /// The bug this replaced: 48k -> 16k over a long buffer returned pure
    /// silence, so every microphone utterance reached the model as zeros.
    #[test]
    fn test_resample_long_buffer_is_not_silent() {
        let input: Vec<f32> = (0..528_000)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5)
            .collect();
        let out = resample_mono(&input, 48_000, 16_000).unwrap();
        assert_eq!(out.len(), 176_000);
        let peak = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak > 0.4, "output is silent: peak {peak}");
    }

    #[test]
    fn test_resample_roundtrip_preserves_audio() {
        let input: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 300.0 * std::f32::consts::TAU / 16_000.0).sin() * 0.5)
            .collect();
        let up = resample_mono(&input, 16_000, 48_000).unwrap();
        let back = resample_mono(&up, 48_000, 16_000).unwrap();
        assert_eq!(back.len(), input.len());

        // Ignore the outermost samples, where the kernel runs off the edge.
        let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let a = rms(&input[200..input.len() - 200]);
        let b = rms(&back[200..back.len() - 200]);
        assert!((a - b).abs() < 0.02, "level drifted: {a} -> {b}");
    }

    #[test]
    fn test_resample_empty() {
        assert!(resample_mono(&[], 48000, 16000).unwrap().is_empty());
    }
}
