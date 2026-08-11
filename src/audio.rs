use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, SizedSample, StreamConfig};
use rubato::{FftFixedOut, Resampler};

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

/// Convert a mono `f32` buffer from one sample rate to another.
///
/// Uses a synchronous FFT resampler (rubato) configured with a chunk output
/// sized to the input, so whole-buffer conversions stay simple.
pub fn resample_mono(input: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(input.to_vec());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }
    if from_rate == 0 || to_rate == 0 {
        return Err(anyhow!("Invalid sample rate: {from_rate} -> {to_rate}"));
    }

    // Size the output chunk so one call consumes the whole input.
    let chunk_size_out = ((input.len() as f64) * (to_rate as f64) / (from_rate as f64)) as usize;
    let chunk_size_out = chunk_size_out.max(1);

    let mut resampler = FftFixedOut::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        chunk_size_out,
        1,
        1,
    )?;

    let frames_needed = resampler.input_frames_next();
    let mut wave_in = vec![input.to_vec()];
    // The FFT resampler wants at least `frames_needed` input frames; zero-pad
    // short buffers, then trim output to the expected length.
    if wave_in[0].len() < frames_needed {
        wave_in[0].resize(frames_needed, 0.0);
    }

    let mut output = resampler.process(&wave_in, None)?;
    let mut samples = output.remove(0);
    samples.truncate(chunk_size_out);
    Ok(samples)
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

    #[test]
    fn test_resample_empty() {
        assert!(resample_mono(&[], 48000, 16000).unwrap().is_empty());
    }
}
