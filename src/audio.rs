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

/// A microphone the settings page can offer.
///
/// `id` is what the config stores: either `pw:<node>` for a sound-server source
/// or a cpal device id such as `alsa:pipewire`. Ids are stable across runs,
/// unlike display names — several ALSA entries share one name.
#[derive(Debug, Clone)]
pub struct InputDevice {
    pub id: String,
    pub label: String,
    pub default: bool,
}

/// Prefix marking a PipeWire/PulseAudio source rather than a raw cpal device.
const SOURCE_PREFIX: &str = "pw:";

/// The ALSA PCM that routes through PipeWire. Opening it with `PIPEWIRE_NODE`
/// set captures from that specific source.
const PIPEWIRE_DEVICE_ID: &str = "alsa:pipewire";

/// ALSA's `null` PCM: it "captures" silence, so it is never a useful mic.
const NULL_DEVICE_ID: &str = "alsa:null";

/// Human-readable name of an input device, or a fallback.
fn device_name(device: &cpal::Device) -> String {
    device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

/// Stable id of a device, e.g. `alsa:hw:CARD=0,DEV=0`.
fn device_id(device: &cpal::Device) -> Option<String> {
    device.id().ok().map(|id| id.to_string())
}

/// Ask `pactl` something, with output in the C locale so keys are not
/// translated. `None` when there is no sound server (or no `pactl`).
fn pactl(args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("pactl")
        .args(args)
        .env("LC_ALL", "C")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

/// Capture sources known to PipeWire/PulseAudio, as `(node, description)`.
///
/// These are the devices users actually think of as microphones. The raw ALSA
/// entries for the same hardware are usually unopenable while the sound server
/// holds the card, so routing through the server is what makes selection work.
fn server_sources() -> Vec<(String, String)> {
    match pactl(&["list", "sources"]) {
        Some(listing) => parse_sources(&listing),
        None => Vec::new(),
    }
}

/// Pull `(node, description)` pairs out of `pactl list sources`.
fn parse_sources(listing: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut node: Option<String> = None;
    for line in listing.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("Name: ") {
            node = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("Description: ") {
            if let Some(node) = node.take() {
                // Monitors record what is being played back, not a mic.
                if node.ends_with(".monitor") {
                    continue;
                }
                let description = value.trim();
                let label = if description.is_empty() || description == "(null)" {
                    node.clone()
                } else {
                    description.to_string()
                };
                out.push((node, label));
            }
        }
    }
    out
}

/// The sound server's default capture source, if any.
fn default_server_source() -> Option<String> {
    let name = pactl(&["get-default-source"])?.trim().to_string();
    (!name.is_empty() && name != "@DEFAULT_SOURCE@").then_some(name)
}

/// Every microphone worth offering, default first.
///
/// Only devices that can actually be opened for capture are listed: a host like
/// ALSA advertises dozens of plugin PCMs and busy hardware nodes that fail the
/// moment you try to record from them.
pub fn list_input_devices() -> Vec<InputDevice> {
    let host = cpal::default_host();
    let Ok(devices) = host.input_devices() else {
        return Vec::new();
    };

    let mut probed: Vec<(String, String)> = Vec::new();
    for device in devices {
        let Some(id) = device_id(&device) else {
            continue;
        };
        if id == NULL_DEVICE_ID || probed.iter().any(|(existing, _)| existing == &id) {
            continue;
        }
        if device.default_input_config().is_err() {
            continue;
        }
        probed.push((id, device_name(&device)));
    }

    let mut out = Vec::new();
    let mut has_default = false;

    // Sound-server sources first: on PipeWire/PulseAudio they are the only way
    // to pick a specific microphone.
    if probed.iter().any(|(id, _)| id == PIPEWIRE_DEVICE_ID) {
        let default_source = default_server_source();
        for (node, label) in server_sources() {
            let is_default = Some(&node) == default_source.as_ref();
            has_default |= is_default;
            out.push(InputDevice {
                id: format!("{SOURCE_PREFIX}{node}"),
                label,
                default: is_default,
            });
        }
    }

    let host_default = host.default_input_device().and_then(|d| device_id(&d));
    for (id, label) in probed {
        let is_default = !has_default && Some(&id) == host_default.as_ref();
        has_default |= is_default;
        out.push(InputDevice {
            id,
            label,
            default: is_default,
        });
    }

    out.sort_by_key(|device| (!device.default, device.label.to_lowercase()));
    out
}

/// Map a saved value written before device ids existed onto a current device.
///
/// Old configs stored the display name, which ALSA hands to several devices at
/// once and which the sound server decorates ("… Analog Stereo"). Returns the
/// id to migrate to, or `None` when the value is already an id, is empty, or
/// matches nothing.
pub fn resolve_device_id(saved: &str) -> Option<String> {
    let saved = saved.trim();
    if saved.is_empty() || saved.starts_with(SOURCE_PREFIX) {
        return None;
    }

    let devices = list_input_devices();
    if devices.iter().any(|d| d.id == saved) {
        return None;
    }

    let wanted = saved.to_lowercase();
    devices
        .iter()
        .find(|d| d.label.to_lowercase() == wanted)
        .or_else(|| {
            devices.iter().find(|d| {
                let label = d.label.to_lowercase();
                label.starts_with(&wanted) || wanted.starts_with(&label)
            })
        })
        .map(|d| d.id.clone())
}

/// Devices matching a saved config value, best match first.
///
/// Accepts a cpal device id and, for configs written before ids were stored, a
/// display name — which can match more than one device, so all are returned.
fn matching_devices(host: &cpal::Host, preferred: &str) -> Vec<cpal::Device> {
    let Ok(devices) = host.input_devices() else {
        return Vec::new();
    };
    devices
        .filter(|d| {
            device_id(d).as_deref() == Some(preferred) || device_name(d) == preferred
        })
        .collect()
}

/// Open an input device and start streaming.
///
/// Tries the configured device first, then the host default, then anything that
/// opens: a mic that was unplugged, renamed, or grabbed by another app must not
/// leave the overlay unable to listen at all.
pub fn open_microphone(preferred: &str) -> Result<Microphone> {
    let host = cpal::default_host();
    let preferred = preferred.trim();
    let mut failures: Vec<String> = Vec::new();
    let mut tried: Vec<String> = Vec::new();

    if let Some(node) = preferred.strip_prefix(SOURCE_PREFIX) {
        match open_source(&host, node) {
            Ok(mic) => return Ok(mic),
            Err(e) => {
                log::warn!("Source {node:?} unavailable ({e}); falling back");
                failures.push(format!("{node}: {e}"));
            }
        }
    } else if !preferred.is_empty() {
        for device in matching_devices(&host, preferred) {
            match try_open(&device, &mut tried) {
                Ok(mic) => return Ok(mic),
                Err(e) => {
                    log::warn!("Input device {preferred:?} failed to open ({e})");
                    failures.push(format!("{preferred}: {e}"));
                }
            }
        }
        if failures.is_empty() {
            log::warn!("Input device {preferred:?} not found");
            failures.push(format!("{preferred}: not found"));
        }
        log::warn!("Falling back to another input device");
    }

    if let Some(device) = host.default_input_device() {
        match try_open(&device, &mut tried) {
            Ok(mic) => return Ok(mic),
            Err(e) => failures.push(format!("default device: {e}")),
        }
    }

    if let Ok(devices) = host.input_devices() {
        for device in devices {
            if device_id(&device).as_deref() == Some(NULL_DEVICE_ID) {
                continue;
            }
            match try_open(&device, &mut tried) {
                Ok(mic) => return Ok(mic),
                Err(e) => failures.push(format!("{}: {e}", device_name(&device))),
            }
        }
    }

    Err(anyhow!(
        "No usable input device. Tried: {}",
        if failures.is_empty() {
            "none found".to_string()
        } else {
            failures.join("; ")
        }
    ))
}

/// Capture a specific PipeWire source.
///
/// The PipeWire ALSA plugin reads `PIPEWIRE_NODE` while the PCM is being
/// opened, which is the only handle cpal's ALSA host gives us for picking a
/// source. The variable is removed again as soon as the stream exists.
fn open_source(host: &cpal::Host, node: &str) -> Result<Microphone> {
    let device = host
        .input_devices()
        .ok()
        .and_then(|mut devices| {
            devices.find(|d| device_id(d).as_deref() == Some(PIPEWIRE_DEVICE_ID))
        })
        .ok_or_else(|| anyhow!("PipeWire is not available on this host"))?;

    // PipeWire silently falls back to the default source for an unknown node,
    // so say so rather than letting the wrong mic look like a bug.
    let sources = server_sources();
    if !sources.is_empty() && !sources.iter().any(|(name, _)| name == node) {
        log::warn!("Source {node:?} is not connected; PipeWire will use its default");
    }

    std::env::set_var("PIPEWIRE_NODE", node);
    let result = try_open(&device, &mut Vec::new());
    std::env::remove_var("PIPEWIRE_NODE");
    if result.is_ok() {
        log::info!("Capturing from source: {node}");
    }
    result
}

/// Open one device, skipping it if it was already attempted this call.
fn try_open(device: &cpal::Device, tried: &mut Vec<String>) -> Result<Microphone> {
    if let Some(id) = device_id(device) {
        if tried.contains(&id) {
            return Err(anyhow!("already tried"));
        }
        tried.push(id);
    }

    let default_config = device.default_input_config()?;
    let sample_rate = default_config.sample_rate();
    let channels = default_config.channels();
    let sample_format = default_config.sample_format();

    let stream_config: StreamConfig = default_config.into();
    let (tx, rx) = crossbeam_channel::unbounded::<AudioChunk>();

    let stream = match sample_format {
        SampleFormat::F32 => build_stream::<f32>(device, stream_config, tx, channels)?,
        SampleFormat::I16 => build_stream::<i16>(device, stream_config, tx, channels)?,
        SampleFormat::U16 => build_stream::<u16>(device, stream_config, tx, channels)?,
        SampleFormat::I32 => build_stream::<i32>(device, stream_config, tx, channels)?,
        SampleFormat::U32 => build_stream::<u32>(device, stream_config, tx, channels)?,
        _ => {
            return Err(anyhow!(
                "Unsupported input sample format: {sample_format:?}"
            ))
        }
    };

    stream.play()?;
    log::info!(
        "Input device: {} ({sample_rate} Hz, {channels} ch, {sample_format:?})",
        device_name(device)
    );
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
    fn parses_sources_and_skips_monitors() {
        let listing = "\
Source #51
\tState: RUNNING
\tName: alsa_input.usb-EMEET.analog-stereo
\tDescription: EMEET SmartCam C960 Ultra Analog Stereo
\tDriver: PipeWire
Source #57
\tName: bluez_output.B0.monitor
\tDescription: Monitor of UGREEN Studio Pro
Source #60
\tName: bluez_input.B0
\tDescription: UGREEN Studio Pro
";
        let sources = parse_sources(listing);
        assert_eq!(
            sources,
            vec![
                (
                    "alsa_input.usb-EMEET.analog-stereo".to_string(),
                    "EMEET SmartCam C960 Ultra Analog Stereo".to_string()
                ),
                ("bluez_input.B0".to_string(), "UGREEN Studio Pro".to_string()),
            ]
        );
    }

    /// pactl prints "(null)" for descriptions it cannot encode; the node name
    /// is still a usable label.
    #[test]
    fn parses_sources_without_a_description() {
        let listing = "\tName: alsa_input.thing\n\tDescription: (null)\n";
        assert_eq!(
            parse_sources(listing),
            vec![("alsa_input.thing".to_string(), "alsa_input.thing".to_string())]
        );
    }

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
