//! Developer tool: inspect the ONNX graph signatures and transcribe a WAV.
//!
//!     cargo run --bin asr-tool -- inspect
//!     cargo run --bin asr-tool -- transcribe audio.wav

use anyhow::{anyhow, Result};
use ort::session::Session;

fn main() -> Result<()> {
    asr_overlay::logger::init();
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("inspect") => inspect(),
        Some("transcribe") => {
            let path = args.next().ok_or_else(|| anyhow!("usage: transcribe <file.wav>"))?;
            transcribe(&path, args.next())
        }
        Some("micpath") => {
            let path = args.next().ok_or_else(|| anyhow!("usage: micpath <file.wav>"))?;
            micpath(&path)
        }
        Some("vad") => {
            let path = args.next().ok_or_else(|| anyhow!("usage: vad <file.wav> [gain]"))?;
            let gain = args.next().and_then(|g| g.parse().ok()).unwrap_or(1.0);
            vad(&path, gain)
        }
        _ => Err(anyhow!(
            "usage: asr-tool [inspect | transcribe <file.wav> [instruction] \
             | micpath <file.wav> | vad <file.wav> [gain]]"
        )),
    }
}

fn inspect() -> Result<()> {
    let dir = asr_overlay::hub::repo_dir(&asr_overlay::asr::ARK_ASR);
    for file in [
        "audio_encoder_whisper_int8.onnx",
        "audio_encoder_adapter_int8.onnx",
        "embedding_fp32.onnx",
        "llm_kv_cpu_fp32_int8.onnx",
    ] {
        let path = dir.join(file);
        println!("\n=== {file}");
        let session = Session::builder()
            .map_err(|e| anyhow!("{e}"))?
            .commit_from_file(&path)
            .map_err(|e| anyhow!("{e}"))?;
        // The LLM has 51 inputs and 49 outputs; only show the non-cache ones.
        for input in session.inputs().iter().take(6) {
            println!("  in  {:<24} {:?}", input.name(), input.dtype());
        }
        if session.inputs().len() > 6 {
            println!("  in  ... {} inputs total", session.inputs().len());
        }
        for output in session.outputs().iter().take(3) {
            println!("  out {:<24} {:?}", output.name(), output.dtype());
        }
        if session.outputs().len() > 3 {
            println!("  out ... {} outputs total", session.outputs().len());
        }
    }
    Ok(())
}

fn transcribe(path: &str, instruction: Option<String>) -> Result<()> {
    let samples = read_wav_16k_mono(path)?;
    println!(
        "Loaded {} samples ({:.2}s)",
        samples.len(),
        samples.len() as f32 / 16_000.0
    );

    let started = std::time::Instant::now();
    let mut asr = asr_overlay::asr::load(5, 256)?;
    if let Some(instruction) = &instruction {
        println!("Instruction: {instruction}");
        asr.set_instruction(instruction);
    }
    println!("Model loaded in {:.1}s", started.elapsed().as_secs_f32());

    let started = std::time::Instant::now();
    let text = asr.transcribe(&samples)?;
    let elapsed = started.elapsed().as_secs_f32();
    println!("\nTranscript: {text}");
    println!(
        "\nTook {elapsed:.2}s for {:.2}s of audio ({:.2}x realtime)",
        samples.len() as f32 / 16_000.0,
        (samples.len() as f32 / 16_000.0) / elapsed
    );
    Ok(())
}

/// Push a WAV through the live capture path — upsampled to a 48 kHz mic, split
/// into callback-sized chunks, segmented by the real VAD — and transcribe each
/// utterance the way the overlay would.
///
/// `gain` scales the input: a headset at 0.15 of full scale is a very different
/// test from a studio recording, and segmentation is where that shows up. If
/// `transcribe` is right and this is wrong, the segmentation is to blame.
fn vad(path: &str, gain: f32) -> Result<()> {
    let samples: Vec<f32> = read_wav_16k_mono(path)?.iter().map(|s| s * gain).collect();
    let mic = asr_overlay::audio::resample_mono(&samples, 16_000, 48_000)?;

    let config = asr_overlay::config::load();
    println!(
        "{:.1}s at gain {gain} — threshold {}, silence {} ms, min {} ms",
        mic.len() as f32 / 48_000.0,
        config.vad_threshold,
        config.vad_silence_ms,
        config.vad_min_ms
    );

    let mut pipeline = asr_overlay::caption::CaptionPipeline::new(48_000, config);
    let mut utterances = Vec::new();
    // 10 ms chunks, the size a cpal callback typically delivers.
    for chunk in mic.chunks(480) {
        if let Some(speech) = pipeline.push_chunk(&chunk.to_vec()) {
            utterances.push(speech);
        }
    }
    if let Some(speech) = pipeline.flush() {
        utterances.push(speech);
    }

    let mut asr = asr_overlay::asr::load(5, 256)?;
    println!("{} utterances", utterances.len());
    for (i, speech) in utterances.iter().enumerate() {
        let audio = asr_overlay::caption::to_stt_rate(speech, 48_000)?;
        let rms = (audio.iter().map(|v| v * v).sum::<f32>() / audio.len().max(1) as f32).sqrt();
        println!(
            "[{i}] {:.2}s rms {rms:.4}: {:?}",
            audio.len() as f32 / 16_000.0,
            asr.transcribe(&audio)?
        );
    }
    Ok(())
}

/// Minimal 16-bit / 32-bit float PCM WAV reader, downmixed to mono and
/// resampled to 16 kHz.
fn read_wav_16k_mono(path: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(anyhow!("Not a RIFF/WAVE file"));
    }

    let mut pos = 12;
    let mut channels = 1u16;
    let mut rate = 16_000u32;
    let mut bits = 16u16;
    let mut format = 1u16;
    let mut data: &[u8] = &[];

    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
        let body = &bytes[pos + 8..(pos + 8 + size).min(bytes.len())];
        match id {
            b"fmt " if body.len() >= 16 => {
                format = u16::from_le_bytes(body[0..2].try_into()?);
                channels = u16::from_le_bytes(body[2..4].try_into()?);
                rate = u32::from_le_bytes(body[4..8].try_into()?);
                bits = u16::from_le_bytes(body[14..16].try_into()?);
            }
            b"data" => data = body,
            _ => {}
        }
        pos += 8 + size + (size & 1);
    }

    let interleaved: Vec<f32> = match (format, bits) {
        (1, 16) => data
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect(),
        (3, 32) => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        _ => return Err(anyhow!("Unsupported WAV format {format} / {bits} bits")),
    };

    let mono: Vec<f32> = if channels <= 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    if rate == 16_000 {
        Ok(mono)
    } else {
        asr_overlay::audio::resample_mono(&mono, rate, 16_000)
    }
}

/// Push a 16 kHz file through the exact conversion the microphone path uses
/// (upsample to the mic's 48 kHz, then back down) and transcribe the result.
/// If this differs from `transcribe`, the resampler is to blame, not the model.
fn micpath(path: &str) -> Result<()> {
    let original = read_wav_16k_mono(path)?;
    let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt();
    let peak = |x: &[f32]| x.iter().fold(0.0f32, |m, v| m.max(v.abs()));

    let at_48k = asr_overlay::audio::resample_mono(&original, 16_000, 48_000)?;
    let back = asr_overlay::audio::resample_mono(&at_48k, 48_000, 16_000)?;

    println!("original : {:6} samples  rms {:.4}  peak {:.4}", original.len(), rms(&original), peak(&original));
    println!("@48k     : {:6} samples  rms {:.4}  peak {:.4}", at_48k.len(), rms(&at_48k), peak(&at_48k));
    println!("back@16k : {:6} samples  rms {:.4}  peak {:.4}", back.len(), rms(&back), peak(&back));

    // How different are they really?
    let n = original.len().min(back.len());
    let mut max_diff = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (a, b) = (original[i], back[i]);
        max_diff = max_diff.max((a - b).abs());
        sum_sq_diff += ((a - b) as f64).powi(2);
        dot += a as f64 * b as f64;
        na += (a as f64).powi(2);
        nb += (b as f64).powi(2);
    }
    println!(
        "max|diff| {:.5}  rms(diff) {:.5}  correlation {:.6}",
        max_diff,
        (sum_sq_diff / n as f64).sqrt(),
        dot / (na.sqrt() * nb.sqrt())
    );

    // Compare what the model actually sees.
    let mel = asr_overlay::asr::MelExtractor::new();
    let (fa, frames_a) = mel.features(&original);
    let (fb, frames_b) = mel.features(&back);
    let mel_diff = fa.iter().zip(&fb).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("mel frames {frames_a} vs {frames_b}, max mel diff {mel_diff:.5}");

    let mut asr = asr_overlay::asr::load(5, 256)?;
    // Reverse order on purpose: if the *first* call always wins, the problem is
    // leftover state between calls, not the audio.
    println!("\nmic path: {:?}", asr.transcribe(&back)?);
    println!("direct  : {:?}", asr.transcribe(&original)?);
    Ok(())
}
