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
            transcribe(&path)
        }
        _ => Err(anyhow!("usage: asr-tool [inspect | transcribe <file.wav>]")),
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

fn transcribe(path: &str) -> Result<()> {
    let samples = read_wav_16k_mono(path)?;
    println!(
        "Loaded {} samples ({:.2}s)",
        samples.len(),
        samples.len() as f32 / 16_000.0
    );

    let started = std::time::Instant::now();
    let mut asr = asr_overlay::asr::load(5, 256)?;
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
