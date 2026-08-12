//! Developer tool: inspect the ONNX graph signatures, transcribe a WAV, and
//! compare the CPU cost of the ASR backends on the same audio.
//!
//!     cargo run --bin asr-tool -- inspect [ark|nemotron]
//!     cargo run --bin asr-tool -- transcribe audio.wav
//!     cargo run --release --bin asr-tool -- bench audio.wav --threads 1,6,12

use anyhow::{anyhow, Result};
use asr_overlay::asr::{Backend, SpeechModel};
use ort::session::Session;

/// Flags shared by every subcommand, pulled out of the argument list before
/// the positional arguments are read.
#[derive(Debug, Clone)]
struct Options {
    backend: Backend,
    /// Thread counts to run; `bench` sweeps them, everything else uses the
    /// first.
    threads: Vec<usize>,
    language: Option<String>,
    backends: Vec<Backend>,
    repeat: usize,
}

fn main() -> Result<()> {
    asr_overlay::logger::init();
    let (options, mut args) = parse_options(std::env::args().skip(1).collect())?;
    let mut args = args.drain(..);

    match args.next().as_deref() {
        Some("inspect") => inspect(options.backend),
        Some("transcribe") => {
            let path = args
                .next()
                .ok_or_else(|| anyhow!("usage: transcribe <file.wav>"))?;
            transcribe(&path, args.next(), &options)
        }
        Some("bench") => {
            let path = args
                .next()
                .ok_or_else(|| anyhow!("usage: bench <file.wav>"))?;
            bench(&path, &options)
        }
        Some("micpath") => {
            let path = args
                .next()
                .ok_or_else(|| anyhow!("usage: micpath <file.wav>"))?;
            micpath(&path, &options)
        }
        Some("vad") => {
            let path = args
                .next()
                .ok_or_else(|| anyhow!("usage: vad <file.wav> [gain] [instruction]"))?;
            let gain = args.next().and_then(|g| g.parse().ok()).unwrap_or(1.0);
            vad(&path, gain, args.next(), &options)
        }
        _ => Err(anyhow!(
            "usage: asr-tool [--backend ark|nemotron] [--threads N[,N...]] \
             [--language es-US] [--repeat N]\n  \
             inspect [ark|nemotron]\n  \
             transcribe <file.wav> [instruction]\n  \
             bench <file.wav>            compare backends on the same audio\n  \
             micpath <file.wav>\n  \
             vad <file.wav> [gain] [instruction]"
        )),
    }
}

/// Split `--flag value` pairs out of the arguments, leaving the positionals.
///
/// Deliberately tiny: the tool is a developer harness, and a full `clap`
/// derive here would be more code than the commands it dispatches to.
fn parse_options(args: Vec<String>) -> Result<(Options, Vec<String>)> {
    let config = asr_overlay::config::load();
    let mut options = Options {
        backend: std::env::var("ASR_BACKEND")
            .map(|name| Backend::parse(&name))
            .unwrap_or_else(|_| Backend::parse(&config.asr_backend)),
        threads: vec![std::env::var("ASR_THREADS")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or(config.threads)],
        language: None,
        backends: Backend::all().to_vec(),
        repeat: 1,
    };

    let mut positional = Vec::new();
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        let mut value = || {
            iter.next()
                .ok_or_else(|| anyhow!("missing value for a flag"))
        };
        match arg.as_str() {
            "--backend" | "-b" => options.backend = Backend::parse(&value()?),
            "--backends" => {
                options.backends = value()?.split(',').map(Backend::parse).collect();
            }
            "--threads" | "-t" => {
                options.threads = value()?
                    .split(',')
                    .map(|t| t.trim().parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--language" | "-l" => options.language = Some(value()?),
            "--repeat" => options.repeat = value()?.parse()?,
            other => positional.push(other.to_string()),
        }
    }
    if options.threads.is_empty() {
        options.threads.push(config.threads);
    }
    Ok((options, positional))
}

/// Load a backend and apply the options that every command shares.
fn load_model(backend: Backend, threads: usize, options: &Options) -> Result<Box<dyn SpeechModel>> {
    let config = asr_overlay::config::load();
    let mut model = asr_overlay::asr::load(backend, threads, config.max_new_tokens)?;
    model.set_min_audio_ms(config.min_audio_ms);
    let language = options.language.clone().unwrap_or(config.language);
    model.set_language(&language)?;
    Ok(model)
}

fn inspect(backend: Backend) -> Result<()> {
    if backend == Backend::Nemotron {
        let dir = asr_overlay::asr::ensure_backend(backend)?;
        return asr_overlay::asr::NemotronAsr::load(&dir, 1).map(|model| model.describe());
    }
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

fn transcribe(path: &str, instruction: Option<String>, options: &Options) -> Result<()> {
    let samples = read_wav_16k_mono(path)?;
    let threads = options.threads[0];
    println!("Backend: {}, threads: {threads}", options.backend.as_str());
    println!(
        "Loaded {} samples ({:.2}s)",
        samples.len(),
        samples.len() as f32 / 16_000.0
    );

    let started = std::time::Instant::now();
    let mut asr = load_model(options.backend, threads, options)?;
    if let Some(instruction) = &instruction {
        println!("Instruction: {instruction}");
        asr.set_instruction(instruction);
    }
    println!("Model loaded in {:.1}s", started.elapsed().as_secs_f32());

    let started = std::time::Instant::now();
    let mut first_word = None;
    let text = asr.transcribe_streaming(&samples, &mut |partial: &str| {
        if first_word.is_none() && !partial.is_empty() {
            first_word = Some(started.elapsed().as_secs_f32());
        }
    })?;
    let elapsed = started.elapsed().as_secs_f32();
    println!("\nTranscript: {text}");
    if let Some(first_word) = first_word {
        println!("First word on screen after {first_word:.2}s");
    }
    println!(
        "\nTook {elapsed:.2}s for {:.2}s of audio ({:.2}x realtime)",
        samples.len() as f32 / 16_000.0,
        (samples.len() as f32 / 16_000.0) / elapsed
    );
    Ok(())
}

/// One measured transcription.
struct Run {
    backend: Backend,
    threads: usize,
    load_secs: f32,
    elapsed_secs: f32,
    first_partial_secs: Option<f32>,
    text: String,
}

/// Run every backend over the same file at every requested thread count and
/// print what each one costs on this CPU.
///
/// The number that matters for a live overlay is the real-time factor: below
/// 1.0 the model keeps up with the microphone, above it the captions fall
/// further behind the longer someone talks.
fn bench(path: &str, options: &Options) -> Result<()> {
    let samples = read_wav_16k_mono(path)?;
    let audio_secs = samples.len() as f32 / 16_000.0;
    println!(
        "Benchmarking {path}: {audio_secs:.2}s of audio, {} thread setting(s), {} run(s) each",
        options.threads.len(),
        options.repeat
    );
    println!("CPU: {} logical cores\n", num_cores());

    let mut runs: Vec<Run> = Vec::new();
    for &backend in &options.backends {
        for &threads in &options.threads {
            let started = std::time::Instant::now();
            let mut asr = match load_model(backend, threads, options) {
                Ok(asr) => asr,
                Err(e) => {
                    println!("{} failed to load: {e}", backend.as_str());
                    continue;
                }
            };
            let load_secs = started.elapsed().as_secs_f32();

            for run in 0..options.repeat.max(1) {
                let started = std::time::Instant::now();
                let mut first_partial = None;
                let text = asr.transcribe_streaming(&samples, &mut |partial: &str| {
                    if first_partial.is_none() && !partial.is_empty() {
                        first_partial = Some(started.elapsed().as_secs_f32());
                    }
                });
                let elapsed = started.elapsed().as_secs_f32();
                match text {
                    Ok(text) => {
                        println!(
                            "  {:<9} threads {:<3} run {}/{}: {elapsed:.2}s  RTF {:.2}",
                            backend.as_str(),
                            threads,
                            run + 1,
                            options.repeat.max(1),
                            elapsed / audio_secs.max(f32::EPSILON)
                        );
                        runs.push(Run {
                            backend,
                            threads,
                            load_secs,
                            elapsed_secs: elapsed,
                            first_partial_secs: first_partial,
                            text,
                        });
                    }
                    Err(e) => println!("  {} threads {threads}: error: {e}", backend.as_str()),
                }
            }
        }
    }

    if runs.is_empty() {
        return Err(anyhow!("No backend produced a transcript"));
    }

    println!("\n{:<10} {:>7} {:>9} {:>7} {:>9} {:>9}", "backend", "threads", "load s", "run s", "RTF", "1st word");
    println!("{}", "-".repeat(56));
    for run in &runs {
        println!(
            "{:<10} {:>7} {:>9.1} {:>7.2} {:>9.2} {:>9}",
            run.backend.as_str(),
            run.threads,
            run.load_secs,
            run.elapsed_secs,
            run.elapsed_secs / audio_secs.max(f32::EPSILON),
            run.first_partial_secs
                .map(|s| format!("{s:.2}s"))
                .unwrap_or_else(|| "-".to_string()),
        );
    }

    println!("\nTranscripts");
    let mut seen: Vec<(Backend, usize)> = Vec::new();
    for run in &runs {
        if seen.contains(&(run.backend, run.threads)) {
            continue;
        }
        seen.push((run.backend, run.threads));
        println!("  [{} @ {}] {}", run.backend.as_str(), run.threads, run.text);
    }
    println!(
        "\nRTF is model time / audio time: below 1.00 the backend keeps up with \
         the microphone."
    );
    Ok(())
}

fn num_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0)
}

/// Push a WAV through the live capture path — upsampled to a 48 kHz mic, split
/// into callback-sized chunks, segmented by the real VAD — and transcribe each
/// utterance the way the overlay would.
///
/// `gain` scales the input: a headset at 0.15 of full scale is a very different
/// test from a studio recording, and segmentation is where that shows up. If
/// `transcribe` is right and this is wrong, the segmentation is to blame.
fn vad(path: &str, gain: f32, instruction: Option<String>, options: &Options) -> Result<()> {
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

    let min_audio_ms = config.min_audio_ms;
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

    let mut asr = load_model(options.backend, options.threads[0], options)?;
    asr.set_min_audio_ms(min_audio_ms);
    if let Some(instruction) = &instruction {
        println!("Instruction: {instruction}");
        asr.set_instruction(instruction);
    }
    println!("{} utterances, padded to {min_audio_ms} ms", utterances.len());
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
fn micpath(path: &str, options: &Options) -> Result<()> {
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

    let mut asr = load_model(options.backend, options.threads[0], options)?;
    // Reverse order on purpose: if the *first* call always wins, the problem is
    // leftover state between calls, not the audio.
    println!("\nmic path: {:?}", asr.transcribe(&back)?);
    println!("direct  : {:?}", asr.transcribe(&original)?);
    Ok(())
}
