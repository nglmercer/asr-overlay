//! Cache-aware FastConformer-RNNT runtime for Nemotron 3.5 ASR streaming 0.6B.

use anyhow::{anyhow, Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{Tensor, TensorRef};
use serde::Deserialize;
use std::path::Path;

use super::mel::NemoMel;
use crate::asr::{Backend, SpeechModel};

/// `genai_config.json` from the model repo.
#[derive(Debug, Clone, Deserialize)]
pub struct GenAiConfig {
    pub model: ModelConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub num_mels: usize,
    pub subsampling_factor: usize,
    /// Encoder frames of attention context kept between chunks.
    pub left_context: usize,
    /// Convolution state kept between chunks.
    pub conv_context: usize,
    /// Mel frames from the previous chunk the pre-encode convolution needs.
    pub pre_encode_cache_size: usize,
    pub sample_rate: u32,
    /// Samples per streaming step — 8960, i.e. 560 ms at 16 kHz.
    pub chunk_samples: usize,
    pub blank_id: i32,
    pub max_symbols_per_step: usize,
    pub encoder: GraphConfig,
    pub decoder: GraphConfig,
    pub joiner: GraphConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GraphConfig {
    pub filename: String,
    #[serde(default)]
    pub hidden_size: usize,
    #[serde(default)]
    pub num_hidden_layers: usize,
}

/// The SentencePiece Unigram vocabulary, read from `vocab.txt` — one piece per
/// line, line number = token id.
///
/// `tokenizer.json` ships in the repo too, but it carries a stray
/// `"pretokenizer"` key that the `tokenizers` crate refuses to parse, and
/// nothing here needs an encoder: an RNNT only ever turns ids back into text.
pub struct Vocab {
    pieces: Vec<String>,
}

impl Vocab {
    fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        let pieces: Vec<String> = text.lines().map(str::to_string).collect();
        if pieces.is_empty() {
            return Err(anyhow!("{} is empty", path.display()));
        }
        Ok(Self { pieces })
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Token id of a `<xx-YY>` language tag.
    pub fn language_id(&self, locale: &str) -> Option<i64> {
        let tag = format!("<{locale}>");
        self.pieces.iter().position(|p| *p == tag).map(|i| i as i64)
    }

    /// Every locale the vocabulary has a tag for, without the angle brackets.
    pub fn languages(&self) -> Vec<String> {
        self.pieces
            .iter()
            .filter(|p| p.len() >= 5 && p.starts_with('<') && p.ends_with('>') && p.contains('-'))
            .map(|p| p[1..p.len() - 1].to_string())
            .collect()
    }

    /// Metaspace decoding: pieces concatenate, and `▁` marks a word boundary.
    pub fn decode(&self, ids: &[i32]) -> String {
        let mut text = String::new();
        for &id in ids {
            if let Some(piece) = self.pieces.get(id as usize) {
                text.push_str(&piece.replace('\u{2581}', " "));
            }
        }
        text.trim().to_string()
    }
}

fn open_session(path: &Path, threads: usize) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(|e| anyhow!("Failed to create ONNX session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| anyhow!("Failed to set the ONNX optimization level: {e}"))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow!("Failed to set ONNX thread count: {e}"))?;
    builder
        .commit_from_file(path)
        .map_err(|e| anyhow!("Failed to load ONNX model {}: {e}", path.display()))
}

/// Mel frames the encoder consumes per call: `pre_encode_cache_size` frames of
/// context plus one 560 ms chunk. The INT4 export fixes this shape, so every
/// call feeds exactly this many frames whether or not there is audio for them.
const ENCODER_FRAMES: usize = 65;

/// Mel frames of new audio per call (`chunk_samples / hop_length`).
const CHUNK_FRAMES: usize = 56;

pub struct NemotronAsr {
    config: ModelConfig,
    vocab: Vocab,
    mel: NemoMel,
    encoder: Session,
    decoder: Session,
    joint: Session,
    /// Value fed to the encoder's `lang_id` input: the vocabulary id of a
    /// `<xx-YY>` tag, or [`AUTO_LANGUAGE`] to let the model decide.
    lang_id: i64,
}

/// `lang_id` for automatic language detection — the model's own default when
/// no language tag conditions the encoder.
pub const AUTO_LANGUAGE: i64 = 0;

/// Encoder state carried between chunks. This is what "cache-aware" means:
/// each 560 ms step sees the previous 56 encoder frames of attention context
/// and 8 frames of convolution state without recomputing them.
struct EncoderCache {
    last_channel: Vec<f32>,
    last_time: Vec<f32>,
    len: i64,
}

impl EncoderCache {
    fn new(config: &ModelConfig) -> Self {
        let layers = config.encoder.num_hidden_layers;
        let hidden = config.encoder.hidden_size;
        Self {
            last_channel: vec![0.0; layers * config.left_context * hidden],
            last_time: vec![0.0; layers * hidden * config.conv_context],
            len: 0,
        }
    }
}

/// Prediction-network state: the LSTM's hidden and cell tensors plus the last
/// emitted token.
struct DecoderState {
    hidden: Vec<f32>,
    cell: Vec<f32>,
    /// `[640]` — the prediction network's output for the tokens so far.
    output: Vec<f32>,
}

impl NemotronAsr {
    pub fn load(dir: &Path, threads: usize) -> Result<Self> {
        let config: GenAiConfig = serde_json::from_str(
            &std::fs::read_to_string(dir.join("genai_config.json"))
                .context("Failed to read genai_config.json")?,
        )
        .context("Invalid genai_config.json")?;
        let config = config.model;

        let vocab = Vocab::load(&dir.join("vocab.txt"))?;
        if vocab.len() != config.vocab_size {
            return Err(anyhow!(
                "vocab.txt has {} pieces, genai_config.json expects {}",
                vocab.len(),
                config.vocab_size
            ));
        }

        log::info!("Loading Nemotron ONNX graphs from {}", dir.display());
        let encoder = open_session(&dir.join(&config.encoder.filename), threads)?;
        let decoder = open_session(&dir.join(&config.decoder.filename), threads)?;
        let joint = open_session(&dir.join(&config.joiner.filename), threads)?;

        log::info!(
            "Nemotron ready: {} encoder layers, {} chunk frames, vocab {}",
            config.encoder.num_hidden_layers,
            CHUNK_FRAMES,
            config.vocab_size
        );

        Ok(Self {
            config,
            vocab,
            mel: NemoMel::new(),
            encoder,
            decoder,
            joint,
            lang_id: AUTO_LANGUAGE,
        })
    }

    /// Locales this model can be pinned to, e.g. `es-US`.
    pub fn languages(&self) -> Vec<String> {
        self.vocab.languages()
    }

    /// Run the encoder over one 65-frame window and return its `[frames, 1024]`
    /// output plus how many of those frames are valid.
    ///
    /// `features` is `[65 * 128]` in `[frame][mel]` order — note that this
    /// encoder wants time first, unlike the ARK one.
    fn encode_chunk(
        &mut self,
        features: Vec<f32>,
        valid_frames: usize,
        cache: &mut EncoderCache,
    ) -> Result<(Vec<f32>, usize)> {
        let layers = self.config.encoder.num_hidden_layers;
        let hidden = self.config.encoder.hidden_size;

        let outputs = self.encoder.run(ort::inputs![
            "audio_signal" => Tensor::from_array((
                [1usize, ENCODER_FRAMES, self.config.num_mels],
                features.into_boxed_slice(),
            ))?,
            "length" => Tensor::from_array(([1usize], vec![valid_frames as i64].into_boxed_slice()))?,
            "cache_last_channel" => TensorRef::from_array_view((
                [1i64, layers as i64, self.config.left_context as i64, hidden as i64],
                cache.last_channel.as_slice(),
            ))?,
            "cache_last_time" => TensorRef::from_array_view((
                [1i64, layers as i64, hidden as i64, self.config.conv_context as i64],
                cache.last_time.as_slice(),
            ))?,
            "cache_last_channel_len" => Tensor::from_array(([1usize], vec![cache.len].into_boxed_slice()))?,
            "lang_id" => Tensor::from_array(([1usize], vec![self.lang_id].into_boxed_slice()))?,
        ])?;

        let (shape, encoded) = outputs["outputs"].try_extract_tensor::<f32>()?;
        let frames = shape[1] as usize;
        let encoded = encoded.to_vec();

        let (_, lengths) = outputs["encoded_lengths"].try_extract_tensor::<i64>()?;
        let valid = (lengths[0].max(0) as usize).min(frames);

        // Carry the state forward for the next chunk.
        let (_, next_channel) = outputs["cache_last_channel_next"].try_extract_tensor::<f32>()?;
        cache.last_channel.copy_from_slice(next_channel);
        let (_, next_time) = outputs["cache_last_time_next"].try_extract_tensor::<f32>()?;
        cache.last_time.copy_from_slice(next_time);
        let (_, next_len) = outputs["cache_last_channel_len_next"].try_extract_tensor::<i64>()?;
        cache.len = next_len[0];

        Ok((encoded, valid))
    }

    /// Advance the prediction network by one token.
    ///
    /// `token` is the previously emitted id; the blank id doubles as the
    /// start-of-sequence symbol, because the prediction network's embedding
    /// uses it as its padding index and so maps it to zeros.
    fn predict(&mut self, token: i32, state: &mut DecoderState) -> Result<()> {
        let outputs = self.decoder.run(ort::inputs![
            "targets" => Tensor::from_array(([1usize, 1usize], vec![token as i64].into_boxed_slice()))?,
            "h_in" => TensorRef::from_array_view(([2i64, 1, 640], state.hidden.as_slice()))?,
            "c_in" => TensorRef::from_array_view(([2i64, 1, 640], state.cell.as_slice()))?,
        ])?;

        // decoder_output is [batch, 640, target_len]; with one target that is
        // just the 640 values, but the joint network wants them as
        // [batch, target_len, 640], so it is fed back transposed.
        let (_, output) = outputs["decoder_output"].try_extract_tensor::<f32>()?;
        state.output = output.to_vec();
        let (_, hidden) = outputs["h_out"].try_extract_tensor::<f32>()?;
        state.hidden.copy_from_slice(hidden);
        let (_, cell) = outputs["c_out"].try_extract_tensor::<f32>()?;
        state.cell.copy_from_slice(cell);
        Ok(())
    }

    /// Joint network for one encoder frame and the current prediction state.
    /// Returns the argmax token id.
    fn joint_argmax(&mut self, frame: &[f32], state: &DecoderState) -> Result<i32> {
        let outputs = self.joint.run(ort::inputs![
            "encoder_output" => TensorRef::from_array_view((
                [1i64, 1, self.config.encoder.hidden_size as i64],
                frame,
            ))?,
            "decoder_output" => TensorRef::from_array_view(([1i64, 1, 640], state.output.as_slice()))?,
        ])?;
        let (_, logits) = outputs["joint_output"].try_extract_tensor::<f32>()?;

        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (id, &score) in logits.iter().enumerate() {
            if score > best_score {
                best_score = score;
                best = id;
            }
        }
        Ok(best as i32)
    }

    /// Print every graph's signature — used by `asr-tool inspect`.
    pub fn describe(&self) {
        for (name, session) in [
            ("encoder", &self.encoder),
            ("decoder", &self.decoder),
            ("joint", &self.joint),
        ] {
            println!("\n=== {name}");
            for input in session.inputs().iter() {
                println!("  in  {:<26} {:?}", input.name(), input.dtype());
            }
            for output in session.outputs().iter() {
                println!("  out {:<26} {:?}", output.name(), output.dtype());
            }
        }
    }
}

impl SpeechModel for NemotronAsr {
    fn backend(&self) -> Backend {
        Backend::Nemotron
    }

    /// Stream the audio through the encoder in 560 ms steps, decoding greedily
    /// after every step.
    ///
    /// Nothing here depends on the utterance length: the encoder state is
    /// carried between chunks, so cost is linear in the audio and a partial
    /// caption is available every 560 ms of audio rather than only at the end.
    fn transcribe_streaming(
        &mut self,
        samples: &[f32],
        on_partial: &mut dyn FnMut(&str),
    ) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let mel_started = std::time::Instant::now();
        let (features, frames) = self.mel.features(samples);
        let mel_ms = mel_started.elapsed().as_millis();
        if frames == 0 {
            return Ok(String::new());
        }
        let n_mels = self.config.num_mels;
        let context = self.config.pre_encode_cache_size;
        let blank = self.config.blank_id;

        let mut cache = EncoderCache::new(&self.config);
        let mut state = DecoderState {
            hidden: vec![0.0; 2 * 640],
            cell: vec![0.0; 2 * 640],
            output: Vec::new(),
        };
        // Blank is the start-of-sequence symbol.
        self.predict(blank, &mut state)?;

        let mut tokens: Vec<i32> = Vec::new();
        let mut chunks = 0usize;
        let started = std::time::Instant::now();

        let mut start = 0usize;
        while start < frames {
            // Frames [start - context, start + CHUNK_FRAMES): the window the
            // pre-encode convolution needs, then the new audio. Missing frames
            // at either end are zeros, matching how the reference runtime
            // primes and drains the stream.
            let mut window = vec![0.0f32; ENCODER_FRAMES * n_mels];
            let mut valid = 0usize;
            for slot in 0..ENCODER_FRAMES {
                let frame = (start + slot).checked_sub(context);
                let Some(frame) = frame else { continue };
                if frame >= frames {
                    break;
                }
                // `features` is [mel][frame]; the encoder wants [frame][mel].
                for mel in 0..n_mels {
                    window[slot * n_mels + mel] = features[mel * frames + frame];
                }
                valid = slot + 1;
            }

            let (encoded, encoded_frames) = self.encode_chunk(window, valid, &mut cache)?;
            chunks += 1;

            let hidden = self.config.encoder.hidden_size;
            for frame in 0..encoded_frames {
                let frame = &encoded[frame * hidden..(frame + 1) * hidden];
                // RNNT greedy: emit until the joint network says blank, with a
                // hard cap so a degenerate frame cannot emit forever.
                for _ in 0..self.config.max_symbols_per_step {
                    let token = self.joint_argmax(frame, &state)?;
                    if token == blank {
                        break;
                    }
                    tokens.push(token);
                    self.predict(token, &mut state)?;
                }
            }

            let text = self.vocab.decode(&tokens);
            if !text.is_empty() {
                on_partial(&text);
            }
            start += CHUNK_FRAMES;
        }

        log::debug!(
            "Nemotron timing: mel {mel_ms} ms ({frames} frames), stream {} ms \
             ({chunks} chunks, {} tokens)",
            started.elapsed().as_millis(),
            tokens.len()
        );

        Ok(self.vocab.decode(&tokens))
    }

    /// `auto`, or a locale the vocabulary has a tag for (`es-US`, `en-US`, …).
    /// A bare language code picks that language's first locale.
    fn set_language(&mut self, language: &str) -> Result<()> {
        let language = language.trim();
        if language.is_empty() || language.eq_ignore_ascii_case("auto") {
            self.lang_id = AUTO_LANGUAGE;
            return Ok(());
        }

        let locales = self.vocab.languages();
        let matched = locales
            .iter()
            .find(|locale| locale.eq_ignore_ascii_case(language))
            .or_else(|| {
                locales
                    .iter()
                    .find(|locale| locale.split('-').next().is_some_and(|code| code.eq_ignore_ascii_case(language)))
            })
            .cloned();

        match matched {
            Some(locale) => {
                self.lang_id = self
                    .vocab
                    .language_id(&locale)
                    .ok_or_else(|| anyhow!("No tag for {locale}"))?;
                log::info!("Nemotron pinned to {locale} (lang_id {})", self.lang_id);
                Ok(())
            }
            None => Err(anyhow!(
                "Nemotron does not support {language}; known locales: {}",
                locales.join(", ")
            )),
        }
    }
}
