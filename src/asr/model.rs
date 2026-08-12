//! ARK-ASR 0.6B INT8 ONNX runtime.
//!
//! Ported from the official `infer_ark_audio_onnx.py`. Four ONNX graphs:
//!
//! 1. `audio_encoder_whisper_int8.onnx` — log-mel `[1,128,T]` → `[1,T',1280]`
//! 2. `audio_encoder_adapter_int8.onnx` — merged `[1,T'/4,5120]` → `[1,N,896]`
//! 3. `embedding_fp32.onnx` — token ids → `[1,L,896]`
//! 4. `llm_kv_cpu_fp32_int8.onnx` — Qwen2 decoder with an external KV cache
//!
//! The decoder is architecturally identical to the Audio8 TTS slow AR
//! (24 layers, hidden 896, 14 heads, 2 KV heads, head_dim 64, max 2048).

use anyhow::{anyhow, Context, Result};
use ort::session::Session;
use ort::value::{Tensor, TensorRef};
use serde::Deserialize;
use std::borrow::Cow;
use std::path::Path;

use super::mel::MelExtractor;

/// `runtime_manifest.json` from the ARK-ASR repo.
#[derive(Debug, Clone, Deserialize)]
pub struct AsrManifest {
    pub sample_rate: u32,
    pub latent_hop_length: usize,
    pub audio_token_id: i64,
    pub user_token_id: i64,
    pub assistant_token_id: i64,
    pub im_end_token_id: i64,
    pub pad_token_id: i64,
    pub stop_token_ids: Vec<i64>,
    pub audio_encoder_default: String,
    pub audio_adapter_default: String,
    pub audio_merge_factor: usize,
    pub audio_encoder_hidden_size: usize,
    pub embedding_default: String,
    pub llm_default: String,
}

/// `llm_kv_cpu_fp32_int8.json` — the decoder's cache geometry.
#[derive(Debug, Clone, Deserialize)]
pub struct LlmMeta {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_total_len: usize,
}

impl LlmMeta {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// One KV cache tensor: `[1, max_total_len, num_kv_heads, head_dim]`.
    pub fn cache_elements(&self) -> usize {
        self.max_total_len * self.num_key_value_heads * self.head_dim()
    }
}

/// Token ids above this are blocked during ASR decoding, matching
/// `DEFAULT_ASR_BLOCK_TOKEN_ID_FROM` in the reference runtime.
const BLOCK_TOKEN_ID_FROM: usize = 151670;

/// The instruction used by the official runtime.
pub const DEFAULT_INSTRUCTION: &str = "Please transcribe this audio.";

/// Default for `min_samples`: 8 s at 16 kHz. Measured on this model — 6 s
/// still drifts into the wrong language, 12 s is no better than 8 s.
const DEFAULT_MIN_SAMPLES: usize = 8 * 16_000;

/// Longest repeating unit the loop guard looks for.
const LOOP_MAX_CYCLE: usize = 4;

/// How many times that unit must repeat back-to-back to count as a loop.
const LOOP_REPEATS: usize = 8;

/// Detect a degenerate tail — "oh, oh, oh, oh, …" — and report the length of
/// the repeating unit.
///
/// Given unclear audio the decoder can fall into a cycle it only leaves at the
/// token budget, which costs ~20 s per utterance and emits a wall of garbage.
/// Eight repeats of the same short cycle is not something a speaker says.
fn looping_cycle(tokens: &[u32]) -> Option<usize> {
    for cycle in 1..=LOOP_MAX_CYCLE {
        let span = cycle * LOOP_REPEATS;
        if tokens.len() < span {
            continue;
        }
        let tail = &tokens[tokens.len() - span..];
        if tail.chunks(cycle).all(|unit| unit == &tail[..cycle]) {
            return Some(cycle);
        }
    }
    None
}

pub struct ArkAsr {
    manifest: AsrManifest,
    meta: LlmMeta,
    tokenizer: tokenizers::Tokenizer,
    mel: MelExtractor,
    encoder: Session,
    adapter: Session,
    embedding: Session,
    llm: Session,
    /// Persistent KV cache buffers, reused across utterances.
    caches: Vec<Vec<f32>>,
    max_new_tokens: usize,
    /// `blocked[id]` = this token may never be generated. Indexed by token id.
    blocked: Vec<bool>,
    /// Instruction given to the decoder after the audio. ARK-ASR is an audio
    /// LLM, so this steers what it produces.
    instruction: String,
    /// Shortest clip handed to the encoder; shorter ones are padded.
    min_samples: usize,
}

fn open_session(path: &Path, threads: usize) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(|e| anyhow!("Failed to create ONNX session builder: {e}"))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow!("Failed to set ONNX thread count: {e}"))?;
    builder
        .commit_from_file(path)
        .map_err(|e| anyhow!("Failed to load ONNX model {}: {e}", path.display()))
}

impl ArkAsr {
    /// Load every graph from a local model directory.
    pub fn load(dir: &Path, threads: usize, max_new_tokens: usize) -> Result<Self> {
        let manifest: AsrManifest = serde_json::from_str(
            &std::fs::read_to_string(dir.join("runtime_manifest.json"))
                .context("Failed to read runtime_manifest.json")?,
        )
        .context("Invalid runtime_manifest.json")?;

        let meta: LlmMeta = serde_json::from_str(
            &std::fs::read_to_string(dir.join("llm_kv_cpu_fp32_int8.json"))
                .context("Failed to read llm_kv_cpu_fp32_int8.json")?,
        )
        .context("Invalid llm_kv_cpu_fp32_int8.json")?;

        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("Failed to load tokenizer.json: {e}"))?;

        log::info!("Loading ARK-ASR ONNX graphs from {}", dir.display());
        let encoder = open_session(&dir.join(&manifest.audio_encoder_default), threads)?;
        let adapter = open_session(&dir.join(&manifest.audio_adapter_default), threads)?;
        let embedding = open_session(&dir.join(&manifest.embedding_default), threads)?;
        let llm = open_session(&dir.join(&manifest.llm_default), threads)?;

        let caches = (0..meta.num_layers * 2)
            .map(|_| vec![0.0f32; meta.cache_elements()])
            .collect();

        // Mirror `build_bad_token_ids`: block every added `<|...|>` token
        // except EOS, which is how generation is meant to terminate.
        let mut blocked = vec![false; BLOCK_TOKEN_ID_FROM];
        let mut blocked_count = 0usize;
        for (token, id) in tokenizer.get_added_vocabulary().get_vocab() {
            let id = *id;
            let id = id as usize;
            if id >= BLOCK_TOKEN_ID_FROM || id == manifest.im_end_token_id as usize {
                continue;
            }
            if token.starts_with('<') && token.ends_with('>') {
                blocked[id] = true;
                blocked_count += 1;
            }
        }
        log::debug!("ASR: blocked {blocked_count} special tokens (EOS stays selectable)");

        log::info!(
            "ARK-ASR ready: {} layers, hidden {}, cache {} tokens",
            meta.num_layers,
            meta.hidden_size,
            meta.max_total_len
        );

        Ok(Self {
            manifest,
            meta,
            tokenizer,
            mel: MelExtractor::new(),
            encoder,
            adapter,
            embedding,
            llm,
            caches,
            max_new_tokens,
            blocked,
            instruction: DEFAULT_INSTRUCTION.to_string(),
            min_samples: DEFAULT_MIN_SAMPLES,
        })
    }

    /// Build the ARK-ASR prompt string.
    ///
    /// Ported from `ArkasrProcessor._build_templates_and_audios`: the audio is
    /// wrapped in begin/end markers and expanded into `n` `<|audio|>`
    /// placeholders, then the generation prompt opens the assistant turn.
    fn prompt(&self, audio_tokens: usize) -> String {
        let mut prompt = String::from("<|user|><|begin_of_audio|>");
        for _ in 0..audio_tokens {
            prompt.push_str("<|audio|>");
        }
        prompt.push_str("<|end_of_audio|>");
        prompt.push_str(&self.instruction);
        prompt.push_str("<|assistant|>");
        prompt
    }

    /// Run the encoder + adapter, returning `[n_tokens, hidden]` audio
    /// embeddings ready to splice into the prompt.
    fn encode_audio(&mut self, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        let (features, frames) = self.mel.features(samples);
        if frames == 0 {
            return Ok((Vec::new(), 0));
        }

        let mel_tensor = Tensor::from_array((
            [1usize, super::mel::N_MELS, frames],
            features.into_boxed_slice(),
        ))?;
        let encoded = self.encoder.run(ort::inputs!["audios" => mel_tensor])?;
        let (shape, hidden) = encoded[0].try_extract_tensor::<f32>()?;
        let seq_len = shape[1] as usize;
        let feat_dim = shape[2] as usize;
        if feat_dim != self.manifest.audio_encoder_hidden_size {
            return Err(anyhow!(
                "Unexpected audio hidden size: {feat_dim} != {}",
                self.manifest.audio_encoder_hidden_size
            ));
        }

        // Merge `merge_factor` consecutive frames along the feature axis.
        let merge = self.manifest.audio_merge_factor;
        let merged_len = seq_len / merge;
        if merged_len == 0 {
            return Ok((Vec::new(), 0));
        }
        let merged: Vec<f32> = hidden[..merged_len * merge * feat_dim].to_vec();

        let merged_tensor = Tensor::from_array((
            [1usize, merged_len, feat_dim * merge],
            merged.into_boxed_slice(),
        ))?;
        let adapted = self
            .adapter
            .run(ort::inputs!["merged_audio_features" => merged_tensor])?;
        let (shape, embeds) = adapted[0].try_extract_tensor::<f32>()?;
        let n_tokens = shape[1] as usize;
        Ok((embeds.to_vec(), n_tokens))
    }

    /// Set the shortest clip handed to the model, in milliseconds. Anything
    /// shorter is padded with silence; zero disables the padding.
    pub fn set_min_audio_ms(&mut self, ms: u64) {
        self.min_samples = (ms as usize) * 16_000 / 1000;
    }

    /// Override the decoder instruction (empty resets to the default).
    pub fn set_instruction(&mut self, instruction: &str) {
        self.instruction = if instruction.trim().is_empty() {
            DEFAULT_INSTRUCTION.to_string()
        } else {
            instruction.trim().to_string()
        };
    }

    /// Look up token embeddings.
    fn embed(&mut self, ids: &[i64]) -> Result<Vec<f32>> {
        let tensor = Tensor::from_array(([1usize, ids.len()], ids.to_vec().into_boxed_slice()))?;
        let out = self.embedding.run(ort::inputs!["input_ids" => tensor])?;
        let (_, data) = out[0].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }

    /// One decoder forward pass over `seq` positions starting at `start_pos`.
    /// Writes the returned KV deltas back into the persistent cache and
    /// returns the logits for the final position.
    fn forward(&mut self, embeds: Vec<f32>, seq: usize, start_pos: usize) -> Result<Vec<f32>> {
        let hidden = self.meta.hidden_size;
        let max_total = self.meta.max_total_len;
        let valid = start_pos + seq;

        let mut attention_mask = vec![0i64; max_total];
        attention_mask[..valid].fill(1);
        let cache_position: Vec<i64> = (start_pos..valid).map(|p| p as i64).collect();

        let mut feeds: Vec<(Cow<str>, ort::session::SessionInputValue)> = Vec::new();
        feeds.push((
            "inputs_embeds".into(),
            Tensor::from_array(([1usize, seq, hidden], embeds.into_boxed_slice()))?.into(),
        ));
        feeds.push((
            "attention_mask".into(),
            Tensor::from_array(([1usize, max_total], attention_mask.into_boxed_slice()))?.into(),
        ));
        feeds.push((
            "cache_position".into(),
            Tensor::from_array(([seq], cache_position.into_boxed_slice()))?.into(),
        ));

        let cache_shape = [
            1i64,
            max_total as i64,
            self.meta.num_key_value_heads as i64,
            self.meta.head_dim() as i64,
        ];
        // Borrow the caches instead of copying them: at 48 tensors of
        // max_total × 2 × 64 floats each, cloning per token would move
        // ~200 MB per generated token.
        let Self { llm, caches, .. } = self;
        for (layer, cache) in caches.iter().enumerate() {
            let name = if layer % 2 == 0 {
                format!("cache_key_{}", layer / 2)
            } else {
                format!("cache_value_{}", layer / 2)
            };
            feeds.push((
                name.into(),
                TensorRef::from_array_view((cache_shape, cache.as_slice()))?.into(),
            ));
        }

        let outputs = llm.run(feeds)?;

        // outputs[0] = logits, outputs[1..] = per-layer key/value deltas.
        let (logits_shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
        let vocab = *logits_shape.last().unwrap() as usize;
        let logits = logits[logits.len() - vocab..].to_vec();

        let stride = self.meta.num_key_value_heads * self.meta.head_dim();
        let offset = start_pos * stride;
        let len = seq * stride;
        for (layer, cache) in caches.iter_mut().enumerate() {
            let (_, delta) = outputs[layer + 1].try_extract_tensor::<f32>()?;
            if offset + len <= cache.len() && delta.len() >= len {
                cache[offset..offset + len].copy_from_slice(&delta[..len]);
            }
        }

        Ok(logits)
    }

    /// Greedy argmax over the allowed token range.
    ///
    /// Ported from `AsrLogitsMask` + `build_bad_token_ids`: ids at or above
    /// `BLOCK_TOKEN_ID_FROM` and every special token except EOS are suppressed.
    ///
    /// Blocking the specials matters. `<|user|>` and the pad token are also
    /// stop tokens, and the model often scores them just below the real first
    /// word — if they stay selectable, decoding stops immediately and the
    /// transcript comes back empty.
    fn pick_token(&self, logits: &[f32]) -> i64 {
        let limit = logits.len().min(BLOCK_TOKEN_ID_FROM);
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (id, &score) in logits[..limit].iter().enumerate() {
            if self.blocked[id] {
                continue;
            }
            if score > best_score {
                best_score = score;
                best = id;
            }
        }
        best as i64
    }

    /// Transcribe mono `f32` audio at 16 kHz.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        // Pad short utterances with silence. Below a few seconds the decoder
        // stops leaning on the audio and starts leaning on its own priors: the
        // same Spanish clip that transcribes correctly at 7 s comes back as
        // Chinese at 2 s ("El alma del Japón" → "El南国的日本"). Whisper pads
        // every clip to 30 s for the same reason; a shorter window is enough
        // here, and the encoder cost scales with what we pad to.
        let padded;
        let samples = if samples.len() < self.min_samples {
            padded = {
                let mut buffer = Vec::with_capacity(self.min_samples);
                buffer.extend_from_slice(samples);
                buffer.resize(self.min_samples, 0.0);
                buffer
            };
            &padded[..]
        } else {
            samples
        };

        // Reset the KV cache between utterances.
        for cache in self.caches.iter_mut() {
            cache.fill(0.0);
        }

        let encode_started = std::time::Instant::now();
        let (audio_embeds, n_audio) = self.encode_audio(samples)?;
        let encode_ms = encode_started.elapsed().as_millis();
        if n_audio == 0 {
            return Ok(String::new());
        }

        let prompt = self.prompt(n_audio);
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("Tokenization failed: {e}"))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        let prompt_len = ids.len();
        if prompt_len + self.max_new_tokens > self.meta.max_total_len {
            return Err(anyhow!(
                "Prompt too long: {prompt_len} + {} > {}",
                self.max_new_tokens,
                self.meta.max_total_len
            ));
        }

        // Token embeddings, with the audio placeholders overwritten by the
        // adapter output (`_inject_audio` in the reference runtime).
        let hidden = self.meta.hidden_size;
        let mut embeds = self.embed(&ids)?;
        let mut audio_slot = 0usize;
        for (pos, &id) in ids.iter().enumerate() {
            if id != self.manifest.audio_token_id {
                continue;
            }
            let src = audio_slot * hidden;
            if src + hidden <= audio_embeds.len() {
                embeds[pos * hidden..(pos + 1) * hidden]
                    .copy_from_slice(&audio_embeds[src..src + hidden]);
            }
            audio_slot += 1;
        }

        let prefill_started = std::time::Instant::now();
        let mut logits = self.forward(embeds, prompt_len, 0)?;
        let prefill_ms = prefill_started.elapsed().as_millis();
        let decode_started = std::time::Instant::now();

        let mut out_ids: Vec<u32> = Vec::new();

        for total_len in prompt_len..prompt_len + self.max_new_tokens {
            let next = self.pick_token(&logits);
            if self.manifest.stop_token_ids.contains(&next) {
                break;
            }
            out_ids.push(next as u32);
            if let Some(cycle) = looping_cycle(&out_ids) {
                log::debug!(
                    "ASR: stopped a {cycle}-token repetition loop after {} tokens",
                    out_ids.len()
                );
                // Keep one copy of the repeated unit and drop the rest.
                out_ids.truncate(out_ids.len() - cycle * (LOOP_REPEATS - 1));
                break;
            }
            let embed = self.embed(&[next])?;
            logits = self.forward(embed, 1, total_len)?;
        }

        log::debug!(
            "ASR timing: encode {encode_ms} ms, prefill {prefill_ms} ms ({prompt_len} tok), \
             decode {} ms ({} tok)",
            decode_started.elapsed().as_millis(),
            out_ids.len()
        );

        let text = self
            .tokenizer
            .decode(&out_ids, true)
            .map_err(|e| anyhow!("Detokenization failed: {e}"))?;
        Ok(text.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = r#"{
        "sample_rate": 16000,
        "latent_hop_length": 320,
        "audio_token_id": 151663,
        "user_token_id": 151665,
        "assistant_token_id": 151668,
        "im_end_token_id": 151645,
        "pad_token_id": 151643,
        "stop_token_ids": [151645, 151643, 151665],
        "audio_encoder_default": "audio_encoder_whisper_int8.onnx",
        "audio_adapter_default": "audio_encoder_adapter_int8.onnx",
        "audio_merge_factor": 4,
        "audio_encoder_hidden_size": 1280,
        "embedding_default": "embedding_fp32.onnx",
        "llm_default": "llm_kv_cpu_fp32_int8.onnx"
    }"#;

    const META: &str = r#"{
        "hidden_size": 896,
        "num_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "max_total_len": 2048,
        "model_type": "arkasr"
    }"#;

    #[test]
    fn loop_guard_spots_a_repeating_tail() {
        // "oh," ×8 — the failure mode this guards against.
        let mut tokens: Vec<u32> = vec![10, 20, 30];
        tokens.extend(std::iter::repeat_n([40u32, 41], 8).flatten());
        assert_eq!(looping_cycle(&tokens), Some(2));

        // A single token repeated is the shortest cycle.
        assert_eq!(looping_cycle(&[7; 8]), Some(1));
    }

    #[test]
    fn loop_guard_leaves_normal_speech_alone() {
        // "no, no, no." repeats, but nowhere near eight times.
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 3, 4, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(looping_cycle(&tokens), None);
        assert_eq!(looping_cycle(&[]), None);
    }

    #[test]
    fn parses_the_real_manifest() {
        let m: AsrManifest = serde_json::from_str(MANIFEST).unwrap();
        assert_eq!(m.sample_rate, 16_000);
        assert_eq!(m.audio_token_id, 151663);
        assert_eq!(m.audio_merge_factor, 4);
        assert_eq!(m.audio_encoder_hidden_size, 1280);
        assert_eq!(m.stop_token_ids, vec![151645, 151643, 151665]);
    }

    #[test]
    fn cache_geometry_matches_the_decoder() {
        let meta: LlmMeta = serde_json::from_str(META).unwrap();
        assert_eq!(meta.head_dim(), 64);
        assert_eq!(meta.cache_elements(), 2048 * 2 * 64);
        // Same geometry as the Audio8 TTS slow AR.
        assert_eq!(meta.num_layers, 24);
        assert_eq!(meta.max_total_len, 2048);
    }
}
