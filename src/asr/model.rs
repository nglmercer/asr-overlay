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
use ort::value::Tensor;
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
}

fn open_session(path: &Path, threads: usize) -> Result<Session> {
    Session::builder()
        .context("Failed to create ONNX session builder")?
        .with_intra_threads(threads)
        .context("Failed to set ONNX thread count")?
        .commit_from_file(path)
        .with_context(|| format!("Failed to load ONNX model {}", path.display()))
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
        prompt.push_str("Please transcribe this audio.");
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
            1usize,
            max_total,
            self.meta.num_key_value_heads,
            self.meta.head_dim(),
        ];
        for layer in 0..self.meta.num_layers {
            feeds.push((
                format!("cache_key_{layer}").into(),
                Tensor::from_array((cache_shape, self.caches[2 * layer].clone().into_boxed_slice()))?
                    .into(),
            ));
            feeds.push((
                format!("cache_value_{layer}").into(),
                Tensor::from_array((
                    cache_shape,
                    self.caches[2 * layer + 1].clone().into_boxed_slice(),
                ))?
                .into(),
            ));
        }

        let outputs = self.llm.run(feeds)?;

        // outputs[0] = logits, outputs[1..] = per-layer key/value deltas.
        let (logits_shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
        let vocab = *logits_shape.last().unwrap() as usize;
        let logits = logits[logits.len() - vocab..].to_vec();

        let stride = self.meta.num_key_value_heads * self.meta.head_dim();
        for layer in 0..self.meta.num_layers * 2 {
            let (_, delta) = outputs[layer + 1].try_extract_tensor::<f32>()?;
            let offset = start_pos * stride;
            let len = seq * stride;
            if offset + len <= self.caches[layer].len() && delta.len() >= len {
                self.caches[layer][offset..offset + len].copy_from_slice(&delta[..len]);
            }
        }

        Ok(logits)
    }

    /// Greedy argmax over the allowed token range.
    /// Ported from `AsrLogitsMask`: everything at or above
    /// `BLOCK_TOKEN_ID_FROM` is suppressed so the model emits text, not
    /// control tokens.
    fn pick_token(&self, logits: &[f32]) -> i64 {
        let limit = logits.len().min(BLOCK_TOKEN_ID_FROM);
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (id, &score) in logits[..limit].iter().enumerate() {
            if score > best_score {
                best_score = score;
                best = id;
            }
        }
        // The stop tokens live above the block threshold, so they are checked
        // separately — otherwise generation could never terminate.
        for &stop in &self.manifest.stop_token_ids {
            let idx = stop as usize;
            if idx < logits.len() && logits[idx] > best_score {
                best_score = logits[idx];
                best = idx;
            }
        }
        best as i64
    }

    /// Transcribe mono `f32` audio at 16 kHz.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        // Reset the KV cache between utterances.
        for cache in self.caches.iter_mut() {
            cache.fill(0.0);
        }

        let (audio_embeds, n_audio) = self.encode_audio(samples)?;
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

        let mut logits = self.forward(embeds, prompt_len, 0)?;
        let mut total_len = prompt_len;
        let mut out_ids: Vec<u32> = Vec::new();

        for _ in 0..self.max_new_tokens {
            let next = self.pick_token(&logits);
            if self.manifest.stop_token_ids.contains(&next) {
                break;
            }
            out_ids.push(next as u32);
            let embed = self.embed(&[next])?;
            logits = self.forward(embed, 1, total_len)?;
            total_len += 1;
        }

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
