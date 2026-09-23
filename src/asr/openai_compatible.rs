//! OpenAI-compatible transcription over HTTP.
//!
//! `POST {base_url}/audio/transcriptions` with a `multipart/form-data` body
//! (`file`, `model`, optional `language`), parsing `{"text": "…"}` back.
//! Covers the OpenAI API itself, self-hosted Whisper servers
//! (whisper.cpp `server`, faster-whisper, …) and Mistral batch Voxtral,
//! which all speak this shape.
//!
//! Endpoints without authentication are supported: when no API key resolves,
//! the request is simply sent without an `Authorization` header.
//!
//! Audio goes out exactly as segmented — no `min_audio_ms` padding. That
//! padding exists only to keep the local ARK decoder on-language.

use std::io::Read;

use anyhow::{anyhow, Context, Result};

use super::provider::UtteranceAsr;

/// Used when no base URL is configured.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Used when no model is configured.
pub const DEFAULT_MODEL: &str = "whisper-1";

/// [`UtteranceAsr`] over `POST {base_url}/audio/transcriptions`.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleAsr {
    base_url: String,
    model: String,
    api_key: Option<String>,
}

impl OpenAiCompatibleAsr {
    pub fn new(base_url: &str, model: &str, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key: api_key.filter(|k| !k.is_empty()),
        }
    }

    pub fn base_url(&self) -> &str {
        if self.base_url.is_empty() {
            DEFAULT_BASE_URL
        } else {
            &self.base_url
        }
    }

    pub fn model(&self) -> &str {
        if self.model.is_empty() {
            DEFAULT_MODEL
        } else {
            &self.model
        }
    }

    pub fn has_api_key(&self) -> bool {
        self.api_key.is_some()
    }

    fn transcriptions_url(&self) -> String {
        format!("{}/audio/transcriptions", self.base_url())
    }

    fn post(&self, url: &str, body: Vec<u8>, content_type: &str) -> Result<String> {
        let mut request = ureq::post(url).header("Content-Type", content_type);
        if let Some(key) = &self.api_key {
            request = request.header("Authorization", format!("Bearer {key}"));
        }
        let response = request
            .send(body)
            .map_err(|e| anyhow!("Transcription request failed: {e}"))?;

        let status = response.status();
        let mut reader = response.into_body().into_reader();
        let mut raw = Vec::new();
        reader.read_to_end(&mut raw)?;
        let body = String::from_utf8_lossy(&raw);
        if !(200..300).contains(&status.as_u16()) {
            return Err(anyhow!(
                "Transcription server returned HTTP {}: {}",
                status.as_u16(),
                body.chars().take(300).collect::<String>()
            ));
        }
        Ok(body.into_owned())
    }
}

impl UtteranceAsr for OpenAiCompatibleAsr {
    fn transcribe(&mut self, pcm_16k: &[f32], language: Option<&str>) -> Result<String> {
        if pcm_16k.is_empty() {
            return Ok(String::new());
        }
        let wav = encode_wav_16k_mono(pcm_16k);
        let boundary = format!("asr-overlay-{}", std::process::id());
        let (body, content_type) = build_multipart(&boundary, self.model(), language, &wav);
        let url = self.transcriptions_url();
        log::debug!(
            "POST {url} model={} ({} samples)",
            self.model(),
            pcm_16k.len()
        );
        let response = self.post(&url, body, &content_type)?;
        parse_transcription_response(&response)
    }
}

/// Encode mono `f32` at 16 kHz as a 16-bit PCM WAV.
pub fn encode_wav_16k_mono(pcm: &[f32]) -> Vec<u8> {
    let samples: Vec<i16> = pcm
        .iter()
        .map(|s| super::provider::f32_to_i16(*s))
        .collect();
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&16_000u32.to_le_bytes());
    wav.extend_from_slice(&32_000u32.to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

/// Build the OpenAI transcription request body. Returns the body and its
/// `Content-Type` (with boundary).
pub fn build_multipart(
    boundary: &str,
    model: &str,
    language: Option<&str>,
    wav: &[u8],
) -> (Vec<u8>, String) {
    let mut body = Vec::new();
    let mut field = |name: &str, value: &[u8], filename: Option<&str>| {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        match filename {
            Some(filename) => body.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                )
                .as_bytes(),
            ),
            None => body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n").as_bytes(),
            ),
        }
        if filename.is_some() {
            body.extend_from_slice(b"Content-Type: audio/wav\r\n");
        }
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(value);
        body.extend_from_slice(b"\r\n");
    };
    field("file", wav, Some("audio.wav"));
    field("model", model.as_bytes(), None);
    if let Some(language) = language.filter(|l| !l.trim().is_empty()) {
        field("language", language.as_bytes(), None);
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (body, format!("multipart/form-data; boundary={boundary}"))
}

/// Pull the transcript out of an OpenAI transcription response.
pub fn parse_transcription_response(body: &str) -> Result<String> {
    let value: serde_json::Value =
        serde_json::from_str(body).context("Transcription server returned invalid JSON")?;
    value
        .get("text")
        .and_then(|t| t.as_str())
        .map(|t| t.trim().to_string())
        .ok_or_else(|| anyhow!("Transcription response has no \"text\" field: {body}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_describes_16khz_mono_pcm16() {
        let wav = encode_wav_16k_mono(&[0.0, 0.5, -0.5, 1.0]);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 1); // PCM
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 1); // mono
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
        assert_eq!(&wav[36..40], b"data");
        // 4 samples × 2 bytes, straight after the 44-byte header.
        assert_eq!(wav.len(), 44 + 8);
        assert_eq!(i16::from_le_bytes([wav[44], wav[45]]), 0);
        assert_eq!(
            i16::from_le_bytes([wav[46], wav[47]]),
            (0.5 * 32767.0) as i16
        );
    }

    #[test]
    fn short_clips_go_out_unpadded() {
        // A 2 s clip stays 2 s: min_audio_ms padding belongs to ARK only.
        let wav = encode_wav_16k_mono(&vec![0.1f32; 32_000]);
        assert_eq!(wav.len(), 44 + 32_000 * 2);
    }

    #[test]
    fn multipart_carries_file_model_and_language() {
        let wav = encode_wav_16k_mono(&[0.0; 16]);
        let (body, content_type) = build_multipart("test-boundary", "whisper-1", Some("en"), &wav);
        assert_eq!(content_type, "multipart/form-data; boundary=test-boundary");
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("--test-boundary\r\n"));
        assert!(body.contains("name=\"file\"; filename=\"audio.wav\""));
        assert!(body.contains("Content-Type: audio/wav"));
        assert!(body.contains("name=\"model\"\r\n\r\nwhisper-1"));
        assert!(body.contains("name=\"language\"\r\n\r\nen"));
        assert!(body.ends_with("--test-boundary--\r\n"));
    }

    #[test]
    fn multipart_omits_blank_language() {
        let wav = encode_wav_16k_mono(&[0.0; 8]);
        for language in [None, Some(""), Some("  ")] {
            let (body, _) = build_multipart("b", "m", language, &wav);
            assert!(
                !String::from_utf8_lossy(&body).contains("name=\"language\""),
                "language sent for {language:?}"
            );
        }
    }

    #[test]
    fn parses_the_openai_response_shape() {
        assert_eq!(
            parse_transcription_response(r#"{"text": "hola mundo"}"#).unwrap(),
            "hola mundo"
        );
        // Verbose responses carry the same field plus extras.
        assert_eq!(
            parse_transcription_response(
                r#"{"task": "transcribe", "language": "es", "duration": 1.2, "text": "  hola  "}"#
            )
            .unwrap(),
            "hola"
        );
        assert_eq!(parse_transcription_response(r#"{"text": ""}"#).unwrap(), "");
    }

    #[test]
    fn rejects_responses_without_text() {
        assert!(parse_transcription_response(r#"{"error": "boom"}"#).is_err());
        assert!(parse_transcription_response(r#"{"text": 42}"#).is_err());
        assert!(parse_transcription_response("not json").is_err());
    }

    #[test]
    fn endpoint_and_defaults() {
        let asr = OpenAiCompatibleAsr::new("http://localhost:8000/v1/", "whisper-large-v3", None);
        assert_eq!(
            asr.transcriptions_url(),
            "http://localhost:8000/v1/audio/transcriptions"
        );
        assert!(!asr.has_api_key());

        let asr = OpenAiCompatibleAsr::new("", "", Some("k".to_string()));
        assert_eq!(asr.base_url(), DEFAULT_BASE_URL);
        assert_eq!(asr.model(), DEFAULT_MODEL);
        assert!(asr.has_api_key());
        // Blank keys count as no key: some servers reject empty bearers.
        assert!(!OpenAiCompatibleAsr::new("", "", Some(String::new())).has_api_key());
    }

    #[test]
    fn empty_audio_short_circuits_without_network() {
        let mut asr = OpenAiCompatibleAsr::new("http://127.0.0.1:1", "m", None);
        assert_eq!(asr.transcribe(&[], None).unwrap(), "");
    }
}
