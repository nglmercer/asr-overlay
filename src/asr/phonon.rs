//! Phonon-2 through Fermion's separately managed CPU speech server.
//! Published weights have no ONNX export; inference stays in the upstream
//! PyTorch/MLX runtime rather than loading the archive with `ort`.

use anyhow::{bail, Context, Result};

use super::{openai_compatible::OpenAiCompatibleAsr, provider::UtteranceAsr};

pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8000/v1";
pub const DEFAULT_MODEL: &str = "phonon-2";

pub struct PhononAsr {
    http: OpenAiCompatibleAsr,
}

impl PhononAsr {
    pub fn new(base_url: &str, model: &str, api_key: Option<String>) -> Self {
        Self {
            http: OpenAiCompatibleAsr::new(
                if base_url.trim().is_empty() {
                    DEFAULT_BASE_URL
                } else {
                    base_url.trim()
                },
                if model.trim().is_empty() {
                    DEFAULT_MODEL
                } else {
                    model.trim()
                },
                api_key,
            ),
        }
    }
}

impl UtteranceAsr for PhononAsr {
    fn transcribe(&mut self, pcm_16k: &[f32], language: Option<&str>) -> Result<String> {
        if let Some(language) = language {
            let language = language.trim();
            if !language.is_empty()
                && !language.eq_ignore_ascii_case("auto")
                && !language.eq_ignore_ascii_case("en")
                && !language.eq_ignore_ascii_case("english")
            {
                bail!("Phonon-2 supports English only; select English/auto or use ARK for other languages");
            }
        }
        // Fermion accepts but ignores language hints. Never imply that the
        // server can change language or translate the audio.
        self.http.transcribe(pcm_16k, None).context(
            "Phonon-2 transcription failed; start `fermion serve phonon-2 --port 8000` and check the configured base URL/model",
        )
    }

    fn prepare(&mut self) -> Result<()> {
        // Models is authenticated and validates both server readiness and
        // the requested model before microphone utterances are consumed.
        let url = format!("{}/models", self.http.base_url());
        let mut request = ureq::get(&url);
        // OpenAiCompatibleAsr owns the key; expose no credentials in logs.
        // Readiness is checked using the same authentication as transcription.
        if let Some(key) = self.http.api_key() {
            request = request.header("Authorization", format!("Bearer {key}"));
        }
        let mut response = request.call().context(
            "Phonon-2 server unavailable; start `fermion serve phonon-2 --port 8000` and check base URL/authentication",
        )?;
        let raw = response.body_mut().read_to_string()?;
        let models: serde_json::Value =
            serde_json::from_str(&raw).context("Phonon server returned an invalid model list")?;
        let requested = self.http.model();
        let found = models["data"].as_array().is_some_and(|models| {
            models.iter().any(|model| {
                let id = model["id"].as_str().unwrap_or("");
                id == requested || (requested == DEFAULT_MODEL && id == "FermionResearch/Phonon-2")
            })
        });
        if !found {
            bail!("Phonon server does not list configured model {requested}; start `fermion serve phonon-2` or correct the model setting");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn defaults_are_local_and_english_only() {
        let mut asr = PhononAsr::new(" ", " ", None);
        assert_eq!(asr.http.base_url(), DEFAULT_BASE_URL);
        assert_eq!(asr.http.model(), DEFAULT_MODEL);
        assert!(asr.transcribe(&[0.0], Some("es")).is_err());
        for language in [None, Some("en"), Some("AUTO")] {
            assert_eq!(asr.transcribe(&[], language).unwrap(), "");
        }
    }

    #[test]
    fn server_contract_checks_readiness_and_sends_unpadded_wav() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            for index in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .unwrap();
                let mut bytes = Vec::new();
                let mut buf = [0; 4096];
                loop {
                    let n = stream.read(&mut buf).unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&buf[..n]);
                    if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&bytes[..end]);
                        let length = headers
                            .lines()
                            .find_map(|line| {
                                let (key, value) = line.split_once(':')?;
                                key.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if bytes.len() >= end + 4 + length {
                            break;
                        }
                    }
                }
                let request = String::from_utf8_lossy(&bytes);
                assert!(request
                    .to_ascii_lowercase()
                    .contains("authorization: bearer test-key"));
                let response = if index == 0 {
                    assert!(request.starts_with("GET /v1/models "));
                    r#"{"data":[{"id":"FermionResearch/Phonon-2"}]}"#
                } else {
                    assert!(request.starts_with("POST /v1/audio/transcriptions "));
                    assert!(request.contains("name=\"model\"\r\n\r\nphonon-2"));
                    assert!(!request.contains("name=\"language\""));
                    let start = bytes.windows(4).position(|w| w == b"RIFF").unwrap();
                    assert_eq!(
                        u32::from_le_bytes(bytes[start + 40..start + 44].try_into().unwrap()),
                        32
                    );
                    r#"{"text":"Hello from Phonon."}"#
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(),
                    response
                )
                .unwrap();
            }
        });
        let mut asr = PhononAsr::new(&url, "", Some("test-key".into()));
        asr.prepare().unwrap();
        assert_eq!(
            asr.transcribe(&[0.1; 16], Some("en")).unwrap(),
            "Hello from Phonon."
        );
        server.join().unwrap();
    }
}
