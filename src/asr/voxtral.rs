//! Mistral Voxtral realtime transcription over WebSocket.
//!
//! Streams 16 kHz mono `pcm_s16le` as binary frames and maps the server's
//! partial/final transcript messages to
//! [`TranscriptEvent`](crate::asr::provider::TranscriptEvent).
//!
//! Session shape: connect with a `Bearer` key, send one JSON config frame,
//! then binary PCM until the pipeline stops. The inbound parser accepts the
//! common realtime shapes (`type`/`text`/`is_final` as well as OpenAI-style
//! `transcription.delta` / `transcription.completed`) so minor server-side
//! variations still caption instead of going silent.
//!
//! Like all streaming providers this reconnects with backoff on a dropped
//! connection until stopped, and receives audio exactly as captured — no VAD
//! buffering, no `min_audio_ms` padding.

use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{Receiver, Sender};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc::UnboundedReceiver;

use super::provider::{push_transcript_chunk, should_stop, StreamingAsr, TranscriptEvent};

/// Mistral realtime transcription endpoint.
pub const REALTIME_WS_URL: &str = "wss://api.mistral.ai/v1/audio/transcriptions/stream";

/// Used when no model is configured.
pub const DEFAULT_MODEL: &str = "voxtral-mini-latest";

/// How often the session loop polls the stop channel.
const STOP_POLL: Duration = Duration::from_millis(50);

/// [`StreamingAsr`] over the Voxtral realtime WebSocket.
pub struct VoxtralRealtimeAsr {
    api_key: String,
    model: String,
    language: Option<String>,
}

impl VoxtralRealtimeAsr {
    pub fn new(api_key: String, model: &str, language: Option<&str>) -> Self {
        let model = if model.trim().is_empty() {
            DEFAULT_MODEL
        } else {
            model.trim()
        };
        Self {
            api_key,
            model: model.to_string(),
            language: language
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_string),
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    fn url(&self) -> String {
        format!("{REALTIME_WS_URL}?model={}", self.model)
    }

    /// The opening config frame describing the audio to follow.
    pub fn config_message(&self) -> String {
        let mut config = serde_json::json!({
            "type": "config",
            "model": self.model,
            "sample_rate": 16000,
            "channels": 1,
            "format": "pcm_s16le",
        });
        if let Some(language) = &self.language {
            config["language"] = serde_json::Value::String(language.clone());
        }
        config.to_string()
    }
}

impl StreamingAsr for VoxtralRealtimeAsr {
    fn run(
        self: Box<Self>,
        audio_rx: Receiver<Vec<i16>>,
        event_tx: Sender<TranscriptEvent>,
        stop_rx: Receiver<()>,
    ) -> anyhow::Result<()> {
        // The pipeline calls this on a plain worker thread, so a private
        // current-thread runtime is enough.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("Failed to start the Voxtral runtime")?;
        runtime.block_on(run_reconnecting(*self, audio_rx, event_tx, stop_rx))
    }
}

enum SessionOutcome {
    Stopped,
    AudioClosed,
    Failed(anyhow::Error),
}

async fn run_reconnecting(
    asr: VoxtralRealtimeAsr,
    audio_rx: Receiver<Vec<i16>>,
    event_tx: Sender<TranscriptEvent>,
    stop_rx: Receiver<()>,
) -> Result<()> {
    // Bridge the sync audio channel into the async session loop.
    let (bridge_tx, mut bridge_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<i16>>();
    let bridge = tokio::task::spawn_blocking(move || {
        while let Ok(chunk) = audio_rx.recv() {
            if bridge_tx.send(chunk).is_err() {
                break;
            }
        }
    });

    let mut backoff_secs = 1u64;
    let mut first_session = true;
    loop {
        // Shutdown also propagates through the audio channel closing and
        // through failed event sends inside the session, so the stop signal
        // is the only thing to poll here.
        if should_stop(&stop_rx) {
            break;
        }
        if !first_session {
            // Drop anything buffered during the outage: live captions must
            // describe the present, not the gap.
            let mut dropped = 0usize;
            while bridge_rx.try_recv().is_ok() {
                dropped += 1;
            }
            if dropped > 0 {
                log::debug!("Voxtral: dropped {dropped} stale audio chunks");
            }
        }
        first_session = false;
        match run_session(&asr, &mut bridge_rx, &event_tx, &stop_rx).await {
            SessionOutcome::Stopped | SessionOutcome::AudioClosed => break,
            SessionOutcome::Failed(e) => {
                if should_stop(&stop_rx) {
                    break;
                }
                log::warn!("Voxtral disconnected ({e:#}); reconnecting in {backoff_secs}s");
                tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
                backoff_secs = (backoff_secs * 2).min(15);
            }
        }
    }
    bridge.abort();
    Ok(())
}

async fn run_session(
    asr: &VoxtralRealtimeAsr,
    audio: &mut UnboundedReceiver<Vec<i16>>,
    events: &Sender<TranscriptEvent>,
    stop: &Receiver<()>,
) -> SessionOutcome {
    let request = match http::Request::builder()
        .uri(asr.url())
        .header("Authorization", format!("Bearer {}", asr.api_key))
        .body(())
    {
        Ok(request) => request,
        Err(e) => return SessionOutcome::Failed(anyhow!("Invalid realtime request: {e}")),
    };
    let (mut ws, _) = match tokio_tungstenite::connect_async(request).await {
        Ok(pair) => pair,
        Err(e) => return SessionOutcome::Failed(anyhow!("Connect failed: {e}")),
    };
    if let Err(e) = ws
        .send(tokio_tungstenite::tungstenite::Message::Text(
            asr.config_message().into(),
        ))
        .await
    {
        return SessionOutcome::Failed(anyhow!("Config send failed: {e}"));
    }
    log::info!("Voxtral realtime session ready (model {})", asr.model());

    let mut transcript = String::new();
    let mut stop_tick = tokio::time::interval(STOP_POLL);
    loop {
        tokio::select! {
            _ = stop_tick.tick() => {
                if should_stop(stop) {
                    let _ = ws.close(None).await;
                    return SessionOutcome::Stopped;
                }
            }
            chunk = audio.recv() => match chunk {
                Some(chunk) => {
                    let mut pcm = Vec::with_capacity(chunk.len() * 2);
                    for sample in &chunk {
                        pcm.extend_from_slice(&sample.to_le_bytes());
                    }
                    if let Err(e) = ws
                        .send(tokio_tungstenite::tungstenite::Message::Binary(pcm.into()))
                        .await
                    {
                        return SessionOutcome::Failed(anyhow!("Audio send failed: {e}"));
                    }
                }
                // The microphone closed: flush what was recognized so far.
                None => {
                    if !transcript.is_empty() {
                        let _ = events.send(TranscriptEvent::Final(std::mem::take(&mut transcript)));
                    }
                    let _ = ws.close(None).await;
                    return SessionOutcome::AudioClosed;
                }
            },
            msg = ws.next() => match msg {
                Some(Ok(tokio_tungstenite::tungstenite::Message::Text(text))) => {
                    let text: &str = text.as_ref();
                    if let Some(detail) = parse_error(text) {
                        return SessionOutcome::Failed(anyhow!("Server error: {detail}"));
                    }
                    for event in parse_server_message(text, &mut transcript) {
                        if events.send(event).is_err() {
                            return SessionOutcome::Stopped;
                        }
                    }
                }
                Some(Ok(_)) => {} // binary/ping/pong/close frames carry no transcripts
                Some(Err(e)) => return SessionOutcome::Failed(anyhow!("Connection failed: {e}")),
                None => return SessionOutcome::Failed(anyhow!("Server closed the connection")),
            },
        }
    }
}

/// Pull a human-readable error out of a server message, if it carries one.
pub fn parse_error(raw: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    if let Some(message) = value
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
    {
        return Some(message.to_string());
    }
    match value.get("type").and_then(|t| t.as_str()) {
        Some("error") => Some(
            value
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error")
                .to_string(),
        ),
        _ => None,
    }
}

/// Fold one server message into the in-progress turn, returning caption events.
///
/// Accepts the plain `{type, text, is_final}` shape, `{transcript, final}`
/// variants, and OpenAI realtime-style `transcription.delta` /
/// `transcription.completed` events.
pub fn parse_server_message(raw: &str, buffer: &mut String) -> Vec<TranscriptEvent> {
    let mut events = Vec::new();
    let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) else {
        return events;
    };
    if value.get("type").and_then(|t| t.as_str()) == Some("error") {
        return events;
    }

    let kind = value.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let text = ["text", "transcript", "delta"]
        .iter()
        .filter_map(|key| value.get(key).and_then(|v| v.as_str()))
        .next()
        .unwrap_or("");

    let mut is_final = value
        .get("is_final")
        .or_else(|| value.get("isFinal"))
        .or_else(|| value.get("final"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    is_final |= matches!(
        value.get("status").and_then(|s| s.as_str()),
        Some("final" | "completed" | "done")
    );
    is_final |= matches!(kind, "final" | "completed" | "done" | "transcript_final")
        || kind.contains("completed")
        || kind == "transcription.done";

    if !text.is_empty() {
        push_transcript_chunk(buffer, text);
        if is_final {
            events.push(TranscriptEvent::Final(std::mem::take(buffer)));
        } else {
            events.push(TranscriptEvent::Partial(buffer.clone()));
        }
    } else if is_final && !buffer.is_empty() {
        // A bare completion marker finalizes whatever was buffered.
        events.push(TranscriptEvent::Final(std::mem::take(buffer)));
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_frame_describes_pcm16_audio() {
        let asr = VoxtralRealtimeAsr::new("k".to_string(), "voxtral-mini-latest", Some("en"));
        let config: serde_json::Value = serde_json::from_str(&asr.config_message()).unwrap();
        assert_eq!(config["sample_rate"], 16000);
        assert_eq!(config["channels"], 1);
        assert_eq!(config["format"], "pcm_s16le");
        assert_eq!(config["model"], "voxtral-mini-latest");
        assert_eq!(config["language"], "en");
        assert!(asr.url().contains("model=voxtral-mini-latest"));
    }

    #[test]
    fn empty_model_falls_back_to_default() {
        let asr = VoxtralRealtimeAsr::new("k".to_string(), "", None);
        assert_eq!(asr.model(), DEFAULT_MODEL);
        let config: serde_json::Value = serde_json::from_str(&asr.config_message()).unwrap();
        assert!(config.get("language").is_none());
    }

    #[test]
    fn plain_partial_final_shape() {
        let mut buf = String::new();
        assert_eq!(
            parse_server_message(r#"{"type": "partial", "text": "hola"}"#, &mut buf),
            vec![TranscriptEvent::Partial("hola".to_string())]
        );
        assert_eq!(
            parse_server_message(
                r#"{"type": "transcript", "text": "hola mundo", "is_final": true}"#,
                &mut buf
            ),
            vec![TranscriptEvent::Final("hola mundo".to_string())]
        );
        assert!(buf.is_empty());
    }

    #[test]
    fn transcript_and_final_variants() {
        let mut buf = String::new();
        assert_eq!(
            parse_server_message(r#"{"transcript": "bon", "final": false}"#, &mut buf),
            vec![TranscriptEvent::Partial("bon".to_string())]
        );
        assert_eq!(
            parse_server_message(r#"{"type": "final", "transcript": "bonjour"}"#, &mut buf),
            vec![TranscriptEvent::Final("bonjour".to_string())]
        );
    }

    #[test]
    fn openai_realtime_style_events() {
        let mut buf = String::new();
        assert_eq!(
            parse_server_message(
                r#"{"type": "conversation.item.input_audio_transcription.delta", "delta": "hel"}"#,
                &mut buf
            ),
            vec![TranscriptEvent::Partial("hel".to_string())]
        );
        assert_eq!(
            parse_server_message(
                r#"{"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello"}"#,
                &mut buf
            ),
            vec![TranscriptEvent::Final("hello".to_string())]
        );
    }

    #[test]
    fn bare_completion_finalizes_the_buffer() {
        let mut buf = String::new();
        parse_server_message(r#"{"type": "partial", "text": "pending"}"#, &mut buf);
        assert_eq!(
            parse_server_message(r#"{"type": "done"}"#, &mut buf),
            vec![TranscriptEvent::Final("pending".to_string())]
        );
    }

    #[test]
    fn noise_and_errors_yield_no_events() {
        let mut buf = String::new();
        for raw in [
            "not json",
            "{}",
            r#"{"type": "connected", "session": "abc"}"#,
            r#"{"type": "error", "message": "bad key"}"#,
            r#"{"error": {"message": "bad key"}}"#,
        ] {
            assert!(parse_server_message(raw, &mut buf).is_empty(), "{raw}");
        }
        assert_eq!(
            parse_error(r#"{"type": "error", "message": "bad key"}"#).as_deref(),
            Some("bad key")
        );
        assert!(parse_error(r#"{"type": "partial", "text": "x"}"#).is_none());
    }
}
