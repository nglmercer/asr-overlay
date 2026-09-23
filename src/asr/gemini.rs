//! Gemini Live streaming transcription and translation.
//!
//! Opens the bidirectional Live API over WebSocket, streams 16 kHz mono
//! PCM (`audio/pcm;rate=16000`, ~100 ms chunks, base64) and maps the server's
//! transcription events to [`TranscriptEvent`](crate::asr::provider::TranscriptEvent):
//!
//! - transcribe mode displays `serverContent.inputTranscription.text`,
//! - translate mode displays `serverContent.outputTranscription.text`.
//!
//! Generated audio is never requested (`responseModalities: ["TEXT"]`) and
//! never played. On a dropped connection the session reconnects with
//! backoff until the pipeline stops; audio buffered while disconnected is
//! dropped rather than transcribed late.

use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{Receiver, Sender};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc::UnboundedReceiver;

use super::provider::{should_stop, StreamingAsr, TranscriptEvent};

/// Gemini Live bidirectional endpoint (see the Live API WebSocket docs).
pub const LIVE_WS_URL: &str = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";

/// Used when no model is configured.
pub const DEFAULT_MODEL: &str = "gemini-2.0-flash-live-001";

/// How long to wait for `setupComplete` before giving up on a session.
const SETUP_TIMEOUT: Duration = Duration::from_secs(15);

/// How often the session loop polls the stop channel.
const STOP_POLL: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeminiMode {
    Transcribe,
    Translate,
}

/// [`StreamingAsr`] over the Gemini Live WebSocket.
pub struct GeminiLiveAsr {
    api_key: String,
    model: String,
    mode: GeminiMode,
    target_language: String,
}

impl GeminiLiveAsr {
    pub fn new(api_key: String, model: &str, mode: GeminiMode, target_language: &str) -> Self {
        let model = if model.trim().is_empty() {
            DEFAULT_MODEL
        } else {
            model.trim()
        };
        Self {
            api_key,
            model: model.to_string(),
            mode,
            target_language: target_language.trim().to_string(),
        }
    }

    pub fn mode(&self) -> GeminiMode {
        self.mode
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    fn model_path(&self) -> String {
        if self.model.starts_with("models/") {
            self.model.clone()
        } else {
            format!("models/{}", self.model)
        }
    }

    /// The session handshake. Translation steering is a system instruction;
    /// the translated text arrives as `outputTranscription`.
    pub fn setup_message(&self) -> String {
        let mut setup = serde_json::json!({
            "model": self.model_path(),
            "generationConfig": { "responseModalities": ["TEXT"] },
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
        });
        if self.mode == GeminiMode::Translate {
            let target = language_name(&self.target_language);
            setup["systemInstruction"] = serde_json::json!({
                "parts": [{ "text": format!(
                    "Translate everything the user says into {target}. Output only the translation, nothing else."
                ) }]
            });
        }
        serde_json::json!({ "setup": setup }).to_string()
    }
}

impl StreamingAsr for GeminiLiveAsr {
    fn run(
        self: Box<Self>,
        audio_rx: Receiver<Vec<i16>>,
        event_tx: Sender<TranscriptEvent>,
        stop_rx: Receiver<()>,
    ) -> anyhow::Result<()> {
        // The pipeline calls this on a plain worker thread, so a private
        // current-thread runtime is enough — no sleeping executors needed.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("Failed to start the Gemini Live runtime")?;
        runtime.block_on(run_reconnecting(*self, audio_rx, event_tx, stop_rx))
    }
}

enum SessionOutcome {
    Stopped,
    AudioClosed,
    Failed(anyhow::Error),
}

async fn run_reconnecting(
    asr: GeminiLiveAsr,
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
                log::debug!("Gemini Live: dropped {dropped} stale audio chunks");
            }
        }
        first_session = false;
        match run_session(&asr, &mut bridge_rx, &event_tx, &stop_rx).await {
            SessionOutcome::Stopped | SessionOutcome::AudioClosed => break,
            SessionOutcome::Failed(e) => {
                if should_stop(&stop_rx) {
                    break;
                }
                log::warn!("Gemini Live disconnected ({e:#}); reconnecting in {backoff_secs}s");
                tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
                backoff_secs = (backoff_secs * 2).min(15);
            }
        }
    }
    bridge.abort();
    Ok(())
}

async fn run_session(
    asr: &GeminiLiveAsr,
    audio: &mut UnboundedReceiver<Vec<i16>>,
    events: &Sender<TranscriptEvent>,
    stop: &Receiver<()>,
) -> SessionOutcome {
    let request = match http::Request::builder()
        .uri(LIVE_WS_URL)
        .header("x-goog-api-key", asr.api_key.as_str())
        .body(())
    {
        Ok(request) => request,
        Err(e) => return SessionOutcome::Failed(anyhow!("Invalid Live API request: {e}")),
    };
    let (mut ws, _) = match tokio_tungstenite::connect_async(request).await {
        Ok(pair) => pair,
        Err(e) => return SessionOutcome::Failed(anyhow!("Connect failed: {e}")),
    };
    if let Err(e) = ws.send(audio_ws_text(asr.setup_message())).await {
        return SessionOutcome::Failed(anyhow!("Setup send failed: {e}"));
    }
    match tokio::time::timeout(SETUP_TIMEOUT, ws.next()).await {
        Ok(Some(Ok(msg))) if is_setup_complete(&ws_text(&msg)) => {}
        Ok(Some(Ok(msg))) => {
            let detail = parse_error(&ws_text(&msg)).unwrap_or_else(|| ws_text(&msg).into_owned());
            return SessionOutcome::Failed(anyhow!("Setup rejected: {detail}"));
        }
        Ok(Some(Err(e))) => return SessionOutcome::Failed(anyhow!("Setup failed: {e}")),
        Ok(None) => {
            return SessionOutcome::Failed(anyhow!("Server closed the connection during setup"))
        }
        Err(_) => return SessionOutcome::Failed(anyhow!("Timed out waiting for setupComplete")),
    }
    log::info!("Gemini Live session ready (model {})", asr.model());

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
                    if let Err(e) = ws.send(audio_ws_text(audio_message(&chunk))).await {
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
                Some(Ok(msg)) => {
                    let text = ws_text(&msg);
                    if let Some(detail) = parse_error(&text) {
                        return SessionOutcome::Failed(anyhow!("Server error: {detail}"));
                    }
                    for event in parse_server_message(&text, asr.mode, &mut transcript) {
                        if events.send(event).is_err() {
                            return SessionOutcome::Stopped;
                        }
                    }
                }
                Some(Err(e)) => return SessionOutcome::Failed(anyhow!("Connection failed: {e}")),
                None => return SessionOutcome::Failed(anyhow!("Server closed the connection")),
            },
        }
    }
}

fn audio_ws_text(text: String) -> tokio_tungstenite::tungstenite::Message {
    tokio_tungstenite::tungstenite::Message::Text(text.into())
}

fn ws_text(msg: &tokio_tungstenite::tungstenite::Message) -> std::borrow::Cow<'_, str> {
    match msg {
        tokio_tungstenite::tungstenite::Message::Text(text) => {
            std::borrow::Cow::Borrowed(text.as_ref())
        }
        _ => std::borrow::Cow::Borrowed(""),
    }
}

/// One realtime audio frame: base64 16-bit LE PCM at 16 kHz mono.
pub fn audio_message(chunk: &[i16]) -> String {
    use base64::Engine as _;
    let mut pcm = Vec::with_capacity(chunk.len() * 2);
    for sample in chunk {
        pcm.extend_from_slice(&sample.to_le_bytes());
    }
    serde_json::json!({
        "realtimeInput": {
            "mediaChunks": [{ "mimeType": "audio/pcm;rate=16000", "data": base64::engine::general_purpose::STANDARD.encode(&pcm) }]
        }
    })
    .to_string()
}

/// True for the handshake acknowledgement.
pub fn is_setup_complete(raw: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .and_then(|v| v.get("setupComplete").cloned())
        .is_some()
}

/// Pull a human-readable error out of a server message, if it carries one.
pub fn parse_error(raw: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    let error = value.get("error")?;
    if let Some(message) = error.get("message").and_then(|m| m.as_str()) {
        return Some(message.to_string());
    }
    Some(error.to_string())
}

/// Fold one server message into the in-progress turn, returning caption events.
///
/// `buffer` accumulates the turn's text across messages; a turn ends on
/// `turnComplete` (or a `finished` transcription), which emits the `Final`.
pub fn parse_server_message(
    raw: &str,
    mode: GeminiMode,
    buffer: &mut String,
) -> Vec<TranscriptEvent> {
    let mut events = Vec::new();
    let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) else {
        return events;
    };
    let Some(content) = value.get("serverContent") else {
        return events;
    };
    let key = match mode {
        GeminiMode::Transcribe => "inputTranscription",
        GeminiMode::Translate => "outputTranscription",
    };
    let transcription = content.get(key);
    if let Some(text) = transcription
        .and_then(|t| t.get("text"))
        .and_then(|t| t.as_str())
    {
        if !text.is_empty() {
            super::provider::push_transcript_chunk(buffer, text);
            events.push(TranscriptEvent::Partial(buffer.clone()));
        }
    }
    // A finished turn always finalizes, even when the last message carries no
    // new text (turnComplete typically arrives on its own message).
    let turn_done = content
        .get("turnComplete")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let finished = transcription
        .and_then(|t| t.get("finished"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if (turn_done || finished) && !buffer.is_empty() {
        events.push(TranscriptEvent::Final(std::mem::take(buffer)));
    }
    events
}

/// Best-effort language name for the translation instruction.
pub fn language_name(code: &str) -> String {
    let code = code.trim().to_lowercase();
    match code.as_str() {
        "en" | "eng" | "english" => "English",
        "es" | "spa" | "spanish" | "español" => "Spanish",
        "fr" | "fra" | "french" | "français" => "French",
        "de" | "deu" | "german" | "deutsch" => "German",
        "it" | "ita" | "italian" | "italiano" => "Italian",
        "pt" | "por" | "portuguese" | "português" => "Portuguese",
        "zh" | "zho" | "chinese" | "中文" => "Chinese",
        "ja" | "jpn" | "japanese" | "日本語" => "Japanese",
        "ko" | "kor" | "korean" | "한국어" => "Korean",
        "nl" | "nld" | "dutch" => "Dutch",
        "pl" | "pol" | "polish" => "Polish",
        "ro" | "ron" | "romanian" => "Romanian",
        "hu" | "hun" | "hungarian" => "Hungarian",
        "cs" | "ces" | "czech" => "Czech",
        "fi" | "fin" | "finnish" => "Finnish",
        "hr" | "hrv" | "croatian" => "Croatian",
        "sk" | "slk" | "slovak" => "Slovak",
        "sl" | "slv" | "slovene" | "slovenian" => "Slovene",
        "et" | "est" | "estonian" => "Estonian",
        "lt" | "lit" | "lithuanian" => "Lithuanian",
        "" => "English",
        other => return code_from(other),
    }
    .to_string()
}

fn code_from(other: &str) -> String {
    // Unknown codes pass through rather than failing: the model understands
    // far more language names than this table lists.
    other.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_requests_text_and_transcriptions() {
        let asr = GeminiLiveAsr::new(
            "k".to_string(),
            "gemini-2.0-flash-live-001",
            GeminiMode::Transcribe,
            "",
        );
        let setup: serde_json::Value = serde_json::from_str(&asr.setup_message()).unwrap();
        let setup = &setup["setup"];
        assert_eq!(setup["model"], "models/gemini-2.0-flash-live-001");
        assert_eq!(
            setup["generationConfig"]["responseModalities"],
            serde_json::json!(["TEXT"])
        );
        assert!(setup.get("inputAudioTranscription").is_some());
        assert!(setup.get("outputAudioTranscription").is_some());
        assert!(setup.get("systemInstruction").is_none());
    }

    #[test]
    fn translate_setup_names_the_target_language() {
        let asr = GeminiLiveAsr::new("k".to_string(), "", GeminiMode::Translate, "es");
        assert_eq!(asr.model(), DEFAULT_MODEL);
        let setup: serde_json::Value = serde_json::from_str(&asr.setup_message()).unwrap();
        let instruction = setup["setup"]["systemInstruction"]["parts"][0]["text"]
            .as_str()
            .unwrap();
        assert!(instruction.contains("Spanish"), "{instruction}");
    }

    #[test]
    fn model_path_prefixing() {
        let asr = GeminiLiveAsr::new("k".to_string(), "models/custom", GeminiMode::Transcribe, "");
        assert!(asr.setup_message().contains("models/custom"));
        assert!(!asr.setup_message().contains("models/models/"));
    }

    #[test]
    fn audio_frames_carry_base64_pcm16() {
        use base64::Engine as _;
        let msg: serde_json::Value = serde_json::from_str(&audio_message(&[0, 1, -1])).unwrap();
        let chunk = &msg["realtimeInput"]["mediaChunks"][0];
        assert_eq!(chunk["mimeType"], "audio/pcm;rate=16000");
        let raw = base64::engine::general_purpose::STANDARD
            .decode(chunk["data"].as_str().unwrap())
            .unwrap();
        assert_eq!(raw, vec![0, 0, 1, 0, 255, 255]);
    }

    #[test]
    fn transcribe_reads_input_transcription() {
        let mut buf = String::new();
        let events = parse_server_message(
            r#"{"serverContent": {"inputTranscription": {"text": "hola"}}}"#,
            GeminiMode::Transcribe,
            &mut buf,
        );
        assert_eq!(events, vec![TranscriptEvent::Partial("hola".to_string())]);
        // …and ignores the translation field.
        let mut buf = String::new();
        let events = parse_server_message(
            r#"{"serverContent": {"outputTranscription": {"text": "hello"}}}"#,
            GeminiMode::Transcribe,
            &mut buf,
        );
        assert!(events.is_empty());
    }

    #[test]
    fn translate_reads_output_transcription() {
        let mut buf = String::new();
        let events = parse_server_message(
            r#"{"serverContent": {"outputTranscription": {"text": "hello"}}}"#,
            GeminiMode::Translate,
            &mut buf,
        );
        assert_eq!(events, vec![TranscriptEvent::Partial("hello".to_string())]);
    }

    #[test]
    fn delta_chunks_accumulate_and_turn_complete_finalizes() {
        let mut buf = String::new();
        let first = parse_server_message(
            r#"{"serverContent": {"inputTranscription": {"text": "hola "}}}"#,
            GeminiMode::Transcribe,
            &mut buf,
        );
        assert_eq!(first, vec![TranscriptEvent::Partial("hola ".to_string())]);
        let second = parse_server_message(
            r#"{"serverContent": {"inputTranscription": {"text": "mundo"}, "turnComplete": true}}"#,
            GeminiMode::Transcribe,
            &mut buf,
        );
        assert_eq!(
            second,
            vec![
                TranscriptEvent::Partial("hola mundo".to_string()),
                TranscriptEvent::Final("hola mundo".to_string()),
            ]
        );
        assert!(buf.is_empty(), "buffer must reset for the next turn");
    }

    #[test]
    fn cumulative_resends_replace_instead_of_doubling() {
        let mut buf = String::new();
        parse_server_message(
            r#"{"serverContent": {"inputTranscription": {"text": "hola"}}}"#,
            GeminiMode::Transcribe,
            &mut buf,
        );
        let events = parse_server_message(
            r#"{"serverContent": {"inputTranscription": {"text": "hola mundo"}}}"#,
            GeminiMode::Transcribe,
            &mut buf,
        );
        assert_eq!(
            events,
            vec![TranscriptEvent::Partial("hola mundo".to_string())]
        );
    }

    #[test]
    fn finished_flag_finalizes_without_turn_complete() {
        let mut buf = String::new();
        let events = parse_server_message(
            r#"{"serverContent": {"inputTranscription": {"text": "done", "finished": true}}}"#,
            GeminiMode::Transcribe,
            &mut buf,
        );
        assert_eq!(
            events.last(),
            Some(&TranscriptEvent::Final("done".to_string()))
        );
    }

    #[test]
    fn bare_turn_complete_finalizes_buffered_text() {
        let mut buf = "pending".to_string();
        let events = parse_server_message(
            r#"{"serverContent": {"turnComplete": true}}"#,
            GeminiMode::Transcribe,
            &mut buf,
        );
        assert_eq!(events, vec![TranscriptEvent::Final("pending".to_string())]);
    }

    #[test]
    fn noise_yields_no_events() {
        let mut buf = String::new();
        for raw in [
            "not json",
            "{}",
            r#"{"setupComplete": {}}"#,
            r#"{"serverContent": {"modelTurn": {"parts": [{"text": "hi"}]}}}"#,
            r#"{"serverContent": {"inputTranscription": {"text": ""}}}"#,
        ] {
            assert!(
                parse_server_message(raw, GeminiMode::Transcribe, &mut buf).is_empty(),
                "{raw}"
            );
        }
        assert!(is_setup_complete(r#"{"setupComplete": {}}"#));
        assert!(!is_setup_complete(r#"{"serverContent": {}}"#));
    }

    #[test]
    fn server_errors_surface() {
        assert_eq!(
            parse_error(r#"{"error": {"message": "quota exceeded"}}"#).as_deref(),
            Some("quota exceeded")
        );
        assert!(parse_error(r#"{"serverContent": {}}"#).is_none());
    }

    #[test]
    fn language_names() {
        assert_eq!(language_name("es"), "Spanish");
        assert_eq!(language_name("EN"), "English");
        assert_eq!(language_name(""), "English");
        assert_eq!(language_name("klingon"), "klingon");
    }
}
