use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use parking_lot::Mutex;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::broadcast;
use tower_http::cors::{self, CorsLayer};

use crate::caption::Caption;
use crate::config::{AsrConfig, OverlayStyle};
use crate::rate_limit::{rate_limit_middleware, RateLimiters};
use crate::render::{render_caption_png, RenderStyle};

mod embedded_assets {
    include!(concat!(env!("OUT_DIR"), "/embedded_assets.rs"));
}

#[derive(Clone, Copy, PartialEq)]
pub enum ListeningState {
    Idle,
    Starting,
    Listening,
}

pub struct AppState {
    pub config: Mutex<AsrConfig>,
    pub caption_tx: broadcast::Sender<Caption>,
    pub last_caption: Arc<Mutex<Option<Caption>>>,
    pub listening: Mutex<ListeningState>,
    pub rate_limiters: RateLimiters,
    pub render_style: Mutex<RenderStyle>,
    /// Style updates pushed to connected overlays so edits in the config page
    /// appear in OBS without reloading the Browser Source.
    pub style_tx: broadcast::Sender<OverlayStyle>,
}

/// True while the pipeline runs, or a start is in flight.
///
/// The controller owns the truth: auto-start and the tray drive it without
/// going through these handlers, so the `listening` field alone would go stale
/// and, among other things, stop a microphone change from restarting capture.
fn is_listening(state: &AppState) -> bool {
    crate::controller::is_listening() || *state.listening.lock() == ListeningState::Starting
}

pub fn build_router(state: Arc<AppState>) -> Router {
    let rl = Arc::new(state.rate_limiters.clone());

    Router::new()
        .route("/", get(serve_index))
        .route("/config", get(|| async { serve_embedded("config.html") }))
        .route("/style", post(set_style))
        .route("/ws", get(ws_stream))
        .route("/caption.png", get(caption_png))
        .route("/settings", get(get_config).post(set_config))
        .route("/status", get(caption_status))
        .route("/devices", get(list_devices))
        .route("/start", post(start_listening))
        .route("/stop", post(stop_listening))
        .route("/index.js", get(|| async { serve_embedded("index.js") }))
        .route("/config.js", get(|| async { serve_embedded("config.js") }))
        .route("/style.js", get(|| async { serve_embedded("style.js") }))
        .route("/vendor.js", get(|| async { serve_embedded("vendor.js") }))
        .route(
            "/assets/{*path}",
            get(
                |axum::extract::Path(path): axum::extract::Path<String>| async move {
                    serve_embedded(&format!("assets/{path}"))
                },
            ),
        )
        .layer(middleware::from_fn_with_state(
            rl.clone(),
            rate_limit_middleware,
        ))
        .layer(
            CorsLayer::new()
                .allow_origin(cors::AllowOrigin::predicate(|origin, _| {
                    origin.as_bytes().starts_with(b"http://localhost")
                        || origin.as_bytes().starts_with(b"http://127.0.0.1")
                }))
                .allow_methods(cors::Any)
                .allow_headers(cors::Any),
        )
        .with_state(state)
}

async fn serve_index() -> Response {
    serve_embedded("index.html")
}

fn serve_embedded(path: &str) -> Response {
    let Some(bytes) = embedded_assets::get(path) else {
        return StatusCode::NOT_FOUND.into_response();
    };

    let mut response = Response::new(Body::from(Bytes::from_static(bytes)));
    response.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static(content_type(path)),
    );
    response
}

fn content_type(path: &str) -> &'static str {
    if path.ends_with(".html") {
        "text/html; charset=utf-8"
    } else if path.ends_with(".js") {
        "application/javascript; charset=utf-8"
    } else if path.ends_with(".css") {
        "text/css; charset=utf-8"
    } else if path.ends_with(".svg") {
        "image/svg+xml"
    } else if path.ends_with(".png") {
        "image/png"
    } else if path.ends_with(".json") {
        "application/json"
    } else {
        "application/octet-stream"
    }
}

/// Messages pushed to overlay clients, tagged so one socket can carry both
/// captions and live style updates.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum WsMessage {
    Caption(Caption),
    Style { style: OverlayStyle },
}

/// WebSocket stream: JSON captions plus style updates.
async fn ws_stream(
    ws: axum::extract::ws::WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> Response {
    let mut caption_rx = state.caption_tx.subscribe();
    let mut style_rx = state.style_tx.subscribe();

    ws.on_upgrade(move |mut socket| async move {
        loop {
            let message = tokio::select! {
                caption = caption_rx.recv() => match caption {
                    Ok(caption) => WsMessage::Caption(caption),
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                },
                style = style_rx.recv() => match style {
                    Ok(style) => WsMessage::Style { style },
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                },
            };

            let Ok(json) = serde_json::to_string(&message) else {
                continue;
            };
            if socket
                .send(axum::extract::ws::Message::Text(json.into()))
                .await
                .is_err()
            {
                break;
            }
        }
    })
}

/// Live style preview: update the in-memory style and broadcast it to every
/// connected overlay. Not persisted — POST /settings saves.
async fn set_style(
    State(state): State<Arc<AppState>>,
    axum::Json(style): axum::Json<OverlayStyle>,
) -> StatusCode {
    state.config.lock().style = style.clone();
    let _ = state.style_tx.send(style);
    StatusCode::OK
}

/// Serve the latest caption rendered as a transparent PNG via cairo.
async fn caption_png(State(state): State<Arc<AppState>>) -> Response {
    let caption = state.last_caption.lock().clone();
    let style = state.render_style.lock().clone();
    let Some(caption) = caption else {
        return StatusCode::NO_CONTENT.into_response();
    };

    match render_caption_png(&caption.text, &style) {
        Ok(png) => {
            let mut response = Response::new(Body::from(png));
            response.headers_mut().insert(
                axum::http::header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("image/png"),
            );
            response
        }
        Err(e) => {
            log::error!("cairo render failed: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn get_config(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let config = state.config.lock().clone();
    (StatusCode::OK, axum::Json(config)).into_response()
}

/// True when a settings change only takes effect on a pipeline restart.
///
/// The capture thread copies its config when it starts, so the input device,
/// the ARK-only knobs, and everything describing the provider require a
/// restart. The rest (VAD windows, style, …) is read live.
pub fn config_needs_restart(old: &AsrConfig, new: &AsrConfig) -> bool {
    old.input_device != new.input_device
        || old.threads != new.threads
        || old.min_audio_ms != new.min_audio_ms
        || old.provider != new.provider
}

async fn set_config(
    State(state): State<Arc<AppState>>,
    axum::Json(config): axum::Json<AsrConfig>,
) -> StatusCode {
    let (saved, needs_restart) = {
        let mut guard = state.config.lock();
        let needs_restart = config_needs_restart(&guard, &config);
        *guard = config;
        (guard.clone(), needs_restart)
    };

    crate::config::save(&saved);
    let _ = state.style_tx.send(saved.style);

    if needs_restart && crate::controller::is_listening() {
        // Restarting joins the pipeline thread and reloads the model, so keep
        // it off the async runtime's worker threads.
        let state = Arc::clone(&state);
        tokio::task::spawn_blocking(move || {
            log::info!("Capture settings changed — restarting the pipeline");
            crate::controller::stop();
            if let Err(e) = crate::controller::start(Arc::clone(&state)) {
                log::error!("Failed to restart capture: {e}");
                *state.listening.lock() = ListeningState::Idle;
            }
        });
    }

    StatusCode::OK
}

#[derive(Serialize)]
struct DeviceEntry {
    id: String,
    label: String,
    default: bool,
}

/// Input devices available for the settings page's picker.
async fn list_devices() -> impl IntoResponse {
    // Probing every device opens PCMs, so keep it off the async worker threads.
    let devices = tokio::task::spawn_blocking(crate::audio::list_input_devices)
        .await
        .unwrap_or_default();
    let devices: Vec<DeviceEntry> = devices
        .into_iter()
        .map(|d| DeviceEntry {
            id: d.id,
            label: d.label,
            default: d.default,
        })
        .collect();
    Json(devices)
}

#[derive(Serialize)]
struct StatusResponse {
    running: bool,
    listening: bool,
    has_caption: bool,
}

async fn caption_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(StatusResponse {
        running: true,
        listening: is_listening(&state),
        has_caption: state.last_caption.lock().is_some(),
    })
}

#[derive(Serialize)]
struct StartResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    listening: bool,
}

async fn start_listening(State(state): State<Arc<AppState>>) -> (StatusCode, Json<StartResponse>) {
    {
        let mut guard = state.listening.lock();
        if crate::controller::is_listening() || *guard == ListeningState::Starting {
            return (
                StatusCode::OK,
                Json(StartResponse {
                    ok: true,
                    error: None,
                    listening: true,
                }),
            );
        }
        *guard = ListeningState::Starting;
    }

    // Opening the microphone blocks; keep it off the async worker threads.
    let starting = Arc::clone(&state);
    let result = match tokio::task::spawn_blocking(move || crate::controller::start(starting)).await
    {
        Ok(result) => result,
        Err(e) => Err(anyhow::anyhow!("start task failed: {e}")),
    };

    let (ok, error, listening) = match result {
        Ok(()) => (true, None, true),
        Err(e) => (false, Some(e.to_string()), false),
    };

    {
        let mut guard = state.listening.lock();
        *guard = if listening {
            ListeningState::Listening
        } else {
            ListeningState::Idle
        };
    }

    if let Some(e) = &error {
        log::error!("Start listening failed: {e}");
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(StartResponse {
                ok,
                error: Some(e.clone()),
                listening,
            }),
        )
    } else {
        (
            StatusCode::OK,
            Json(StartResponse {
                ok: true,
                error: None,
                listening,
            }),
        )
    }
}

#[derive(Serialize)]
struct StopResponse {
    ok: bool,
    listening: bool,
}

async fn stop_listening(State(state): State<Arc<AppState>>) -> Json<StopResponse> {
    crate::controller::stop();
    {
        let mut guard = state.listening.lock();
        *guard = ListeningState::Idle;
    }
    Json(StopResponse {
        ok: true,
        listening: is_listening(&state),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderKind;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use std::sync::Arc;
    use tower::ServiceExt;

    #[test]
    fn provider_changes_need_a_restart() {
        let old = AsrConfig::default();
        // Provider kind, model, base URL, API-key env, target language and
        // streaming delay each restart the pipeline when listening.
        for mutate in [
            |c: &mut AsrConfig| c.provider.kind = ProviderKind::GeminiLiveTranscribe,
            |c: &mut AsrConfig| c.provider.model = "other".to_string(),
            |c: &mut AsrConfig| c.provider.base_url = "http://x:8000/v1".to_string(),
            |c: &mut AsrConfig| c.provider.api_key_env = "OTHER_KEY".to_string(),
            |c: &mut AsrConfig| c.provider.target_language = "fr".to_string(),
            |c: &mut AsrConfig| c.provider.streaming_delay_ms = 200,
            |c: &mut AsrConfig| c.input_device = "pw:mic".to_string(),
            |c: &mut AsrConfig| c.threads += 1,
            |c: &mut AsrConfig| c.min_audio_ms = 1000,
        ] {
            let mut new = old.clone();
            mutate(&mut new);
            assert!(config_needs_restart(&old, &new));
        }
    }

    #[test]
    fn live_settings_do_not_restart() {
        let old = AsrConfig::default();
        for mutate in [
            |c: &mut AsrConfig| c.vad_threshold = 0.5,
            |c: &mut AsrConfig| c.vad_silence_ms = 1,
            |c: &mut AsrConfig| c.language = "en".to_string(),
            |c: &mut AsrConfig| c.style.font_size = 99,
            |c: &mut AsrConfig| c.max_new_tokens = 64,
        ] {
            let mut new = old.clone();
            mutate(&mut new);
            assert!(!config_needs_restart(&old, &new));
        }
        assert!(!config_needs_restart(&old, &old.clone()));
    }

    #[test]
    fn settings_never_contain_secrets() {
        std::env::set_var("ASR_OVERLAY_TEST_SERVER_KEY", "sk-server-secret");
        let config = AsrConfig {
            provider: crate::config::ProviderConfig {
                api_key_env: "ASR_OVERLAY_TEST_SERVER_KEY".to_string(),
                ..crate::config::ProviderConfig::default()
            },
            ..AsrConfig::default()
        };
        // GET /settings serializes this struct, so whatever it holds is what
        // the endpoint returns.
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("sk-server-secret"), "{json}");
        std::env::remove_var("ASR_OVERLAY_TEST_SERVER_KEY");
    }

    fn test_state() -> Arc<AppState> {
        let (caption_tx, _rx) = broadcast::channel(16);
        Arc::new(AppState {
            config: Mutex::new(AsrConfig::default()),
            caption_tx,
            last_caption: Arc::new(Mutex::new(None)),
            listening: Mutex::new(ListeningState::Idle),
            rate_limiters: crate::rate_limit::RateLimiters::new(),
            render_style: Mutex::new(RenderStyle::default()),
            style_tx: broadcast::channel(16).0,
        })
    }

    #[tokio::test]
    async fn test_get_index() {
        let state = test_state();
        let app = build_router(state);

        let response = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert!(body.windows(16).any(|w| w == b"<div id=\"caption"));
    }

    #[tokio::test]
    async fn test_get_status() {
        let state = test_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["running"], true);
        assert_eq!(json["listening"], false);
    }

    #[tokio::test]
    async fn test_get_settings() {
        let state = test_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/settings")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["port"], 8080);
    }

    #[tokio::test]
    async fn test_get_ws() {
        let state = test_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/ws")
                    .header("connection", "Upgrade")
                    .header("upgrade", "websocket")
                    .header("sec-websocket-version", "13")
                    .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert!(
            response.status() == StatusCode::SWITCHING_PROTOCOLS
                || response.status() == StatusCode::UPGRADE_REQUIRED
        );
    }
}
