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
use crate::config::AsrConfig;
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
}

pub fn build_router(state: Arc<AppState>) -> Router {
    let rl = Arc::new(state.rate_limiters.clone());

    Router::new()
        .route("/", get(serve_index))
        .route("/ws", get(ws_stream))
        .route("/caption.png", get(caption_png))
        .route("/settings", get(get_config).post(set_config))
        .route("/status", get(caption_status))
        .route("/start", post(start_listening))
        .route("/stop", post(stop_listening))
        .route("/index.js", get(|| async { serve_embedded("index.js") }))
        .route(
            "/assets/{*path}",
            get(|axum::extract::Path(path): axum::extract::Path<String>| async move {
                serve_embedded(&format!("assets/{path}"))
            }),
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

/// WebSocket caption stream. Emits JSON captions to overlay clients.
async fn ws_stream(
    ws: axum::extract::ws::WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> Response {
    let caption_rx = state.caption_tx.subscribe();
    ws.on_upgrade(move |mut socket| async move {
        let mut caption_rx = caption_rx;
        loop {
            match caption_rx.recv().await {
                Ok(caption) => {
                    let Ok(json) = serde_json::to_string(&caption) else {
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
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    })
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

async fn set_config(
    State(state): State<Arc<AppState>>,
    axum::Json(config): axum::Json<AsrConfig>,
) -> StatusCode {
    let saved = {
        let mut guard = state.config.lock();
        *guard = config;
        guard.clone()
    };
    crate::config::save(&saved);
    StatusCode::OK
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
        listening: *state.listening.lock() == ListeningState::Listening,
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
        if *guard == ListeningState::Listening {
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

    let result = crate::controller::start(Arc::clone(&state));

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
        listening: *state.listening.lock() == ListeningState::Listening,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use std::sync::Arc;
    use tower::ServiceExt;

    fn test_state() -> Arc<AppState> {
        let (caption_tx, _rx) = broadcast::channel(16);
        Arc::new(AppState {
            config: Mutex::new(AsrConfig::default()),
            caption_tx,
            last_caption: Arc::new(Mutex::new(None)),
            listening: Mutex::new(ListeningState::Idle),
            rate_limiters: crate::rate_limit::RateLimiters::new(),
            render_style: Mutex::new(RenderStyle::default()),
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
