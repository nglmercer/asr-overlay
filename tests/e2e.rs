use std::sync::Arc;
use std::time::Duration;

use asr_overlay::config::AsrConfig;
use asr_overlay::server::{build_router, AppState, ListeningState};

fn pick_unused_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

struct TestServer {
    base_url: String,
    client: reqwest::Client,
}

impl TestServer {
    async fn new() -> Self {
        let port = pick_unused_port();
        let (caption_tx, _rx) = tokio::sync::broadcast::channel(16);

        let state = Arc::new(AppState {
            config: parking_lot::Mutex::new(AsrConfig::default()),
            caption_tx,
            last_caption: Arc::new(parking_lot::Mutex::new(None)),
            listening: parking_lot::Mutex::new(ListeningState::Idle),
            rate_limiters: asr_overlay::rate_limit::RateLimiters::new(),
            render_style: parking_lot::Mutex::new(asr_overlay::render::RenderStyle::default()),
            style_tx: tokio::sync::broadcast::channel(16).0,
        });

        let app = build_router(state);
        let addr = format!("127.0.0.1:{port}");
        let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();

        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let base_url = format!("http://127.0.0.1:{port}");
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        let server = Self { base_url, client };
        server.wait_ready().await;
        server
    }

    async fn wait_ready(&self) {
        for _ in 0..50 {
            if self.client.get(&self.base_url).send().await.is_ok() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        panic!("server did not start within 5s");
    }

    async fn get(&self, path: &str) -> reqwest::Response {
        self.client
            .get(format!("{}/{}", self.base_url, path))
            .send()
            .await
            .unwrap()
    }
}

#[tokio::test]
async fn e2e_status_reports_not_listening() {
    let server = TestServer::new().await;
    let resp = server.get("status").await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["running"], true);
    assert_eq!(json["listening"], false);
    assert_eq!(json["has_caption"], false);
}

#[tokio::test]
async fn e2e_serves_overlay_assets_from_the_binary() {
    let server = TestServer::new().await;

    for (path, content_type) in [
        ("", "text/html"),
        ("index.js", "application/javascript"),
    ] {
        let response = server.get(path).await;
        assert_eq!(response.status(), reqwest::StatusCode::OK, "GET /{path}");
        assert_eq!(
            response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .split(';')
                .next()
                .unwrap_or_default(),
            content_type,
            "content type for /{path}"
        );
        assert!(
            !response.bytes().await.unwrap().is_empty(),
            "GET /{path} was empty"
        );
    }
}

#[tokio::test]
async fn e2e_settings_post_and_get() {
    let server = TestServer::new().await;
    let resp = server.get("settings").await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let body = r#"{"port": 8080, "language": "en", "vad_threshold": 0.01}"#;
    let resp = server
        .client
        .post(format!("{}/settings", server.base_url))
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let resp = server.get("settings").await;
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["port"], 8080);
    assert_eq!(json["language"], "en");
    assert_eq!(json["vad_threshold"], 0.01);
}

#[tokio::test]
async fn e2e_stop_is_idempotent() {
    let server = TestServer::new().await;
    let resp = server
        .client
        .post(format!("{}/stop", server.base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["ok"], true);
    assert_eq!(json["listening"], false);
}

#[tokio::test]
async fn e2e_serves_the_config_page_and_its_assets() {
    let server = TestServer::new().await;

    // The config page is a Vue app: its headings render client-side, so the
    // served shell is pinned by its title instead of a heading. The bundle
    // needles below still pin the real built assets.
    for (path, needle) in [
        ("/config", "ASR Overlay"),
        ("/config.js", "OverlayStyle"),
        ("/style.js", "caption-text"),
    ] {
        let response = server
            .client
            .get(format!("{}{path}", server.base_url))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "{path}");
        let body = response.text().await.unwrap();
        assert!(body.contains(needle), "{path} is missing {needle}");
    }
}

#[tokio::test]
async fn e2e_style_endpoint_updates_the_live_config() {
    let server = TestServer::new().await;

    let style = asr_overlay::config::OverlayStyle {
        color: "#ff0000".to_string(),
        font_size: 44,
        ..asr_overlay::config::OverlayStyle::default()
    };

    let response = server
        .client
        .post(format!("{}/style", server.base_url))
        .json(&style)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);

    // The live style must be visible through /settings straight away.
    let config: serde_json::Value = server
        .client
        .get(format!("{}/settings", server.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(config["style"]["color"], "#ff0000");
    assert_eq!(config["style"]["font_size"], 44);
}
