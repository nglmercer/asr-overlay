use std::sync::Arc;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "asr-overlay", version, about = "OBS overlay that detects speech and generates live subtitles")]
struct Cli {
    /// Whisper ggml model file.
    #[arg(long, default_value = "models/whisper/ggml-small.bin")]
    model: String,

    /// Transcription language.
    #[arg(long, default_value = "es")]
    language: String,

    /// HTTP port for the overlay.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Auto-start listening on launch.
    #[arg(long)]
    no_auto_start: bool,
}

#[tokio::main]
async fn main() {
    asr_overlay::logger::init();

    let cli = Cli::parse();

    let mut config = asr_overlay::config::load();
    config.model_path = cli.model;
    config.language = cli.language;
    config.port = cli.port;
    config.auto_start = !cli.no_auto_start;
    asr_overlay::config::save(&config);

    let (caption_tx, _caption_rx) = tokio::sync::broadcast::channel::<asr_overlay::caption::Caption>(16);

    let state = Arc::new(asr_overlay::server::AppState {
        config: parking_lot::Mutex::new(config.clone()),
        caption_tx,
        last_caption: std::sync::Arc::new(parking_lot::Mutex::new(None)),
        listening: parking_lot::Mutex::new(asr_overlay::server::ListeningState::Idle),
        rate_limiters: asr_overlay::rate_limit::RateLimiters::new(),
        render_style: parking_lot::Mutex::new(asr_overlay::render::RenderStyle::default()),
    });

    let port = config.port;
    let addr = format!("0.0.0.0:{port}");

    log::info!("ASR Overlay running at http://localhost:{port}");
    log::info!("Overlay (OBS Browser Source): http://localhost:{port}/");
    log::info!("Model:  {}", config.model_path);
    log::info!("Lang:   {}", config.language);

    if let Err(e) = asr_overlay::tray::create_tray(Arc::clone(&state), port) {
        log::warn!("Failed to create tray icon: {e}");
    }

    if config.auto_start {
        log::info!("Auto-start: listening");
        if let Err(e) = asr_overlay::controller::start(Arc::clone(&state)) {
            log::error!("Auto-start failed: {e}");
        }
    }

    let app = asr_overlay::server::build_router(state);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            log::error!("Failed to bind to {addr}: {e}");
            return;
        }
    };

    if let Err(e) = axum::serve(listener, app).await {
        log::error!("Server error: {e}");
    }
}
