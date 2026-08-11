use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::audio::open_microphone;
use crate::caption::run_pipeline;
use crate::server::AppState;
use crate::stt::WhisperStt;

/// Global, singly-created pipeline controller. The server and tray both use it
/// to start/stop listening without racing on the microphone.
static CONTROLLER: Mutex<Option<PipelineHandle>> = Mutex::new(None);
static STARTING: AtomicBool = AtomicBool::new(false);

struct PipelineHandle {
    thread: JoinHandle<()>,
    stop_tx: crossbeam_channel::Sender<()>,
}

/// True if the pipeline is currently active.
pub fn is_listening() -> bool {
    CONTROLLER.lock().is_some()
}

/// Start (or restart) the mic → STT pipeline. Idempotent.
pub fn start(state: Arc<AppState>) -> anyhow::Result<()> {
    {
        let guard = CONTROLLER.lock();
        if guard.is_some() {
            return Ok(());
        }
    }

    if STARTING.swap(true, Ordering::SeqCst) {
        // Another caller is starting; treat as success (they will finish).
        return Ok(());
    }

    let config = state.config.lock().clone();
    let caption_tx = state.caption_tx.clone();
    let last_caption = Arc::clone(&state.last_caption);

    // The whole Microphone (stream + receiver) moves into the thread so the
    // cpal stream stays alive for as long as the pipeline runs.
    let mic = open_microphone()?;
    let mic_rate = mic.sample_rate;

    let (stop_tx, stop_rx) = crossbeam_channel::bounded::<()>(1);

    let thread = std::thread::Builder::new()
        .name("asr-pipeline".into())
        .spawn(move || {
            // Keep the mic (and its stream) alive across run_pipeline.
            let mic = mic;

            let stt = match WhisperStt::load(&config.model_path, &config.language, config.threads)
            {
                Ok(s) => s,
                Err(e) => {
                    log::error!("Failed to load Whisper model: {e}");
                    log::error!(
                        "Download a model, e.g. ggml-small.bin, and point --model / config at it."
                    );
                    return;
                }
            };

            let emit = move |caption: crate::caption::Caption| {
                *last_caption.lock() = Some(caption.clone());
                let _ = caption_tx.send(caption);
            };

            if let Err(e) = run_pipeline(mic.rx, stop_rx, mic_rate, config, stt, emit) {
                log::error!("Pipeline stopped with error: {e}");
            }
            // Dropping `mic` closes the cpal stream, releasing the device.
        })?;

    {
        let mut guard = CONTROLLER.lock();
        *guard = Some(PipelineHandle { thread, stop_tx });
    }
    STARTING.store(false, Ordering::SeqCst);
    log::info!("Listening started");
    Ok(())
}

/// Stop the pipeline. The microphone stream is dropped.
pub fn stop() {
    let handle = CONTROLLER.lock().take();
    if let Some(handle) = handle {
        let _ = handle.stop_tx.send(());
        if handle.thread.join().is_err() {
            log::warn!("Pipeline thread panicked while stopping");
        }
        log::info!("Listening stopped");
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn idle_by_default() {
        assert!(!super::is_listening());
    }
}
