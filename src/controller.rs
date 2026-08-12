use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::audio::open_microphone;
use crate::caption::run_pipeline;
use crate::server::AppState;

/// Global, singly-created pipeline controller. The server and tray both use it
/// to start/stop listening without racing on the microphone.
static CONTROLLER: Mutex<Option<PipelineHandle>> = Mutex::new(None);
static STARTING: AtomicBool = AtomicBool::new(false);

struct PipelineHandle {
    thread: JoinHandle<()>,
    stop_tx: crossbeam_channel::Sender<()>,
}

/// Resets `STARTING` on every exit path from `start`, including errors.
struct StartGuard;

impl Drop for StartGuard {
    fn drop(&mut self) {
        STARTING.store(false, Ordering::SeqCst);
    }
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
    // Clears the flag however we leave this function: an early `?` used to
    // leave it set, so every later start silently did nothing.
    let _starting = StartGuard;

    let config = state.config.lock().clone();
    let caption_tx = state.caption_tx.clone();
    let last_caption = Arc::clone(&state.last_caption);

    // The whole Microphone (stream + receiver) moves into the thread so the
    // cpal stream stays alive for as long as the pipeline runs.
    let mic = open_microphone(&config.input_device)?;
    let mic_rate = mic.sample_rate;

    let (stop_tx, stop_rx) = crossbeam_channel::bounded::<()>(1);

    let thread = std::thread::Builder::new()
        .name("asr-pipeline".into())
        .spawn(move || {
            // Keep the mic (and its stream) alive across run_pipeline.
            let mic = mic;

            let backend = crate::asr::Backend::parse(&config.asr_backend);
            let asr = match crate::asr::load(backend, config.threads, config.max_new_tokens) {
                Ok(mut asr) => {
                    asr.set_min_audio_ms(config.min_audio_ms);
                    if let Err(e) = asr.set_language(&config.language) {
                        log::warn!("Ignoring the configured language: {e}");
                    }
                    asr
                }
                Err(e) => {
                    log::error!("Failed to load the ASR model: {e}");
                    return;
                }
            };

            let emit = move |caption: crate::caption::Caption| {
                *last_caption.lock() = Some(caption.clone());
                let _ = caption_tx.send(caption);
            };

            if let Err(e) = run_pipeline(mic.rx, stop_rx, mic_rate, config, asr, emit) {
                log::error!("Pipeline stopped with error: {e}");
            }
            // Dropping `mic` closes the cpal stream, releasing the device.
        })?;

    {
        let mut guard = CONTROLLER.lock();
        *guard = Some(PipelineHandle { thread, stop_tx });
    }
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
