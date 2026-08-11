use std::sync::Arc;
use std::time::Duration;

use tray_icon::{
    menu::{Menu, MenuEvent, MenuItem},
    Icon, TrayIconBuilder,
};

use crate::server::AppState;

fn make_icon() -> Icon {
    const ICON_PNG: &[u8] = include_bytes!("../assets/tray-icon.png");
    let image = image::load_from_memory(ICON_PNG)
        .expect("failed to decode embedded tray icon")
        .into_rgba8();
    let (width, height) = image.dimensions();
    Icon::from_rgba(image.into_raw(), width, height).expect("failed to create tray icon")
}

pub fn create_tray(state: Arc<AppState>, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    std::thread::spawn(move || {
        if let Err(e) = gtk::init() {
            log::warn!("Failed to initialize GTK: {e}");
            return;
        }

        let icon = make_icon();

        let start_stop = MenuItem::new("Start Listening", true, None);
        let open_overlay = MenuItem::new("Open Overlay", true, None);
        let restart = MenuItem::new("Restart", true, None);
        let quit = MenuItem::new("Quit", true, None);

        let menu = Menu::new();
        let _ = menu.append(&start_stop);
        let _ = menu.append(&open_overlay);
        let _ = menu.append(&MenuItem::new("", false, None));
        let _ = menu.append(&restart);
        let _ = menu.append(&MenuItem::new("", false, None));
        let _ = menu.append(&quit);

        let _tray = match TrayIconBuilder::new()
            .with_icon(icon)
            .with_tooltip("ASR Overlay")
            .with_menu(Box::new(menu))
            .build()
        {
            Ok(t) => t,
            Err(e) => {
                log::warn!("Failed to create tray icon: {e}");
                return;
            }
        };

        log::info!("Tray icon created");

        let start_stop_clone = start_stop.clone();
        let open_overlay_clone = open_overlay.clone();
        let restart_clone = restart.clone();
        let quit_clone = quit.clone();
        let state_clone = Arc::clone(&state);

        // Polling at a modest interval keeps the GTK loop responsive without the
        // 100%-CPU idle callback that used to run try_recv() continuously.
        glib::timeout_add_local(Duration::from_millis(100), move || {
            while let Ok(event) = MenuEvent::receiver().try_recv() {
                if event.id() == start_stop_clone.id() {
                    if crate::controller::is_listening() {
                        log::info!("Tray requested stop");
                        crate::controller::stop();
                    } else {
                        log::info!("Tray requested start");
                        let state = Arc::clone(&state_clone);
                        std::thread::spawn(move || {
                            if let Err(e) = crate::controller::start(state) {
                                log::warn!("Listening did not start: {e}");
                            }
                        });
                    }
                } else if event.id() == open_overlay_clone.id() {
                    let url = format!("http://localhost:{port}/");
                    let _ = open::that(url);
                } else if event.id() == restart_clone.id() {
                    let _ = std::process::Command::new(std::env::current_exe().unwrap()).spawn();
                    std::process::exit(0);
                } else if event.id() == quit_clone.id() {
                    std::process::exit(0);
                }
            }

            sync_start_stop_item(&start_stop_clone);
            glib::ControlFlow::Continue
        });

        gtk::main();
    });

    Ok(())
}

fn sync_start_stop_item(item: &MenuItem) {
    let listening = crate::controller::is_listening();
    let text = if listening {
        "Stop Listening"
    } else {
        "Start Listening"
    };

    if item.text() != text {
        item.set_text(text);
    }
    if !item.is_enabled() {
        item.set_enabled(true);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn embedded_tray_icon_is_a_valid_png() {
        const ICON_PNG: &[u8] = include_bytes!("../assets/tray-icon.png");
        let icon = image::load_from_memory(ICON_PNG)
            .expect("embedded tray icon should decode")
            .into_rgba8();

        assert_eq!(icon.dimensions(), (64, 64));
    }
}
