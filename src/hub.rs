//! Minimal Hugging Face model downloader with an on-disk cache.
//!
//! Mirrors what `transformers` / `transformers.js` do: the first time a model
//! is needed it is fetched into a local cache directory, and every run after
//! that resolves straight from disk. No Python, no `huggingface_hub` CLI.

use anyhow::{anyhow, Context, Result};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const ENDPOINT: &str = "https://huggingface.co";

/// A model repo on the Hub plus the files this app needs from it.
#[derive(Debug, Clone)]
pub struct RepoSpec {
    /// e.g. `Audio8/ark-asr-0.6b-int8-onnx`
    pub repo: &'static str,
    pub revision: &'static str,
    /// Repo-relative paths to fetch, in download order.
    pub files: &'static [&'static str],
}

/// Cache root: `$ASR_OVERLAY_CACHE`, else `$XDG_CACHE_HOME/asr-overlay/models`,
/// else `./models`.
pub fn cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("ASR_OVERLAY_CACHE") {
        return PathBuf::from(dir);
    }
    match dirs::cache_dir() {
        Some(d) => d.join("asr-overlay").join("models"),
        None => PathBuf::from("models"),
    }
}

/// Local directory for a repo's snapshot.
pub fn repo_dir(spec: &RepoSpec) -> PathBuf {
    // `org/name` → `org--name` so it stays a single directory level.
    cache_root().join(spec.repo.replace('/', "--"))
}

/// True when every file in the spec is already on disk and non-empty.
pub fn is_cached(spec: &RepoSpec) -> bool {
    let dir = repo_dir(spec);
    spec.files.iter().all(|f| {
        std::fs::metadata(dir.join(f))
            .map(|m| m.len() > 0)
            .unwrap_or(false)
    })
}

/// Ensure every file in `spec` is present locally, downloading what is missing.
/// Returns the local snapshot directory.
///
/// `progress` is called as `(file, bytes_done, total_bytes_or_none)`.
pub fn ensure_repo(spec: &RepoSpec, mut progress: impl FnMut(&str, u64, Option<u64>)) -> Result<PathBuf> {
    let dir = repo_dir(spec);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create cache dir {}", dir.display()))?;

    for file in spec.files {
        let dest = dir.join(file);
        if std::fs::metadata(&dest).map(|m| m.len() > 0).unwrap_or(false) {
            continue;
        }
        let url = format!("{ENDPOINT}/{}/resolve/{}/{file}", spec.repo, spec.revision);
        log::info!("Downloading {} → {}", url, dest.display());
        download(&url, &dest, |done, total| progress(file, done, total))
            .with_context(|| format!("Failed to download {file} from {}", spec.repo))?;
    }

    Ok(dir)
}

/// Stream a URL to `dest`, writing through a `.part` file so an interrupted
/// download never leaves a truncated model in the cache.
fn download(url: &str, dest: &Path, mut progress: impl FnMut(u64, Option<u64>)) -> Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part = dest.with_extension(format!(
        "{}part",
        dest.extension()
            .map(|e| format!("{}.", e.to_string_lossy()))
            .unwrap_or_default()
    ));

    let response = ureq::get(url)
        .call()
        .map_err(|e| anyhow!("HTTP request failed: {e}"))?;

    let total: Option<u64> = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok());

    let mut reader = response.into_body().into_reader();
    let mut file = std::fs::File::create(&part)
        .with_context(|| format!("Failed to create {}", part.display()))?;

    let mut buf = vec![0u8; 1 << 20];
    let mut done: u64 = 0;
    let mut last_report: u64 = 0;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        done += n as u64;
        // Report at most once per 32 MiB to keep logs readable.
        if done - last_report >= 32 << 20 {
            last_report = done;
            progress(done, total);
        }
    }
    file.flush()?;
    drop(file);

    if let Some(total) = total {
        if done != total {
            let _ = std::fs::remove_file(&part);
            return Err(anyhow!("Truncated download: got {done} of {total} bytes"));
        }
    }

    std::fs::rename(&part, dest)
        .with_context(|| format!("Failed to finalize {}", dest.display()))?;
    progress(done, total);
    Ok(())
}

/// Human-readable byte size for progress logs.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[0])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: RepoSpec = RepoSpec {
        repo: "Audio8/ark-asr-0.6b-int8-onnx",
        revision: "main",
        files: &["runtime_manifest.json"],
    };

    #[test]
    fn repo_dir_flattens_the_org_separator() {
        let dir = repo_dir(&SPEC);
        assert!(dir.ends_with("Audio8--ark-asr-0.6b-int8-onnx"));
    }

    #[test]
    fn cache_root_honours_the_env_override() {
        // Safe here: the test only reads the variable it just set.
        unsafe { std::env::set_var("ASR_OVERLAY_CACHE", "/tmp/asr-overlay-cache-test") };
        assert_eq!(cache_root(), PathBuf::from("/tmp/asr-overlay-cache-test"));
        unsafe { std::env::remove_var("ASR_OVERLAY_CACHE") };
    }

    #[test]
    fn missing_files_are_not_cached() {
        unsafe { std::env::set_var("ASR_OVERLAY_CACHE", "/tmp/asr-overlay-cache-empty") };
        assert!(!is_cached(&SPEC));
        unsafe { std::env::remove_var("ASR_OVERLAY_CACHE") };
    }

    #[test]
    fn human_bytes_scales() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1536), "1.5 KiB");
        assert_eq!(human_bytes(640 * 1024 * 1024), "640.0 MiB");
    }
}
