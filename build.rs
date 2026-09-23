use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn collect_files(root: &Path, current: &Path, files: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(current).expect("failed to read asset directory");
    for entry in entries {
        let path = entry.expect("failed to read asset entry").path();
        if path.is_dir() {
            collect_files(root, &path, files);
        } else {
            files.push(
                path.strip_prefix(root)
                    .expect("asset path is outside the asset root")
                    .to_path_buf(),
            );
        }
    }
}

fn npm_program() -> &'static str {
    if cfg!(windows) { "npm.cmd" } else { "npm" }
}

fn mtime(path: &Path) -> Option<std::time::SystemTime> {
    fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn run_npm(frontend: &Path, args: &[&str], what: &str) {
    let status = Command::new(npm_program())
        .args(args)
        .current_dir(frontend)
        .status()
        .unwrap_or_else(|e| {
            panic!(
                "Failed to run `npm {}` ({e}). Building the frontend needs Node.js \
                 (v20.19+ or v22.12+) and npm on PATH; alternatively, build \
                 frontend/dist yourself and set ASR_OVERLAY_SKIP_FRONTEND_BUILD=1.",
                args.join(" ")
            )
        });
    if !status.success() {
        panic!("`npm {}` failed with status {status} ({what})", args.join(" "));
    }
}

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("missing manifest dir"));
    let frontend = manifest_dir.join("frontend");
    let dist = frontend.join("dist");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("missing OUT_DIR"));
    let generated_path = out_dir.join("embedded_assets.rs");

    println!("cargo:rerun-if-changed=build.rs");
    // Frontend sources only — never node_modules/ or dist/ (the build writes
    // dist/, which would otherwise trigger a rebuild on every invocation).
    println!("cargo:rerun-if-changed=frontend/package.json");
    println!("cargo:rerun-if-changed=frontend/package-lock.json");
    println!("cargo:rerun-if-changed=frontend/vite.config.ts");
    println!("cargo:rerun-if-changed=frontend/tsconfig.json");
    println!("cargo:rerun-if-changed=frontend/index.html");
    println!("cargo:rerun-if-changed=frontend/config.html");
    println!("cargo:rerun-if-changed=frontend/src");

    if env::var_os("ASR_OVERLAY_SKIP_FRONTEND_BUILD").is_some() {
        assert!(
            dist.join("index.html").exists(),
            "ASR_OVERLAY_SKIP_FRONTEND_BUILD is set but frontend/dist is missing; \
             run `npm run build` in frontend/ first."
        );
    } else {
        // Install dependencies when missing or older than the lockfile.
        let node_modules = frontend.join("node_modules");
        let lock = frontend.join("package-lock.json");
        let stale = match (mtime(&lock), mtime(&node_modules)) {
            (Some(lock), Some(deps)) => lock > deps,
            _ => true,
        };
        if stale {
            run_npm(&frontend, &["ci", "--no-fund", "--no-audit"], "install frontend deps");
        }
        run_npm(&frontend, &["run", "build"], "build frontend");
    }

    let mut files = Vec::new();
    collect_files(&dist, &dist, &mut files);
    files.sort();

    let mut generated =
        String::from("pub fn get(path: &str) -> Option<&'static [u8]> {\n    match path {\n");

    for relative_path in files {
        let key = relative_path.to_string_lossy().replace('\\', "/");
        let absolute_path = dist.join(&relative_path);
        generated.push_str(&format!(
            "        {key:?} => Some(include_bytes!({absolute:?})),\n",
            key = key,
            absolute = absolute_path.to_string_lossy(),
        ));
    }

    generated.push_str("        _ => None,\n    }\n}\n");
    fs::write(generated_path, generated).expect("failed to write embedded asset map");
}
