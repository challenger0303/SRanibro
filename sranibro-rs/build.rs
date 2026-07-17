//! Embed the Windows application icon (assets/sranibro.ico) into the exe so Explorer,
//! the taskbar, and Alt-Tab show it. Best-effort: if the resource compiler isn't found
//! the build still succeeds (just without the embedded icon) rather than failing.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=assets/sranibro.ico");
    println!("cargo:rerun-if-env-changed=PROFILE");
    let profile = std::env::var("PROFILE").expect("Cargo did not provide PROFILE to build.rs");
    println!("cargo:rustc-env=SRANIBRO_BUILD_PROFILE={profile}");
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("Cargo did not provide CARGO_MANIFEST_DIR to build.rs");
    let manifest_dir = Path::new(&manifest_dir);
    let commit = git(manifest_dir, &["rev-parse", "--verify", "HEAD"])
        .unwrap_or_else(|| "unavailable".into());
    println!("cargo:rustc-env=SRANIBRO_BUILD_COMMIT={commit}");
    watch_git_identity(manifest_dir);
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/sranibro.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=app icon not embedded ({e})");
        }
    }
}

fn git(repo: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn watch_git_identity(repo: &Path) {
    for name in ["HEAD", "index", "packed-refs"] {
        if let Some(path) = git(
            repo,
            &["rev-parse", "--path-format=absolute", "--git-path", name],
        ) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(reference) = git(repo, &["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git(
            repo,
            &[
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                &reference,
            ],
        ) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
}
