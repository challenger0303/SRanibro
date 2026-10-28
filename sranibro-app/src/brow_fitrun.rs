//! In-app eyebrow HEAD-fit runner — the pure-Rust counterpart to [`crate::brow_train`].
//!
//! Where `brow_train` shells out to the user's PyTorch venv for a full retrain, this fits
//! ONLY the output head (fc1 + fc2) onto the captured `brow_data/`, reusing an existing
//! `brow.bin`'s FROZEN conv backbone ([`crate::ml::brow_calfit`]). No subprocess, no Python:
//! it decodes the PNGs, runs them through the backbone, fits the head, and writes a fresh
//! `BROWNET1` `brow.bin` — seconds, ideal for a quick per-user recalibration.
//!
//! Same scaffolding as [`crate::brow_train::BrowTrainer`]: a background thread writes a
//! bounded log + status behind an `Arc<Mutex>` that the UI polls each frame; `Drop` detaches.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// Max lines kept in the live log ring buffer (older lines drop off the front).
const LOG_CAP: usize = 200;

/// Fully-resolved inputs for a fit run (paths pre-validated by [`BrowFitter::start`]).
pub struct FitInputs {
    /// The existing `brow.bin` whose conv backbone is frozen and reused.
    pub backbone_bin: PathBuf,
    /// The captured `brow_data` dir (holds `images/` + `labels.csv`).
    pub brow_data_dir: PathBuf,
    /// Seed for the deterministic head init / shuffle.
    pub seed: u64,
}

/// Which stage the fit is in (drives the log tags / labels).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    Load,
    Features,
    Fit,
    Write,
}

impl Stage {
    /// Short lowercase log tag (`[load]`, `[features]`, …).
    fn tag(self) -> &'static str {
        match self {
            Stage::Load => "load",
            Stage::Features => "features",
            Stage::Fit => "fit",
            Stage::Write => "write",
        }
    }

    /// Human-readable stage label (for a status line, mirroring [`crate::brow_train::Stage`]).
    pub fn label(self) -> &'static str {
        match self {
            Stage::Load => "loading backbone",
            Stage::Features => "decoding frames",
            Stage::Fit => "fitting head",
            Stage::Write => "writing brow.bin",
        }
    }
}

/// Public status snapshot for the UI (cheap to clone each frame).
#[derive(Clone, Debug)]
pub enum Status {
    /// Never started, or reset back to idle.
    Idle,
    /// A run is in progress; `log` is the recent tail (bounded).
    Running { log: Vec<String> },
    /// Finished: the produced `brow.bin`. `log` is the full tail for review.
    Done { brow_bin: PathBuf, log: Vec<String> },
    /// Failed with a human-readable reason; `log` is the tail.
    Failed { msg: String, log: Vec<String> },
}

impl Status {
    pub fn is_running(&self) -> bool {
        matches!(self, Status::Running { .. })
    }
}

/// Shared, thread-owned state behind a single mutex (worker writes, UI reads a clone).
struct Shared {
    status: Status,
    /// Bounded ring buffer of the fit's granular log lines.
    log: Vec<String>,
}

impl Shared {
    fn push_log(&mut self, line: String) {
        if self.log.len() >= LOG_CAP {
            let drop = self.log.len() - LOG_CAP + 1;
            self.log.drain(0..drop);
        }
        self.log.push(line);
        // Keep the live Running status' log tail in sync so the UI sees new lines.
        if let Status::Running { .. } = self.status {
            self.status = Status::Running {
                log: self.log.clone(),
            };
        }
    }
}

/// The in-app head-fit runner. Owns a background thread; poll [`Self::status`] each UI frame.
pub struct BrowFitter {
    shared: Arc<Mutex<Shared>>,
    handle: Option<JoinHandle<()>>,
}

impl Default for BrowFitter {
    fn default() -> Self {
        Self::new()
    }
}

impl BrowFitter {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(Shared {
                status: Status::Idle,
                log: Vec::new(),
            })),
            handle: None,
        }
    }

    /// Current status (cheap clone). Call each UI frame.
    pub fn status(&self) -> Status {
        self.shared
            .lock()
            .map(|s| s.status.clone())
            .unwrap_or(Status::Idle)
    }

    /// True while a run is in progress.
    pub fn is_running(&self) -> bool {
        self.shared
            .lock()
            .map(|s| s.status.is_running())
            .unwrap_or(false)
    }

    /// Validate inputs and, if OK, spawn the background fit. Returns `Err(msg)` (and stays
    /// Idle) when a precondition fails — the UI shows the message and nothing is launched. A
    /// no-op (Err) if a run is already in progress.
    pub fn start(&mut self, inputs: FitInputs) -> Result<(), String> {
        if self.is_running() {
            return Err("a fit run is already in progress".into());
        }
        if !inputs.backbone_bin.is_file() {
            return Err(format!(
                "base eyebrow model not found: {}",
                inputs.backbone_bin.display()
            ));
        }
        let labels = inputs.brow_data_dir.join("labels.csv");
        if !labels.is_file() {
            return Err(format!(
                "labels.csv not found — capture a dataset first ({})",
                labels.display()
            ));
        }
        // Reset shared state to a fresh Running.
        {
            let mut s = self.shared.lock().unwrap();
            s.log.clear();
            s.status = Status::Running { log: Vec::new() };
        }
        // Join a finished previous thread (if any) so we don't leak the handle.
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let shared = self.shared.clone();
        let handle = std::thread::Builder::new()
            .name("brow-fitter".into())
            .spawn(move || run_fit(shared, inputs))
            .map_err(|e| format!("could not spawn fitter thread: {e}"))?;
        self.handle = Some(handle);
        Ok(())
    }
}

impl Drop for BrowFitter {
    fn drop(&mut self) {
        // Don't block app shutdown on an in-flight fit — detach the handle.
        if let Some(h) = self.handle.take() {
            drop(h);
        }
    }
}

/// The whole fit, on the worker thread: load backbone -> features -> fit -> write -> Done/Failed.
/// Calls the lower-level [`crate::ml::brow_calfit::load_dataset`] + [`crate::ml::brow_fit::fit_head`]
/// + [`crate::ml::brow_net::BrowNet::to_bytes_with_head`] directly so each stage gets its own log line.
fn run_fit(shared: Arc<Mutex<Shared>>, inputs: FitInputs) {
    use crate::ml::brow_net::BrowNet;

    // 1) Load the frozen backbone.
    log(
        &shared,
        format!(
            "[{}] loading backbone {}",
            Stage::Load.tag(),
            inputs.backbone_bin.display()
        ),
    );
    let mut backbone = match BrowNet::load(&inputs.backbone_bin) {
        Ok(n) => n,
        Err(e) => return fail(&shared, format!("backbone: {e}")),
    };
    let out_dim = backbone.out_dim();

    // 2) Decode + feature-extract every labeled frame through the frozen conv backbone.
    log(
        &shared,
        format!("[{}] decoding frames…", Stage::Features.tag()),
    );
    let ds = match crate::ml::brow_calfit::load_dataset(&inputs.brow_data_dir, &mut backbone) {
        Ok(d) => d,
        Err(e) => return fail(&shared, format!("dataset: {e}")),
    };
    log(
        &shared,
        format!(
            "[{}] loaded {} frames (skipped {})",
            Stage::Features.tag(),
            ds.features.len(),
            ds.skipped
        ),
    );

    // 3) Fit the head.
    log(
        &shared,
        format!("[{}] training head (out_dim={out_dim})", Stage::Fit.tag()),
    );
    let head = crate::ml::brow_fit::fit_head(&ds.features, &ds.labels, out_dim, inputs.seed);

    // 4) Serialize (frozen conv + new head) and write brow.bin.
    log(&shared, format!("[{}] brow.bin", Stage::Write.tag()));
    let bytes = match backbone.to_bytes_with_head(&head) {
        Ok(b) => b,
        Err(e) => return fail(&shared, format!("serialize: {e}")),
    };
    let brow_bin = inputs.brow_data_dir.join("brow.bin");
    if let Err(e) = std::fs::write(&brow_bin, &bytes) {
        return fail(&shared, format!("write {}: {e}", brow_bin.display()));
    }
    if !brow_bin.is_file() {
        return fail(
            &shared,
            format!("brow.bin was not written: {}", brow_bin.display()),
        );
    }

    let mut s = shared.lock().unwrap();
    s.push_log(format!("[done] {}", brow_bin.display()));
    let log = s.log.clone();
    s.status = Status::Done { brow_bin, log };
}

/// Append a line to the shared log (no-op if the lock is poisoned).
fn log(shared: &Arc<Mutex<Shared>>, line: String) {
    if let Ok(mut s) = shared.lock() {
        s.push_log(line);
    }
}

/// Transition the shared state to `Failed`, keeping the log tail for the UI.
fn fail(shared: &Arc<Mutex<Shared>>, msg: String) {
    let mut s = shared.lock().unwrap();
    s.push_log(format!("[error] {msg}"));
    let log = s.log.clone();
    s.status = Status::Failed { msg, log };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("sranibro_fitrun_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn start_errs_without_backbone_or_labels() {
        let dir = tmp_dir("start_bad");
        let mut f = BrowFitter::new();
        // No backbone file yet.
        let e = f
            .start(FitInputs {
                backbone_bin: dir.join("nope.bin"),
                brow_data_dir: dir.clone(),
                seed: 0,
            })
            .unwrap_err();
        assert!(e.contains("base eyebrow model"), "{e}");
        assert!(!f.is_running());
        // Backbone present but labels.csv missing.
        std::fs::write(dir.join("brow.bin"), b"BROWNET1").unwrap();
        let e = f
            .start(FitInputs {
                backbone_bin: dir.join("brow.bin"),
                brow_data_dir: dir.clone(),
                seed: 0,
            })
            .unwrap_err();
        assert!(e.contains("labels.csv"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stage_tags_and_labels_are_distinct() {
        let stages = [Stage::Load, Stage::Features, Stage::Fit, Stage::Write];
        for s in stages {
            assert!(!s.tag().is_empty());
            assert!(!s.label().is_empty());
        }
    }
}
