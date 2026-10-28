//! Background in-app EyeWide head fitter.
//!
//! This is deliberately separate from eyebrow fitting: it requires two completed
//! sessions and validates on the newest whole session. Output filenames are unique, so
//! a failed run can never truncate or replace the model currently selected in config.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

const LOG_CAP: usize = 120;

pub struct FitInputs {
    pub backbone_bin: PathBuf,
    pub wide_data_dir: PathBuf,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub enum Status {
    Idle,
    Running {
        log: Vec<String>,
    },
    Done {
        wide_bin: PathBuf,
        sessions: usize,
        train_frames: usize,
        val_frames: usize,
        train_rmse: f64,
        val_rmse: f64,
        log: Vec<String>,
    },
    Failed {
        msg: String,
        log: Vec<String>,
    },
}

impl Status {
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running { .. })
    }
}

struct Shared {
    status: Status,
    log: Vec<String>,
}

impl Shared {
    fn push(&mut self, line: String) {
        if self.log.len() >= LOG_CAP {
            self.log.drain(0..self.log.len() - LOG_CAP + 1);
        }
        self.log.push(line);
        if matches!(self.status, Status::Running { .. }) {
            self.status = Status::Running {
                log: self.log.clone(),
            };
        }
    }
}

pub struct WideFitter {
    shared: Arc<Mutex<Shared>>,
    handle: Option<JoinHandle<()>>,
}

impl Default for WideFitter {
    fn default() -> Self {
        Self::new()
    }
}

impl WideFitter {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(Shared {
                status: Status::Idle,
                log: Vec::new(),
            })),
            handle: None,
        }
    }

    pub fn status(&self) -> Status {
        self.shared
            .lock()
            .map(|s| s.status.clone())
            .unwrap_or(Status::Idle)
    }

    pub fn is_running(&self) -> bool {
        self.status().is_running()
    }

    pub fn start(&mut self, inputs: FitInputs) -> Result<(), String> {
        if self.is_running() {
            return Err("a Wide fit is already running".into());
        }
        if !inputs.backbone_bin.is_file() {
            return Err(format!(
                "base Wide model not found: {}",
                inputs.backbone_bin.display()
            ));
        }
        let sessions = crate::ml::wide_calfit::completed_sessions(&inputs.wide_data_dir)?;
        if sessions.len() < 2 {
            return Err(format!(
                "need 2 completed Wide sessions; found {} (capture again after reseating)",
                sessions.len()
            ));
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        {
            let mut shared = self.shared.lock().unwrap();
            shared.log.clear();
            shared.status = Status::Running { log: Vec::new() };
        }
        let shared = self.shared.clone();
        self.handle = Some(
            std::thread::Builder::new()
                .name("wide-fitter".into())
                .spawn(move || run(shared, inputs))
                .map_err(|e| format!("could not spawn Wide fitter: {e}"))?,
        );
        Ok(())
    }
}

impl Drop for WideFitter {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            drop(handle);
        }
    }
}

fn run(shared: Arc<Mutex<Shared>>, inputs: FitInputs) {
    log(&shared, format!("[load] {}", inputs.backbone_bin.display()));
    log(&shared, "[features] decoding all completed sessions".into());
    let fit = match crate::ml::wide_calfit::fit_to_bytes(
        &inputs.wide_data_dir,
        &inputs.backbone_bin,
        inputs.seed,
    ) {
        Ok(fit) => fit,
        Err(error) => return fail(&shared, error),
    };
    let train_rmse = fit.train_mse.sqrt();
    let val_rmse = fit.val_mse.sqrt();
    log(
        &shared,
        format!(
            "[fit] {} sessions; train={} RMSE={train_rmse:.4}; held-out={} RMSE={val_rmse:.4}",
            fit.sessions, fit.train_frames, fit.val_frames
        ),
    );
    // A catastrophically non-generalizing head should remain a diagnostic result rather
    // than becoming selectable. The threshold is intentionally lenient; normal 0..1
    // regressors should be comfortably below it.
    if val_rmse > 0.40 {
        return fail(
            &shared,
            format!("held-out session RMSE {val_rmse:.3} is too high (limit 0.400)"),
        );
    }
    let unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let models = inputs.wide_data_dir.join("models");
    if let Err(error) = std::fs::create_dir_all(&models) {
        return fail(&shared, format!("create {}: {error}", models.display()));
    }
    let wide_bin = models.join(format!("wide-{unix_ms}-{}.bin", std::process::id()));
    let candidate = wide_bin.with_extension("bin.candidate");
    if let Err(error) = std::fs::write(&candidate, &fit.bytes) {
        return fail(&shared, format!("write {}: {error}", candidate.display()));
    }
    if let Err(error) = crate::ml::wide_net::WideNet::load(&candidate) {
        let _ = std::fs::remove_file(&candidate);
        return fail(&shared, format!("written model verification: {error}"));
    }
    if let Err(error) = std::fs::rename(&candidate, &wide_bin) {
        let _ = std::fs::remove_file(&candidate);
        return fail(&shared, format!("publish {}: {error}", wide_bin.display()));
    }
    let mut state = shared.lock().unwrap();
    state.push(format!("[done] {}", wide_bin.display()));
    let log = state.log.clone();
    state.status = Status::Done {
        wide_bin,
        sessions: fit.sessions,
        train_frames: fit.train_frames,
        val_frames: fit.val_frames,
        train_rmse,
        val_rmse,
        log,
    };
}

fn log(shared: &Arc<Mutex<Shared>>, line: String) {
    if let Ok(mut state) = shared.lock() {
        state.push(line);
    }
}

fn fail(shared: &Arc<Mutex<Shared>>, msg: String) {
    let mut state = shared.lock().unwrap();
    state.push(format!("[error] {msg}"));
    let log = state.log.clone();
    state.status = Status::Failed { msg, log };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_requires_base_model_before_spawning() {
        let root =
            std::env::temp_dir().join(format!("sranibro_wide_fitrun_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let mut fitter = WideFitter::new();
        let error = fitter
            .start(FitInputs {
                backbone_bin: root.join("missing.bin"),
                wide_data_dir: root.clone(),
                seed: 1,
            })
            .unwrap_err();
        assert!(error.contains("base Wide model"));
        assert!(!fitter.is_running());
        let _ = std::fs::remove_dir_all(root);
    }
}
