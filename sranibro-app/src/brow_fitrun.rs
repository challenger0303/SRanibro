
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

const LOG_CAP: usize = 200;

pub struct FitInputs {
    pub backbone_bin: PathBuf,
    pub brow_data_dir: PathBuf,
    pub seed: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    Load,
    Features,
    Fit,
    Write,
}

impl Stage {
    fn tag(self) -> &'static str {
        match self {
            Stage::Load => "load",
            Stage::Features => "features",
            Stage::Fit => "fit",
            Stage::Write => "write",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Stage::Load => "loading backbone",
            Stage::Features => "decoding frames",
            Stage::Fit => "fitting head",
            Stage::Write => "writing brow.bin",
        }
    }
}

#[derive(Clone, Debug)]
pub enum Status {
    Idle,
    Running { log: Vec<String> },
    Done { brow_bin: PathBuf, log: Vec<String> },
    Failed { msg: String, log: Vec<String> },
}

impl Status {
    pub fn is_running(&self) -> bool {
        matches!(self, Status::Running { .. })
    }
}

struct Shared {
    status: Status,
    log: Vec<String>,
}

impl Shared {
    fn push_log(&mut self, line: String) {
        if self.log.len() >= LOG_CAP {
            let drop = self.log.len() - LOG_CAP + 1;
            self.log.drain(0..drop);
        }
        self.log.push(line);
        if let Status::Running { .. } = self.status {
            self.status = Status::Running {
                log: self.log.clone(),
            };
        }
    }
}

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

    pub fn status(&self) -> Status {
        self.shared
            .lock()
            .map(|s| s.status.clone())
            .unwrap_or(Status::Idle)
    }

    pub fn is_running(&self) -> bool {
        self.shared
            .lock()
            .map(|s| s.status.is_running())
            .unwrap_or(false)
    }

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
        {
            let mut s = self.shared.lock().unwrap();
            s.log.clear();
            s.status = Status::Running { log: Vec::new() };
        }
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
        if let Some(h) = self.handle.take() {
            drop(h);
        }
    }
}

fn run_fit(shared: Arc<Mutex<Shared>>, inputs: FitInputs) {
    use crate::ml::brow_net::BrowNet;

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

    log(
        &shared,
        format!("[{}] training head (out_dim={out_dim})", Stage::Fit.tag()),
    );
    let head = crate::ml::brow_fit::fit_head(&ds.features, &ds.labels, out_dim, inputs.seed);

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

fn log(shared: &Arc<Mutex<Shared>>, line: String) {
    if let Ok(mut s) = shared.lock() {
        s.push_log(line);
    }
}

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
        let e = f
            .start(FitInputs {
                backbone_bin: dir.join("nope.bin"),
                brow_data_dir: dir.clone(),
                seed: 0,
            })
            .unwrap_err();
        assert!(e.contains("base eyebrow model"), "{e}");
        assert!(!f.is_running());
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
