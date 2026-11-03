
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

const LOG_CAP: usize = 200;

const BAKE_PY: &str = include_str!("../tools/bake_brow_weights.py");

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    Split,
    Train,
    Bake,
}

impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Stage::Split => "splitting labels",
            Stage::Train => "training (PyTorch)",
            Stage::Bake => "baking weights",
        }
    }
}

#[derive(Clone, Debug)]
pub enum Status {
    Idle,
    Running { stage: Stage, log: Vec<String> },
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
    stage: Stage,
}

impl Shared {
    fn push_log(&mut self, line: String) {
        if self.log.len() >= LOG_CAP {
            let drop = self.log.len() - LOG_CAP + 1;
            self.log.drain(0..drop);
        }
        self.log.push(line);
        if let Status::Running { stage, .. } = self.status {
            self.status = Status::Running {
                stage,
                log: self.log.clone(),
            };
        }
    }

    fn set_stage(&mut self, stage: Stage) {
        self.stage = stage;
        self.status = Status::Running {
            stage,
            log: self.log.clone(),
        };
    }
}

pub struct BrowTrainer {
    shared: Arc<Mutex<Shared>>,
    handle: Option<JoinHandle<()>>,
}

impl Default for BrowTrainer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct TrainInputs {
    pub python_exe: PathBuf,
    pub vr_eyebrow_dir: PathBuf,
    pub brow_data_dir: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedCmd {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub cwd: PathBuf,
    pub stage: Stage,
}

impl BrowTrainer {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(Shared {
                status: Status::Idle,
                log: Vec::new(),
                stage: Stage::Split,
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

    pub fn start(&mut self, inputs: TrainInputs) -> Result<(), String> {
        if self.is_running() {
            return Err("a training run is already in progress".into());
        }
        validate_inputs(&inputs)?;
        let bake = resolve_bake_script().map_err(|e| format!("bake tool: {e}"))?;
        let plan = build_commands(&inputs, &bake.path);

        {
            let mut s = self.shared.lock().unwrap();
            s.log.clear();
            s.stage = Stage::Split;
            s.status = Status::Running {
                stage: Stage::Split,
                log: Vec::new(),
            };
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }

        let shared = self.shared.clone();
        let inputs2 = inputs.clone();
        let handle = std::thread::Builder::new()
            .name("brow-trainer".into())
            .spawn(move || run_all(shared, inputs2, plan, bake))
            .map_err(|e| format!("could not spawn trainer thread: {e}"))?;
        self.handle = Some(handle);
        Ok(())
    }
}

impl Drop for BrowTrainer {
    fn drop(&mut self) {
        if let Some(h) = self.handle.take() {
            drop(h);
        }
    }
}

fn validate_inputs(inputs: &TrainInputs) -> Result<(), String> {
    if !inputs.python_exe.is_file() {
        return Err(format!(
            "Python interpreter not found: {}",
            inputs.python_exe.display()
        ));
    }
    let train_py = inputs.vr_eyebrow_dir.join("train.py");
    if !train_py.is_file() {
        return Err(format!(
            "train.py not found in vr_eyebrow dir: {}",
            train_py.display()
        ));
    }
    let labels = inputs.brow_data_dir.join("labels.csv");
    if !labels.is_file() {
        return Err(format!(
            "labels.csv not found — capture a dataset first ({})",
            labels.display()
        ));
    }
    if !inputs.brow_data_dir.join("images").is_dir() {
        return Err(format!(
            "images/ not found under {}",
            inputs.brow_data_dir.display()
        ));
    }
    Ok(())
}

pub fn build_commands(inputs: &TrainInputs, bake_script: &Path) -> Vec<PlannedCmd> {
    let images = inputs.brow_data_dir.join("images");
    let train_csv = inputs.brow_data_dir.join("train.csv");
    let val_csv = inputs.brow_data_dir.join("val.csv");
    let model_pth = inputs.brow_data_dir.join("model.pth");
    let train_py = inputs.vr_eyebrow_dir.join("train.py");

    let s = |p: &Path| p.to_string_lossy().into_owned();

    let train = PlannedCmd {
        program: inputs.python_exe.clone(),
        args: vec![
            s(&train_py),
            "--data-dir".into(),
            s(&images),
            "--train-csv".into(),
            s(&train_csv),
            "--val-csv".into(),
            s(&val_csv),
            "--save-path".into(),
            s(&model_pth),
        ],
        cwd: inputs.vr_eyebrow_dir.clone(),
        stage: Stage::Train,
    };
    let bake = PlannedCmd {
        program: inputs.python_exe.clone(),
        args: vec![
            s(bake_script),
            "--pth".into(),
            s(&model_pth),
            "--out".into(),
            s(&inputs.brow_data_dir),
        ],
        cwd: inputs.brow_data_dir.clone(),
        stage: Stage::Bake,
    };
    vec![train, bake]
}

fn run_all(
    shared: Arc<Mutex<Shared>>,
    inputs: TrainInputs,
    plan: Vec<PlannedCmd>,
    _bake: BakeScript,
) {
    {
        let mut s = shared.lock().unwrap();
        s.set_stage(Stage::Split);
        s.push_log("[split] labels.csv -> train.csv + val.csv".into());
    }
    match crate::brow_calib::split_labels(&inputs.brow_data_dir) {
        Ok((t, v)) => {
            let mut s = shared.lock().unwrap();
            s.push_log(format!("[split] wrote {} and {}", t.display(), v.display()));
        }
        Err(e) => {
            fail(&shared, format!("split failed: {e}"));
            return;
        }
    }

    for cmd in &plan {
        {
            let mut s = shared.lock().unwrap();
            s.set_stage(cmd.stage);
            s.push_log(format!(
                "[{}] {}",
                stage_tag(cmd.stage),
                render_cmdline(cmd)
            ));
        }
        match run_streamed(cmd, &shared) {
            Ok(0) => {}
            Ok(code) => {
                fail(
                    &shared,
                    format!("{} exited with code {code}", cmd.stage.label()),
                );
                return;
            }
            Err(e) => {
                fail(&shared, format!("{}: {e}", cmd.stage.label()));
                return;
            }
        }
    }

    let brow_bin = inputs.brow_data_dir.join("brow.bin");
    if !brow_bin.is_file() {
        fail(
            &shared,
            format!("bake finished but {} was not written", brow_bin.display()),
        );
        return;
    }
    let mut s = shared.lock().unwrap();
    s.push_log(format!("[done] {}", brow_bin.display()));
    let log = s.log.clone();
    s.status = Status::Done { brow_bin, log };
}

fn fail(shared: &Arc<Mutex<Shared>>, msg: String) {
    let mut s = shared.lock().unwrap();
    s.push_log(format!("[error] {msg}"));
    let log = s.log.clone();
    s.status = Status::Failed { msg, log };
}

fn stage_tag(stage: Stage) -> &'static str {
    match stage {
        Stage::Split => "split",
        Stage::Train => "train",
        Stage::Bake => "bake",
    }
}

fn render_cmdline(cmd: &PlannedCmd) -> String {
    let q = |s: &str| {
        if s.contains(' ') {
            format!("\"{s}\"")
        } else {
            s.to_string()
        }
    };
    let mut out = q(&cmd.program.to_string_lossy());
    for a in &cmd.args {
        out.push(' ');
        out.push_str(&q(a));
    }
    out
}

#[cfg(windows)]
fn run_streamed(cmd: &PlannedCmd, shared: &Arc<Mutex<Shared>>) -> std::io::Result<i32> {
    use std::io::{BufRead, BufReader};
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    let mut child = std::process::Command::new(&cmd.program)
        .args(&cmd.args)
        .current_dir(&cmd.cwd)
        .env("PYTHONUNBUFFERED", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let se_shared = shared.clone();
    let se_thread = stderr.map(|se| {
        std::thread::spawn(move || {
            for line in BufReader::new(se).lines().map_while(Result::ok) {
                push_shared(&se_shared, line);
            }
        })
    });
    if let Some(so) = stdout {
        for line in BufReader::new(so).lines().map_while(Result::ok) {
            push_shared(shared, line);
        }
    }
    if let Some(t) = se_thread {
        let _ = t.join();
    }
    let status = child.wait()?;
    Ok(status.code().unwrap_or(-1))
}

#[cfg(not(windows))]
fn run_streamed(cmd: &PlannedCmd, shared: &Arc<Mutex<Shared>>) -> std::io::Result<i32> {
    use std::io::{BufRead, BufReader};
    let mut child = std::process::Command::new(&cmd.program)
        .args(&cmd.args)
        .current_dir(&cmd.cwd)
        .env("PYTHONUNBUFFERED", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let se_shared = shared.clone();
    let se_thread = stderr.map(|se| {
        std::thread::spawn(move || {
            for line in BufReader::new(se).lines().map_while(Result::ok) {
                push_shared(&se_shared, line);
            }
        })
    });
    if let Some(so) = stdout {
        for line in BufReader::new(so).lines().map_while(Result::ok) {
            push_shared(shared, line);
        }
    }
    if let Some(t) = se_thread {
        let _ = t.join();
    }
    let status = child.wait()?;
    Ok(status.code().unwrap_or(-1))
}

fn push_shared(shared: &Arc<Mutex<Shared>>, line: String) {
    if let Ok(mut s) = shared.lock() {
        s.push_log(line);
    }
}

pub struct BakeScript {
    pub path: PathBuf,
    temp: Option<PathBuf>,
}

impl Drop for BakeScript {
    fn drop(&mut self) {
        if let Some(p) = &self.temp {
            let _ = std::fs::remove_file(p);
        }
    }
}

pub fn resolve_bake_script() -> std::io::Result<BakeScript> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for rel in ["bake_brow_weights.py", "tools/bake_brow_weights.py"] {
                let p = dir.join(rel);
                if p.is_file() {
                    return Ok(BakeScript {
                        path: p,
                        temp: None,
                    });
                }
            }
        }
    }
    let tmp = std::env::temp_dir().join(format!(
        "sranibro_bake_brow_weights_{}.py",
        std::process::id()
    ));
    std::fs::write(&tmp, BAKE_PY)?;
    Ok(BakeScript {
        path: tmp.clone(),
        temp: Some(tmp),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(base: &Path) -> TrainInputs {
        TrainInputs {
            python_exe: base.join("py").join("python.exe"),
            vr_eyebrow_dir: base.join("vr_eyebrow"),
            brow_data_dir: base.join("brow_data"),
        }
    }

    #[test]
    fn build_commands_uses_1channel_train_then_bake() {
        let base = Path::new("C:\\t");
        let bake = Path::new("C:\\ship\\bake_brow_weights.py");
        let plan = build_commands(&inputs(base), bake);
        assert_eq!(plan.len(), 2);

        let train = &plan[0];
        assert_eq!(train.stage, Stage::Train);
        assert!(train.args[0].ends_with("train.py"));
        assert_eq!(
            train.cwd,
            base.join("vr_eyebrow"),
            "train runs in the vr_eyebrow dir"
        );
        let joined = train.args.join(" ");
        assert!(joined.contains("--data-dir"));
        assert!(joined.contains("--train-csv"));
        assert!(joined.contains("--val-csv"));
        assert!(joined.contains("--save-path"));
        assert!(
            !joined.contains("--reference"),
            "1-channel: no reference args: {joined}"
        );
        assert!(train.args.iter().any(|a| a.ends_with("images")));
        assert!(train.args.iter().any(|a| a.ends_with("train.csv")));
        assert!(train.args.iter().any(|a| a.ends_with("val.csv")));
        assert!(train.args.iter().any(|a| a.ends_with("model.pth")));

        let b = &plan[1];
        assert_eq!(b.stage, Stage::Bake);
        assert_eq!(b.args[0], bake.to_string_lossy());
        assert!(b.args.iter().any(|a| a.ends_with("model.pth")));
        let out_idx = b.args.iter().position(|a| a == "--out").unwrap();
        assert_eq!(
            b.args[out_idx + 1],
            base.join("brow_data").to_string_lossy()
        );
    }

    #[test]
    fn validate_reports_each_missing_input() {
        let base = std::env::temp_dir().join(format!("sranibro_bt_val_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let inp = inputs(&base);
        let e = validate_inputs(&inp).unwrap_err();
        assert!(e.contains("Python interpreter"), "{e}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn resolve_bake_script_always_yields_a_readable_script() {
        let bs = resolve_bake_script().expect("resolve");
        let text = std::fs::read_to_string(&bs.path).expect("read");
        assert!(text.contains("BROWNET1"), "resolved file is the bake tool");
    }

    #[test]
    fn embedded_bake_script_is_the_bake_tool() {
        assert!(BAKE_PY.contains("BROWNET1"));
        assert!(BAKE_PY.contains("--pth"));
        assert!(BAKE_PY.contains("--out"));
    }
}
