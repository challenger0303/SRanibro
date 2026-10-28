//! Private, supervised compute process. No camera, config, log capture or network.
//!
//! wgpu/driver panics and blocked calls must not run on the tracking thread. The
//! parent exchanges one bounded job at a time over anonymous pipes and falls back
//! to CPU on a deadline, EOF, malformed reply or failure. There is no auto-restart.
//! A single process-wide lease remains held until the I/O thread has exited AND
//! its child has been reaped; a broken teardown cannot accumulate more workers.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::eye_net::{EyeNet, OUT_DIM};
use super::tvm_params::Tensor;

const INPUT_LEN: usize = 2 * 100 * 100;
const MAGIC: &[u8; 8] = b"SRNGPU01";
const WORKER_ARG: &str = "--internal-gpu-worker";
const MAX_TEXT: usize = 4096;
const STARTUP_TIMEOUT: Duration = Duration::from_secs(8);
const SYNC_TIMEOUT: Duration = Duration::from_secs(2);
const LIVE_TIMEOUT: Duration = Duration::from_millis(250);
const LIVE_POLL: Duration = Duration::from_millis(2);

// Protocol dimensions are fixed by the already-validated model, never by an
// unchecked pipe length. We pass the exact loaded weights, not a path that could
// be replaced between the CPU validation and child startup.
const SPEC: &[(&str, &[i64])] = &[
    ("conv1_weight", &[20, 2, 5, 5]),
    ("conv1_bias", &[20]),
    ("conv2_weight", &[48, 20, 5, 5]),
    ("conv2_bias", &[48]),
    ("conv3_weight", &[64, 48, 3, 3]),
    ("conv3_bias", &[64]),
    ("fc5_ft_weight", &[500, 25600]),
    ("fc5_ft_bias", &[500]),
    ("fc6_10_weight", &[5, 500]),
    ("fc6_10_bias", &[5]),
];

static HOST: OnceLock<PathBuf> = OnceLock::new();
static OCCUPIED: AtomicBool = AtomicBool::new(false);
// If thread creation/termination itself fails, fail closed for this application
// session instead of repeatedly creating resources against a broken driver.
static QUARANTINED: AtomicBool = AtomicBool::new(false);

struct Lease;
impl Lease {
    fn acquire() -> Result<Arc<Self>, String> {
        if QUARANTINED.load(Ordering::Acquire) {
            return Err("GPU worker teardown failed; restart SRanibro to retry".into());
        }
        OCCUPIED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| "Previous GPU worker is still stopping; continuing on CPU".to_string())?;
        Ok(Arc::new(Self))
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        OCCUPIED.store(false, Ordering::Release);
    }
}

struct WorkerProcess {
    child: Mutex<Option<Child>>,
    stopping: AtomicBool,
    _lease: Arc<Lease>,
    #[cfg(windows)]
    _job: WorkerJob,
}

// An unnamed, non-inheritable job ensures a parent crash/forced exit also stops
// its GPU helper, including a helper blocked inside the driver or native Drop.
#[cfg(windows)]
pub(super) struct WorkerJob(isize);
#[cfg(windows)]
impl WorkerJob {
    #[cfg(any(feature = "xr5-research-model", feature = "external-eyelid-model"))]
    pub(super) fn terminate_tree(&self) {
        // The Python model can own a GPU child that inherited its output pipe.
        // Kill the entire tree before waiting for pipe EOF, not on final Drop.
        unsafe { windows_sys::Win32::System::JobObjects::TerminateJobObject(self.0 as _, 1); }
    }
    pub(super) fn attach(child: &Child) -> io::Result<Self> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::JobObjects::*;
        unsafe {
            let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if handle.is_null() {
                return Err(io::Error::last_os_error());
            }
            let job = Self(handle as isize);
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                &limits as *const _ as _,
                std::mem::size_of_val(&limits) as u32,
            ) == 0
                || AssignProcessToJobObject(handle, child.as_raw_handle() as _) == 0
            {
                return Err(io::Error::last_os_error());
            }
            Ok(job)
        }
    }
}
#[cfg(windows)]
impl Drop for WorkerJob {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0 as _);
        }
    }
}
impl WorkerProcess {
    fn exit_note(&self) -> String {
        let mut slot = self.child.lock().unwrap_or_else(|p| p.into_inner());
        match slot.as_mut().map(Child::try_wait) {
            Some(Ok(Some(status))) => format!("; worker {status}"),
            _ => String::new(),
        }
    }
    // No wait/join here. The owner I/O thread reaps after pipe EOF; any stalled
    // cleanup keeps the lease occupied and prevents additional GPU processes.
    fn terminate(&self) {
        if self.stopping.swap(true, Ordering::AcqRel) {
            return;
        }
        let mut slot = self.child.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(child) = slot.as_mut() {
            if !matches!(child.try_wait(), Ok(Some(_))) && child.kill().is_err() {
                QUARANTINED.store(true, Ordering::Release);
            }
        }
    }

    fn reap(&self) {
        self.terminate();
        let child = self.child.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(mut child) = child {
            // This is ONLY the bounded-count pipe thread, never the GUI/tracker.
            // Its lease stays occupied if the OS cannot finish reaping the child.
            if child.wait().is_err() {
                QUARANTINED.store(true, Ordering::Release);
            }
        }
    }
}
impl Drop for WorkerProcess {
    fn drop(&mut self) {
        self.terminate();
    }
}

struct Job {
    sequence: u64,
    batch: usize,
    input: Vec<f32>,
}
struct Completed {
    sequence: u64,
    batch: usize,
    raw: [[f32; OUT_DIM]; 2],
    input: Vec<f32>,
}
enum Reply {
    Ready(String),
    Completed(Completed),
}

#[derive(Clone, Copy)]
struct Pending {
    sequence: u64,
    batch: usize,
    started: Instant,
    budget: Duration,
}
impl Pending {
    fn remaining(self, now: Instant) -> Duration {
        self.budget
            .saturating_sub(now.saturating_duration_since(self.started))
    }
    fn accepts(self, sequence: u64, batch: usize, now: Instant) -> bool {
        sequence == self.sequence && batch == self.batch && !self.remaining(now).is_zero()
    }
}

pub(crate) struct GpuWorkerClient {
    process: Arc<WorkerProcess>,
    sender: Option<mpsc::SyncSender<Job>>,
    receiver: mpsc::Receiver<Result<Reply, String>>,
    pending: Option<Pending>,
    upload: Vec<f32>,
    sequence: u64,
    failed: bool,
    adapter_name: String,
}

impl GpuWorkerClient {
    pub(crate) fn new(net: &EyeNet) -> Result<Self, String> {
        let executable = HOST
            .get()
            .ok_or("This executable has no isolated GPU worker; using CPU")?;
        let lease = Lease::acquire()?;
        let weights = net.weights();
        let owned: Vec<Vec<f32>> = [
            weights.conv1_w,
            weights.conv1_b,
            weights.conv2_w,
            weights.conv2_b,
            weights.conv3_w,
            weights.conv3_b,
            weights.fc1_w,
            weights.fc1_b,
            weights.fc2_w,
            weights.fc2_b,
        ]
        .iter()
        .map(|w| w.to_vec())
        .collect();
        let mut command = Command::new(executable);
        command
            .arg(WORKER_ARG)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let mut child = command
            .spawn()
            .map_err(|e| format!("Could not start isolated GPU worker: {e}"))?;
        #[cfg(windows)]
        let job = match WorkerJob::attach(&child) {
            Ok(job) => job,
            Err(error) => {
                let _ = child.kill();
                QUARANTINED.store(true, Ordering::Release);
                return Err(format!("Could not contain GPU worker lifetime: {error}"));
            }
        };
        let pipes = child.stdin.take().zip(child.stdout.take());
        let Some((mut input, mut output)) = pipes else {
            let _ = child.kill();
            QUARANTINED.store(true, Ordering::Release);
            return Err("GPU worker pipes unavailable; continuing on CPU".into());
        };
        let process = Arc::new(WorkerProcess {
            child: Mutex::new(Some(child)),
            stopping: AtomicBool::new(false),
            _lease: lease,
            #[cfg(windows)]
            _job: job,
        });
        let (sender, jobs) = mpsc::sync_channel::<Job>(1);
        let (replies, receiver) = mpsc::sync_channel(1);
        let thread_process = process.clone();
        let spawned = std::thread::Builder::new()
            .name("eyelid-gpu-ipc".into())
            .spawn(move || {
                let result = (|| -> Result<(), String> {
                    write_weights(&mut input, &owned)
                        .map_err(|e| format!("GPU worker model transfer: {e}"))?;
                    drop(owned);
                    let adapter =
                        read_ready(&mut output).map_err(|e| format!("GPU worker startup: {e}"))?;
                    replies
                        .try_send(Ok(Reply::Ready(adapter)))
                        .map_err(|_| "GPU startup receiver closed")?;
                    while let Ok(job) = jobs.recv() {
                        write_job(&mut input, &job)
                            .map_err(|e| format!("GPU worker request: {e}"))?;
                        let (sequence, batch, raw) = read_result(&mut output)
                            .map_err(|e| format!("GPU worker reply: {e}"))?;
                        let reply = Completed {
                            sequence,
                            batch,
                            raw,
                            input: job.input,
                        };
                        replies
                            .try_send(Ok(Reply::Completed(reply)))
                            .map_err(|_| "GPU result receiver closed")?;
                    }
                    Ok(())
                })();
                if let Err(error) = result {
                    let _ = replies.try_send(Err(format!("{error}{}", thread_process.exit_note())));
                }
                // Closing pipes first also releases any waiting producer/consumer in
                // the child. Kill handles a blocked driver or blocked native Drop.
                drop(input);
                drop(output);
                thread_process.reap();
            });
        if let Err(error) = spawned {
            process.terminate();
            QUARANTINED.store(true, Ordering::Release);
            return Err(format!("Could not start GPU pipe thread: {error}"));
        }
        let mut client = Self {
            process,
            sender: Some(sender),
            receiver,
            pending: None,
            upload: vec![0.0; 2 * INPUT_LEN],
            sequence: 0,
            failed: false,
            adapter_name: String::new(),
        };
        match client.receiver.recv_timeout(STARTUP_TIMEOUT) {
            Ok(Ok(Reply::Ready(name))) => {
                client.adapter_name = name;
                Ok(client)
            }
            Ok(Err(error)) => {
                client.fail();
                Err(error)
            }
            _ => {
                client.fail();
                Err("GPU worker initialization did not complete within 8 seconds".into())
            }
        }
    }

    pub(crate) fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    fn fail(&mut self) {
        self.failed = true;
        self.pending = None;
        self.sender.take();
        self.process.terminate();
    }

    fn submit(
        &mut self,
        primary: &[f32],
        secondary: Option<&[f32]>,
        budget: Duration,
    ) -> Result<(), String> {
        if self.failed {
            return Err("GPU worker is disabled for this model instance".into());
        }
        if self.pending.is_some() {
            return Err("GPU worker already has one job in flight".into());
        }
        if primary.len() != INPUT_LEN || secondary.is_some_and(|v| v.len() != INPUT_LEN) {
            return Err("GPU worker received an invalid input shape".into());
        }
        if !primary
            .iter()
            .chain(secondary.into_iter().flatten())
            .all(|v| v.is_finite())
        {
            return Err("GPU worker received non-finite input".into());
        }
        if budget.is_zero() {
            self.fail();
            return Err("GPU inference deadline expired".into());
        }
        let Some(sequence) = self.sequence.checked_add(1) else {
            self.fail();
            return Err("GPU sequence exhausted".into());
        };
        self.sequence = sequence;
        let batch = if secondary.is_some() { 2 } else { 1 };
        self.upload[..INPUT_LEN].copy_from_slice(primary);
        if let Some(secondary) = secondary {
            self.upload[INPUT_LEN..].copy_from_slice(secondary);
        }
        let job = Job {
            sequence,
            batch,
            input: std::mem::take(&mut self.upload),
        };
        self.pending = Some(Pending {
            sequence,
            batch,
            started: Instant::now(),
            budget,
        });
        if self
            .sender
            .as_ref()
            .is_none_or(|sender| sender.try_send(job).is_err())
        {
            self.fail();
            return Err("GPU worker request channel unavailable".into());
        }
        Ok(())
    }

    fn receive(&mut self, wait: Duration) -> Result<Option<(usize, [[f32; OUT_DIM]; 2])>, String> {
        if self.failed {
            return Err("GPU worker is disabled for this model instance".into());
        }
        let Some(pending) = self.pending else {
            return Ok(None);
        };
        let remaining = pending.remaining(Instant::now());
        if remaining.is_zero() {
            self.fail();
            return Err("GPU inference timed out; continuing on CPU".into());
        }
        match self.receiver.recv_timeout(wait.min(remaining)) {
            Ok(Ok(Reply::Completed(completed)))
                if pending.accepts(completed.sequence, completed.batch, Instant::now()) =>
            {
                self.pending = None;
                self.upload = completed.input;
                Ok(Some((completed.batch, completed.raw)))
            }
            Err(mpsc::RecvTimeoutError::Timeout)
                if !pending.remaining(Instant::now()).is_zero() =>
            {
                Ok(None)
            }
            reply => {
                let message = match reply {
                    Ok(Err(error)) => error,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        "GPU inference timed out; continuing on CPU".into()
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        "GPU worker exited unexpectedly".into()
                    }
                    _ => "GPU worker returned a late, unexpected or mismatched result".into(),
                };
                self.fail();
                Err(message)
            }
        }
    }

    pub(crate) fn infer_batch(
        &mut self,
        primary: &[f32],
        secondary: Option<&[f32]>,
    ) -> Result<[[f32; OUT_DIM]; 2], String> {
        let started = Instant::now();
        if self.pending.is_some() {
            let _ = self.receive(SYNC_TIMEOUT)?;
        }
        let remaining = SYNC_TIMEOUT.saturating_sub(started.elapsed());
        self.submit(primary, secondary, remaining)?;
        match self.receive(remaining)? {
            Some((_, raw)) => Ok(raw),
            None => {
                self.fail();
                Err("GPU synchronous inference deadline expired".into())
            }
        }
    }

    pub(crate) fn infer_live_batch(
        &mut self,
        primary: &[f32],
        secondary: Option<&[f32]>,
    ) -> Result<Option<[[f32; OUT_DIM]; 2]>, String> {
        if self.pending.is_none() {
            self.submit(primary, secondary, LIVE_TIMEOUT)?;
            return Ok(None);
        }
        let Some((batch, raw)) = self.receive(LIVE_POLL)? else {
            return Ok(None);
        };
        let requested_batch = if secondary.is_some() { 2 } else { 1 };
        self.submit(primary, secondary, LIVE_TIMEOUT)?;
        Ok((batch == requested_batch).then_some(raw))
    }
}
impl Drop for GpuWorkerClient {
    fn drop(&mut self) {
        self.fail();
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn read_u8(reader: &mut impl Read) -> io::Result<u8> {
    let mut bytes = [0];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}
fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}
fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}
fn write_floats(writer: &mut impl Write, values: &[f32]) -> io::Result<()> {
    #[cfg(target_endian = "little")]
    return writer.write_all(bytemuck::cast_slice(values));
    #[cfg(target_endian = "big")]
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    #[cfg(target_endian = "big")]
    Ok(())
}
fn read_floats(reader: &mut impl Read, values: &mut [f32]) -> io::Result<()> {
    reader.read_exact(bytemuck::cast_slice_mut(values))?;
    #[cfg(target_endian = "big")]
    for value in values.iter_mut() {
        *value = f32::from_bits(value.to_bits().swap_bytes());
    }
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(invalid("non-finite GPU protocol value"))
    }
}
fn write_text(writer: &mut impl Write, text: &str) -> io::Result<()> {
    let mut len = text.len().min(MAX_TEXT);
    while !text.is_char_boundary(len) {
        len -= 1;
    }
    writer.write_all(&(len as u32).to_le_bytes())?;
    writer.write_all(&text.as_bytes()[..len])
}
fn read_text(reader: &mut impl Read) -> io::Result<String> {
    let len = read_u32(reader)? as usize;
    if len > MAX_TEXT {
        return Err(invalid("oversized GPU protocol text"));
    }
    let mut bytes = vec![0; len];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|_| invalid("invalid GPU protocol text"))
}
fn write_weights(writer: &mut impl Write, weights: &[Vec<f32>]) -> io::Result<()> {
    if weights.len() != SPEC.len() {
        return Err(invalid("invalid GPU model tensor count"));
    }
    writer.write_all(MAGIC)?;
    for ((_, shape), values) in SPEC.iter().zip(weights) {
        let len: usize = shape.iter().map(|n| *n as usize).product();
        if values.len() != len {
            return Err(invalid("invalid GPU model dimensions"));
        }
        writer.write_all(&(len as u32).to_le_bytes())?;
        write_floats(writer, values)?;
    }
    writer.flush()
}
fn read_weights(reader: &mut impl Read) -> io::Result<EyeNet> {
    let mut magic = [0; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid("GPU worker protocol mismatch"));
    }
    let mut tensors = HashMap::new();
    for (name, shape) in SPEC {
        let len: usize = shape.iter().map(|n| *n as usize).product();
        if read_u32(reader)? as usize != len {
            return Err(invalid("invalid GPU model dimensions"));
        }
        let mut data = vec![0.0; len];
        read_floats(reader, &mut data)?;
        tensors.insert(
            (*name).into(),
            Tensor {
                name: (*name).into(),
                shape: shape.to_vec(),
                data,
            },
        );
    }
    EyeNet::new(tensors).map_err(|e| invalid(&e))
}
fn read_ready(reader: &mut impl Read) -> io::Result<String> {
    match read_u8(reader)? {
        0 => read_text(reader),
        2 => Err(invalid(&read_text(reader)?)),
        _ => Err(invalid("invalid GPU startup reply")),
    }
}
fn write_job(writer: &mut impl Write, job: &Job) -> io::Result<()> {
    writer.write_all(&job.sequence.to_le_bytes())?;
    writer.write_all(&[job.batch as u8])?;
    write_floats(writer, &job.input[..job.batch * INPUT_LEN])?;
    writer.flush()
}
fn read_result(reader: &mut impl Read) -> io::Result<(u64, usize, [[f32; OUT_DIM]; 2])> {
    match read_u8(reader)? {
        1 => {}
        2 => return Err(invalid(&read_text(reader)?)),
        _ => return Err(invalid("invalid GPU result tag")),
    }
    let sequence = read_u64(reader)?;
    let batch = read_u8(reader)? as usize;
    if !(1..=2).contains(&batch) {
        return Err(invalid("invalid GPU result batch"));
    }
    let mut raw = [[0.0; OUT_DIM]; 2];
    for sample in &mut raw[..batch] {
        read_floats(reader, sample)?;
    }
    Ok((sequence, batch, raw))
}

fn run_worker(reader: &mut impl Read, writer: &mut impl Write) -> Result<(), String> {
    let net = read_weights(reader).map_err(|e| e.to_string())?;
    let mut gpu = super::eye_net_gpu::GpuEyeNet::new(&net)?;
    drop(net);
    writer.write_all(&[0]).map_err(|e| e.to_string())?;
    write_text(writer, gpu.adapter_name()).map_err(|e| e.to_string())?;
    writer.flush().map_err(|e| e.to_string())?;
    let mut input = vec![0.0; 2 * INPUT_LEN];
    loop {
        let sequence = match read_u64(reader) {
            Ok(sequence) => sequence,
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(error.to_string()),
        };
        let batch = read_u8(reader).map_err(|e| e.to_string())? as usize;
        if !(1..=2).contains(&batch) {
            return Err("invalid GPU request batch".into());
        }
        read_floats(reader, &mut input[..batch * INPUT_LEN]).map_err(|e| e.to_string())?;
        let raw = gpu.infer_batch(
            &input[..INPUT_LEN],
            (batch == 2).then_some(&input[INPUT_LEN..]),
        )?;
        writer.write_all(&[1]).map_err(|e| e.to_string())?;
        writer
            .write_all(&sequence.to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&[batch as u8])
            .map_err(|e| e.to_string())?;
        for sample in &raw[..batch] {
            write_floats(writer, sample).map_err(|e| e.to_string())?;
        }
        writer.flush().map_err(|e| e.to_string())?;
    }
}

/// Main binary only: call before diagnostics/config/services. Tools that do not
/// implement this entry point safely stay CPU-only rather than spawning themselves.
#[doc(hidden)]
pub fn gpu_worker_entrypoint() -> bool {
    if std::env::args().nth(1).as_deref() != Some(WORKER_ARG) {
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        // PR_SET_PDEATHSIG follows the *creating thread*, not the parent process.
        // Reload starts the helper on a short-lived thread, so that signal would
        // kill a healthy helper as soon as Reload finished. Observe reparenting
        // independently instead; this thread remains runnable if compute blocks.
        let parent = unsafe { libc::getppid() };
        if parent <= 1 {
            return true;
        }
        if std::thread::Builder::new()
            .name("gpu-parent-watch".into())
            .spawn(move || loop {
                if unsafe { libc::getppid() } != parent {
                    // Do not enter a potentially blocked GPU Drop on parent death.
                    unsafe {
                        libc::_exit(0);
                    }
                }
                std::thread::sleep(Duration::from_millis(250));
            })
            .is_err()
        {
            return true;
        }
    }
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    if let Err(error) = run_worker(&mut input, &mut output) {
        let _ = output.write_all(&[2]);
        let _ = write_text(&mut output, &error);
        let _ = output.flush();
    }
    true
}

#[doc(hidden)]
pub fn enable_gpu_worker_host() {
    if let Ok(path) = std::env::current_exe() {
        let _ = HOST.set(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deadline_and_sequence_reject_late_or_mismatched_results() {
        let now = Instant::now();
        let job = Pending {
            sequence: 42,
            batch: 2,
            started: now,
            budget: LIVE_TIMEOUT,
        };
        assert!(job.accepts(42, 2, now + LIVE_TIMEOUT - Duration::from_nanos(1)));
        assert!(!job.accepts(42, 2, now + LIVE_TIMEOUT));
        assert!(!job.accepts(41, 2, now));
        assert!(!job.accepts(42, 1, now));
        assert_eq!(job.remaining(now + Duration::from_secs(99)), Duration::ZERO);
    }
    #[test]
    fn protocol_rejects_unbounded_dimensions_before_allocating() {
        let mut bad = MAGIC.to_vec();
        bad.extend(u32::MAX.to_le_bytes());
        assert!(read_weights(&mut bad.as_slice()).is_err());
        assert!(read_text(&mut u32::MAX.to_le_bytes().as_slice()).is_err());
        assert!(read_result(&mut [1, 0, 0, 0, 0, 0, 0, 0, 0, 255].as_slice()).is_err());
    }
    #[test]
    fn protocol_preserves_asymmetric_pair_and_rejects_nan_or_truncated() {
        let mut bytes = vec![1];
        bytes.extend(7_u64.to_le_bytes());
        bytes.push(2);
        for x in 0..10 {
            bytes.extend((x as f32 * 0.125).to_le_bytes());
        }
        let (seq, batch, raw) = read_result(&mut bytes.as_slice()).unwrap();
        assert_eq!((seq, batch), (7, 2));
        assert_eq!(raw[0][1], 0.125);
        assert_eq!(raw[1][4], 1.125);
        assert!(read_result(&mut &bytes[..bytes.len() - 1]).is_err());
        bytes[10..14].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(read_result(&mut bytes.as_slice()).is_err());
    }
    #[test]
    fn text_is_bounded_and_utf8_safe() {
        let mut bytes = Vec::new();
        write_text(&mut bytes, &"眼".repeat(MAX_TEXT)).unwrap();
        let decoded = read_text(&mut bytes.as_slice()).unwrap();
        assert!(decoded.len() <= MAX_TEXT);
        assert!(decoded.chars().all(|c| c == '眼'));
    }
}
