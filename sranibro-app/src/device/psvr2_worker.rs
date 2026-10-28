//! Isolated owner of the installed PSVR2Toolkit CAPI.
//!
//! Native initialization, mutex waits and destruction may block or crash. They
//! never execute in the UI/tracking process. A bounded pipe reader keeps only
//! the latest camera/gaze frame, while control events are preserved or fail
//! explicitly on overflow. A process-wide lease prevents overlapping sessions.

use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::psvr2_capi::{
    EyeImageFrame, FoveatedGaze, GazeStatus, GazeVec2, GazeVec3, LensConfig, Psvr2Capi,
    WearableData, WearableEye, IMAGE_HEADER_BYTES, IMAGE_PIXELS, RESULT_DRIVER_INACTIVE,
    RESULT_NO_SLOT, RESULT_OK,
};

const WORKER_ARG: &str = "--internal-psvr2-worker";
const MAGIC: &[u8; 8] = b"SRNPSV01";
const MAX_TEXT: usize = 4096;
const MAX_PACKET: usize = IMAGE_HEADER_BYTES + IMAGE_PIXELS;
const MAX_CONTROL: usize = 32;
const STOP_BUDGET: Duration = Duration::from_millis(500);
const GRACE_BUDGET: Duration = Duration::from_millis(40);
const HEARTBEAT: Duration = Duration::from_millis(100);

static HOST: OnceLock<PathBuf> = OnceLock::new();
static OCCUPIED: AtomicBool = AtomicBool::new(false);
static QUARANTINED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StreamKind {
    Gaze,
    Image,
}

#[derive(Debug)]
pub(crate) enum WorkerEvent {
    /// Emitted before discovery and again with the selected path before init.
    Loading {
        path: String,
    },
    Initialized {
        path: String,
    },
    /// A heartbeat, including when the state has not changed.
    Driver(bool),
    Gaze(GazeStatus),
    Image(EyeImageFrame),
    StreamError {
        stream: StreamKind,
        message: String,
    },
    Fatal(String),
}

struct Lease;
impl Lease {
    fn acquire() -> io::Result<Self> {
        if QUARANTINED.load(Ordering::Acquire) {
            return Err(io::Error::other(
                "PSVR2 helper cleanup failed; restart SRanibro before reconnecting",
            ));
        }
        OCCUPIED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "Previous PSVR2 helper is still stopping",
                )
            })?;
        Ok(Self)
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        // Preserve the occupied gate permanently if OS cleanup could not be
        // established. Repeated retries must not accumulate helper processes.
        if !QUARANTINED.load(Ordering::Acquire) {
            OCCUPIED.store(false, Ordering::Release);
        }
    }
}

#[cfg(windows)]
struct WorkerJob(isize);
#[cfg(windows)]
impl WorkerJob {
    fn attach(child: &Child) -> io::Result<Self> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::JobObjects::*;
        unsafe {
            let h = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if h.is_null() {
                return Err(io::Error::last_os_error());
            }
            let job = Self(h as isize);
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                h,
                JobObjectExtendedLimitInformation,
                &limits as *const _ as _,
                std::mem::size_of_val(&limits) as u32,
            ) == 0
                || AssignProcessToJobObject(h, child.as_raw_handle() as _) == 0
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

struct Process {
    child: Mutex<Option<Child>>,
    stopping: AtomicBool,
    reaped: AtomicBool,
    reader_done: AtomicBool,
    lease: Mutex<Option<Lease>>,
    #[cfg(windows)]
    _job: WorkerJob,
}
impl Process {
    fn terminate(&self) {
        self.stopping.store(true, Ordering::Release);
        let mut slot = self.child.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(child) = slot.as_mut() {
            // Exit may race the kill. An already-exited process is safe to reap;
            // do not permanently quarantine a healthy host for that OS error.
            if !matches!(child.try_wait(), Ok(Some(_)))
                && child.kill().is_err()
                && !matches!(child.try_wait(), Ok(Some(_)))
            {
                QUARANTINED.store(true, Ordering::Release);
            }
        }
    }
    // Only the one pipe reader calls this. Its process/lease remains owned if
    // the OS cannot reap, rather than blocking any caller or launching another.
    fn reap_on_reader(&self) {
        self.terminate();
        let child = self.child.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(mut child) = child {
            if child.wait().is_err() {
                QUARANTINED.store(true, Ordering::Release);
                return;
            }
        }
        self.reaped.store(true, Ordering::Release);
    }
    fn poll_reap_without_reader(&self) -> bool {
        let mut slot = self.child.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(child) = slot.as_mut() {
            if matches!(child.try_wait(), Ok(Some(_))) {
                slot.take();
                self.reaped.store(true, Ordering::Release);
            }
        }
        self.reaped.load(Ordering::Acquire)
    }
    fn finish_reader(&self) {
        self.reader_done.store(true, Ordering::Release);
        if self.reaped.load(Ordering::Acquire) {
            // A bounded caller may already have returned while OS cleanup or
            // reader scheduling was delayed. Only this confirmed completion
            // releases its lease; a timeout alone never permits another child.
            self.lease.lock().unwrap_or_else(|p| p.into_inner()).take();
        }
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        if !self.reaped.load(Ordering::Acquire) {
            self.terminate();
            QUARANTINED.store(true, Ordering::Release);
        }
    }
}

#[derive(Default)]
struct Pending {
    control: VecDeque<WorkerEvent>,
    gaze: Option<GazeStatus>,
    image: Option<EyeImageFrame>,
    prefer_image: bool,
    ended: bool,
    failed: bool,
}
impl Pending {
    fn push(&mut self, event: WorkerEvent) {
        if self.failed {
            return;
        }
        match event {
            WorkerEvent::Gaze(frame) => self.gaze = Some(frame),
            WorkerEvent::Image(frame) => self.image = Some(frame),
            event => {
                if matches!(event, WorkerEvent::Driver(false) | WorkerEvent::Fatal(_)) {
                    self.gaze = None;
                    self.image = None;
                }
                if self.control.len() >= MAX_CONTROL {
                    self.control.clear();
                    self.gaze = None;
                    self.image = None;
                    self.failed = true;
                    self.control.push_back(WorkerEvent::Fatal(
                        "PSVR2 helper control queue overflow; reconnect required".into(),
                    ));
                } else {
                    self.failed = matches!(event, WorkerEvent::Fatal(_));
                    self.control.push_back(event);
                }
            }
        }
    }
    fn pop(&mut self) -> Option<WorkerEvent> {
        if let Some(event) = self.control.pop_front() {
            return Some(event);
        }
        if self.prefer_image {
            if let Some(frame) = self.image.take() {
                self.prefer_image = false;
                return Some(WorkerEvent::Image(frame));
            }
        }
        if let Some(frame) = self.gaze.take() {
            self.prefer_image = true;
            return Some(WorkerEvent::Gaze(frame));
        }
        self.image.take().map(WorkerEvent::Image)
    }
}
#[derive(Default)]
struct Mailbox {
    pending: Mutex<Pending>,
    ready: Condvar,
}
impl Mailbox {
    fn push(&self, event: WorkerEvent) {
        self.pending
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(event);
        self.ready.notify_all();
    }
    fn ended(&self) {
        self.pending.lock().unwrap_or_else(|p| p.into_inner()).ended = true;
        self.ready.notify_all();
    }
}

pub(crate) struct WorkerClient {
    process: Arc<Process>,
    mailbox: Arc<Mailbox>,
    input: Option<ChildStdin>,
    reader: Option<JoinHandle<()>>,
}
impl WorkerClient {
    pub(crate) fn spawn() -> io::Result<Self> {
        let executable = HOST.get().ok_or_else(|| {
            io::Error::other("This executable has no isolated PSVR2 helper entry point")
        })?;
        let lease = Lease::acquire()?;
        let mut command = Command::new(executable);
        command
            .arg(WORKER_ARG)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let mut child = command.spawn()?;
        #[cfg(windows)]
        let job = match WorkerJob::attach(&child) {
            Ok(job) => job,
            Err(error) => {
                cleanup_unstarted(&mut child);
                return Err(error);
            }
        };
        let Some((mut input, mut output)) = child.stdin.take().zip(child.stdout.take()) else {
            cleanup_unstarted(&mut child);
            return Err(io::Error::other("PSVR2 helper pipes unavailable"));
        };
        // The child waits for this handshake before loading any native library.
        // It is smaller than an empty anonymous pipe's capacity and sent once.
        if let Err(error) = input
            .write_all(MAGIC)
            .and_then(|_| input.write_all(&std::process::id().to_le_bytes()))
        {
            cleanup_unstarted(&mut child);
            return Err(error);
        }
        let process = Arc::new(Process {
            child: Mutex::new(Some(child)),
            stopping: AtomicBool::new(false),
            reaped: AtomicBool::new(false),
            reader_done: AtomicBool::new(false),
            lease: Mutex::new(Some(lease)),
            #[cfg(windows)]
            _job: job,
        });
        let mailbox = Arc::new(Mailbox::default());
        let owner = process.clone();
        let replies = mailbox.clone();
        let reader = thread::Builder::new()
            .name("psvr2-helper-ipc".into())
            .spawn(move || {
                loop {
                    match read_event(&mut output) {
                        Ok(event) => replies.push(event),
                        Err(error) => {
                            if !owner.stopping.load(Ordering::Acquire) {
                                replies.push(WorkerEvent::Fatal(format!(
                                    "PSVR2 helper disconnected: {error}"
                                )));
                            }
                            break;
                        }
                    }
                }
                owner.reap_on_reader();
                drop(output);
                replies.ended();
                owner.finish_reader();
            });
        let reader = match reader {
            Ok(reader) => reader,
            Err(error) => {
                process.terminate();
                let until = Instant::now() + STOP_BUDGET;
                while !process.poll_reap_without_reader() && Instant::now() < until {
                    thread::sleep(Duration::from_millis(5));
                }
                process.reader_done.store(true, Ordering::Release);
                if !process.reaped.load(Ordering::Acquire) {
                    QUARANTINED.store(true, Ordering::Release);
                }
                return Err(error);
            }
        };
        Ok(Self {
            process,
            mailbox,
            input: Some(input),
            reader: Some(reader),
        })
    }

    pub(crate) fn recv_timeout(&self, timeout: Duration) -> io::Result<Option<WorkerEvent>> {
        let deadline = Instant::now() + timeout;
        let mut pending = self
            .mailbox
            .pending
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(event) = pending.pop() {
                return Ok(Some(event));
            }
            if pending.ended {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "PSVR2 helper exited",
                ));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            let result = self
                .mailbox
                .ready
                .wait_timeout(pending, remaining)
                .unwrap_or_else(|p| p.into_inner());
            pending = result.0;
        }
    }

    pub(crate) fn stop(&mut self) -> io::Result<()> {
        let deadline = Instant::now() + STOP_BUDGET;
        self.process.stopping.store(true, Ordering::Release);
        if let Some(mut input) = self.input.take() {
            // One byte after the handshake, never an unbounded input backlog.
            let _ = input.write_all(b"X");
        }
        let graceful = Instant::now() + GRACE_BUDGET;
        while !self.process.reader_done.load(Ordering::Acquire) && Instant::now() < graceful {
            thread::sleep(Duration::from_millis(2));
        }
        if !self.process.reaped.load(Ordering::Acquire) {
            self.process.terminate();
        }
        while !(self.process.reaped.load(Ordering::Acquire)
            && self.process.reader_done.load(Ordering::Acquire)
            && self.reader.as_ref().is_none_or(JoinHandle::is_finished))
        {
            if Instant::now() >= deadline {
                // Keep ownership in the background reader. Slow scheduling is
                // not a permanent native failure: manual reconnect is allowed
                // only after it confirms exit and releases the occupied lease.
                return Err(io::Error::new(io::ErrorKind::TimedOut, "PSVR2 helper cleanup is still pending after 500 ms; reconnect remains blocked until cleanup finishes. Restart SRanibro if it does not finish"));
            }
            thread::sleep(Duration::from_millis(2));
        }
        if self.reader.as_ref().is_some_and(JoinHandle::is_finished) {
            if let Some(reader) = self.reader.take() {
                let _ = reader.join();
            }
        }
        self.process
            .lease
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take();
        Ok(())
    }
}
impl Drop for WorkerClient {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn cleanup_unstarted(child: &mut Child) {
    let _ = child.kill();
    let deadline = Instant::now() + STOP_BUDGET;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return,
            _ if Instant::now() >= deadline => {
                QUARANTINED.store(true, Ordering::Release);
                return;
            }
            _ => thread::sleep(Duration::from_millis(5)),
        }
    }
}

/// Opt in only from an executable that calls `worker_entrypoint` before normal
/// argument parsing, logger setup, configuration loading, and UI initialization.
pub fn enable_worker_host() {
    if let Ok(path) = std::env::current_exe() {
        let _ = HOST.set(path);
    }
}

pub fn worker_entrypoint() -> bool {
    if std::env::args_os().nth(1).as_deref() != Some(std::ffi::OsStr::new(WORKER_ARG)) {
        return false;
    }
    if std::env::args_os().count() != 2 {
        std::process::exit(2);
    }
    let result = run_worker();
    if let Err(error) = &result {
        eprintln!("[psvr2:helper] {error}");
    }
    std::process::exit(if result.is_ok() { 0 } else { 1 });
}

fn run_worker() -> io::Result<()> {
    let mut magic = [0u8; 8];
    let mut parent = [0u8; 4];
    {
        let mut input = io::stdin().lock();
        input.read_exact(&mut magic)?;
        input.read_exact(&mut parent)?;
    }
    if &magic != MAGIC {
        return Err(invalid("PSVR2 helper handshake mismatch"));
    }
    let parent = u32::from_le_bytes(parent);
    #[cfg(target_os = "linux")]
    {
        if parent <= 1 || unsafe { libc::getppid() } as u32 != parent {
            unsafe {
                libc::_exit(1);
            }
        }
        thread::Builder::new()
            .name("psvr2-parent-watch".into())
            .spawn(move || loop {
                if unsafe { libc::getppid() } as u32 != parent {
                    unsafe {
                        libc::_exit(0);
                    }
                }
                thread::sleep(Duration::from_millis(100));
            })?;
    }
    #[cfg(not(target_os = "linux"))]
    let _ = parent;
    let stop = Arc::new(AtomicBool::new(false));
    let read_stop = stop.clone();
    thread::Builder::new()
        .name("psvr2-stop-watch".into())
        .spawn(move || {
            let mut command = [0u8; 1];
            let _ = io::stdin().read_exact(&mut command);
            read_stop.store(true, Ordering::Release);
        })?;
    let mut output = io::stdout().lock();
    write_event(
        &mut output,
        &WorkerEvent::Loading {
            path: "discovering PSVR2Toolkit".into(),
        },
    )?;
    let mut capi = match Psvr2Capi::load() {
        Ok(capi) => capi,
        Err(error) => {
            write_event(
                &mut output,
                &WorkerEvent::Fatal(format!("PSVR2 CAPI discovery/load failed: {error}")),
            )?;
            return Ok(());
        }
    };
    let path = capi.module_path().to_string_lossy().into_owned();
    write_event(&mut output, &WorkerEvent::Loading { path: path.clone() })?;
    if stop.load(Ordering::Acquire) {
        return Ok(());
    }
    let code = capi.initialize();
    if code != RESULT_OK {
        let note = match code {
            RESULT_DRIVER_INACTIVE => {
                "SteamVR PSVR2Toolkit driver is inactive; start SteamVR first"
            }
            RESULT_NO_SLOT => "PSVR2Toolkit client slots are full; close unused tracking clients",
            _ => "PSVR2Toolkit initialization failed",
        };
        write_event(
            &mut output,
            &WorkerEvent::Fatal(format!("{note} (code {code})")),
        )?;
        return Ok(());
    }
    write_event(&mut output, &WorkerEvent::Initialized { path })?;
    let mut heartbeat = Instant::now();
    while !stop.load(Ordering::Acquire) {
        if Instant::now() >= heartbeat {
            // Supported Toolkit releases export this. Absence is a protocol
            // capability error, not evidence that a driver is active.
            let Some(active) = capi.driver_active() else {
                write_event(
                    &mut output,
                    &WorkerEvent::Fatal("PSVR2 CAPI has no driver activity export".into()),
                )?;
                return Ok(());
            };
            write_event(&mut output, &WorkerEvent::Driver(active))?;
            heartbeat = Instant::now() + HEARTBEAT;
        }
        if stop.load(Ordering::Acquire) {
            break;
        }
        let mut received = false;
        match capi.next_gaze(0) {
            Ok(Some(frame)) => {
                write_event(&mut output, &WorkerEvent::Gaze(frame))?;
                received = true;
            }
            Ok(None) => {}
            Err(error) => write_event(
                &mut output,
                &WorkerEvent::StreamError {
                    stream: StreamKind::Gaze,
                    message: error.to_string(),
                },
            )?,
        }
        if stop.load(Ordering::Acquire) {
            break;
        }
        match capi.next_image(0) {
            Ok(Some(frame)) => {
                write_event(&mut output, &WorkerEvent::Image(frame))?;
                received = true;
            }
            Ok(None) => {}
            Err(error) => write_event(
                &mut output,
                &WorkerEvent::StreamError {
                    stream: StreamKind::Image,
                    message: error.to_string(),
                },
            )?,
        }
        if !received {
            thread::sleep(Duration::from_millis(1));
        }
    }
    // Native deinit may hang here. The parent still owns the 500 ms kill/reap
    // deadline; the entire native heap/mappings vanish with this process.
    drop(capi);
    Ok(())
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn bounded_text(text: &str) -> &[u8] {
    let mut end = text.len().min(MAX_TEXT);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text.as_bytes()[..end]
}
fn write_event(writer: &mut impl Write, event: &WorkerEvent) -> io::Result<()> {
    let (tag, payload) = match event {
        WorkerEvent::Loading { path } => (1, bounded_text(path).to_vec()),
        WorkerEvent::Initialized { path } => (2, bounded_text(path).to_vec()),
        WorkerEvent::Driver(active) => (3, vec![u8::from(*active)]),
        WorkerEvent::Gaze(frame) => (4, encode_gaze(frame)),
        WorkerEvent::Image(frame) => {
            if frame.pixels.len() != IMAGE_PIXELS {
                return Err(invalid("PSVR2 helper image size mismatch"));
            }
            let mut bytes = Vec::with_capacity(MAX_PACKET);
            bytes.extend_from_slice(&frame.header);
            bytes.extend_from_slice(&frame.pixels);
            (5, bytes)
        }
        WorkerEvent::StreamError { stream, message } => (
            if *stream == StreamKind::Gaze { 6 } else { 7 },
            bounded_text(message).to_vec(),
        ),
        WorkerEvent::Fatal(message) => (8, bounded_text(message).to_vec()),
    };
    writer.write_all(MAGIC)?;
    writer.write_all(&[tag])?;
    writer.write_all(&(payload.len() as u32).to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()
}
fn read_event(reader: &mut impl Read) -> io::Result<WorkerEvent> {
    let mut header = [0u8; 13];
    reader.read_exact(&mut header)?;
    if &header[..8] != MAGIC {
        return Err(invalid("PSVR2 helper reply magic mismatch"));
    }
    let tag = header[8];
    let size = u32::from_le_bytes(header[9..13].try_into().unwrap()) as usize;
    let valid_size = match tag {
        1 | 2 | 6 | 7 | 8 => size <= MAX_TEXT,
        3 => size == 1,
        4 => size == 327,
        5 => size == MAX_PACKET,
        _ => false,
    };
    if !valid_size {
        return Err(invalid("PSVR2 helper reply type/length mismatch"));
    }
    let mut bytes = vec![0u8; size];
    reader.read_exact(&mut bytes)?;
    let text = |bytes: Vec<u8>| {
        String::from_utf8(bytes).map_err(|_| invalid("PSVR2 helper text is not UTF-8"))
    };
    Ok(match tag {
        1 => WorkerEvent::Loading { path: text(bytes)? },
        2 => WorkerEvent::Initialized { path: text(bytes)? },
        3 => {
            if bytes[0] > 1 {
                return Err(invalid("PSVR2 helper invalid driver flag"));
            }
            WorkerEvent::Driver(bytes[0] != 0)
        }
        4 => WorkerEvent::Gaze(decode_gaze(&bytes)?),
        5 => {
            let header: [u8; IMAGE_HEADER_BYTES] = bytes[..IMAGE_HEADER_BYTES].try_into().unwrap();
            if header[..2] != *b"VI" {
                return Err(invalid("PSVR2 helper invalid image magic"));
            }
            let total_size = u32::from_le_bytes(header[4..8].try_into().unwrap());
            if total_size != 0 && total_size < MAX_PACKET as u32 {
                return Err(invalid("PSVR2 helper image block is too small"));
            }
            WorkerEvent::Image(EyeImageFrame {
                version: u16::from_le_bytes(header[2..4].try_into().unwrap()),
                total_size,
                timestamp: u32::from_le_bytes(header[8..12].try_into().unwrap()),
                image_type: u16::from_le_bytes(header[16..18].try_into().unwrap()),
                header,
                pixels: bytes[IMAGE_HEADER_BYTES..].to_vec(),
            })
        }
        6 | 7 => WorkerEvent::StreamError {
            stream: if tag == 6 {
                StreamKind::Gaze
            } else {
                StreamKind::Image
            },
            message: text(bytes)?,
        },
        8 => WorkerEvent::Fatal(text(bytes)?),
        _ => unreachable!(),
    })
}

struct Encoder(Vec<u8>);
impl Encoder {
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
    fn vec2(&mut self, v: GazeVec2) {
        self.f32(v.x);
        self.f32(v.y);
    }
    fn vec3(&mut self, v: GazeVec3) {
        self.f32(v.x);
        self.f32(v.y);
        self.f32(v.z);
    }
    fn eye(&mut self, v: WearableEye) {
        self.u32(v.is_gaze_origin_valid);
        self.vec3(v.gaze_origin_mm);
        self.u32(v.is_gaze_dir_valid);
        self.vec3(v.gaze_dir_norm);
        self.u32(v.is_pupil_dia_valid);
        self.f32(v.pupil_dia_mm);
        self.u32(v.is_pupil_pos_in_sensor_area_valid);
        self.vec2(v.pupil_pos_in_sensor_area);
        self.u32(v.is_pos_guide_valid);
        self.vec2(v.pos_guide);
        self.u32(v.is_blink_valid);
        self.u32(v.blink);
    }
}
fn encode_gaze(v: &GazeStatus) -> Vec<u8> {
    let mut e = Encoder(Vec::with_capacity(327));
    e.0.extend_from_slice(&v.magic);
    e.0.extend_from_slice(&v.version.to_le_bytes());
    e.u32(v.size);
    e.f32(v.exp_l);
    e.f32(v.exp_r);
    e.u32(v.led_status);
    e.u32(v.exp_counter_l);
    e.u32(v.exp_counter_r);
    e.u32(v.led_counter);
    e.u32(v.dsp_return_code as u32);
    e.vec3(v.lens_config.left);
    e.vec3(v.lens_config.right);
    e.u32(v.user_calibration_id);
    e.vec3(v.fr_gaze_origin);
    e.0.extend_from_slice(&[v.enabled_eye, v.motor_sequence, v.motor_strength]);
    e.0.extend_from_slice(&v.wearable.timestamp.to_le_bytes());
    e.u32(v.wearable.frame_counter);
    e.eye(v.wearable.left);
    e.eye(v.wearable.right);
    e.u32(v.wearable.is_gaze_origin_combined_valid);
    e.vec3(v.wearable.gaze_origin_combined_mm);
    e.u32(v.wearable.is_gaze_dir_combined_valid);
    e.vec3(v.wearable.gaze_dir_combined_norm);
    e.u32(v.wearable.is_convergence_distance_valid);
    e.0.extend_from_slice(&v.foveated.timestamp.to_le_bytes());
    e.u32(v.foveated.frame_counter);
    e.u32(v.foveated.tracking_state);
    e.vec3(v.foveated.gaze_dir_left_norm);
    e.vec3(v.foveated.gaze_dir_right_norm);
    e.vec3(v.foveated.gaze_dir_combined_norm);
    e.f32(v.foveated.convergence_distance_mm);
    e.0
}
struct Decoder<'a>(&'a [u8]);
impl Decoder<'_> {
    fn take<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        if self.0.len() < N {
            return Err(invalid("truncated PSVR2 gaze fields"));
        }
        let out = self.0[..N].try_into().unwrap();
        self.0 = &self.0[N..];
        Ok(out)
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn vec2(&mut self) -> io::Result<GazeVec2> {
        Ok(GazeVec2 {
            x: self.f32()?,
            y: self.f32()?,
        })
    }
    fn vec3(&mut self) -> io::Result<GazeVec3> {
        Ok(GazeVec3 {
            x: self.f32()?,
            y: self.f32()?,
            z: self.f32()?,
        })
    }
    fn eye(&mut self) -> io::Result<WearableEye> {
        Ok(WearableEye {
            is_gaze_origin_valid: self.u32()?,
            gaze_origin_mm: self.vec3()?,
            is_gaze_dir_valid: self.u32()?,
            gaze_dir_norm: self.vec3()?,
            is_pupil_dia_valid: self.u32()?,
            pupil_dia_mm: self.f32()?,
            is_pupil_pos_in_sensor_area_valid: self.u32()?,
            pupil_pos_in_sensor_area: self.vec2()?,
            is_pos_guide_valid: self.u32()?,
            pos_guide: self.vec2()?,
            is_blink_valid: self.u32()?,
            blink: self.u32()?,
        })
    }
}
fn decode_gaze(bytes: &[u8]) -> io::Result<GazeStatus> {
    let mut d = Decoder(bytes);
    let value = GazeStatus {
        magic: d.take()?,
        version: u16::from_le_bytes(d.take()?),
        size: d.u32()?,
        exp_l: d.f32()?,
        exp_r: d.f32()?,
        led_status: d.u32()?,
        exp_counter_l: d.u32()?,
        exp_counter_r: d.u32()?,
        led_counter: d.u32()?,
        dsp_return_code: d.u32()? as i32,
        lens_config: LensConfig {
            left: d.vec3()?,
            right: d.vec3()?,
        },
        user_calibration_id: d.u32()?,
        fr_gaze_origin: d.vec3()?,
        enabled_eye: d.take::<1>()?[0],
        motor_sequence: d.take::<1>()?[0],
        motor_strength: d.take::<1>()?[0],
        wearable: WearableData {
            timestamp: i64::from_le_bytes(d.take()?),
            frame_counter: d.u32()?,
            left: d.eye()?,
            right: d.eye()?,
            is_gaze_origin_combined_valid: d.u32()?,
            gaze_origin_combined_mm: d.vec3()?,
            is_gaze_dir_combined_valid: d.u32()?,
            gaze_dir_combined_norm: d.vec3()?,
            is_convergence_distance_valid: d.u32()?,
        },
        foveated: FoveatedGaze {
            timestamp: i64::from_le_bytes(d.take()?),
            frame_counter: d.u32()?,
            tracking_state: d.u32()?,
            gaze_dir_left_norm: d.vec3()?,
            gaze_dir_right_norm: d.vec3()?,
            gaze_dir_combined_norm: d.vec3()?,
            convergence_distance_mm: d.f32()?,
        },
    };
    if !d.0.is_empty() || value.magic != *b"GS" || (value.size != 0 && value.size < 328) {
        return Err(invalid("PSVR2 helper invalid gaze layout"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn delayed_reader_completion_retains_lease_without_permanent_quarantine() {
        // Model an exited child whose pipe reader has not been scheduled yet.
        // No runtime or headset is loaded by this lifecycle test.
        let lease = Lease::acquire().unwrap();
        #[cfg(windows)]
        let job = unsafe {
            let handle = windows_sys::Win32::System::JobObjects::CreateJobObjectW(
                std::ptr::null(),
                std::ptr::null(),
            );
            assert!(!handle.is_null());
            WorkerJob(handle as isize)
        };
        let process = Arc::new(Process {
            child: Mutex::new(None),
            stopping: AtomicBool::new(false),
            reaped: AtomicBool::new(true),
            reader_done: AtomicBool::new(false),
            lease: Mutex::new(Some(lease)),
            #[cfg(windows)]
            _job: job,
        });
        let owner = process.clone();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let reader = thread::spawn(move || {
            resume_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            owner.finish_reader();
        });
        let mut client = WorkerClient {
            process,
            mailbox: Arc::new(Mailbox::default()),
            input: None,
            reader: Some(reader),
        };
        assert_eq!(client.stop().unwrap_err().kind(), io::ErrorKind::TimedOut);
        assert!(!QUARANTINED.load(Ordering::Acquire));
        assert!(
            Lease::acquire().is_err(),
            "pending reader must retain ownership"
        );
        resume_tx.send(()).unwrap();
        let until = Instant::now() + Duration::from_secs(2);
        while !client.reader.as_ref().unwrap().is_finished() {
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(2));
        }
        // Completion allows a new session even while the old client is still
        // alive. Dropping that old client must not release the new owner's gate.
        let next = Lease::acquire().unwrap();
        drop(client);
        assert!(OCCUPIED.load(Ordering::Acquire));
        drop(next);
        assert!(!OCCUPIED.load(Ordering::Acquire));
    }
    #[test]
    fn gaze_wire_preserves_marker_and_float_bits_without_padding() {
        let mut frame = GazeStatus::default();
        frame.magic = *b"GS";
        frame.size = 328;
        frame.version = 7;
        frame.wearable.timestamp = i64::MIN + 7;
        frame.wearable.frame_counter = u32::MAX;
        frame.wearable.left.pupil_dia_mm = f32::from_bits(0x7fc01234);
        frame.wearable.right.gaze_dir_norm.x = -0.0;
        frame.foveated.timestamp = i64::MAX;
        frame.motor_strength = 19;
        let encoded = encode_gaze(&frame);
        assert_eq!(encoded.len(), 327);
        let decoded = decode_gaze(&encoded).unwrap();
        assert_eq!(encode_gaze(&decoded), encoded);
        assert_eq!(decoded.wearable.left.pupil_dia_mm.to_bits(), 0x7fc01234);
        assert_eq!(
            decoded.wearable.right.gaze_dir_norm.x.to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(decoded.wearable.frame_counter, u32::MAX);
    }
    #[test]
    fn malformed_lengths_rejected_before_payload_allocation() {
        for (tag, length) in [(5, u32::MAX), (4, 328), (3, 2), (8, 4097), (99, 0)] {
            let mut bytes = MAGIC.to_vec();
            bytes.push(tag);
            bytes.extend_from_slice(&length.to_le_bytes());
            assert_eq!(
                read_event(&mut &bytes[..]).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
        }
    }
    #[test]
    fn mailbox_coalesces_frames_but_keeps_control_errors() {
        let mut pending = Pending::default();
        for marker in 0..1000 {
            let mut gaze = GazeStatus::default();
            gaze.wearable.frame_counter = marker;
            pending.push(WorkerEvent::Gaze(gaze));
        }
        pending.push(WorkerEvent::StreamError {
            stream: StreamKind::Image,
            message: "bad image".into(),
        });
        assert!(matches!(
            pending.pop(),
            Some(WorkerEvent::StreamError {
                stream: StreamKind::Image,
                ..
            })
        ));
        assert!(
            matches!(pending.pop(), Some(WorkerEvent::Gaze(g)) if g.wearable.frame_counter == 999)
        );
        assert!(pending.pop().is_none());
    }
    #[test]
    fn overflow_is_fatal_and_driver_stop_discards_cached_frames() {
        let mut pending = Pending::default();
        pending.push(WorkerEvent::Gaze(GazeStatus::default()));
        pending.push(WorkerEvent::Driver(false));
        assert!(pending.gaze.is_none());
        for _ in 0..MAX_CONTROL {
            pending.push(WorkerEvent::Driver(true));
        }
        assert!(matches!(pending.pop(), Some(WorkerEvent::Fatal(_))));
        pending.push(WorkerEvent::Gaze(GazeStatus::default()));
        assert!(pending.pop().is_none());
    }
    #[test]
    fn protocol_roundtrips_text_driver_and_exact_image_timestamp() {
        let mut header = [0u8; IMAGE_HEADER_BYTES];
        header[..2].copy_from_slice(b"VI");
        header[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        let image = EyeImageFrame {
            header,
            pixels: vec![23; IMAGE_PIXELS],
            version: 0,
            total_size: 0,
            timestamp: u32::MAX,
            image_type: 0,
        };
        let mut bytes = Vec::new();
        write_event(&mut bytes, &WorkerEvent::Image(image)).unwrap();
        assert!(
            matches!(read_event(&mut &bytes[..]).unwrap(), WorkerEvent::Image(i) if i.timestamp == u32::MAX && i.pixels.len() == IMAGE_PIXELS)
        );
        for event in [
            WorkerEvent::Loading {
                path: "日本語 path".into(),
            },
            WorkerEvent::Driver(false),
            WorkerEvent::Fatal("reason".into()),
        ] {
            let mut bytes = Vec::new();
            write_event(&mut bytes, &event).unwrap();
            assert!(read_event(&mut &bytes[..]).is_ok());
        }
    }
}
