//! In-app capture of the process's OWN stdout/stderr, so the egui "Console" tab can
//! show the runtime log (`eprintln!("[xr5] …")`, `[vr4]`, `[ml]`, `[brokeneye]`, …)
//! without launching from a terminal.
//!
//! How it works (Windows): [`init`] creates an anonymous pipe, then points BOTH std
//! handles at the write end via `SetStdHandle(STD_OUTPUT_HANDLE/STD_ERROR_HANDLE, …)`.
//! Rust's `println!`/`eprintln!` fetch the std handle *per write* on Windows, so any
//! print issued AFTER `init()` flows into the pipe instead of the console. A background
//! thread `ReadFile`s the read end, splits on newlines, and pushes lines into the bounded
//! [`LOG`] ring buffer (oldest dropped past the cap). The UI renders that buffer.
//!
//! Interaction with `AttachConsole`: `main` normally re-attaches to the parent terminal
//! so CLI subcommands print there. When we redirect stdout/stderr into the pipe (UI path
//! only), those prints go to the Console tab instead of the terminal — the intended
//! behaviour. CLI subcommands (`status`, `mlcheck`, …) never call `init`, so they keep
//! printing to the console as before.
//!
//! This is additive: the `sranibro.log` file writer (`append_log` in `main`) is untouched.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

/// Max lines retained in the ring buffer. Oldest are dropped past this.
const CAP: usize = 2000;

/// Global capture buffer. Lazily created on first access so [`log_buffer`] works even
/// if [`init`] was never called (e.g. non-Windows, or a CLI path) — it just stays empty.
static LOG: OnceLock<Arc<Mutex<VecDeque<String>>>> = OnceLock::new();
static FILE_LOG: OnceLock<Option<Mutex<FileLog>>> = OnceLock::new();

const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024;

struct FileLog {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    bytes_written: u64,
    lines_since_flush: u8,
}

fn open_log_file(path: &std::path::Path) -> std::io::Result<File> {
    let mut options = std::fs::OpenOptions::new();
    options.create(true).append(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::{
            FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
        };
        options.share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE);
    }
    options.open(path)
}

impl FileLog {
    fn rotate(&mut self) {
        if let Some(mut writer) = self.writer.take() {
            let _ = writer.flush();
        }
        let old = self.path.with_extension("log.1");
        let _ = std::fs::remove_file(&old);
        let _ = std::fs::rename(&self.path, old);
        self.writer = open_log_file(&self.path)
            .ok()
            .map(|file| BufWriter::with_capacity(64 * 1024, file));
        self.bytes_written = 0;
        self.lines_since_flush = 0;
    }
}

fn file_log() -> Option<&'static Mutex<FileLog>> {
    FILE_LOG
        .get_or_init(|| {
            let path = crate::config::base_dir().join("sranibro.log");
            if std::fs::metadata(&path).is_ok_and(|m| m.len() >= MAX_LOG_BYTES) {
                let old = path.with_extension("log.1");
                let _ = std::fs::remove_file(&old);
                let _ = std::fs::rename(&path, old);
            }
            let file = open_log_file(&path).ok()?;
            let bytes_written = file.metadata().map_or(0, |m| m.len());
            Some(Mutex::new(FileLog {
                path,
                writer: Some(BufWriter::with_capacity(64 * 1024, file)),
                bytes_written,
                lines_since_flush: 0,
            }))
        })
        .as_ref()
}

/// Handle to the shared capture buffer (newest line last). Cheap to clone.
pub fn log_buffer() -> Arc<Mutex<VecDeque<String>>> {
    LOG.get_or_init(|| Arc::new(Mutex::new(VecDeque::with_capacity(CAP + 1))))
        .clone()
}

/// Push one line into the ring buffer, dropping the oldest past [`CAP`].
fn push_line(line: String) {
    // The UI redirects stderr into this pipe, so without an explicit tee all
    // adapter diagnostics disappear when the process exits. Keep the documented
    // `%APPDATA%\SRanibro\sranibro.log` useful for remote hardware bring-up.
    append_file_log(&line);
    let buf = log_buffer();
    let mut q = match buf.lock() {
        Ok(q) => q,
        Err(_) => return, // poisoned — drop the line rather than panic in the reader thread
    };
    q.push_back(line);
    while q.len() > CAP {
        q.pop_front();
    }
}

/// Best-effort file side of the UI log tee. Never writes to stderr (that would
/// recurse back into this pipe) and never lets a logging failure affect the app.
fn append_file_log(line: &str) {
    if let Some(log) = file_log() {
        let Ok(mut log) = log.lock() else { return };
        if log.bytes_written >= MAX_LOG_BYTES {
            log.rotate();
        }
        let Some(writer) = log.writer.as_mut() else {
            return;
        };
        let _ = writeln!(writer, "{line}");
        log.bytes_written = log.bytes_written.saturating_add(line.len() as u64 + 1);
        log.lines_since_flush = log.lines_since_flush.saturating_add(1);
        let important = line.contains("ERROR")
            || line.contains("failed")
            || line.contains("panic")
            || line.contains("gaze diag")
            || line.contains("[starvr] streaming")
            || line.contains("[starvr] first image callback")
            || line.contains("[starvr] first wearable callback");
        if important || log.lines_since_flush >= 32 {
            if let Some(writer) = log.writer.as_mut() {
                let _ = writer.flush();
            }
            log.lines_since_flush = 0;
        }
    }
}

/// Redirect this process's stdout+stderr into the in-memory [`LOG`] ring buffer and spawn
/// the reader thread. Call ONCE, EARLY in `main` (before the first print) on the UI path.
/// Idempotent: a second call is a no-op. Non-Windows builds are a no-op.
#[cfg(windows)]
pub fn init() {
    use std::os::windows::io::{FromRawHandle, OwnedHandle};

    use windows_sys::Win32::Foundation::{HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Console::{SetStdHandle, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE};
    use windows_sys::Win32::System::Pipes::CreatePipe;

    static STARTED: OnceLock<()> = OnceLock::new();
    if STARTED.set(()).is_err() {
        return; // already initialised
    }

    // Prime the buffer so `log_buffer()` returns the same instance the reader fills.
    let _ = log_buffer();

    unsafe {
        let mut read: HANDLE = INVALID_HANDLE_VALUE;
        let mut write: HANDLE = INVALID_HANDLE_VALUE;
        // Default pipe buffer size (0). No SECURITY_ATTRIBUTES: the handles stay in-process
        // (we don't hand them to a child), so inheritability doesn't matter here.
        if CreatePipe(&mut read, &mut write, std::ptr::null(), 0) == 0 {
            return; // pipe creation failed — leave std handles alone
        }

        // Point BOTH std handles at the write end so println!/eprintln! flow into the pipe.
        // (Rust fetches the std handle per write on Windows, so prints AFTER this redirect.)
        SetStdHandle(STD_OUTPUT_HANDLE, write);
        SetStdHandle(STD_ERROR_HANDLE, write);

        // Own the read end so it is closed if the thread ever unwinds. The write end stays
        // open for the process lifetime (it's now the std handle target), so the reader
        // never sees a clean EOF — it blocks in ReadFile until real output arrives.
        let read_owned = OwnedHandle::from_raw_handle(read as *mut _);

        std::thread::Builder::new()
            .name("logcap".into())
            .spawn(move || reader_loop(read_owned))
            .ok();
    }
}

/// Blocking read loop: pull bytes from the pipe, split into lines, push into [`LOG`].
/// Robust — a 0-byte read loops, an error exits quietly, non-UTF8 is lossily decoded.
#[cfg(windows)]
fn reader_loop(read: std::os::windows::io::OwnedHandle) {
    use std::os::windows::io::AsRawHandle;

    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::Storage::FileSystem::ReadFile;

    let handle = read.as_raw_handle() as HANDLE;
    let mut buf = [0u8; 4096];
    // Carry an incomplete trailing line across reads so a line split across two ReadFiles
    // is emitted whole (not as two fragments).
    let mut pending: Vec<u8> = Vec::new();

    loop {
        let mut n_read: u32 = 0;
        let ok = unsafe {
            ReadFile(
                handle,
                buf.as_mut_ptr(),
                buf.len() as u32,
                &mut n_read,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            return; // broken pipe / error — stop quietly
        }
        if n_read == 0 {
            continue; // no data this call — keep waiting
        }
        pending.extend_from_slice(&buf[..n_read as usize]);

        // Emit every complete `\n`-terminated line; keep the remainder in `pending`.
        loop {
            let Some(nl) = pending.iter().position(|&b| b == b'\n') else {
                break;
            };
            let mut line: Vec<u8> = pending.drain(..=nl).collect();
            line.pop(); // drop '\n'
            if line.last() == Some(&b'\r') {
                line.pop(); // drop '\r' (CRLF)
            }
            push_line(String::from_utf8_lossy(&line).into_owned());
        }
        // Guard against a pathological no-newline flood: cap the carry, flushing as a line.
        if pending.len() > 64 * 1024 {
            let line = std::mem::take(&mut pending);
            push_line(String::from_utf8_lossy(&line).into_owned());
        }
    }
}

/// Non-Windows: nothing to redirect. `log_buffer()` still works (stays empty).
#[cfg(not(windows))]
pub fn init() {}
