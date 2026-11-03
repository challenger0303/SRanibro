
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

const CAP: usize = 2000;

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

pub fn log_buffer() -> Arc<Mutex<VecDeque<String>>> {
    LOG.get_or_init(|| Arc::new(Mutex::new(VecDeque::with_capacity(CAP + 1))))
        .clone()
}

fn push_line(line: String) {
    append_file_log(&line);
    let buf = log_buffer();
    let mut q = match buf.lock() {
        Ok(q) => q,
        Err(_) => return,
    };
    q.push_back(line);
    while q.len() > CAP {
        q.pop_front();
    }
}

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
            || line.contains("[ui:")
            || line.starts_with("[ml]")
            || line.starts_with("[psvr2]")
            || line.starts_with("[wear-memory]")
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

#[cfg(windows)]
pub fn init() {
    use std::os::windows::io::{FromRawHandle, OwnedHandle};

    use windows_sys::Win32::Foundation::{HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Console::{SetStdHandle, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE};
    use windows_sys::Win32::System::Pipes::CreatePipe;

    static STARTED: OnceLock<()> = OnceLock::new();
    if STARTED.set(()).is_err() {
        return;
    }

    let _ = log_buffer();

    unsafe {
        let mut read: HANDLE = INVALID_HANDLE_VALUE;
        let mut write: HANDLE = INVALID_HANDLE_VALUE;
        if CreatePipe(&mut read, &mut write, std::ptr::null(), 0) == 0 {
            return;
        }

        SetStdHandle(STD_OUTPUT_HANDLE, write);
        SetStdHandle(STD_ERROR_HANDLE, write);

        let read_owned = OwnedHandle::from_raw_handle(read as *mut _);

        std::thread::Builder::new()
            .name("logcap".into())
            .spawn(move || reader_loop(read_owned))
            .ok();
    }
}

#[cfg(windows)]
fn reader_loop(read: std::os::windows::io::OwnedHandle) {
    use std::os::windows::io::AsRawHandle;

    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::Storage::FileSystem::ReadFile;

    let handle = read.as_raw_handle() as HANDLE;
    let mut buf = [0u8; 4096];
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
            return;
        }
        if n_read == 0 {
            continue;
        }
        pending.extend_from_slice(&buf[..n_read as usize]);

        loop {
            let Some(nl) = pending.iter().position(|&b| b == b'\n') else {
                break;
            };
            let mut line: Vec<u8> = pending.drain(..=nl).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            push_line(String::from_utf8_lossy(&line).into_owned());
        }
        if pending.len() > 64 * 1024 {
            let line = std::mem::take(&mut pending);
            push_line(String::from_utf8_lossy(&line).into_owned());
        }
    }
}

#[cfg(target_os = "linux")]
pub fn init() {
    use std::io::Read;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    static STARTED: OnceLock<()> = OnceLock::new();
    if STARTED.set(()).is_err() {
        return;
    }
    let _ = log_buffer();
    unsafe {
        let mut fds = [-1; 2];
        if libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) != 0 {
            return;
        }
        let read = OwnedFd::from_raw_fd(fds[0]);
        let write = OwnedFd::from_raw_fd(fds[1]);
        let old_stdout = libc::fcntl(libc::STDOUT_FILENO, libc::F_DUPFD_CLOEXEC, 3);
        if old_stdout < 0 {
            return;
        }
        let old_stdout = OwnedFd::from_raw_fd(old_stdout);
        if std::thread::Builder::new()
            .name("logcap".into())
            .spawn(move || {
                let mut file = File::from(read);
                let mut buf = [0u8; 4096];
                let mut pending = Vec::new();
                loop {
                    let size = match file.read(&mut buf) {
                        Ok(0) => break,
                        Ok(size) => size,
                        Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(_) => break,
                    };
                    for byte in &buf[..size] {
                        if *byte == b'\n' || pending.len() == 64 * 1024 {
                            push_line(
                                String::from_utf8_lossy(&pending)
                                    .trim_end_matches('\r')
                                    .into(),
                            );
                            pending.clear();
                        }
                        if *byte != b'\n' {
                            pending.push(*byte);
                        }
                    }
                }
                if !pending.is_empty() {
                    push_line(String::from_utf8_lossy(&pending).into());
                }
            })
            .is_err()
        {
            return;
        }
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        if libc::dup2(write.as_raw_fd(), libc::STDOUT_FILENO) < 0 {
            return;
        }
        if libc::dup2(write.as_raw_fd(), libc::STDERR_FILENO) < 0 {
            let _ = libc::dup2(old_stdout.as_raw_fd(), libc::STDOUT_FILENO);
        }
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn init() {}
