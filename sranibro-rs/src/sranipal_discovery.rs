//! SRanipal installation discovery.
//!
//! Users should not need to remember an installer-specific directory. A valid root
//! is the folder containing `sr_runtime.exe` and the EyePrediction model expected by
//! SRanibro. Discovery is deliberately bounded and runs only when requested.

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::config::MODEL_REL;

const RUNTIME_EXE: &str = "sr_runtime.exe";
const USER_SEARCH_MAX_DEPTH: usize = 4;
const USER_SEARCH_MAX_DIRS: usize = 4_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Found {
    pub root: PathBuf,
    pub source: &'static str,
}

/// Resolve a selected folder, `sr_runtime.exe`, or nested model path to a valid
/// SRanipal root. A model is the actual requirement; `sr_runtime.exe` provides the
/// concrete landmark users can recognize in a file picker.
pub fn resolve(candidate: &Path) -> Option<PathBuf> {
    let start = if candidate.is_file() {
        candidate.parent()?
    } else {
        candidate
    };
    start
        .ancestors()
        .take(6)
        .find(|root| root.join(MODEL_REL).is_file())
        .map(Path::to_path_buf)
}

pub fn discover() -> Option<Found> {
    for (path, source) in direct_candidates() {
        if let Some(root) = resolve(&path) {
            return Some(Found { root, source });
        }
    }

    // A recursive uninstall-registry query is comparatively slow. Keep it after
    // all cheap candidates so normal installs return without paying that cost.
    for path in uninstall_registry_candidates() {
        if let Some(root) = resolve(&path) {
            return Some(Found {
                root,
                source: "installed-program registry",
            });
        }
    }

    for root in user_search_roots() {
        if let Some(found) = bounded_find(&root, USER_SEARCH_MAX_DEPTH, USER_SEARCH_MAX_DIRS) {
            return Some(Found {
                root: found,
                source: "limited user-folder search",
            });
        }
    }
    None
}

#[cfg(windows)]
fn system_executable(name: &str) -> PathBuf {
    ["SystemRoot", "WINDIR"]
        .into_iter()
        .filter_map(std::env::var_os)
        .map(PathBuf::from)
        .find(|path| path.is_absolute())
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"))
        .join("System32")
        .join(name)
}

fn direct_candidates() -> Vec<(PathBuf, &'static str)> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();
    let mut push = |path: PathBuf, source: &'static str| {
        let key = path.to_string_lossy().to_ascii_lowercase();
        if !key.is_empty() && seen.insert(key) {
            candidates.push((path, source));
        }
    };

    for name in ["SRANIPAL_DIR", "SRANIPAL_ROOT"] {
        if let Some(value) = std::env::var_os(name) {
            push(PathBuf::from(value), "environment variable");
        }
    }
    for path in running_runtime_paths() {
        push(path, "running sr_runtime.exe");
    }
    #[cfg(windows)]
    if let Ok(output) = Command::new(system_executable("where.exe"))
        .arg(RUNTIME_EXE)
        .output()
    {
        if output.status.success() {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                push(PathBuf::from(line.trim()), "PATH");
            }
        }
    }
    const RELATIVE_CANDIDATES: &[&str] = &[
        "SRanipal",
        r"VIVE\SRanipal",
        r"HTC\SRanipal",
        r"VIVE\SR_Runtime",
        r"VIVE\ViveSR",
        r"VIVE\VIVE Eye and Facial Tracking SDK\SRanipal",
        r"Steam\steamapps\common\SRanipal",
        r"Steam\steamapps\common\VIVE SRanipal",
    ];
    for env_name in [
        "ProgramFiles",
        "ProgramFiles(x86)",
        "ProgramData",
        "LOCALAPPDATA",
    ] {
        if let Some(base) = std::env::var_os(env_name) {
            let base = PathBuf::from(base);
            for relative in RELATIVE_CANDIDATES {
                push(base.join(relative), "common install location");
            }
        }
    }

    // Downloaded standalone copies are common. Check recognizable immediate
    // children before the broader bounded walk so the usual case returns quickly.
    for root in user_search_roots() {
        for name in ["SRanipal", "SR_Runtime", "VIVE SRanipal"] {
            push(root.join(name), "user-folder install");
        }
        if let Ok(entries) = std::fs::read_dir(&root) {
            for entry in entries.flatten().take(512) {
                let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                if entry.file_type().is_ok_and(|kind| kind.is_dir())
                    && (name.contains("sranipal") || name.contains("sr_runtime"))
                {
                    push(entry.path(), "user-folder install");
                }
            }
        }
    }

    candidates
}

fn user_search_roots() -> Vec<PathBuf> {
    let Some(profile) = std::env::var_os("USERPROFILE").map(PathBuf::from) else {
        return Vec::new();
    };
    [profile.join("Downloads"), profile.join("Desktop")]
        .into_iter()
        .filter(|path| path.is_dir())
        .collect()
}

fn bounded_find(root: &Path, max_depth: usize, max_dirs: usize) -> Option<PathBuf> {
    if let Some(found) = resolve(root) {
        return Some(found);
    }
    let mut queue = VecDeque::from([(root.to_path_buf(), 0usize)]);
    let mut visited = 0usize;
    while let Some((dir, depth)) = queue.pop_front() {
        if visited >= max_dirs {
            break;
        }
        visited += 1;
        if dir.join(RUNTIME_EXE).is_file() {
            if let Some(found) = resolve(&dir) {
                return Some(found);
            }
        }
        if depth >= max_depth {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                queue.push_back((entry.path(), depth + 1));
            }
        }
    }
    None
}

fn uninstall_registry_candidates() -> Vec<PathBuf> {
    #[cfg(not(windows))]
    {
        Vec::new()
    }
    #[cfg(windows)]
    {
        const KEYS: &[&str] = &[
            r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"HKLM\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
            r"HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
        ];
        let mut paths = Vec::new();
        for key in KEYS {
            let Ok(output) = Command::new(system_executable("reg.exe"))
                .args(["query", key, "/s"])
                .output()
            else {
                continue;
            };
            let text = String::from_utf8_lossy(&output.stdout);
            let mut block = Vec::new();
            for line in text.lines().chain(std::iter::once("")) {
                if line.trim().is_empty() {
                    append_registry_block(&block, &mut paths);
                    block.clear();
                } else {
                    block.push(line.to_owned());
                }
            }
        }
        paths
    }
}

#[cfg(windows)]
fn append_registry_block(lines: &[String], out: &mut Vec<PathBuf>) {
    let relevant = lines.iter().any(|line| {
        let lower = line.to_ascii_lowercase();
        lower.contains("sranipal") || lower.contains("sr_runtime")
    });
    if !relevant {
        return;
    }
    for line in lines {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if !(lower.starts_with("installlocation")
            || lower.starts_with("displayicon")
            || lower.starts_with("uninstallstring"))
        {
            continue;
        }
        let Some((_, value)) = trimmed.split_once("REG_SZ") else {
            continue;
        };
        let value = value.trim().trim_matches('"');
        if !value.is_empty() {
            out.push(PathBuf::from(value));
        }
    }
}

#[cfg(windows)]
fn running_runtime_paths() -> Vec<PathBuf> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W,
        TH32CS_SNAPPROCESS,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    let mut paths = Vec::new();
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if snapshot == INVALID_HANDLE_VALUE {
            return paths;
        }
        let mut entry: PROCESSENTRY32W = std::mem::zeroed();
        entry.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
        let mut ok = Process32FirstW(snapshot, &mut entry);
        while ok != 0 {
            let end = entry
                .szExeFile
                .iter()
                .position(|&ch| ch == 0)
                .unwrap_or(entry.szExeFile.len());
            let name = String::from_utf16_lossy(&entry.szExeFile[..end]);
            if name.eq_ignore_ascii_case(RUNTIME_EXE) {
                let process =
                    OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, entry.th32ProcessID);
                if !process.is_null() {
                    let mut buffer = vec![0u16; 32_768];
                    let mut len = buffer.len() as u32;
                    if QueryFullProcessImageNameW(process, 0, buffer.as_mut_ptr(), &mut len) != 0 {
                        paths.push(PathBuf::from(String::from_utf16_lossy(
                            &buffer[..len as usize],
                        )));
                    }
                    CloseHandle(process);
                }
            }
            ok = Process32NextW(snapshot, &mut entry);
        }
        CloseHandle(snapshot);
    }
    paths
}

#[cfg(not(windows))]
fn running_runtime_paths() -> Vec<PathBuf> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn fixture() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "sranibro_sranipal_discovery_{}_{}",
            std::process::id(),
            FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("model").join("EyePrediction")).unwrap();
        std::fs::write(root.join(RUNTIME_EXE), b"runtime").unwrap();
        std::fs::write(root.join(MODEL_REL), b"model").unwrap();
        root
    }

    #[test]
    fn resolves_root_from_runtime_and_nested_model() {
        let root = fixture();
        assert_eq!(resolve(&root.join(RUNTIME_EXE)), Some(root.clone()));
        assert_eq!(resolve(&root.join(MODEL_REL)), Some(root.clone()));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn bounded_search_finds_a_nested_install() {
        let outer =
            std::env::temp_dir().join(format!("sranibro_sranipal_outer_{}", std::process::id()));
        let root = outer.join("vendor").join("SRanipal");
        let _ = std::fs::remove_dir_all(&outer);
        std::fs::create_dir_all(root.join("model").join("EyePrediction")).unwrap();
        std::fs::write(root.join(RUNTIME_EXE), b"runtime").unwrap();
        std::fs::write(root.join(MODEL_REL), b"model").unwrap();
        assert_eq!(bounded_find(&outer, 3, 32), Some(root));
        let _ = std::fs::remove_dir_all(outer);
    }
}
