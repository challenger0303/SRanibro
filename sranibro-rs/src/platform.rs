//! Windows platform control — free the EyeChip from the Tobii Platform Runtime
//! so SRanibro's DLL-free WinUSB reader can claim it ("CUSTOM BRIDGE" mode,
//! mirroring `tobii_launcher.py`).
//!
//! Pimax auto-starts the `Tobii VR4PIMAXP3B Platform Runtime` service, which
//! holds the EyeChip and is *auto-revived* whenever a client connects over
//! `tobii-ttp://`. To capture the device DLL-free we DISABLE + stop the Tobii
//! services and kill the platform-runtime exe (and Broken Eye, which also holds
//! TCP 5555). It is a HOT switch — the Pimax HMD / SteamVR session stays up.
//!
//! It deliberately NEVER touches `sr_runtime.exe` (the Vive facial / lip tracker)
//! or `VRCFaceTracking.exe` (our own downstream consumer).

/// Whether the Tobii runtime currently holds the EyeChip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TobiiMode {
    /// Tobii Platform Runtime stopped/disabled — the EyeChip is free for us.
    Custom,
    /// Tobii Platform Runtime running — it owns the device (default Pimax state).
    BrokenEye,
    /// No Tobii services found (non-Pimax host, or nothing installed).
    Unknown,
}

#[cfg(windows)]
mod sys {
    use std::io;
    use std::process::Command;
    use std::time::Duration;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::Security::{GetTokenInformation, TokenElevation, TOKEN_ELEVATION};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    const TOKEN_QUERY: u32 = 0x0008;

    /// True if this process token is elevated (admin).
    pub fn is_elevated() -> bool {
        unsafe {
            let mut token: HANDLE = std::ptr::null_mut();
            if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
                return false;
            }
            let mut elevation = TOKEN_ELEVATION { TokenIsElevated: 0 };
            let mut ret_len = 0u32;
            let ok = GetTokenInformation(
                token,
                TokenElevation,
                &mut elevation as *mut _ as *mut core::ffi::c_void,
                core::mem::size_of::<TOKEN_ELEVATION>() as u32,
                &mut ret_len,
            );
            CloseHandle(token);
            ok != 0 && elevation.TokenIsElevated != 0
        }
    }

    /// Re-launch this exe elevated (UAC) with `args`, block until it exits,
    /// return its exit code. Uses PowerShell's `Start-Process -Verb RunAs -Wait`
    /// (no Shell FFI). A declined UAC prompt surfaces as a non-zero code.
    pub fn run_elevated_and_wait(args: &[&str]) -> io::Result<i32> {
        let exe = std::env::current_exe()?;
        let exe_s = exe.to_string_lossy().replace('\'', "''");
        let ps = if args.is_empty() {
            format!("Start-Process -FilePath '{exe_s}' -Verb RunAs -Wait")
        } else {
            let list = args
                .iter()
                .map(|a| format!("'{}'", a.replace('\'', "''")))
                .collect::<Vec<_>>()
                .join(",");
            format!("Start-Process -FilePath '{exe_s}' -ArgumentList {list} -Verb RunAs -Wait")
        };
        Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &ps])
            .status()
            .map(|s| s.code().unwrap_or(-1))
    }

    fn run(cmd: &str, args: &[&str]) {
        match Command::new(cmd).args(args).output() {
            Ok(o) => {
                let txt = format!(
                    "{}{}",
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr)
                );
                let line = txt
                    .lines()
                    .map(str::trim)
                    .find(|l| !l.is_empty())
                    .unwrap_or("");
                eprintln!("[platform] $ {cmd} {} -> {line}", args.join(" "));
            }
            Err(e) => eprintln!("[platform] $ {cmd} {} -> ERROR {e}", args.join(" ")),
        }
    }

    fn sc_output(args: &[&str]) -> String {
        Command::new("sc.exe")
            .args(args)
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default()
    }

    /// All installed service names beginning with "Tobii".
    pub fn list_tobii_services() -> Vec<String> {
        sc_output(&["query", "type=", "service", "state=", "all"])
            .lines()
            .filter_map(|l| {
                l.trim()
                    .strip_prefix("SERVICE_NAME:")
                    .map(|n| n.trim().to_string())
            })
            .filter(|n| n.to_lowercase().starts_with("tobii"))
            .collect()
    }

    /// Numeric code from an `sc` field line (STATE / START_TYPE). LOCALE-INDEPENDENT:
    /// the localized word (RUNNING/STOPPED/DISABLED) is translated on non-English
    /// Windows, but the numeric code is stable. Finds the line by the field label, then
    /// reads the first integer after the colon. (STATE 1=stopped/4=running;
    /// START_TYPE 4=disabled.)
    fn sc_field_code(out: &str, field: &str) -> Option<u32> {
        out.lines()
            .find(|l| l.to_uppercase().contains(field))
            .and_then(|l| l.split(':').nth(1))
            .and_then(|rhs| rhs.split_whitespace().next())
            .and_then(|tok| tok.parse::<u32>().ok())
    }

    fn service_running(name: &str) -> bool {
        sc_field_code(&sc_output(&["query", name]), "STATE") == Some(4)
    }

    /// True if the service state is explicitly STOPPED (not STOP_PENDING/PAUSED, which
    /// still hold the device).
    fn service_stopped(name: &str) -> bool {
        sc_field_code(&sc_output(&["query", name]), "STATE") == Some(1)
    }

    /// True if the service's start type is DISABLED (won't auto-start / revive).
    fn service_start_disabled(name: &str) -> bool {
        sc_field_code(&sc_output(&["qc", name]), "START_TYPE") == Some(4)
    }

    /// Does this Tobii service KEY denote a "platform runtime"? Normalizes away
    /// spaces/underscores/case so it matches "Platform Runtime", "platform_runtime",
    /// and "PlatformRuntime" keys (the old space-literal substring missed several,
    /// e.g. underscore/XR5 keys).
    fn is_platform_runtime(name: &str) -> bool {
        let norm: String = name
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        norm.contains("platformruntime")
    }

    /// True if a Tobii "*Platform Runtime*" service is currently RUNNING (= it
    /// owns the EyeChip and we must switch before capture).
    pub fn platform_runtime_blocking() -> bool {
        list_tobii_services()
            .into_iter()
            .any(|n| is_platform_runtime(&n) && service_running(&n))
    }

    /// True if a Tobii "*Platform Runtime*" service exists AND could still grab the
    /// EyeChip — i.e. it is NOT both DISABLED and fully STOPPED. A plain stop is NOT
    /// enough because the runtime AUTO-REVIVES on the next tobii-ttp client; only
    /// DISABLING + STOPPING makes the device safely ours. Gating the handoff on this
    /// closes the revive race (running / stopped-but-enabled / stop-pending all need
    /// it) while still skipping the UAC prompt when already disabled+stopped (inert).
    pub fn platform_runtime_needs_handoff() -> bool {
        list_tobii_services()
            .into_iter()
            .filter(|n| is_platform_runtime(&n))
            .any(|n| !(service_start_disabled(&n) && service_stopped(&n)))
    }

    /// DISABLE + stop every Tobii service, then force-kill the platform-runtime
    /// exe(s) and Broken Eye. Requires elevation. Never touches sr_runtime or
    /// VRCFaceTracking.
    pub fn switch_to_custom() {
        let svcs = list_tobii_services();
        eprintln!("[platform] CUSTOM: disabling + stopping Tobii service(s): {svcs:?}");
        for s in &svcs {
            run("sc.exe", &["config", s, "start=", "disabled"]);
            run("sc.exe", &["stop", s]);
        }
        eprintln!(
            "[platform] killing platform runtime + Broken Eye (sr_runtime / VRCFT left alone)..."
        );
        for exe in [
            "platform_runtime_VR4PIMAXP3B_service.exe",
            "platform_runtime_XR5EYECHIP_WIN10_x64.exe",
            "Broken Eye.exe",
        ] {
            run("taskkill.exe", &["/F", "/IM", exe]);
        }
        std::thread::sleep(Duration::from_millis(600));
    }

    /// Re-enable (demand) + start the Tobii services — back to BROKENEYE mode.
    pub fn restore_brokeneye() {
        let svcs = list_tobii_services();
        eprintln!("[platform] RESTORE: re-enabling + starting Tobii service(s): {svcs:?}");
        for s in &svcs {
            run("sc.exe", &["config", s, "start=", "demand"]);
            run("sc.exe", &["start", s]);
        }
    }

    /// Whether StarVR's generic Tobii broker is installed but not running.
    pub fn starvr_service_needs_start() -> bool {
        list_tobii_services()
            .into_iter()
            .find(|name| name.eq_ignore_ascii_case("Tobii Service"))
            .is_some_and(|name| !service_running(&name))
    }

    /// Re-enable/start only the generic StarVR Tobii broker. Do not wake any
    /// Pimax platform-runtime service, which would contend for a Pimax EyeChip.
    pub fn restore_starvr_service() {
        let Some(name) = list_tobii_services()
            .into_iter()
            .find(|name| name.eq_ignore_ascii_case("Tobii Service"))
        else {
            eprintln!("[platform] StarVR: generic Tobii Service is not installed");
            return;
        };
        eprintln!("[platform] StarVR: re-enabling + starting {name}");
        run("sc.exe", &["config", &name, "start=", "demand"]);
        run("sc.exe", &["start", &name]);
    }
}

// --- public API (thin cfg shims over `sys`) ---

/// Report whether the Tobii runtime is holding the EyeChip.
#[cfg(windows)]
pub fn detect_mode() -> TobiiMode {
    if sys::list_tobii_services().is_empty() {
        TobiiMode::Unknown
    } else if sys::platform_runtime_blocking() {
        TobiiMode::BrokenEye
    } else {
        TobiiMode::Custom
    }
}

#[cfg(windows)]
pub fn is_elevated() -> bool {
    sys::is_elevated()
}

/// Pre-flight before opening the EyeChip: if the Tobii Platform Runtime owns the
/// device, switch to CUSTOM — self-elevating via UAC if we aren't admin.
#[cfg(windows)]
pub fn ensure_capture_ready() {
    if !sys::platform_runtime_needs_handoff() {
        return; // already disabled + stopped -> inert, won't revive
    }
    eprintln!(
        "[platform] Tobii Platform Runtime present (running or revivable) -> switching to CUSTOM"
    );
    if sys::is_elevated() {
        sys::switch_to_custom();
    } else {
        match sys::run_elevated_and_wait(&["custom"]) {
            Ok(code) => eprintln!("[platform] elevated CUSTOM switch finished (exit {code})"),
            Err(e) => eprintln!("[platform] UAC elevation failed: {e} — EyeChip may stay busy"),
        }
    }
}

/// `custom` subcommand — free the EyeChip (self-elevates if needed).
#[cfg(windows)]
pub fn cmd_custom() {
    if sys::is_elevated() {
        sys::switch_to_custom();
        println!("CUSTOM BRIDGE: EyeChip freed (Tobii runtime disabled + stopped).");
    } else {
        eprintln!("[platform] elevating (UAC) to switch to CUSTOM...");
        match sys::run_elevated_and_wait(&["custom"]) {
            Ok(code) => std::process::exit(code),
            Err(e) => eprintln!("[platform] elevation failed: {e}"),
        }
    }
}

/// `restore` subcommand — hand the EyeChip back to the Tobii runtime (BROKENEYE).
#[cfg(windows)]
pub fn cmd_restore() {
    if sys::is_elevated() {
        sys::restore_brokeneye();
        println!("BROKENEYE: Tobii runtime re-enabled + started.");
    } else {
        eprintln!("[platform] elevating (UAC) to restore BROKENEYE...");
        match sys::run_elevated_and_wait(&["restore"]) {
            Ok(code) => std::process::exit(code),
            Err(e) => eprintln!("[platform] elevation failed: {e}"),
        }
    }
}

/// Ensure the generic Tobii broker required by StarVR is running. Pimax CUSTOM
/// mode disables all Tobii services, so switching HMDs must repair this one.
#[cfg(windows)]
pub fn ensure_starvr_ready() {
    if !sys::starvr_service_needs_start() {
        return;
    }
    eprintln!("[platform] StarVR requires the stopped/disabled generic Tobii Service");
    if sys::is_elevated() {
        sys::restore_starvr_service();
    } else {
        match sys::run_elevated_and_wait(&["starvr-service"]) {
            Ok(code) => {
                eprintln!("[platform] elevated StarVR service restore finished (exit {code})")
            }
            Err(e) => eprintln!("[platform] UAC elevation failed: {e}"),
        }
    }
}

/// Elevated helper target used by [`ensure_starvr_ready`].
#[cfg(windows)]
pub fn cmd_starvr_service() {
    if sys::is_elevated() {
        sys::restore_starvr_service();
    } else {
        eprintln!("[platform] starvr-service requires elevation");
    }
}

#[cfg(not(windows))]
pub fn detect_mode() -> TobiiMode {
    TobiiMode::Unknown
}
#[cfg(not(windows))]
pub fn is_elevated() -> bool {
    false
}
#[cfg(not(windows))]
pub fn ensure_capture_ready() {}
#[cfg(not(windows))]
pub fn cmd_custom() {
    eprintln!("`custom` mode is Windows-only.");
}
#[cfg(not(windows))]
pub fn cmd_restore() {
    eprintln!("`restore` mode is Windows-only.");
}
#[cfg(not(windows))]
pub fn ensure_starvr_ready() {}
#[cfg(not(windows))]
pub fn cmd_starvr_service() {
    eprintln!("`starvr-service` mode is Windows-only.");
}
