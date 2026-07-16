//! SRanibro app entry — first-run / startup view.
//!
//! Demonstrates the distribution model: load `sranibro.toml`, validate the
//! user-supplied assets (SRanipal ML weights + patched Tobii DLLs — none of
//! which we bundle), and report readiness. The device adapter + egui UI are the
//! next keystones; this entry already gives an honest "what's missing and why".
//!
//! Double-clicking the exe (no args) launches the GUI. In a *release* build we use the
//! Windows GUI subsystem so no console window pops up; the diagnostic CLI subcommands
//! (`status`, `mlcheck`, `capture`, …) still print because `main` re-attaches to the
//! parent terminal's console when launched from one. Debug builds keep the console.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use sranibro_rs::config::Config;

/// Append a diagnostic line to the app-dir log AND stderr, so failures are visible on
/// another machine even when launched by double-click (no console) under panic="abort".
/// Append a line to the app-dir log file (best-effort, NEVER panics — no eprintln here,
/// so it is safe to call from inside the panic hook without double-panicking).
fn append_log(msg: &str) {
    use std::io::Write;
    let path = sranibro_rs::config::base_dir().join("sranibro.log");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(f, "{msg}");
    }
}

/// Log a diagnostic to the file AND stderr, so failures are visible on another machine
/// even when launched by double-click (no console) under panic="abort".
fn log_diag(msg: &str) {
    append_log(msg); // file first (survives a broken-stdio eprintln)
    eprintln!("{msg}");
}

/// Route panics to the log file — the hook runs before `panic="abort"` terminates, so a
/// crash on another machine leaves a breadcrumb. The hook writes ONLY the file (no
/// eprintln) so it can't double-panic, then chains the previous hook for the stderr print.
fn install_diagnostics() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        append_log(&format!("[panic] {info}"));
        prev(info);
    }));
}

fn main() {
    // Under the windowed subsystem (release) there is no console of our own. If we were
    // launched FROM a terminal, re-attach to it so CLI subcommands' stdout/stderr are
    // visible; a no-op (no parent console) when double-clicked, so the GUI stays clean.
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Console::{AttachConsole, ATTACH_PARENT_PROCESS};
        let _ = AttachConsole(ATTACH_PARENT_PROCESS);
    }

    install_diagnostics();
    // Device-handoff subcommands: free / restore the EyeChip from the Tobii
    // Platform Runtime (self-elevating via UAC). `mode` just reports.
    match std::env::args().nth(1).as_deref() {
        // Console asset/readiness report (was the old no-arg default; the GUI is now the
        // default, so this is opt-in via `sranibro status` / `check`).
        Some("status") | Some("check") => {
            run_status_report();
            return;
        }
        Some("custom") => {
            sranibro_rs::platform::cmd_custom();
            return;
        }
        Some("restore") => {
            sranibro_rs::platform::cmd_restore();
            return;
        }
        Some("starvr-service") => {
            sranibro_rs::platform::cmd_starvr_service();
            return;
        }
        Some("mode") => {
            println!("{:?}", sranibro_rs::platform::detect_mode());
            return;
        }
        // VPE eyechip native-wake validator (diagnostic): dumps the device/interface
        // layout + runs the RE'd keepalive to test if SRanibro can hold the VPE eyechip
        // without SRanipal running in the background. Windows-only.
        #[cfg(windows)]
        Some("vpetest") => {
            sranibro_rs::device::vpe_probe::run();
            return;
        }
        // VPE read-only interface/driver map (no service changes) — run it WITH and
        // WITHOUT SRanipal to see which interface is always present.
        #[cfg(windows)]
        Some("vpescan") => {
            sranibro_rs::device::vpe_probe::scan();
            return;
        }
        // VPE eyechip wake via the HID API (the always-present 0BB4:0309 HidUsb interface).
        #[cfg(windows)]
        Some("vpewake") => {
            sranibro_rs::device::vpe_hid::run();
            return;
        }
        Some("mlcheck") => {
            // Optional label -> short labeled CSV capture (e.g. `mlcheck wide`);
            // no label -> 120s stdout stream.
            run_mlcheck(std::env::args().nth(2));
            return;
        }
        Some("capture") => {
            // Dedicated single-session acquisition: chip opened ONCE, spacebar
            // drives labeled segments. Args after `capture` override the labels.
            run_capture(std::env::args().skip(2).collect());
            return;
        }
        _ => {}
    }

    // `vr4` subcommand: live hardware acquisition test (Windows + VR4 only).
    if std::env::args().nth(1).as_deref() == Some("vr4") {
        run_vr4();
        return;
    }
    // `run` subcommand: full chain (acquisition -> ML -> SRanipalState -> sinks).
    if std::env::args().nth(1).as_deref() == Some("run") {
        run_full();
        return;
    }
    // `ui` subcommand: in-process egui dashboard over the live pipeline.
    if std::env::args().nth(1).as_deref() == Some("ui") {
        run_ui_mode();
        return;
    }

    // Default (double-click / no subcommand): launch the GUI. First-run still drops a
    // commented starter sranibro.toml so the file is discoverable next to the app.
    let cfg_path = sranibro_rs::config::config_path();
    let _ = Config::write_template_if_absent(&cfg_path);
    #[cfg(windows)]
    run_ui_mode();
    #[cfg(not(windows))]
    run_status_report();
}

/// Console asset/readiness report (the `status` / `check` subcommand). Loads the config,
/// writes a starter template on first run, and prints each asset's presence + what it
/// gates — the "what's missing and why" view, without launching the GUI.
fn run_status_report() {
    println!("SRanibro v{}", env!("CARGO_PKG_VERSION"));

    let cfg_path = sranibro_rs::config::config_path();
    match Config::write_template_if_absent(&cfg_path) {
        Ok(true) => println!(
            "[config] wrote starter {} — edit it to point at your assets",
            cfg_path.display()
        ),
        Ok(false) => {}
        Err(e) => eprintln!("[config] could not write template: {e}"),
    }

    let (cfg, warn) = Config::load(&cfg_path);
    if let Some(w) = warn {
        eprintln!("[config] parse warning: {w} (using defaults)");
    }
    println!("[config] device = {}", cfg.hmd.device);

    println!("\nAssets (you supply these — nothing proprietary is bundled):");
    for a in cfg.check_assets() {
        let mark = if a.present {
            "OK"
        } else if a.required {
            "!!"
        } else {
            "--"
        };
        let p = a
            .path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(not set)".into());
        println!("  [{mark}] {:<28} {p}", a.label);
        if !a.present {
            println!("         -> {}", a.gates);
        }
    }

    let missing = cfg.missing_required();
    println!();
    if missing.is_empty() {
        println!(
            "Ready: required assets present. (Run `sranibro-rs vr4` to test VR4 acquisition.)"
        );
    } else {
        println!(
            "Not ready: {} required asset(s) missing — fill them into {} and re-run.",
            missing.len(),
            cfg_path.display()
        );
    }
}

/// Live VR4 acquisition test: stream over WinUSB and print per-second rates.
/// Requires the platform service stopped: `net stop "Tobii VR4PIMAXP3B Platform Runtime"`.
#[cfg(windows)]
fn merge_sample(
    dst: &mut sranibro_rs::core::types::GazeSample,
    src: sranibro_rs::core::types::GazeSample,
) {
    if src.timestamp_us != 0 {
        dst.timestamp_us = src.timestamp_us;
    }
    merge_eye(&mut dst.left, src.left);
    merge_eye(&mut dst.right, src.right);
}

#[cfg(windows)]
fn merge_eye(
    dst: &mut sranibro_rs::core::types::EyeSample,
    src: sranibro_rs::core::types::EyeSample,
) {
    if src.gaze_valid {
        dst.gaze = src.gaze;
        dst.gaze_valid = true;
    }
    if src.origin_valid {
        dst.origin_mm = src.origin_mm;
        dst.origin_valid = true;
    }
    if src.pupil_valid {
        dst.pupil_mm = src.pupil_mm;
        dst.pupil_valid = true;
    }
    if src.pupil_pos_valid {
        dst.pupil_pos = src.pupil_pos;
        dst.pupil_pos_valid = true;
    }
    if src.openness_reported {
        dst.openness = src.openness;
        dst.openness_valid = src.openness_valid;
        dst.openness_reported = true;
    }
}

#[cfg(windows)]
fn run_vr4() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use sranibro_rs::core::types::{Eye, GazeSample};
    use sranibro_rs::device::vr4_adapter::Vr4Adapter;
    use sranibro_rs::device::HmdAdapter;

    println!("SRanibro VR4 acquisition test (WinUSB, DLL-free)");
    println!("If this hangs/fails: stop the service first ->");
    println!("  net stop \"Tobii VR4PIMAXP3B Platform Runtime\"\n");

    let gaze_n = Arc::new(AtomicU64::new(0));
    let l_img = Arc::new(AtomicU64::new(0));
    let r_img = Arc::new(AtomicU64::new(0));
    let latest = Arc::new(Mutex::new(GazeSample::default()));

    // Free the EyeChip from the Tobii runtime before opening it.
    sranibro_rs::platform::ensure_capture_ready();

    let (gn, ln, rn, lt) = (gaze_n.clone(), l_img.clone(), r_img.clone(), latest.clone());
    let on_frame = Box::new(move |eye: Eye, _w: u32, _h: u32, _px: &[u8]| {
        match eye {
            Eye::Left => &ln,
            Eye::Right => &rn,
        }
        .fetch_add(1, Ordering::Relaxed);
    });
    let on_gaze = Box::new(move |g: GazeSample| {
        gn.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut slot) = lt.lock() {
            merge_sample(&mut slot, g);
        }
    });

    let mut adapter = Vr4Adapter::new();
    if let Err(e) = adapter.start(on_frame, on_gaze) {
        eprintln!("start failed: {e}");
        return;
    }

    println!("Streaming ~30s...\n");
    for s in 0..30 {
        std::thread::sleep(Duration::from_secs(1));
        let g = latest.lock().map(|s| *s).unwrap_or_default();
        println!(
            "[{:2}s] gaze {:3}/s | img L {:3}/s R {:3}/s | L_gaze=({:+.3},{:+.3},{:+.3}) valid={} | pupil L {:.2}({}) R {:.2}({})",
            s + 1,
            gaze_n.swap(0, Ordering::Relaxed),
            l_img.swap(0, Ordering::Relaxed),
            r_img.swap(0, Ordering::Relaxed),
            g.left.gaze[0],
            g.left.gaze[1],
            g.left.gaze[2],
            g.left.gaze_valid,
            g.left.pupil_mm,
            g.left.pupil_valid,
            g.right.pupil_mm,
            g.right.pupil_valid,
        );
    }
    adapter.stop();
    println!("\nDone.");
}

#[cfg(not(windows))]
fn run_vr4() {
    eprintln!("vr4 mode is Windows-only.");
}

/// Horizontal mirror of a `w`x`w` grayscale image (matches the pipeline's flip).
#[cfg(windows)]
fn mirror_h(src: &[u8], w: usize) -> Vec<u8> {
    let h = src.len() / w;
    let mut out = vec![0u8; src.len()];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            out[row + x] = src[row + (w - 1 - x)];
        }
    }
    out
}

/// 5-channel ML diagnostic: stream ALL raw model outputs per eye (~10Hz) so we
/// can map which channel encodes which expression. Frees the EyeChip first, then
/// the user performs/holds expressions on cue while this records.
#[cfg(windows)]
fn run_mlcheck(label: Option<String>) {
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use sranibro_rs::core::types::{Eye, GazeSample};
    use sranibro_rs::device::vr4_adapter::Vr4Adapter;
    use sranibro_rs::device::HmdAdapter;
    use sranibro_rs::ml::preprocess;

    println!("SRanibro ML 5-channel diagnostic");
    sranibro_rs::platform::ensure_capture_ready();

    let mut net = match load_net() {
        Some(n) => n,
        None => {
            eprintln!("[mlcheck] no ML weights — set [assets].sranipal_dir in sranibro.toml");
            return;
        }
    };
    let map = device_map();
    let sz = preprocess::SRC * preprocess::SRC;

    let frames: Arc<Mutex<[Option<Vec<u8>>; 2]>> = Arc::new(Mutex::new([None, None]));
    let fc = frames.clone();
    let (swap, flip) = (map.swap_eyes, map.flip_image);
    let on_frame = Box::new(move |eye: Eye, _w: u32, _h: u32, px: &[u8]| {
        let eye = if swap { eye.opposite() } else { eye };
        let stored = if flip && px.len() >= sz {
            mirror_h(&px[..sz], preprocess::SRC)
        } else {
            px.to_vec()
        };
        if let Ok(mut f) = fc.lock() {
            f[eye.idx()] = Some(stored);
        }
    });
    let on_gaze = Box::new(|_g: GazeSample| {});

    let mut adapter = Vr4Adapter::new();
    if let Err(e) = adapter.start(on_frame, on_gaze) {
        eprintln!("[mlcheck] start failed: {e}");
        return;
    }

    // Labeled mode -> short capture to a CSV; unlabeled -> 120s stdout stream.
    let cap_secs = if label.is_some() { 8.0 } else { 120.0 };
    if let Some(lbl) = label.as_deref() {
        println!("[mlcheck] hold expression '{lbl}' — capturing {cap_secs:.0}s, starting in 3s...");
        for k in (1..=3).rev() {
            println!("  {k}...");
            std::thread::sleep(Duration::from_secs(1));
        }
        println!("[mlcheck] GO — hold '{lbl}'");
    } else {
        println!("cols: t  L[c0..c4] R[c0..c4] (per-eye dup feed)  S[c0..c4] (interleaved L+R = real SRanipal feed)");
        println!("       c0=presence c1=openness; watch which S channel rises on WIDE-open to resolve the wide source.");
    }

    let t0 = Instant::now();
    let mut n = 0u64;
    let mut rows: Vec<String> = Vec::new();
    loop {
        std::thread::sleep(Duration::from_millis(50));
        let secs = t0.elapsed().as_secs_f32();
        if secs > cap_secs {
            break;
        }
        let (l, r) = {
            let f = frames.lock().unwrap();
            (f[0].clone(), f[1].clone())
        };
        if let (Some(l), Some(r)) = (l, r) {
            if l.len() >= sz && r.len() >= sz {
                let lo = net.forward_one(&preprocess::vr4_to_input(&l));
                let ro = net.forward_one(&preprocess::vr4_to_input(&r));
                // Real SRanipal feeding: one pass with L in ch0, R in ch1.
                let so = net.forward_one(&preprocess::vr4_to_input_stereo(&l, &r));
                if let Some(lbl) = label.as_deref() {
                    rows.push(format!(
                        "{secs:.3},{lbl},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},\
                         {:.5},{:.5},{:.5},{:.5},{:.5}",
                        lo[0], lo[1], lo[2], lo[3], lo[4],
                        ro[0], ro[1], ro[2], ro[3], ro[4],
                        so[0], so[1], so[2], so[3], so[4]
                    ));
                } else {
                    println!(
                        "{secs:6.1} L {:.3} {:.3} {:.3} {:.3} {:.3}  R {:.3} {:.3} {:.3} {:.3} {:.3}  \
                         S {:.3} {:.3} {:.3} {:.3} {:.3}",
                        lo[0], lo[1], lo[2], lo[3], lo[4],
                        ro[0], ro[1], ro[2], ro[3], ro[4],
                        so[0], so[1], so[2], so[3], so[4]
                    );
                }
                n += 1;
            }
        }
    }
    adapter.stop();

    if let Some(lbl) = label.as_deref() {
        let path = format!("ml_captures/cap_{lbl}_stereo.csv");
        let _ = std::fs::create_dir_all("ml_captures");
        let mut out = String::from("ts_sec,label,L0,L1,L2,L3,L4,R0,R1,R2,R3,R4,S0,S1,S2,S3,S4\n");
        out.push_str(&rows.join("\n"));
        out.push('\n');
        match std::fs::write(&path, out) {
            Ok(()) => println!("[mlcheck] wrote {path} ({n} rows)"),
            Err(e) => eprintln!("[mlcheck] write {path} failed: {e}"),
        }
    } else {
        println!("[mlcheck] done ({n} samples)");
    }
}

#[cfg(not(windows))]
fn run_mlcheck(_label: Option<String>) {
    eprintln!("mlcheck mode is Windows-only.");
}

/// Dedicated interactive acquisition pipeline. Opens the EyeChip ONCE and keeps
/// it streaming for the whole session — re-opening per expression is unstable
/// (the WinUSB pipe goes stale: 1st open works, 2nd yields no frames, 3rd dies
/// with ReadPipe error 31). You drive timing with the SPACEBAR: SPACE starts the
/// current label's segment, SPACE again stops it and advances; ESC ends early.
/// Everything lands in one `ml_captures/cap_session_stereo.csv` with per-eye L/R
/// (duplicated feed) and interleaved S (the real SRanipal L-in-ch0 / R-in-ch1 feed).
#[cfg(windows)]
fn run_capture(labels: Vec<String>) {
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use sranibro_rs::core::types::{Eye, GazeSample};
    use sranibro_rs::device::vr4_adapter::Vr4Adapter;
    use sranibro_rs::device::HmdAdapter;
    use sranibro_rs::ml::preprocess;

    const VK_SPACE: i32 = 0x20;
    const VK_ESC: i32 = 0x1B;
    // Global key state (no console focus / message loop needed).
    fn key_down(vk: i32) -> bool {
        unsafe {
            (windows_sys::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState(vk) as u16 & 0x8000)
                != 0
        }
    }
    // Wait for a fresh SPACE press; false if ESC pressed instead.
    fn wait_space() -> bool {
        // Drain only SPACE — an ESC held/tapped at the prompt must still register
        // and end capture, so don't wait it out here.
        while key_down(VK_SPACE) {
            std::thread::sleep(Duration::from_millis(15));
        }
        loop {
            if key_down(VK_ESC) {
                return false;
            }
            if key_down(VK_SPACE) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(15));
        }
    }

    let labels: Vec<String> = if labels.is_empty() {
        ["relax", "wide", "squeeze", "closed", "wink_l", "wink_r"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        labels
    };

    println!("SRanibro interactive capture — single EyeChip session");
    sranibro_rs::platform::ensure_capture_ready();
    let mut net = match load_net() {
        Some(n) => n,
        None => {
            eprintln!("[capture] no ML weights — set [assets].sranipal_dir in sranibro.toml");
            return;
        }
    };
    let map = device_map();
    let sz = preprocess::SRC * preprocess::SRC;

    let frames: Arc<Mutex<[Option<Vec<u8>>; 2]>> = Arc::new(Mutex::new([None, None]));
    let fc = frames.clone();
    let (swap, flip) = (map.swap_eyes, map.flip_image);
    let on_frame = Box::new(move |eye: Eye, _w: u32, _h: u32, px: &[u8]| {
        let eye = if swap { eye.opposite() } else { eye };
        let stored = if flip && px.len() >= sz {
            mirror_h(&px[..sz], preprocess::SRC)
        } else {
            px.to_vec()
        };
        if let Ok(mut f) = fc.lock() {
            f[eye.idx()] = Some(stored);
        }
    });
    let on_gaze = Box::new(|_g: GazeSample| {});

    let mut adapter = Vr4Adapter::new();
    if let Err(e) = adapter.start(on_frame, on_gaze) {
        eprintln!("[capture] start failed: {e}");
        return;
    }
    std::thread::sleep(Duration::from_millis(500)); // let the stream come up

    println!("\nControls: SPACE = start/stop a segment, ESC = finish.");
    println!("Sequence: {}", labels.join(" -> "));
    let mut rows: Vec<String> = Vec::new();
    let mut aborted = false;
    for lbl in &labels {
        println!("\n[{lbl}] hold the expression, press SPACE to START (ESC to finish)");
        if !wait_space() {
            aborted = true;
            break;
        }
        while key_down(VK_SPACE) {
            std::thread::sleep(Duration::from_millis(15)); // release the START press
        }
        println!("[{lbl}] RECORDING — press SPACE to stop & advance");
        let mut n = 0u64;
        let t0 = Instant::now();
        loop {
            std::thread::sleep(Duration::from_millis(40));
            // Don't unwrap: a poisoned lock (adapter thread panic) must not crash
            // the capture before adapter.stop() runs.
            let (l, r) = match frames.lock() {
                Ok(f) => (f[0].clone(), f[1].clone()),
                Err(_) => (None, None),
            };
            if let (Some(l), Some(r)) = (l, r) {
                if l.len() >= sz && r.len() >= sz {
                    let lo = net.forward_one(&preprocess::vr4_to_input(&l));
                    let ro = net.forward_one(&preprocess::vr4_to_input(&r));
                    let so = net.forward_one(&preprocess::vr4_to_input_stereo(&l, &r));
                    rows.push(format!(
                        "{:.3},{lbl},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},\
                         {:.5},{:.5},{:.5},{:.5},{:.5}",
                        t0.elapsed().as_secs_f32(),
                        lo[0],
                        lo[1],
                        lo[2],
                        lo[3],
                        lo[4],
                        ro[0],
                        ro[1],
                        ro[2],
                        ro[3],
                        ro[4],
                        so[0],
                        so[1],
                        so[2],
                        so[3],
                        so[4]
                    ));
                    n += 1;
                }
            }
            if key_down(VK_ESC) {
                aborted = true;
                break;
            }
            if key_down(VK_SPACE) {
                while key_down(VK_SPACE) {
                    std::thread::sleep(Duration::from_millis(15));
                }
                break;
            }
        }
        println!("[{lbl}] {n} rows ({:.1}s)", t0.elapsed().as_secs_f32());
        if aborted {
            break;
        }
    }
    adapter.stop();

    let path = "ml_captures/cap_session_stereo.csv";
    let _ = std::fs::create_dir_all("ml_captures");
    let mut out = String::from("ts_sec,label,L0,L1,L2,L3,L4,R0,R1,R2,R3,R4,S0,S1,S2,S3,S4\n");
    out.push_str(&rows.join("\n"));
    out.push('\n');
    match std::fs::write(path, out) {
        Ok(()) => println!(
            "\n[capture] wrote {path} ({} rows){}",
            rows.len(),
            if aborted { " [finished early]" } else { "" }
        ),
        Err(e) => eprintln!("[capture] write {path} failed: {e}"),
    }
    println!("[capture] run `sranibro-rs restore` to give the EyeChip back to Tobii.");
}

#[cfg(not(windows))]
fn run_capture(_labels: Vec<String>) {
    eprintln!("capture mode is Windows-only.");
}

/// Full end-to-end chain on real hardware: VR4 acquisition -> ML openness ->
/// SRanipalState post-process -> debug sink. Requires the platform service
/// stopped and `[assets].sranipal_dir` set in sranibro.toml.
#[cfg(windows)]
fn run_full() {
    use std::time::Duration;

    use sranibro_rs::output::{BrokenEyeSink, DebugSink, OscSink, OutputSink};
    use sranibro_rs::pipeline::Pipeline;

    println!("SRanibro full chain (VR4 -> ML -> SRanipalState -> debug + BrokenEye)\n");

    let net = load_net();
    let (mut cfg, _) = Config::load(&sranibro_rs::config::config_path());
    sranibro_rs::platform::ensure_capture_ready();
    let adapter = match sranibro_rs::device::make_adapter(&cfg) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("adapter init failed: {e}");
            return;
        }
    };
    let status = adapter.status_arc();
    let device_key = sranibro_rs::config::running_device_key(&cfg.hmd.device, adapter.name());
    if cfg.migrate_auto_device_settings(&device_key) {
        let _ = cfg.save(&sranibro_rs::config::config_path());
    }
    let mut sinks: Vec<Box<dyn OutputSink>> = vec![Box::new(DebugSink::new(60))];
    match BrokenEyeSink::new(5555, cfg.output.vrcft_filter_samples) {
        Ok(s) => sinks.push(Box::new(s)),
        Err(e) => eprintln!("[brokeneye] could not start server: {e}"),
    }
    if cfg.output.osc {
        match OscSink::new(
            cfg.output.osc_host.clone(),
            cfg.output.osc_port,
            "/avatar/parameters",
        ) {
            Ok(s) => sinks.push(Box::new(s)),
            Err(e) => eprintln!("[osc] could not open socket: {e}"),
        }
    } else if cfg.output.eyebrow_osc {
        match OscSink::new_brow_only(
            cfg.output.osc_host.clone(),
            cfg.output.osc_port,
            "/avatar/parameters",
        ) {
            Ok(s) => sinks.push(Box::new(s)),
            Err(e) => eprintln!("[osc] could not open eyebrow-only socket: {e}"),
        }
    }

    // Brow needs the eye net's blink signal; only load it when the eye net is present.
    let brow = if net.is_some() {
        sranibro_rs::engine::load_brow_net(&cfg)
    } else {
        None
    };
    // The custom Wide model is XR5-only. Load it even while SRanipal remains the
    // selected source so the diagnostic path can compare both on the same frame.
    let wide = if device_key == "pimax_xr5" && net.is_some() {
        sranibro_rs::engine::load_wide_net(&cfg)
    } else {
        None
    };
    let (map, init) = sranibro_rs::engine::pipeline_start_settings(&cfg, &device_key);
    let mut pipeline = match Pipeline::run(
        adapter,
        net,
        brow,
        wide,
        sinks,
        map,
        status,
        device_key.clone(),
        init,
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("pipeline start failed: {e}");
            return;
        }
    };
    println!("Running ~30s (wear the headset and blink/look around)...\n");
    std::thread::sleep(Duration::from_secs(30));
    pipeline.stop();
    println!("\nDone.");
}

#[cfg(not(windows))]
fn run_full() {
    eprintln!("run mode is Windows-only.");
}

/// Per-unit eye-image mapping from config (swap L/R, mirror image).
fn device_map_for(cfg: &Config, device_key: &str) -> sranibro_rs::pipeline::DeviceMap {
    let m = cfg.mapping_for(device_key);
    sranibro_rs::pipeline::DeviceMap {
        swap_eyes: m.swap_eyes,
        flip_image: m.flip_image,
        flip_gaze_x: m.flip_gaze_x,
    }
}

/// Mapping helper for the fixed-VR4 diagnostic commands, which do not run the adapter
/// auto-selector used by the product pipeline.
fn device_map() -> sranibro_rs::pipeline::DeviceMap {
    let (cfg, _) = Config::load(&sranibro_rs::config::config_path());
    let key = sranibro_rs::config::canonical_device_key(&cfg.hmd.device);
    device_map_for(&cfg, &key)
}

/// Load the EyePrediction net from the configured weights path (None if absent).
/// Thin wrapper over [`sranibro_rs::engine::load_net`] so both entrypoints and the
/// UI build the net identically.
fn load_net() -> Option<sranibro_rs::ml::eye_net::EyeNet> {
    let (cfg, _) = Config::load(&sranibro_rs::config::config_path());
    sranibro_rs::engine::load_net(&cfg)
}

/// Launch the egui dashboard over a live pipeline (rebuilt in-process when the
/// user edits asset paths in Settings — see `ui::App::apply_and_reload`).
#[cfg(windows)]
fn run_ui_mode() {
    // Redirect our own stdout/stderr into the in-app Console tab's ring buffer BEFORE the
    // engine/UI print anything. Only on the UI path, so CLI subcommands still print to the
    // terminal. Trade-off: a terminal launch of the UI now shows its logs in-app, not in
    // the console (AttachConsole'd earlier) — that is the intended behaviour.
    sranibro_rs::logcap::init();
    let (cfg, _) = Config::load(&sranibro_rs::config::config_path());
    if cfg.ui.steamvr_overlay {
        eprintln!("[ui] steamvr_overlay=true, but the in-headset SteamVR overlay isn't ported to the Rust build yet — desktop dashboard only.");
    }
    let engine = match sranibro_rs::engine::build_engine(&cfg) {
        Ok(e) => e,
        Err(e) => {
            log_diag(&format!("engine start failed: {e}"));
            return;
        }
    };
    if let Err(e) = sranibro_rs::ui::run_ui(engine.pipeline, engine.be_status) {
        // Most likely a graphics-init failure (no usable GPU adapter / driver) — log it
        // so a blank-window launch on another machine is diagnosable, not a silent flash.
        log_diag(&format!(
            "ui error: {e} — graphics init may have failed (update your GPU driver; a DX12/Vulkan-capable GPU is required)"
        ));
    }
}

#[cfg(not(windows))]
fn run_ui_mode() {
    eprintln!("ui mode is Windows-only.");
}
