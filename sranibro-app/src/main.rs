#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use sranibro_rs::config::Config;

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

fn log_diag(msg: &str) {
    append_log(msg);
    eprintln!("{msg}");
}

fn install_diagnostics() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        append_log(&format!("[panic] {info}"));
        prev(info);
    }));
}

fn main() {
    #[cfg(all(any(windows, target_os = "linux"), feature = "external-eyelid-model"))]
    if sranibro_rs::ml::external_eyelid::worker_entrypoint() { return; }
    #[cfg(any(windows, target_os = "linux"))]
    {
        if sranibro_rs::device::psvr2_worker::worker_entrypoint() { return; }
        sranibro_rs::device::psvr2_worker::enable_worker_host();
    }
    #[cfg(any(windows, target_os = "linux"))]
    {
        if sranibro_rs::ml::gpu_worker_entrypoint() { return; }
        sranibro_rs::ml::enable_gpu_worker_host();
    }
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Console::{AttachConsole, ATTACH_PARENT_PROCESS};
        let _ = AttachConsole(ATTACH_PARENT_PROCESS);
    }

    install_diagnostics();
    match std::env::args().nth(1).as_deref() {
        #[cfg(all(target_os = "linux", feature = "xr5-only"))]
        Some("xr5-check") => {
            std::process::exit(if xr5_connection_check() { 0 } else { 1 });
        }
        Some("status") | Some("check") => {
            run_status_report();
            return;
        }
        #[cfg(not(feature = "psvr2-only"))]
        Some("custom") => {
            sranibro_rs::platform::cmd_custom();
            return;
        }
        #[cfg(not(feature = "psvr2-only"))]
        Some("restore") => {
            sranibro_rs::platform::cmd_restore();
            return;
        }
        #[cfg(not(any(feature = "psvr2-only", feature = "xr5-only")))]
        Some("starvr-direct") => {
            sranibro_rs::platform::cmd_starvr_direct();
            return;
        }
        #[cfg(not(any(feature = "psvr2-only", feature = "xr5-only")))]
        Some("starvr-service") => {
            sranibro_rs::platform::cmd_starvr_service();
            return;
        }
        #[cfg(not(feature = "psvr2-only"))]
        Some("mode") => {
            println!("{:?}", sranibro_rs::platform::detect_mode());
            return;
        }
        #[cfg(all(windows, not(any(feature = "psvr2-only", feature = "xr5-only"))))]
        Some("vpetest") => {
            sranibro_rs::device::vpe_probe::run();
            return;
        }
        #[cfg(all(windows, not(any(feature = "psvr2-only", feature = "xr5-only"))))]
        Some("vpescan") => {
            sranibro_rs::device::vpe_probe::scan();
            return;
        }
        #[cfg(all(windows, not(any(feature = "psvr2-only", feature = "xr5-only"))))]
        Some("vpewake") => {
            sranibro_rs::device::vpe_hid::run();
            return;
        }
        Some("mlcheck") => {
            run_mlcheck(std::env::args().nth(2));
            return;
        }
        Some("capture") => {
            run_capture(std::env::args().skip(2).collect());
            return;
        }
        #[cfg(feature = "psvr2-only")]
        Some(
            "custom" | "restore" | "starvr-direct" | "starvr-service" | "mode" | "vpetest"
            | "vpescan" | "vpewake" | "vr4",
        ) => {
            eprintln!("This command is unavailable in the PSVR2-only beta.");
            return;
        }
        #[cfg(feature = "xr5-only")]
        Some("starvr-direct" | "starvr-service" | "vpetest" | "vpescan" | "vpewake" | "vr4") => {
            eprintln!("This command is unavailable in the XR5-only build.");
            return;
        }
        _ => {}
    }

    #[cfg(not(any(feature = "psvr2-only", feature = "xr5-only")))]
    if std::env::args().nth(1).as_deref() == Some("vr4") {
        run_vr4();
        return;
    }
    if std::env::args().nth(1).as_deref() == Some("run") {
        run_full();
        return;
    }
    if std::env::args().nth(1).as_deref() == Some("ui") {
        run_ui_mode();
        return;
    }

    let cfg_path = sranibro_rs::config::config_path();
    let _ = Config::write_template_if_absent(&cfg_path);
    #[cfg(any(windows, target_os = "linux"))]
    run_ui_mode();
    #[cfg(not(any(windows, target_os = "linux")))]
    run_status_report();
}

#[cfg(all(target_os = "linux", feature = "xr5-only"))]
fn xr5_connection_check() -> bool {
    use sranibro_rs::{core::types::Eye, device::HmdAdapter};
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    use std::time::{Duration, Instant};
    let counts: Arc<[AtomicUsize; 4]> = Arc::new(std::array::from_fn(|_| AtomicUsize::new(0)));
    let frames = counts.clone();
    let gaze = counts.clone();
    let mut adapter = sranibro_rs::device::xr5_adapter::Xr5Adapter::new(&sranibro_rs::config::Config::default());
    let status = adapter.status_arc();
    let result = adapter.start(Box::new(move |eye, width, height, bytes| {
        if width == 200 && height == 200 && bytes.len() == 40000 {
            frames[if eye == Eye::Left { 0 } else { 1 }].fetch_add(1, Ordering::Relaxed);
        }
    }), Box::new(move |sample| {
        if sample.left.gaze_valid { gaze[2].fetch_add(1, Ordering::Relaxed); }
        if sample.right.gaze_valid { gaze[3].fetch_add(1, Ordering::Relaxed); }
    }));
    if let Err(error) = result { eprintln!("XR5 check: {error}"); return false; }
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(20) {
        if counts[0].load(Ordering::Relaxed) >= 120 && counts[1].load(Ordering::Relaxed) >= 120
            && counts[2].load(Ordering::Relaxed) >= 30 && counts[3].load(Ordering::Relaxed) >= 30 { break; }
        if status.lock().map(|s| s.contains("connection failed") || s.contains("stream error")).unwrap_or(false) { break; }
        std::thread::sleep(Duration::from_millis(50));
    }
    adapter.stop();
    let counts = counts.each_ref().map(|n| n.load(Ordering::Relaxed));
    let passed = counts.into_iter().all(|count| count > 0);
    println!("XR5_CHECK={} left_frames={} right_frames={} left_gaze={} right_gaze={} hardware_streaming_verified={}",
        if passed { "PASS" } else { "FAIL" }, counts[0], counts[1], counts[2], counts[3], passed);
    if !passed { eprintln!("XR5 status: {}", status.lock().map(|s| s.clone()).unwrap_or_default()); }
    passed
}

fn run_status_report() {
    #[cfg(feature = "psvr2-only")]
    println!("SRanibro PSVR2 Beta v{}", env!("CARGO_PKG_VERSION"));
    #[cfg(feature = "xr5-only")]
    println!("SRanibro XR5 Beta v{}", env!("CARGO_PKG_VERSION"));
    #[cfg(not(any(feature = "psvr2-only", feature = "xr5-only")))]
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

    println!("\nModels and optional tools:");
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
        #[cfg(feature = "psvr2-only")]
        println!("Ready: required model assets present. Launch SRanibro with PSVR2Toolkit active.");
        #[cfg(all(feature = "xr5-only", not(target_os = "linux")))]
        println!("Ready: XR5 research variant assets present.");
        #[cfg(all(feature = "xr5-only", target_os = "linux"))]
        println!("Ready: XR5 model assets present. Run `./sranibro-xr5 xr5-check` to check the headset connection.");
        #[cfg(all(
            not(target_os = "linux"),
            not(any(feature = "psvr2-only", feature = "xr5-only"))
        ))]
        println!(
            "Ready: required assets present. (Run `sranibro-rs vr4` to test VR4 acquisition.)"
        );
        #[cfg(all(
            target_os = "linux",
            not(any(feature = "psvr2-only", feature = "xr5-only"))
        ))]
        println!(
            "Model assets ready. Launch SteamVR with Ignition and PSVR2Toolkit before tracking."
        );
    } else {
        println!(
            "Not ready: {} required asset(s) missing — fill them into {} and re-run.",
            missing.len(),
            cfg_path.display()
        );
    }
}

#[cfg(all(windows, not(any(feature = "psvr2-only", feature = "xr5-only"))))]
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

#[cfg(all(windows, not(any(feature = "psvr2-only", feature = "xr5-only"))))]
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
    if src.pupil_pos_reported {
        if src.pupil_pos_valid {
            dst.pupil_pos = src.pupil_pos;
        }
        dst.pupil_pos_valid = src.pupil_pos_valid;
        dst.pupil_pos_reported = true;
    }
    if src.openness_reported {
        dst.openness = src.openness;
        dst.openness_valid = src.openness_valid;
        dst.openness_reported = true;
    }
}

#[cfg(all(windows, not(any(feature = "psvr2-only", feature = "xr5-only"))))]
fn run_vr4() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use sranibro_rs::core::types::{Eye, GazeSample};
    use sranibro_rs::device::vr4_adapter::Vr4Adapter;
    use sranibro_rs::device::HmdAdapter;

    println!("SRanibro VR4 acquisition test (WinUSB)");
    println!("If this hangs/fails: stop the service first ->");
    println!("  net stop \"Tobii VR4PIMAXP3B Platform Runtime\"\n");

    let gaze_n = Arc::new(AtomicU64::new(0));
    let l_img = Arc::new(AtomicU64::new(0));
    let r_img = Arc::new(AtomicU64::new(0));
    let latest = Arc::new(Mutex::new(GazeSample::default()));

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
    fn key_down(vk: i32) -> bool {
        unsafe {
            (windows_sys::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState(vk) as u16 & 0x8000)
                != 0
        }
    }
    fn wait_space() -> bool {
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
    std::thread::sleep(Duration::from_millis(500));

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
            std::thread::sleep(Duration::from_millis(15));
        }
        println!("[{lbl}] RECORDING — press SPACE to stop & advance");
        let mut n = 0u64;
        let t0 = Instant::now();
        loop {
            std::thread::sleep(Duration::from_millis(40));
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
        Ok(s) => {
            s.status().sranipal_brow_link.store(
                cfg.output.vrcft_sranipal_brow_link,
                std::sync::atomic::Ordering::Relaxed,
            );
            sinks.push(Box::new(s));
        }
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

    let brow = if net.is_some() {
        sranibro_rs::engine::load_brow_net(&cfg)
    } else {
        None
    };
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

#[cfg(target_os = "linux")]
fn run_full() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static STOP: AtomicBool = AtomicBool::new(false);
    extern "C" fn request_stop(_: i32) {
        STOP.store(true, Ordering::Relaxed);
    }
    unsafe {
        libc::signal(
            libc::SIGINT,
            request_stop as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            request_stop as *const () as libc::sighandler_t,
        );
    }
    let (cfg, _) = Config::load(&sranibro_rs::config::config_path());
    match sranibro_rs::engine::build_engine(&cfg) {
        Ok(mut engine) => {
            println!("SRanibro PSVR2 tracking active. Press Ctrl+C to exit.");
            while !STOP.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            engine.pipeline.stop();
        }
        Err(error) => {
            log_diag(&format!("engine start failed: {error}"));
            std::process::exit(1);
        }
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
fn run_full() {
    eprintln!("run mode is Windows-only.");
}

fn device_map_for(cfg: &Config, device_key: &str) -> sranibro_rs::pipeline::DeviceMap {
    let m = cfg.mapping_for(device_key);
    sranibro_rs::pipeline::DeviceMap {
        swap_eyes: m.swap_eyes,
        flip_image: m.flip_image,
        flip_gaze_x: m.flip_gaze_x,
    }
}

fn device_map() -> sranibro_rs::pipeline::DeviceMap {
    let (cfg, _) = Config::load(&sranibro_rs::config::config_path());
    let key = sranibro_rs::config::canonical_device_key(&cfg.hmd.device);
    device_map_for(&cfg, &key)
}

fn load_net() -> Option<sranibro_rs::ml::eye_net::EyeNet> {
    let (cfg, _) = Config::load(&sranibro_rs::config::config_path());
    sranibro_rs::engine::load_net(&cfg)
}

#[cfg(any(windows, target_os = "linux"))]
fn run_ui_mode() {
    sranibro_rs::logcap::init();
    let (cfg, _) = Config::load(&sranibro_rs::config::config_path());
    if let Err(e) = sranibro_rs::ui::run_ui_from_config(cfg) {
        log_diag(&format!(
            "ui error: {e} — the UI renderer could not initialize or continue; check your GPU driver and [ui:renderer]/[ui:gpu] log (no automatic renderer restart was attempted)"
        ));
        #[cfg(target_os = "linux")]
        std::process::exit(1);
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
fn run_ui_mode() {
    eprintln!("ui mode is Windows-only.");
}
