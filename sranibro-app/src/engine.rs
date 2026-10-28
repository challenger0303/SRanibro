//! Engine assembly: build the full acquisition -> ML -> post-process -> sinks
//! pipeline from a [`Config`]. Extracted so both the binary entrypoint and the
//! UI's live "Apply & reload" build the engine identically — the UI rebuilds it
//! in-process when the user edits asset paths, with no app restart.

use std::sync::Arc;

use crate::config::{Config, WideSource};
use crate::ml::brow_net::BrowNet;
use crate::ml::eye_net::EyeNet;
use crate::ml::tvm_params;
use crate::ml::wide_net::WideNet;
use crate::output::BrokenEyeStatus;
use crate::pipeline::{DeviceMap, EyelidModelIdentity, Pipeline, PipelineInit};

/// A running engine plus the handles the UI needs to render/swap it.
pub struct Engine {
    pub pipeline: Pipeline,
    pub be_status: Option<Arc<BrokenEyeStatus>>,
    /// Non-fatal startup information surfaced by the UI (for example, that the
    /// alternate StarVR connection route recovered the device).
    pub startup_notice: Option<String>,
    /// The StarVR route actually used, so an in-process reload stays in sync.
    pub starvr_direct_mode: Option<bool>,
}

/// Resolve every live value needed before [`Pipeline::run`] starts its threads. The
/// geometry's tri-state mirror wins when explicitly present; otherwise legacy mapping
/// values remain compatible for non-XR5 configs.
pub fn pipeline_start_settings(cfg: &Config, device_key: &str) -> (DeviceMap, PipelineInit) {
    let mapping = cfg.mapping_for(device_key);
    let geometry = cfg.geometry_for(device_key);
    let is_xr5 = crate::config::canonical_device_key(device_key) == "pimax_xr5";
    let is_vr4 = crate::config::canonical_device_key(device_key) == "pimax_vr4";
    let ml_mirror = [
        geometry[0].mirror_h.unwrap_or(mapping.ml_mirror_l),
        geometry[1].mirror_h.unwrap_or(mapping.ml_mirror_r),
    ];
    let mut eyelid_response_profile = cfg.eyelid_response_profile_for(device_key);
    // A profile created before this feature stored closing time in global tuning.
    // Migrate it only while this HMD has no explicit response profile yet.
    if !cfg.has_eyelid_response_profile(device_key) && cfg.tuning.blink_close_ms.is_finite() {
        eyelid_response_profile.blink_close_ms = cfg.tuning.blink_close_ms.clamp(0.0, 160.0);
    }

    (
        DeviceMap {
            swap_eyes: mapping.swap_eyes,
            flip_image: mapping.flip_image,
            flip_gaze_x: mapping.flip_gaze_x,
        },
        PipelineInit {
            inference_enabled: cfg.ui.inference_enabled,
            #[cfg(feature = "external-eyelid-model")]
            xr5_gaze_package: (is_xr5 && cfg.ui.inference_enabled && cfg.hmd.xr5_model_gaze)
                .then(|| cfg.hmd.xr5_gaze_package.clone()),
            #[cfg(feature = "external-eyelid-model")]
            xr5_gaze_hybrid: is_xr5 && cfg.ui.inference_enabled && cfg.hmd.xr5_model_gaze && cfg.hmd.xr5_gaze_hybrid,
            #[cfg(any(feature = "xr5-research-model", feature = "external-eyelid-model"))]
            xr5_research: (is_xr5 && cfg.ui.inference_enabled).then(|| {
                let mut settings = cfg.xr5_research.clone();
                settings.gpu = cfg.ui.eyelid_inference_backend.resolved() != crate::config::EyelidInferenceBackend::Cpu;
                settings
            }),
            raw_gaze_osc_port: cfg.output.raw_gaze_osc.then_some(cfg.output.raw_gaze_osc_port),
            gaze_smoothing: !is_vr4 || cfg.hmd.vr4_gaze_smoothing,
            raw_gaze_osc_smoothing: is_vr4 && cfg.hmd.vr4_gaze_smoothing,
            eyebrow_enabled: cfg.ui.eyebrow_enabled,
            eye_wide_enabled: cfg.ui.eye_wide_enabled,
            right_eye_left_head: cfg.right_eye_left_head_for(device_key),
            eyelid_inference_backend: cfg.ui.eyelid_inference_backend.resolved(),
            ml_mirror,
            tuning: cfg.tuning,
            geometry,
            despeckle: cfg.despeckle_for(device_key),
            flatten: cfg.flatten_for(device_key),
            brightness: cfg.brightness_for(device_key),
            photometric_correction: if crate::config::supports_photometric_fit(device_key) {
                cfg.photometric_correction_for(device_key)
            } else {
                crate::core::types::PhotometricCorrection::default()
            },
            gaze_correction: if cfg!(feature = "external-eyelid-model") && is_xr5 && cfg.ui.inference_enabled && cfg.hmd.xr5_model_gaze {
                if cfg.hmd.xr5_gaze_hybrid { cfg.hmd.xr5_hybrid_gaze_correction }
                else { cfg.hmd.xr5_model_gaze_correction }
            } else if crate::config::supports_gaze_correction(device_key) {
                cfg.gaze_correction_for(device_key)
            } else {
                crate::config::GazeCorrection::default()
            },
            gaze_eyelid_profile: cfg.gaze_eyelid_profile_for(device_key),
            wink_profile: cfg.wink_profile_for(device_key),
            blink_timing_profile: cfg.blink_timing_profile_for(device_key),
            eyelid_response_profile,
            wide_enabled: if is_xr5 && !native_xr5_selected(cfg, device_key) {
                cfg.dream_air_profile_for(device_key)
                    .map(|profile| profile.wide_supported)
                    .unwrap_or([true; 2])
            } else {
                [true; 2]
            },
            wide_source: cfg.wide_source_for(device_key),
            eyelid_model_identity: None,
        },
    )
}

fn native_xr5_selected(cfg: &Config, device_key: &str) -> bool {
    #[cfg(any(feature = "xr5-research-model", feature = "external-eyelid-model"))]
    { cfg.ui.inference_enabled
        && crate::config::canonical_device_key(device_key) == "pimax_xr5"
        && cfg.xr5_research.enabled && cfg.xr5_research.avatar }
    #[cfg(not(any(feature = "xr5-research-model", feature = "external-eyelid-model")))]
    { let _ = (cfg, device_key); false }
}

/// XR5 snapshots its gaze provider when the adapter is constructed. Legacy `auto`
/// buckets are migrated only after auto-detection reveals the real device key, so a
/// migrated provider change must be reflected by constructing the still-unstarted
/// adapter again from the effective config.
fn xr5_adapter_needs_rebuild_after_migration(
    before: &Config,
    after: &Config,
    device_key: &str,
) -> bool {
    crate::config::canonical_device_key(device_key) == "pimax_xr5"
        && before.gaze_source_for(device_key) != after.gaze_source_for(device_key)
}

/// Load the EyePrediction net from the configured weights path (direct `ml_model`
/// file, else `<sranipal_dir>/MODEL_REL`). Returns `None` — and prints a precise
/// reason — when absent or invalid, so the engine simply runs gaze-only.
pub fn load_net(cfg: &Config) -> Option<EyeNet> {
    load_net_identified(cfg).0
}

/// Read once, fingerprint those exact bytes, then parse the same buffer. This makes the
/// runtime identity stronger than hashing a path later after its contents may have changed.
fn load_net_identified(cfg: &Config) -> (Option<EyeNet>, Option<EyelidModelIdentity>) {
    if !cfg.ui.inference_enabled { return (None, None); }
    match cfg.ml_params_path() {
        Some(p) if p.is_file() => match std::fs::read(&p) {
            Ok(bytes) => {
                let identity = EyelidModelIdentity {
                    crc32: crate::diagnostics::crc32_fingerprint(&bytes),
                    bytes: bytes.len() as u64,
                };
                match tvm_params::parse_map_bytes(&bytes) {
                    Ok(map) => match EyeNet::new(map) {
                        Ok(n) => {
                            println!("[ml] loaded weights from {}", p.display());
                            (Some(n), Some(identity))
                        }
                        Err(e) => {
                            eprintln!("[ml] model invalid: {e} (running without ML)");
                            (None, None)
                        }
                    },
                    Err(e) => {
                        eprintln!("[ml] parse failed: {e} (running without ML)");
                        (None, None)
                    }
                }
            }
            Err(e) => {
                eprintln!("[ml] read failed: {e} (running without ML)");
                (None, None)
            }
        },
        Some(p) => {
            eprintln!(
                "[ml] weights not found at {} (running gaze-only)",
                p.display()
            );
            (None, None)
        }
        None => {
            eprintln!("[ml] no SRanipal weights configured (running gaze-only)");
            (None, None)
        }
    }
}

/// Load the optional eyebrow model (explicit override or packaged `eyebrow.bin`).
/// `None` (with a printed reason) when absent/invalid, so the engine simply runs
/// without brow output.
pub fn load_brow_net(cfg: &Config) -> Option<BrowNet> {
    if !cfg.ui.inference_enabled { return None; }
    let p = cfg.brow_model_path()?;
    if !p.is_file() {
        eprintln!("[brow] model not found at {} (no brow output)", p.display());
        return None;
    }
    match BrowNet::load(&p) {
        Ok(n) => {
            println!(
                "[brow] loaded eyebrow model from {} (out_dim={})",
                p.display(),
                n.out_dim()
            );
            Some(n)
        }
        Err(e) => {
            eprintln!("[brow] model invalid: {e} (no brow output)");
            None
        }
    }
}

/// Load the optional task-tagged Dream Air/XR5 EyeWide model. It is still loaded while
/// SRanipal is selected so same-frame A/B telemetry is available before cutover.
pub fn load_wide_net(cfg: &Config) -> Option<WideNet> {
    if !cfg.ui.inference_enabled { return None; }
    let path = cfg.wide_model_path()?;
    if !path.is_file() {
        eprintln!("[wide] model not found at {}", path.display());
        return None;
    }
    match WideNet::load(&path) {
        Ok(net) => {
            println!(
                "[wide] loaded custom XR5 EyeWide model from {}",
                path.display()
            );
            Some(net)
        }
        Err(e) => {
            eprintln!("[wide] model invalid: {e}");
            None
        }
    }
}

/// Resolve a persisted EyeWide preference to a runtime-safe provider. A missing or
/// invalid optional model must never prevent the main app and camera from starting.
fn runtime_wide_source(
    requested: WideSource,
    is_xr5: bool,
    eyelid_model_loaded: bool,
    wide_model_loaded: bool,
) -> WideSource {
    if is_xr5 && eyelid_model_loaded && wide_model_loaded && requested != WideSource::Sranipal {
        requested
    } else {
        WideSource::Sranipal
    }
}

#[cfg(windows)]
fn resolved_config_device_key(cfg: &Config) -> String {
    let configured = crate::config::canonical_device_key(&cfg.hmd.device);
    if configured != "auto" {
        return configured;
    }
    let serial = crate::device::usb::peek_serial();
    crate::device::auto_device_key_from_serial(serial.as_deref()).into()
}

#[cfg(target_os = "linux")]
fn resolved_config_device_key(cfg: &Config) -> String {
    let configured = crate::config::canonical_device_key(&cfg.hmd.device);
    if crate::config::PSVR2_ONLY_BUILD || configured == "auto" {
        "psvr2".into()
    } else {
        configured
    }
}

/// Route fallback is intentionally limited to errors that can plausibly be caused
/// by the Tobii broker owning (or not owning) the tracker. DLL/path/export errors
/// cannot be repaired by another UAC service switch.
#[cfg(windows)]
fn is_starvr_route_error(error: &std::io::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    [
        "tobii_device_create",
        "device enumeration",
        "no starvr device",
        "connection_failed",
        "stream readiness timed out",
    ]
    .iter()
    .any(|marker| message.contains(marker))
}

#[cfg(windows)]
fn persist_starvr_route(direct: bool) {
    let path = crate::config::config_path();
    let (mut persisted, warning) = Config::load(&path);
    if let Some(warning) = warning {
        eprintln!("[starvr] could not safely persist connection route: {warning}");
        return;
    }
    persisted.hmd.starvr_direct = direct;
    if let Err(error) = persisted.save(&path) {
        eprintln!("[starvr] connected, but could not save the working route: {error}");
    }
}

/// Build + start the engine for `cfg`: free the EyeChip (pre-flight), open the
/// configured adapter, wire the ML net and the VRCFT/OSC sinks, and run the
/// pipeline. The caller owns the returned [`Engine`] (call `pipeline.stop()` /
/// drop to tear it down). StarVR automatically retries the opposite service/direct
/// route for connection-level failures and remembers the route that succeeds.
#[cfg(any(windows, target_os = "linux"))]
pub fn build_engine(cfg: &Config) -> std::io::Result<Engine> {
    build_engine_with_gpu(cfg, None)
}

/// GUI engine builder. The legacy context parameter is retained for call-site
/// compatibility; EyeNet runs in its isolated worker, never on the UI device.
#[cfg(windows)]
pub(crate) fn build_engine_with_gpu(
    cfg: &Config,
    gpu_context: Option<crate::ml::eyelid_model::EyelidGpuContext>,
) -> std::io::Result<Engine> {
    let is_starvr = resolved_config_device_key(cfg) == "starvr";
    match build_engine_once(cfg, gpu_context.clone()) {
        Ok(engine) => Ok(engine),
        Err(first_error) if is_starvr && is_starvr_route_error(&first_error) => {
            let first_direct = cfg.hmd.starvr_direct;
            let mut alternate = cfg.clone();
            alternate.hmd.starvr_direct = !first_direct;
            let first_name = if first_direct { "direct" } else { "service" };
            let second_name = if first_direct { "service" } else { "direct" };
            eprintln!(
                "[starvr] {first_name} route failed: {first_error}; trying {second_name} route"
            );
            match build_engine_once(&alternate, gpu_context) {
                Ok(mut engine) => {
                    persist_starvr_route(alternate.hmd.starvr_direct);
                    engine.starvr_direct_mode = Some(alternate.hmd.starvr_direct);
                    engine.startup_notice = Some(format!(
                        "StarVR recovered using the {second_name} connection route. \
                         The working route was saved for the next launch."
                    ));
                    Ok(engine)
                }
                Err(second_error) => Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionRefused,
                    format!(
                        "StarVR connection failed via {first_name} mode ({first_error}); \
                         {second_name} mode also failed ({second_error})"
                    ),
                )),
            }
        }
        Err(error) => Err(error),
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn build_engine_with_gpu(
    cfg: &Config,
    gpu_context: Option<crate::ml::eyelid_model::EyelidGpuContext>,
) -> std::io::Result<Engine> {
    build_engine_once(cfg, gpu_context)
}

#[cfg(any(windows, target_os = "linux"))]
fn build_engine_once(
    cfg: &Config,
    gpu_context: Option<crate::ml::eyelid_model::EyelidGpuContext>,
) -> std::io::Result<Engine> {
    use crate::device::make_adapter;
    use crate::output::{BrokenEyeSink, OscSink, OutputSink};

    let mut adapter = make_adapter(cfg)?;
    let device_key = crate::config::running_device_key(&cfg.hmd.device, adapter.name());
    #[cfg(any(feature = "xr5-research-model", feature = "external-eyelid-model"))]
    let is_xr5 = crate::config::canonical_device_key(&device_key) == "pimax_xr5";
    #[cfg(all(windows, feature = "external-eyelid-model", not(feature = "xr5-only")))]
    if is_xr5 && cfg.ui.inference_enabled && cfg.xr5_research.enabled && cfg.xr5_research.avatar {
        if cfg.xr5_research.model_file().is_empty() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput,
                "Choose an XR5 eyelid model file in Tracking settings, then Apply + Reload"));
        }
        crate::ml::external_eyelid::Package::read(&crate::ml::external_eyelid::resolve(cfg.xr5_research.model_file()))?;
        let runtime = std::env::current_exe()?.with_file_name("model-runtime").join("onnxruntime.dll");
        if !runtime.is_file() { return Err(std::io::Error::new(std::io::ErrorKind::NotFound,
            "Model runtime missing; restore the model-runtime folder from the app ZIP")); }
    }
    let use_research_avatar = {
        #[cfg(any(feature = "xr5-research-model", feature = "external-eyelid-model"))]
        { cfg.ui.inference_enabled && is_xr5 && cfg.xr5_research.enabled && cfg.xr5_research.avatar }
        #[cfg(not(any(feature = "xr5-research-model", feature = "external-eyelid-model")))]
        { false }
    };
    let (net, eyelid_model_identity) = if use_research_avatar {
        println!("[ml] XR5 research model selected; SRanipal inference is bypassed");
        (None, None)
    } else { load_net_identified(cfg) };
    // Either eyelid model supplies blink/validity to the common brow output gate.
    // Do not disable an independent brow model merely because EyeNet was replaced.
    let brow = if net.is_some() || use_research_avatar {
        load_brow_net(cfg)
    } else {
        if cfg.brow_model_path().is_some() {
            eprintln!("[brow] disabled: needs an eyelid model too (for blink gating)");
        }
        None
    };

    // `auto` is only a selector, not a real HMD calibration bucket. Resolve it from the
    // adapter selected by the serial sniff before loading any per-device settings.
    let mut effective = cfg.clone();
    if effective.migrate_auto_device_settings(&device_key) {
        if let Err(e) = effective.save(&crate::config::config_path()) {
            eprintln!("[config] migrated legacy auto settings in memory but could not save: {e}");
        } else {
            eprintln!("[config] migrated legacy auto settings -> {device_key}");
        }
    }
    if xr5_adapter_needs_rebuild_after_migration(cfg, &effective, &device_key) {
        // Do not run the `auto` serial sniff a second time. The first adapter already
        // resolved the device, and constructors do not open hardware until `run`.
        let mut resolved = effective.clone();
        resolved.hmd.device = device_key.clone();
        adapter = make_adapter(&resolved)?;
        eprintln!("[config] applied migrated XR5 gaze source to adapter");
    }
    let status = adapter.status_arc();
    let cfg = &effective;

    let is_xr5 = crate::config::canonical_device_key(&device_key) == "pimax_xr5";
    let requested_wide_source = cfg.wide_source_for(&device_key);
    let wide = if is_xr5 && net.is_some() {
        load_wide_net(cfg)
    } else {
        None
    };
    let live_wide_source =
        runtime_wide_source(requested_wide_source, is_xr5, net.is_some(), wide.is_some());
    if !use_research_avatar && live_wide_source != requested_wide_source {
        eprintln!(
            "[wide] requested {} is unavailable; starting safely with sranipal",
            requested_wide_source.as_str()
        );
    }
    // Free the EyeChip from the Tobii runtime (may disable services / trigger UAC)
    // only for adapters that access the device directly (WinUSB `pimax_vr4` AND the
    // direct stream-engine `pimax_dll`), and only once the gating Tobii DLL is present
    // on disk (otherwise we'd cause those side effects just to then refuse).
    #[cfg(windows)]
    if adapter.needs_eyechip_handoff()
        && cfg
            .tobii_runtime_path()
            .map(|p| p.is_file())
            .unwrap_or(false)
    {
        crate::platform::ensure_capture_ready();
    }

    // VRCFT-compatible BrokenEye TCP sink (+ optional VRChat OSC direct).
    let (mut sinks, be_status): (Vec<Box<dyn OutputSink>>, Option<Arc<BrokenEyeStatus>>) =
        if !cfg.output.brokeneye || !cfg.ui.inference_enabled {
            (Vec::new(), None)
        } else {
            match BrokenEyeSink::new(cfg.output.brokeneye_port, cfg.output.vrcft_filter_samples) {
                Ok(s) => {
                    let st = s.status();
                    st.sranipal_brow_link.store(
                        cfg.output.vrcft_sranipal_brow_link,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    (vec![Box::new(s) as Box<dyn OutputSink>], Some(st))
                }
                Err(e) => {
                    eprintln!("[brokeneye] could not start server: {e}");
                    (Vec::new(), None)
                }
            }
        };
    if cfg.output.vrchat_eye_osc && cfg.ui.inference_enabled {
        match crate::output::VrchatEyeOscSink::new(cfg.output.osc_host.clone(), cfg.output.osc_port)
        {
            Ok(sink) => sinks.push(Box::new(sink)),
            Err(error) => eprintln!("[osc:eye] could not open socket: {error}"),
        }
    }
    if cfg.output.osc && cfg.ui.inference_enabled {
        match OscSink::new(
            cfg.output.osc_host.clone(),
            cfg.output.osc_port,
            "/avatar/parameters",
        ) {
            Ok(s) => sinks.push(Box::new(s)),
            Err(e) => eprintln!("[osc] could not open socket: {e}"),
        }
    } else if cfg.output.eyebrow_osc && cfg.ui.inference_enabled {
        // VRCFT remains the eye/gaze source. This sink emits only the eight
        // FT/v2 Brow* parameters used by the original vr_eyebrow application.
        match OscSink::new_brow_only(
            cfg.output.osc_host.clone(),
            cfg.output.osc_port,
            "/avatar/parameters",
        ) {
            Ok(s) => sinks.push(Box::new(s)),
            Err(e) => eprintln!("[osc] could not open eyebrow-only socket: {e}"),
        }
    }

    // Per-device eye mapping: each HMD's saved orientation, else its built-in preset
    // (Pimax flips gaze X, Varjo does not). Switching devices swaps in the right one.
    let (map, mut init) = pipeline_start_settings(cfg, &device_key);
    init.wide_source = live_wide_source;
    init.eyelid_model_identity = eyelid_model_identity;
    let mut pipeline = Pipeline::run_with_gpu_context(
        adapter,
        net,
        brow,
        wide,
        sinks,
        map,
        status,
        device_key.clone(),
        init,
        gpu_context,
    )?;
    if cfg.output.eye_image_http {
        match crate::eye_image_http::EyeImageHttpServer::new(
            &cfg.output.eye_image_host,
            cfg.output.eye_image_port,
            pipeline.tele.clone(),
        ) {
            Ok(server) => pipeline.eye_image_http = Some(server),
            Err(error) => eprintln!("[eye-image] could not start local output: {error}"),
        }
    }
    Ok(Engine {
        pipeline,
        be_status,
        startup_notice: None,
        starvr_direct_mode: (crate::config::canonical_device_key(&device_key) == "starvr")
            .then_some(cfg.hmd.starvr_direct),
    })
}

/// Build a deliberately disconnected pipeline so a hardware/startup error never
/// closes the application before the user can see the message and edit Settings.
/// It has no sinks and does not load models or touch the EyeChip.
#[cfg(any(windows, target_os = "linux"))]
pub fn build_recovery_engine(cfg: &Config, error: impl Into<String>) -> std::io::Result<Engine> {
    use std::sync::Mutex;

    use crate::core::types::DeviceProfile;
    use crate::device::{FrameFn, GazeFn, HmdAdapter};

    struct RecoveryAdapter {
        profile: DeviceProfile,
        status: Arc<Mutex<String>>,
    }

    impl HmdAdapter for RecoveryAdapter {
        fn name(&self) -> &'static str {
            "unavailable"
        }

        fn profile(&self) -> &DeviceProfile {
            &self.profile
        }

        fn start(&mut self, _on_frame: FrameFn, _on_gaze: GazeFn) -> std::io::Result<()> {
            Ok(())
        }

        fn stop(&mut self) {}

        fn status_arc(&self) -> Arc<Mutex<String>> {
            self.status.clone()
        }
    }

    let error = error.into();
    let device_key = resolved_config_device_key(cfg);
    let status = Arc::new(Mutex::new(format!("not connected: {error}")));
    let adapter: Box<dyn HmdAdapter> = Box::new(RecoveryAdapter {
        profile: DeviceProfile {
            name: format!("{} (not connected)", device_key.replace('_', " ")),
            transport: "Not connected".into(),
            streams: "No camera or gaze stream".into(),
            gaze_src: "Unavailable".into(),
            ..DeviceProfile::default()
        },
        status: status.clone(),
    });
    let (map, init) = pipeline_start_settings(cfg, &device_key);
    let pipeline = Pipeline::run(
        adapter,
        None,
        None,
        None,
        Vec::new(),
        map,
        status,
        device_key.clone(),
        init,
    )?;
    Ok(Engine {
        pipeline,
        be_status: None,
        startup_notice: Some(format!(
            "Eye tracker connection failed. SRanibro stayed open in recovery mode: {error}"
        )),
        starvr_direct_mode: (device_key == "starvr").then_some(cfg.hmd.starvr_direct),
    })
}

/// Other platforms have no acquisition adapter yet.
#[cfg(not(any(windows, target_os = "linux")))]
pub fn build_engine(_cfg: &Config) -> std::io::Result<Engine> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "engine acquisition is supported on Windows and Linux (PSVR2)",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "external-eyelid-model")]
    #[test]
    fn model_gaze_is_opt_in_xr5_only_and_uses_separate_trim() {
        let mut cfg=Config::default();
        assert!(pipeline_start_settings(&cfg,"pimax_xr5").1.xr5_gaze_package.is_none());
        cfg.hmd.xr5_model_gaze=true;
        assert!(!pipeline_start_settings(&cfg,"pimax_xr5").1.xr5_gaze_hybrid);
        cfg.hmd.xr5_gaze_hybrid=true;
        cfg.hmd.xr5_gaze_package="personal.xr5gaze".into();
        cfg.hmd.xr5_model_gaze_correction.offset_x_deg=[2.,-2.];
        cfg.hmd.xr5_hybrid_gaze_correction.offset_x_deg=[1.,-1.];
        let xr5=pipeline_start_settings(&cfg,"pimax_xr5").1;
        assert!(xr5.xr5_gaze_package.is_some());
        assert!(xr5.xr5_gaze_hybrid);
        assert_eq!(xr5.gaze_correction.offset_x_deg,[1.,-1.]);
        cfg.hmd.xr5_gaze_hybrid=false;
        assert_eq!(pipeline_start_settings(&cfg,"pimax_xr5").1.gaze_correction.offset_x_deg,[2.,-2.]);
        cfg.hmd.xr5_gaze_hybrid=true;
        for device in ["pimax_vr4","starvr","varjo","psvr2","vpe"] {
            let init=pipeline_start_settings(&cfg,device).1;
            assert!(init.xr5_gaze_package.is_none(),"{device}");
            assert!(!init.xr5_gaze_hybrid,"{device}");
            assert_ne!(init.gaze_correction.offset_x_deg,[2.,-2.],"{device}");
            assert_ne!(init.gaze_correction.offset_x_deg,[1.,-1.],"{device}");
        }
        cfg.ui.inference_enabled=false;
        assert!(!pipeline_start_settings(&cfg,"pimax_xr5").1.xr5_gaze_hybrid);
        assert!(pipeline_start_settings(&cfg,"pimax_xr5").1.xr5_gaze_package.is_none());
        cfg.ui.inference_enabled=true;
        cfg.hmd.xr5_model_gaze=false;
        assert!(!pipeline_start_settings(&cfg,"pimax_xr5").1.xr5_gaze_hybrid);
        assert_ne!(pipeline_start_settings(&cfg,"pimax_xr5").1.gaze_correction.offset_x_deg,[2.,-2.]);
        let serialized=toml::to_string(&cfg).unwrap();
        let restored:Config=toml::from_str(&serialized).unwrap();
        assert_eq!(restored.hmd.xr5_gaze_package,cfg.hmd.xr5_gaze_package);
        assert!(!restored.hmd.xr5_model_gaze);
    }

    #[test]
    fn eyelid_processing_preference_is_shared_by_sranipal_and_native_xr5() {
        use crate::config::EyelidInferenceBackend::{Auto, Cpu, Gpu};
        for selected in [Gpu, Cpu, Auto] {
            let mut cfg = Config::default();
            cfg.ui.eyelid_inference_backend = selected;
            #[cfg(any(feature = "xr5-research-model", feature = "external-eyelid-model"))]
            { cfg.xr5_research.gpu = selected == Cpu; } // stale opposite legacy choice
            let (_, vr4) = pipeline_start_settings(&cfg, "pimax_vr4");
            let (_, xr5) = pipeline_start_settings(&cfg, "pimax_xr5");
            assert_eq!(vr4.eyelid_inference_backend, selected.resolved());
            assert_eq!(xr5.eyelid_inference_backend, selected.resolved());
            #[cfg(any(feature = "xr5-research-model", feature = "external-eyelid-model"))]
            {
                assert!(vr4.xr5_research.is_none());
                assert_eq!(xr5.xr5_research.unwrap().gpu, selected != Cpu);
            }
        }
    }

    #[test]
    fn camera_only_never_loads_any_models() {
        let mut c=Config::default();c.ui.inference_enabled=false;
        assert!(load_net(&c).is_none());assert!(load_brow_net(&c).is_none());assert!(load_wide_net(&c).is_none());
        let (_,init)=pipeline_start_settings(&c,"pimax_xr5");
        assert!(!init.inference_enabled);
        #[cfg(any(feature="xr5-research-model",feature="external-eyelid-model"))]
        assert!(init.xr5_research.is_none());
        let (_,fallback)=pipeline_start_settings(&Config::default(),"pimax_xr5");
        assert_eq!(fallback.geometry,c.geometry_for("pimax_xr5"));
    }

    #[test]
    fn vr4_gaze_smoothing_switch_controls_both_outputs_only_on_vr4() {
        let mut cfg = Config::default();
        // Auto is resolved by the running adapter before assembling the pipeline.
        cfg.hmd.device = "auto".into();
        for enabled in [true, false] {
            cfg.hmd.vr4_gaze_smoothing = enabled;
            for key in ["pimax_vr4", "pimax-vr4", "crystal_super", "Crystal Super"] {
                let (_, init) = pipeline_start_settings(&cfg, key);
                assert_eq!(init.gaze_smoothing, enabled);
                assert_eq!(init.raw_gaze_osc_smoothing, enabled);
            }
            for key in ["pimax_xr5", "psvr2", "starvr", "varjo"] {
                let (_, init) = pipeline_start_settings(&cfg, key);
                assert!(init.gaze_smoothing, "{key} retains its avatar EMA");
                assert!(!init.raw_gaze_osc_smoothing, "{key} retains native OSC");
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn starvr_route_retry_filter_rejects_asset_errors() {
        let connection = std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            "tobii_device_create failed: 5 (CONNECTION_FAILED)",
        );
        let asset = std::io::Error::new(std::io::ErrorKind::Unsupported, "DLL missing export");
        let no_live_data = std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "StarVR stream readiness timed out (images=1, stereo_pairs=0, wearable=0)",
        );
        assert!(is_starvr_route_error(&connection));
        assert!(is_starvr_route_error(&no_live_data));
        assert!(!is_starvr_route_error(&asset));
    }

    #[test]
    fn xr5_mirror_is_atomic_with_geometry_and_explicit_off_wins() {
        let mut cfg = Config::default();
        // An old mapping serializes false but has no way to mean "explicit off". The
        // new XR5 geometry preset must still install its required right-eye mirror.
        cfg.set_mapping("pimax_xr5", crate::config::EyeMapping::default());
        let (_, initial) = pipeline_start_settings(&cfg, "pimax_xr5");
        assert_eq!(initial.ml_mirror, [false, true]);
        assert_eq!(initial.geometry[1].mirror_h, Some(true));

        let mut geometry = cfg.geometry_for("pimax_xr5");
        geometry[1].mirror_h = Some(false);
        cfg.set_geometry("pimax_xr5", geometry);
        let (_, overridden) = pipeline_start_settings(&cfg, "pimax_xr5");
        assert_eq!(overridden.ml_mirror, [false, false]);
    }

    #[test]
    fn pipeline_resolves_one_coherent_eye_stream_swap() {
        let mut cfg = Config::default();
        let (vr4, _) = pipeline_start_settings(&cfg, "pimax_vr4");
        let (xr5, _) = pipeline_start_settings(&cfg, "pimax_xr5");
        assert!(!vr4.swap_eyes && !xr5.swap_eyes);
        assert!(vr4.flip_gaze_x && xr5.flip_gaze_x);

        cfg.set_mapping(
            "pimax_vr4",
            crate::config::EyeMapping {
                swap_eyes: true,
                flip_gaze_x: true,
                ..Default::default()
            },
        );
        let (custom, _) = pipeline_start_settings(&cfg, "pimax_vr4");
        assert!(custom.swap_eyes);
        assert!(custom.flip_gaze_x);
    }

    #[test]
    fn gaze_correction_is_loaded_for_vr4_and_xr5_only() {
        let mut cfg = Config::default();
        let xr5_correction = crate::config::GazeCorrection {
            enabled: true,
            vergence_deg: 2.5,
            ..Default::default()
        };
        let vr4_correction = crate::config::GazeCorrection {
            enabled: true,
            vergence_deg: -1.5,
            ..Default::default()
        };
        cfg.set_gaze_correction("pimax_xr5", xr5_correction);
        cfg.set_gaze_correction("pimax_vr4", vr4_correction);
        cfg.set_gaze_correction("starvr", vr4_correction);
        assert_eq!(
            pipeline_start_settings(&cfg, "pimax_xr5").1.gaze_correction,
            xr5_correction
        );
        assert_eq!(
            pipeline_start_settings(&cfg, "pimax_vr4").1.gaze_correction,
            vr4_correction
        );
        assert_eq!(
            pipeline_start_settings(&cfg, "starvr").1.gaze_correction,
            crate::config::GazeCorrection::default()
        );
    }

    #[test]
    fn custom_wide_selection_is_effective_only_for_xr5() {
        let mut cfg = Config::default();
        cfg.hmd.wide_source = WideSource::Custom;

        assert_eq!(
            pipeline_start_settings(&cfg, "pimax_xr5").1.wide_source,
            WideSource::Custom
        );
        for other in ["pimax_vr4", "starvr", "varjo", "varjo_mjpeg", "vpe"] {
            assert_eq!(
                pipeline_start_settings(&cfg, other).1.wide_source,
                WideSource::Sranipal,
                "{other} must never inherit the XR5 custom selector"
            );
        }
    }

    #[test]
    fn unavailable_auto_or_custom_wide_falls_back_without_blocking_startup() {
        for requested in [WideSource::Auto, WideSource::Custom] {
            assert_eq!(
                runtime_wide_source(requested, true, true, false),
                WideSource::Sranipal,
                "missing Wide model must be recoverable"
            );
            assert_eq!(
                runtime_wide_source(requested, true, false, true),
                WideSource::Sranipal,
                "missing eyelid model must be recoverable"
            );
            assert_eq!(
                runtime_wide_source(requested, true, true, true),
                requested,
                "complete XR5 model stack keeps the requested source"
            );
            assert_eq!(
                runtime_wide_source(requested, false, true, true),
                WideSource::Sranipal,
                "non-XR5 devices never inherit image Wide"
            );
        }
    }
    #[test]
    fn guided_wide_capability_is_loaded_only_for_xr5_sranipal_fallback() {
        let mut cfg = Config::default();
        #[cfg(any(feature = "xr5-research-model", feature = "external-eyelid-model"))]
        { cfg.xr5_research.enabled = false; cfg.xr5_research.avatar = false; }
        let profile = crate::config::DreamAirProfile {
            wide_supported: [true, false],
            ..Default::default()
        };
        cfg.set_dream_air_profile("pimax_xr5", profile.clone());
        cfg.set_dream_air_profile("pimax_vr4", profile);
        assert_eq!(
            pipeline_start_settings(&cfg, "pimax_xr5").1.wide_enabled,
            [true, false]
        );
        assert_eq!(
            pipeline_start_settings(&cfg, "pimax_vr4").1.wide_enabled,
            [true, true]
        );
    }

    #[cfg(any(feature = "xr5-research-model", feature = "external-eyelid-model"))]
    #[test]
    fn native_xr5_ignores_legacy_wide_capability_and_never_selects_on_other_devices() {
        let mut cfg = Config::default();
        cfg.xr5_research.enabled = true;
        cfg.xr5_research.avatar = true;
        cfg.hmd.wide_source = WideSource::Custom;
        cfg.set_dream_air_profile("pimax_xr5", crate::config::DreamAirProfile {
            wide_supported: [false; 2], ..Default::default()
        });
        assert!(native_xr5_selected(&cfg, "pimax_xr5"));
        assert_eq!(pipeline_start_settings(&cfg, "pimax_xr5").1.wide_enabled, [true; 2]);
        for other in ["pimax_vr4", "psvr2", "starvr", "varjo", "vive_pro_eye"] {
            assert!(!native_xr5_selected(&cfg, other));
            assert!(pipeline_start_settings(&cfg, other).1.xr5_research.is_none());
        }
        cfg.ui.inference_enabled = false;
        assert!(!native_xr5_selected(&cfg, "pimax_xr5"));
    }

    #[test]
    fn migrated_auto_gaze_source_requires_an_xr5_adapter_rebuild_only_when_changed() {
        let mut before = Config::default();
        before
            .hmd
            .gaze_source
            .insert("auto".into(), crate::config::GazeSource::Combined);

        let mut after = before.clone();
        assert!(after.migrate_auto_device_settings("pimax_xr5"));
        assert!(xr5_adapter_needs_rebuild_after_migration(
            &before,
            &after,
            "pimax_xr5"
        ));
        assert!(!xr5_adapter_needs_rebuild_after_migration(
            &before,
            &after,
            "pimax_vr4"
        ));

        // A real device-specific setting wins during migration, so the adapter's
        // already-snapshotted value remains correct and no rebuild is needed.
        let mut explicit = before.clone();
        explicit.set_gaze_source("pimax_xr5", crate::config::GazeSource::PerEye);
        let mut migrated = explicit.clone();
        assert!(migrated.migrate_auto_device_settings("pimax_xr5"));
        assert!(!xr5_adapter_needs_rebuild_after_migration(
            &explicit,
            &migrated,
            "pimax_xr5"
        ));
    }

    #[cfg(windows)]
    #[test]
    #[cfg(feature = "xr5-only")]
    fn migrated_auto_combined_source_reaches_the_rebuilt_xr5_adapter() {
        let mut before = Config::default();
        before
            .hmd
            .gaze_source
            .insert("auto".into(), crate::config::GazeSource::Combined);
        let mut effective = before.clone();
        assert!(effective.migrate_auto_device_settings("pimax_xr5"));

        // Mirror build_engine's no-resniff reconstruction path.
        effective.hmd.device = "pimax_xr5".into();
        let adapter = crate::device::make_adapter(&effective).expect("construct XR5 adapter");
        assert!(adapter.profile().gaze_src.contains("combined"));
    }

    #[test]
    fn eye_wide_master_is_global_across_every_hmd() {
        let mut cfg = Config::default();
        cfg.ui.eye_wide_enabled = false;

        for device in [
            "auto",
            "pimax_xr5",
            "pimax_vr4",
            "starvr",
            "varjo",
            "varjo_mjpeg",
            "vpe",
        ] {
            assert!(
                !pipeline_start_settings(&cfg, device).1.eye_wide_enabled,
                "{device} must inherit the global EyeWide output master"
            );
        }
    }
}
