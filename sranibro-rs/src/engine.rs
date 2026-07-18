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
use crate::pipeline::{DeviceMap, Pipeline, PipelineInit};

/// A running engine plus the handles the UI needs to render/swap it.
pub struct Engine {
    pub pipeline: Pipeline,
    pub be_status: Option<Arc<BrokenEyeStatus>>,
}

/// Resolve every live value needed before [`Pipeline::run`] starts its threads. The
/// geometry's tri-state mirror wins when explicitly present; otherwise legacy mapping
/// values remain compatible for non-XR5 configs.
pub fn pipeline_start_settings(cfg: &Config, device_key: &str) -> (DeviceMap, PipelineInit) {
    let mapping = cfg.mapping_for(device_key);
    let geometry = cfg.geometry_for(device_key);
    let is_xr5 = crate::config::canonical_device_key(device_key) == "pimax_xr5";
    let ml_mirror = [
        geometry[0].mirror_h.unwrap_or(mapping.ml_mirror_l),
        geometry[1].mirror_h.unwrap_or(mapping.ml_mirror_r),
    ];
    (
        DeviceMap {
            swap_eyes: mapping.swap_eyes,
            flip_image: mapping.flip_image,
            flip_gaze_x: mapping.flip_gaze_x,
        },
        PipelineInit {
            eyebrow_enabled: cfg.ui.eyebrow_enabled,
            ml_mirror,
            tuning: cfg.tuning,
            geometry,
            despeckle: cfg.despeckle_for(device_key),
            flatten: cfg.flatten_for(device_key),
            brightness: cfg.brightness_for(device_key),
            gaze_correction: if is_xr5 {
                cfg.gaze_correction_for(device_key)
            } else {
                crate::config::GazeCorrection::default()
            },
            wide_enabled: if is_xr5 {
                cfg.dream_air_profile_for(device_key)
                    .map(|profile| profile.wide_supported)
                    .unwrap_or([true; 2])
            } else {
                [true; 2]
            },
            wide_source: cfg.wide_source_for(device_key),
        },
    )
}

/// Load the EyePrediction net from the configured weights path (direct `ml_model`
/// file, else `<sranipal_dir>/MODEL_REL`). Returns `None` — and prints a precise
/// reason — when absent or invalid, so the engine simply runs gaze-only.
pub fn load_net(cfg: &Config) -> Option<EyeNet> {
    match cfg.ml_params_path() {
        Some(p) if p.is_file() => match tvm_params::parse_map(&p.to_string_lossy()) {
            Ok(map) => match EyeNet::new(map) {
                Ok(n) => {
                    println!("[ml] loaded weights from {}", p.display());
                    Some(n)
                }
                Err(e) => {
                    eprintln!("[ml] model invalid: {e} (running without ML)");
                    None
                }
            },
            Err(e) => {
                eprintln!("[ml] parse failed: {e} (running without ML)");
                None
            }
        },
        Some(p) => {
            eprintln!(
                "[ml] weights not found at {} (running gaze-only)",
                p.display()
            );
            None
        }
        None => {
            eprintln!("[ml] no SRanipal weights configured (running gaze-only)");
            None
        }
    }
}

/// Load the optional eyebrow model (`[assets].brow_model`). `None` (with a printed
/// reason) when absent/invalid, so the engine simply runs without brow output.
pub fn load_brow_net(cfg: &Config) -> Option<BrowNet> {
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

/// Build + start the engine for `cfg`: free the EyeChip (pre-flight), open the
/// configured adapter, wire the ML net and the VRCFT/OSC sinks, and run the
/// pipeline. The caller owns the returned [`Engine`] (call `pipeline.stop()` /
/// drop to tear it down). Errors if the adapter can't be created or started.
#[cfg(windows)]
pub fn build_engine(cfg: &Config) -> std::io::Result<Engine> {
    use crate::device::make_adapter;
    use crate::output::{BrokenEyeSink, OscSink, OutputSink};

    let net = load_net(cfg);
    // Brow inference needs the eye net's blink signal to gate output; if there's no eye
    // model, skip brow (rather than blink-gate on a constant).
    let brow = if net.is_some() {
        load_brow_net(cfg)
    } else {
        if cfg.brow_model_path().is_some() {
            eprintln!("[brow] disabled: needs the SRanipal eye model too (for blink gating)");
        }
        None
    };

    let adapter = make_adapter(cfg)?;
    let status = adapter.status_arc();
    // `auto` is only a selector, not a real HMD calibration bucket. Resolve it from the
    // adapter selected by the serial sniff before loading any per-device settings.
    let device_key = crate::config::running_device_key(&cfg.hmd.device, adapter.name());
    let mut effective = cfg.clone();
    if effective.migrate_auto_device_settings(&device_key) {
        if let Err(e) = effective.save(&crate::config::config_path()) {
            eprintln!("[config] migrated legacy auto settings in memory but could not save: {e}");
        } else {
            eprintln!("[config] migrated legacy auto settings -> {device_key}");
        }
    }
    let cfg = &effective;

    let is_xr5 = crate::config::canonical_device_key(&device_key) == "pimax_xr5";
    let wide_source = cfg.wide_source_for(&device_key);
    if wide_source == WideSource::Custom && net.is_none() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "custom EyeWide needs the SRanipal eyelid model for blink gating",
        ));
    }
    let wide = if is_xr5 && net.is_some() {
        load_wide_net(cfg)
    } else {
        None
    };
    if wide_source == WideSource::Custom && wide.is_none() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "wide_source=custom needs a valid [assets].wide_model",
        ));
    }

    // Free the EyeChip from the Tobii runtime (may disable services / trigger UAC)
    // only for adapters that access the device directly (WinUSB `pimax_vr4` AND the
    // direct stream-engine `pimax_dll`), and only once the gating Tobii DLL is present
    // on disk (otherwise we'd cause those side effects just to then refuse).
    if adapter.needs_eyechip_handoff() && cfg.tobii_dll_path().map(|p| p.is_file()).unwrap_or(false)
    {
        crate::platform::ensure_capture_ready();
    }

    // VRCFT-compatible BrokenEye TCP sink (+ optional VRChat OSC direct).
    let (mut sinks, be_status): (Vec<Box<dyn OutputSink>>, Option<Arc<BrokenEyeStatus>>) =
        match BrokenEyeSink::new(cfg.output.brokeneye_port, cfg.output.vrcft_filter_samples) {
            Ok(s) => {
                let st = s.status();
                (vec![Box::new(s) as Box<dyn OutputSink>], Some(st))
            }
            Err(e) => {
                eprintln!("[brokeneye] could not start server: {e}");
                (Vec::new(), None)
            }
        };
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
    let (map, init) = pipeline_start_settings(cfg, &device_key);
    let pipeline = Pipeline::run(
        adapter, net, brow, wide, sinks, map, status, device_key, init,
    )?;
    Ok(Engine {
        pipeline,
        be_status,
    })
}

/// Non-Windows stub (acquisition is Windows-only); keeps the lib cross-compilable.
#[cfg(not(windows))]
pub fn build_engine(_cfg: &Config) -> std::io::Result<Engine> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "engine acquisition is Windows-only",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn gaze_correction_is_loaded_only_for_xr5() {
        let mut cfg = Config::default();
        let correction = crate::config::GazeCorrection {
            enabled: true,
            vergence_deg: 2.5,
            ..Default::default()
        };
        cfg.set_gaze_correction("pimax_xr5", correction);
        cfg.set_gaze_correction("pimax_vr4", correction);
        assert_eq!(
            pipeline_start_settings(&cfg, "pimax_xr5").1.gaze_correction,
            correction
        );
        assert_eq!(
            pipeline_start_settings(&cfg, "pimax_vr4").1.gaze_correction,
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
    fn guided_wide_capability_is_loaded_only_for_xr5() {
        let mut cfg = Config::default();
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
}
