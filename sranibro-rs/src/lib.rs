//! SRanibro — all-Rust, distribution-safe eye/face tracker.
//!
//! Library crate holding the HMD-agnostic engine. Nothing proprietary is bundled:
//! ML weights load from the user's SRanipal directory at runtime ([`ml`]), and
//! the post-processor ([`core`]) is original code. The `sranibro-rs` binary (and
//! later the egui app) build on this.

#[cfg(all(feature = "psvr2-only", feature = "xr5-only"))]
compile_error!("psvr2-only and xr5-only are mutually exclusive build variants");

pub mod blink_timing_fit;
pub mod brow_calib;
pub mod brow_fitrun;
pub mod brow_train;
pub(crate) mod bundled_tobii;
pub mod calib_session;
pub mod config;
pub mod core;
pub mod device;
pub mod diagnostics;
pub mod endpoint_fit;
pub mod engine;
pub mod eye_image_http;
pub mod gaze_eyelid_fit;
pub mod gaze_residual_calib;
pub mod geometry_calib;
pub mod geometry_discovery;
pub mod geometry_fitrun;
#[cfg(feature = "research-synthetic-eye-lab")]
pub mod geometry_landmarks;
pub mod logcap;
pub mod ml;
pub mod output;
pub mod pipeline;
pub mod platform;
pub mod recording_audio;
pub mod reseat_assist;
pub mod sranipal_discovery;
pub mod theme;
pub mod ui;
pub mod vr_research_overlay;
pub mod wear_memory;
pub mod wide_calib;
pub mod wide_fitrun;
pub mod wink_fit;

#[cfg(test)]
pub(crate) mod test_alloc;
