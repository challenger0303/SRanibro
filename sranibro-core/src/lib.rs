//! SRanibro core — the HMD-agnostic eye-tracking post-processor and ML front-end.
//!
//! This is the openness / wide / squeeze / gaze post-processor ([`core`]) plus the
//! image preprocessing and eyelid-model inference ([`ml`]) from the SRanibro app,
//! carved out as a standalone, MIT-licensed library. It is device-independent: feed
//! it the eye model's per-eye output and a gaze sample and it returns smoothed,
//! calibrated [`core::types::EyeResult`]s.
//!
//! No proprietary assets are bundled. The eyelid model weights load at runtime from
//! a SRanipal install you supply; nothing here distributes them.
//!
//! The device-access layer (camera capture, gaze decode, connection handling) lives
//! in the closed-source SRanibro app and is intentionally not part of this crate.

pub mod core;
pub mod ml;
