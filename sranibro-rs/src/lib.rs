//! SRanibro — all-Rust, distribution-safe eye/face tracker.
//!
//! Library crate holding the HMD-agnostic engine. Nothing proprietary is bundled:
//! ML weights load from the user's SRanipal directory at runtime ([`ml`]), and
//! the post-processor ([`core`]) is original code. The `sranibro-rs` binary (and
//! later the egui app) build on this.

pub mod brow_calib;
pub mod brow_fitrun;
pub mod brow_train;
pub mod config;
pub mod core;
pub mod device;
pub mod diagnostics;
pub mod engine;
pub mod geometry_calib;
pub mod geometry_discovery;
pub mod geometry_fitrun;
pub mod logcap;
pub mod ml;
pub mod output;
pub mod pipeline;
pub mod platform;
pub mod theme;
pub mod ui;
pub mod wide_calib;
pub mod wide_fitrun;

#[cfg(test)]
pub(crate) mod test_alloc;
