//! Public-source boundary for the optional official Tobii runtime payload.
//!
//! The reviewable source build deliberately contains neither a Tobii DLL nor the
//! private materializer used by an official binary. Private release builds replace
//! this file in an isolated staging tree. Source builds therefore keep supporting
//! the user-configured DLL path without acquiring any proprietary payload.

use std::path::PathBuf;

/// Materialized official Tobii runtime, when this source file has been replaced by
/// the private release implementation. The public-source build always returns `None`.
pub(crate) fn path() -> Option<PathBuf> {
    None
}

/// Materialized StarVR-specific runtime in an official build. It is deliberately
/// a separate capability from [`path`]: a runtime validated for direct Pimax
/// access is not assumed to expose StarVR's image and wearable subscriptions.
/// Public-source builds contain neither payload and therefore return `None`.
pub(crate) fn starvr_path() -> Option<PathBuf> {
    None
}
