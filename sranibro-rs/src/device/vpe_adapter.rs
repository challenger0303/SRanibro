//! Vive Pro Eye adapter (STUB — fundamentally constrained; researched 2026-06-26).
//!
//! VPE is Tobii-based but SPECIAL in a decisive way: HTC's stack DELIBERATELY
//! never exposes the raw eye-camera images (SDK Developer Privacy Guidelines —
//! the SDK gives "gaze position, pupil size and eye openness, but not actual
//! images of the face, eyes or lips"). There is NO `tobii_wearable_image_subscribe`
//! equivalent surfaced for VPE and no developer-accessible WinUSB image endpoint
//! (the camera lives behind HTC's SR_Runtime). So unlike Pimax VR4 / StarVR, the
//! "grab the IR image -> run our EyePrediction CNN" path is STRUCTURALLY IMPOSSIBLE
//! on VPE; a direct-USB image bypass is not feasible.
//!
//! Therefore VPE is a CONSUME-PROCESSED-DATA adapter, not an acquire-image one.
//! The only inputs are HTC's already-processed signals: gaze + the openness/wide/
//! squeeze that `EyePredictionModule.dll`'s PostProcessor computes (the very thing
//! we reverse-engineered — see project_sranipal_findings) via the SRanipal SDK, or
//! less-filtered gaze via the Tobii Pro/XR SDK. There is NO image to feed our
//! model, so the ML core is bypassed entirely for VPE.
//!
//! TODO (needs a Vive Pro Eye + SRanipal/SR_Runtime): bind SRanipal eye data
//! (`ViveSR_GetEyeData_v2` / the SRanipalTrackingModule path) and map EyeData_v2
//! (openness/wide/squeeze/gaze/pupil) DIRECTLY to [`crate::core::types::EyeResult`],
//! bypassing our ML (`ml_device` is irrelevant here). This makes VPE the one
//! adapter that emits HTC's NATIVE expression values rather than our reconstruction.

#![cfg(windows)]

use std::io;

use super::{FrameFn, GazeFn, HmdAdapter};
use crate::core::types::DeviceProfile;

pub struct VpeAdapter {
    profile: DeviceProfile,
}

impl Default for VpeAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl VpeAdapter {
    pub fn new() -> Self {
        Self {
            profile: DeviceProfile {
                name: "Vive Pro Eye".into(),
                ml_device: "vr4".into(),
                ..DeviceProfile::default()
            },
        }
    }
}

impl HmdAdapter for VpeAdapter {
    fn name(&self) -> &'static str {
        "vive-pro-eye"
    }
    fn profile(&self) -> &DeviceProfile {
        &self.profile
    }

    fn start(&mut self, _on_frame: FrameFn, _on_gaze: GazeFn) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Vive Pro Eye adapter deferred (special): HTC never exposes raw eye \
             images, so our image->ML path is impossible. Plan: consume SRanipal \
             EyeData_v2 (openness/wide/squeeze/gaze/pupil) directly, bypassing our \
             ML. Needs a Vive Pro Eye + SRanipal runtime.",
        ))
    }

    fn stop(&mut self) {}
}
