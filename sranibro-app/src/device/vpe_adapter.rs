
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
