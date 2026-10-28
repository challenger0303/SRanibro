//! Explicit boundary for the non-public direct EyeChip access implementation.
pub(crate) fn direct_access_available() -> bool { false }
pub(crate) fn unavailable() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Unsupported,
        "Direct EyeChip access is excluded from this public source build; see SOURCE_BOUNDARY.md")
}
#[cfg(test)]
mod tests {
    use crate::device::{vr4_adapter::Vr4Adapter,HmdAdapter};
    #[test] fn direct_acquisition_fails_before_opening_hardware() {
        let mut adapter=Vr4Adapter::new();
        let error=adapter.start(Box::new(|_,_,_,_| panic!("No frame expected")),Box::new(|_| panic!("No gaze expected"))).unwrap_err();
        assert_eq!(error.kind(),std::io::ErrorKind::Unsupported);
    }
}
