//! Runtime intervention is deliberately excluded from this public checkout.
//! No service state, process, DLL or device ownership is changed here.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum TobiiMode { Custom, BrokenEye, Unknown }
pub fn detect_mode() -> TobiiMode { TobiiMode::Unknown }
pub fn is_elevated() -> bool { false }
pub fn ensure_capture_ready() {}
pub fn ensure_starvr_ready() {}
pub fn ensure_starvr_direct_ready() {}
fn unavailable() { eprintln!("Runtime management is excluded from the public source build; see SOURCE_BOUNDARY.md."); }
pub fn cmd_custom() { unavailable(); }
pub fn cmd_restore() { unavailable(); }
pub fn cmd_starvr_direct() { unavailable(); }
pub fn cmd_starvr_service() { unavailable(); }
