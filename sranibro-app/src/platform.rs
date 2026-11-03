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
