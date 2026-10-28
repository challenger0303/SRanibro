//! Public source provider: no embedded or extracted vendor runtime.
use std::path::PathBuf;
pub(crate) fn path() -> Option<PathBuf> { None }
pub(crate) fn starvr_path() -> Option<PathBuf> { None }
#[cfg(test)]
mod tests {
    #[test] fn public_source_has_no_embedded_runtime() {
        assert!(super::path().is_none());
        assert!(super::starvr_path().is_none());
    }
}
