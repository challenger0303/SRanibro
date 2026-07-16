//! Dream Air/XR5 image-based EyeWide network.
//!
//! The convolution implementation is intentionally shared byte-for-byte with the
//! proven eyebrow TinyEyeNet runtime (`BrowNet`). Only the task header and input crop
//! differ. New models use `TINYEYE1 + task_id + out_dim`; the task id prevents a brow
//! model from being selected accidentally as an EyeWide model.

use std::io::Read;
use std::path::Path;

use crate::ml::brow_fit::HeadWeights;
use crate::ml::brow_net::BrowNet;

const GENERIC_MAGIC: &[u8; 8] = b"TINYEYE1";
const LEGACY_WIDE_MAGIC: &[u8; 8] = b"WIDENET1";
const BROW_MAGIC: &[u8; 8] = b"BROWNET1";
const TASK_WIDE: u32 = 2;

/// Thin task wrapper around the shared TinyEyeNet convolution runtime.
pub struct WideNet {
    inner: BrowNet,
}

impl WideNet {
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut file| file.read_to_end(&mut bytes))
            .map_err(|e| format!("wide weights '{}': {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let legacy = if bytes.len() >= 16 && &bytes[..8] == GENERIC_MAGIC {
            let task = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
            let out_dim = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
            if task != TASK_WIDE {
                return Err(format!(
                    "TinyEyeNet task {task}, expected EyeWide ({TASK_WIDE})"
                ));
            }
            if out_dim != 1 {
                return Err(format!("EyeWide out_dim {out_dim}, expected 1"));
            }
            let mut converted = Vec::with_capacity(bytes.len() - 4);
            converted.extend_from_slice(BROW_MAGIC);
            converted.extend_from_slice(&out_dim.to_le_bytes());
            converted.extend_from_slice(&bytes[16..]);
            converted
        } else if bytes.len() >= 12 && &bytes[..8] == LEGACY_WIDE_MAGIC {
            let out_dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
            if out_dim != 1 {
                return Err(format!("EyeWide out_dim {out_dim}, expected 1"));
            }
            let mut converted = bytes.to_vec();
            converted[..8].copy_from_slice(BROW_MAGIC);
            converted
        } else if bytes.len() >= 8 && &bytes[..8] == BROW_MAGIC {
            return Err("BROWNET1 is an eyebrow model, not an EyeWide model".into());
        } else {
            return Err("not a TINYEYE1 EyeWide weights file".into());
        };
        let inner = BrowNet::from_bytes(&legacy)?;
        if inner.out_dim() != 1 {
            return Err(format!("EyeWide out_dim {}, expected 1", inner.out_dim()));
        }
        Ok(Self { inner })
    }

    pub fn forward_one(&mut self, input64: &[f32]) -> f32 {
        self.inner.forward_one(input64)[0]
    }

    pub fn features(&mut self, input64: &[f32]) -> Vec<f32> {
        self.inner.brow_features(input64)
    }

    /// Serialize the shared frozen backbone plus a fitted one-output head in the
    /// task-tagged generic format.
    pub fn to_bytes_with_head(&self, head: &HeadWeights) -> Result<Vec<u8>, String> {
        let brow = self.inner.to_bytes_with_head(head)?;
        let mut out = Vec::with_capacity(brow.len() + 4);
        out.extend_from_slice(GENERIC_MAGIC);
        out.extend_from_slice(&TASK_WIDE.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&brow[12..]);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_wide_blob(generic: bool) -> Vec<u8> {
        let mut out = Vec::new();
        if generic {
            out.extend_from_slice(GENERIC_MAGIC);
            out.extend_from_slice(&TASK_WIDE.to_le_bytes());
            out.extend_from_slice(&1u32.to_le_bytes());
        } else {
            out.extend_from_slice(LEGACY_WIDE_MAGIC);
            out.extend_from_slice(&1u32.to_le_bytes());
        }
        let conv = [(1usize, 16usize), (16, 32), (32, 64), (64, 64)];
        let floats = conv.iter().map(|(ic, oc)| oc * ic * 9 + oc).sum::<usize>()
            + 128 * 1024
            + 128
            + 128
            + 1;
        out.resize(out.len() + floats * 4, 0);
        out
    }

    #[test]
    fn generic_and_legacy_wide_models_use_shared_runtime() {
        for bytes in [zero_wide_blob(true), zero_wide_blob(false)] {
            let mut net = WideNet::from_bytes(&bytes).unwrap();
            assert_eq!(net.forward_one(&[0.0; 64 * 64]), 0.0);
            assert_eq!(net.features(&[0.0; 64 * 64]).len(), 1024);
        }
    }

    #[test]
    fn eyebrow_magic_is_rejected() {
        let mut bytes = zero_wide_blob(false);
        bytes[..8].copy_from_slice(BROW_MAGIC);
        assert!(WideNet::from_bytes(&bytes)
            .err()
            .unwrap()
            .contains("eyebrow"));
    }
}
