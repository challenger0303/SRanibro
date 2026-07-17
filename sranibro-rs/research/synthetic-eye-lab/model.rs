use std::error::Error;
use std::path::Path;

use sha2::{Digest, Sha256};
use sranibro_rs::ml::{eye_net::EyeNet, tvm_params};

use crate::renderer::{RenderedStereo, SIDE};

pub const CHANNELS: usize = 2;
pub const INPUT_LEN: usize = CHANNELS * SIDE * SIDE;

pub struct LoadedModel {
    pub net: EyeNet,
    pub identity_sha256: String,
}

pub fn load_exact(path: &Path) -> Result<LoadedModel, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let identity_sha256 = sha256_hex(&bytes);
    let map = tvm_params::parse_map_bytes(&bytes)?;
    let net = EyeNet::new(map).map_err(|error| format!("EyeNet model invalid: {error}"))?;
    Ok(LoadedModel {
        net,
        identity_sha256,
    })
}

/// Exact CHW tensor: left u8 image -> channel 0, right u8 image -> channel 1,
/// each converted only by division by 255. No production geometry or normalization
/// is applied in canonical mode.
pub fn canonical_tensor(stereo: &RenderedStereo) -> Vec<f32> {
    assert_eq!(stereo.left.pixels.len(), SIDE * SIDE);
    assert_eq!(stereo.right.pixels.len(), SIDE * SIDE);
    let mut tensor = vec![0.0f32; INPUT_LEN];
    for (dst, src) in tensor[..SIDE * SIDE].iter_mut().zip(&stereo.left.pixels) {
        *dst = *src as f32 / 255.0;
    }
    for (dst, src) in tensor[SIDE * SIDE..].iter_mut().zip(&stereo.right.pixels) {
        *dst = *src as f32 / 255.0;
    }
    tensor
}

pub fn tensor_sha256(tensor: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"sranibro-synthetic-eye-input-f32le-v1\0");
    for value in tensor {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hex(&hasher.finalize())
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::{render_stereo, StereoPolicy, SyntheticEyeSpec};

    #[test]
    fn tensor_channel_order_is_exact_and_not_swapped() {
        let mut left = SyntheticEyeSpec::default();
        let mut right = left.clone();
        left.canvas_level = 0.0;
        right.canvas_level = 1.0;
        left.components = crate::renderer::EyeComponents::none();
        right.components = crate::renderer::EyeComponents::none();
        let stereo = render_stereo(&left, &right, StereoPolicy::Independent);
        let tensor = canonical_tensor(&stereo);
        assert_eq!(tensor.len(), INPUT_LEN);
        assert!(tensor[..SIDE * SIDE].iter().all(|v| *v == 0.0));
        assert!(tensor[SIDE * SIDE..].iter().all(|v| *v == 1.0));
    }

    #[test]
    fn tensor_hash_is_stable_and_bit_sensitive() {
        let stereo = render_stereo(
            &SyntheticEyeSpec::default(),
            &SyntheticEyeSpec::default(),
            StereoPolicy::AnatomicalMirror,
        );
        let mut tensor = canonical_tensor(&stereo);
        let first = tensor_sha256(&tensor);
        assert_eq!(
            first,
            "2a924b579e1ba714916f7878432d2c36c9c4a9f842dc905ede120236f55fe28e"
        );
        assert_eq!(
            sha256_hex(&stereo.left.pixels),
            "eb7f667fc62145d3200095cc45c981a7e7ce0a46472b75408e51cbd0c4e88372"
        );
        tensor[0] = f32::from_bits(tensor[0].to_bits() ^ 1);
        assert_ne!(first, tensor_sha256(&tensor));
    }
}
