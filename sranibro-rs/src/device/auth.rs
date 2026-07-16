//! EyeChip authentication: HMAC-MD5 of the device challenge.
//!
//! The EyeChip gates its streams behind a challenge/response handshake (TTP
//! packets 1900/1911). The key + pad were recovered in `tobii_usb_direct.py`;
//! we compute the response ourselves so no patched DLL / SDK is needed.

use hmac::{Hmac, Mac};
use md5::Md5;

pub const HMAC_KEY: [u8; 16] = [
    112, 240, 94, 207, 10, 40, 102, 12, 173, 90, 105, 133, 170, 159, 133, 130,
];
pub const AUTH_PAD: [u8; 16] = [
    153, 48, 199, 152, 25, 147, 218, 135, 58, 222, 43, 10, 98, 25, 235, 49,
];

/// HMAC-MD5(HMAC_KEY, challenge) -> 16-byte digest (the 1911 response payload).
pub fn auth_digest(challenge: &[u8]) -> [u8; 16] {
    let mut mac = Hmac::<Md5>::new_from_slice(&HMAC_KEY).expect("HMAC accepts any key length");
    mac.update(challenge);
    let out = mac.finalize().into_bytes();
    let mut digest = [0u8; 16];
    digest.copy_from_slice(&out);
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn matches_python_reference_vector() {
        // Cross-checked against Python: hmac.new(HMAC_KEY, bytes(range(16)), md5).
        let challenge: Vec<u8> = (0u8..16).collect();
        assert_eq!(
            hex(&auth_digest(&challenge)),
            "6f9755ec23f2f6323f7fd13f3bc36540"
        );
    }
}
