//! Internal contract between the canonical stereo eyelid tensor and raw eyelid inference.
//!
//! Phase 1 deliberately keeps preprocessing, gaze, wide, brow, blink, calibration, and
//! smoothing outside this boundary. The legacy adapter only gives names to the existing
//! five EyeNet outputs; it does not reinterpret them.

use super::eye_net::EyeNet;

pub(crate) const EYELID_INPUT_CHANNELS: usize = 2;
pub(crate) const EYELID_INPUT_WIDTH: usize = 100;
pub(crate) const EYELID_INPUT_HEIGHT: usize = 100;
pub(crate) const EYELID_INPUT_LEN: usize =
    EYELID_INPUT_CHANNELS * EYELID_INPUT_WIDTH * EYELID_INPUT_HEIGHT;

/// Already-preprocessed contiguous CHW tensor with shape `[2, 100, 100]`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CanonicalStereoInput<'a> {
    tensor: &'a [f32; EYELID_INPUT_LEN],
}

impl<'a> CanonicalStereoInput<'a> {
    pub(crate) fn new(tensor: &'a [f32; EYELID_INPUT_LEN]) -> Self {
        Self { tensor }
    }

    pub(crate) fn as_slice(self) -> &'a [f32] {
        self.tensor
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InputShapeError {
    pub(crate) actual: usize,
}

impl<'a> TryFrom<&'a [f32]> for CanonicalStereoInput<'a> {
    type Error = InputShapeError;

    fn try_from(tensor: &'a [f32]) -> Result<Self, Self::Error> {
        let actual = tensor.len();
        let tensor = tensor.try_into().map_err(|_| InputShapeError { actual })?;
        Ok(Self::new(tensor))
    }
}

/// Raw model capabilities. `None` means unsupported by the backend, never invalid,
/// low-confidence, or non-finite. The legacy adapter therefore always returns `Some`,
/// including `Some(NaN)` and `Some(infinity)` when those are the raw model values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RawEyelidPrediction {
    pub(crate) presence: Option<f32>,
    pub(crate) openness: [Option<f32>; 2],
    pub(crate) squeeze: [Option<f32>; 2],
}

/// Phase 1's Legacy backend is infallible, so this is intentionally uninhabited.
/// Future backends may add fixed, allocation-free error variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RequiredOutput {
    Presence,
    LeftOpenness,
    RightOpenness,
    LeftSqueeze,
    RightSqueeze,
}

/// Exact bridge into the existing post-processor input layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LegacyPublishFrame {
    pub(crate) ml_raw: [f32; 2],
    pub(crate) ml5: [[f32; 5]; 2],
}

impl RawEyelidPrediction {
    fn from_legacy_raw(raw: [f32; 5]) -> Self {
        Self {
            presence: Some(raw[0]),
            openness: [Some(raw[1]), Some(raw[2])],
            squeeze: [Some(raw[3]), Some(raw[4])],
        }
    }

    /// Require the five capabilities used by the current legacy post-processor.
    /// Structural channels 2 and 4 stay literal zeroes exactly as before Phase 1.
    pub(crate) fn require_legacy_frame(self) -> Result<LegacyPublishFrame, RequiredOutput> {
        let presence = self.presence.ok_or(RequiredOutput::Presence)?;
        let left_open = self.openness[0].ok_or(RequiredOutput::LeftOpenness)?;
        let right_open = self.openness[1].ok_or(RequiredOutput::RightOpenness)?;
        let left_squeeze = self.squeeze[0].ok_or(RequiredOutput::LeftSqueeze)?;
        let right_squeeze = self.squeeze[1].ok_or(RequiredOutput::RightSqueeze)?;
        Ok(LegacyPublishFrame {
            ml_raw: [left_open, right_open],
            ml5: [
                [presence, left_open, 0.0, left_squeeze, 0.0],
                [presence, right_open, 0.0, right_squeeze, 0.0],
            ],
        })
    }
}

pub(crate) trait EyelidModel: Send {
    fn infer(&mut self, input: CanonicalStereoInput<'_>)
        -> Result<RawEyelidPrediction, ModelError>;
}

/// Thin behavior-preserving adapter over the existing SRanipal-compatible EyeNet.
pub(crate) struct LegacyEyelidModel {
    net: EyeNet,
}

impl LegacyEyelidModel {
    pub(crate) fn new(net: EyeNet) -> Self {
        Self { net }
    }
}

impl EyelidModel for LegacyEyelidModel {
    fn infer(
        &mut self,
        input: CanonicalStereoInput<'_>,
    ) -> Result<RawEyelidPrediction, ModelError> {
        Ok(RawEyelidPrediction::from_legacy_raw(
            self.net.forward_one(input.as_slice()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::hint::black_box;

    use super::*;
    use crate::ml::tvm_params::Tensor;
    use crate::test_alloc::count_current_thread_allocations;

    fn tensor(name: &str, shape: &[i64], data: Vec<f32>) -> Tensor {
        Tensor {
            name: name.into(),
            shape: shape.to_vec(),
            data,
        }
    }

    fn zero_tensor(name: &str, shape: &[i64]) -> Tensor {
        let len = shape.iter().product::<i64>() as usize;
        tensor(name, shape, vec![0.0; len])
    }

    fn synthetic_net(output: [f32; 5]) -> EyeNet {
        let mut map = HashMap::new();
        for (name, shape) in [
            ("conv1_weight", &[20, 2, 5, 5][..]),
            ("conv1_bias", &[20][..]),
            ("conv2_weight", &[48, 20, 5, 5][..]),
            ("conv2_bias", &[48][..]),
            ("conv3_weight", &[64, 48, 3, 3][..]),
            ("conv3_bias", &[64][..]),
            ("fc5_ft_weight", &[500, 25_600][..]),
            ("fc5_ft_bias", &[500][..]),
            ("fc6_10_weight", &[5, 500][..]),
        ] {
            map.insert(name.into(), zero_tensor(name, shape));
        }
        map.insert(
            "fc6_10_bias".into(),
            tensor("fc6_10_bias", &[5], output.to_vec()),
        );
        EyeNet::new(map).expect("synthetic tensors match the legacy architecture")
    }

    #[test]
    fn canonical_input_enforces_the_exact_shape() {
        let exact = [0.0; EYELID_INPUT_LEN];
        assert!(CanonicalStereoInput::try_from(exact.as_slice()).is_ok());
        assert_eq!(
            CanonicalStereoInput::try_from(&exact[..EYELID_INPUT_LEN - 1]).unwrap_err(),
            InputShapeError {
                actual: EYELID_INPUT_LEN - 1
            }
        );
    }

    #[test]
    fn legacy_mapping_is_asymmetric_and_structural_channels_stay_zero() {
        let prediction = RawEyelidPrediction::from_legacy_raw([0.91, 0.12, 0.78, 0.34, 0.67]);
        assert_eq!(prediction.presence, Some(0.91));
        assert_eq!(prediction.openness, [Some(0.12), Some(0.78)]);
        assert_eq!(prediction.squeeze, [Some(0.34), Some(0.67)]);
        let frame = prediction.require_legacy_frame().unwrap();
        assert_eq!(frame.ml_raw, [0.12, 0.78]);
        assert_eq!(
            frame.ml5,
            [[0.91, 0.12, 0.0, 0.34, 0.0], [0.91, 0.78, 0.0, 0.67, 0.0]]
        );
    }

    #[test]
    fn legacy_publish_sequence_keeps_brightness_feedback_and_ml5_golden() {
        let sequence = [
            [0.91, 0.12, 0.78, 0.34, 0.67],
            [0.04, 0.56, 0.23, 0.89, 0.45],
            [1.20, -0.10, 1.40, 0.00, 0.99],
        ];
        let expected_raw = [[0.12, 0.78], [0.56, 0.23], [-0.10, 1.40]];
        let expected_ml5 = [
            [[0.91, 0.12, 0.0, 0.34, 0.0], [0.91, 0.78, 0.0, 0.67, 0.0]],
            [[0.04, 0.56, 0.0, 0.89, 0.0], [0.04, 0.23, 0.0, 0.45, 0.0]],
            [[1.20, -0.10, 0.0, 0.00, 0.0], [1.20, 1.40, 0.0, 0.99, 0.0]],
        ];
        let mut prev_open = [0.5; 2];
        assert_eq!(prev_open, [0.5; 2]);
        for (index, raw) in sequence.into_iter().enumerate() {
            let frame = RawEyelidPrediction::from_legacy_raw(raw)
                .require_legacy_frame()
                .unwrap();
            // This is the exact assignment used by the runtime brightness feedback loop.
            prev_open = frame.ml_raw;
            assert_eq!(prev_open, expected_raw[index]);
            assert_eq!(frame.ml5, expected_ml5[index]);
        }
    }

    #[test]
    fn legacy_mapping_preserves_nonfinite_and_low_presence_bits_as_some() {
        let raw = [0.001, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.25];
        let prediction = RawEyelidPrediction::from_legacy_raw(raw);
        for (actual, expected) in [
            (prediction.presence.unwrap(), raw[0]),
            (prediction.openness[0].unwrap(), raw[1]),
            (prediction.openness[1].unwrap(), raw[2]),
            (prediction.squeeze[0].unwrap(), raw[3]),
            (prediction.squeeze[1].unwrap(), raw[4]),
        ] {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn missing_capability_never_becomes_a_valid_looking_zero() {
        let prediction = RawEyelidPrediction {
            presence: Some(0.9),
            openness: [Some(0.2), None],
            squeeze: [Some(0.3), Some(0.4)],
        };
        assert_eq!(
            prediction.require_legacy_frame(),
            Err(RequiredOutput::RightOpenness)
        );
    }

    #[test]
    fn direct_and_adapter_paths_are_bit_identical() {
        let expected = [0.91, 0.12, 0.78, 0.34, 0.67];
        let input = [0.25; EYELID_INPUT_LEN];
        let mut net = synthetic_net(expected);
        let direct = net.forward_one(&input);
        let mut adapter = LegacyEyelidModel::new(net);
        let prediction = adapter
            .infer(CanonicalStereoInput::new(&input))
            .expect("legacy inference is infallible");
        let frame = prediction
            .require_legacy_frame()
            .expect("legacy provides every required channel");
        assert_eq!(direct.map(f32::to_bits), expected.map(f32::to_bits));
        assert_eq!(
            prediction.presence.map(f32::to_bits),
            Some(direct[0].to_bits())
        );
        assert_eq!(
            prediction.openness.map(|value| value.map(f32::to_bits)),
            [Some(direct[1].to_bits()), Some(direct[2].to_bits())]
        );
        assert_eq!(
            prediction.squeeze.map(|value| value.map(f32::to_bits)),
            [Some(direct[3].to_bits()), Some(direct[4].to_bits())]
        );
        assert_eq!(frame.ml5[0][2].to_bits(), 0.0f32.to_bits());
        assert_eq!(frame.ml5[0][4].to_bits(), 0.0f32.to_bits());
        assert_eq!(frame.ml5[1][2].to_bits(), 0.0f32.to_bits());
        assert_eq!(frame.ml5[1][4].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn adapter_adds_no_steady_state_allocation_after_warmup() {
        let input = [0.25; EYELID_INPUT_LEN];
        let mut net = synthetic_net([0.91, 0.12, 0.78, 0.34, 0.67]);
        black_box(net.forward_one(&input));
        let (_, direct_allocations) = count_current_thread_allocations(|| {
            black_box(net.forward_one(black_box(&input)));
        });

        let mut model: Box<dyn EyelidModel> = Box::new(LegacyEyelidModel::new(net));
        black_box(
            model
                .infer(CanonicalStereoInput::new(&input))
                .expect("warm adapter inference"),
        );
        let (_, adapter_allocations) = count_current_thread_allocations(|| {
            let canonical = CanonicalStereoInput::try_from(black_box(input.as_slice()))
                .expect("fixed-size canonical input");
            black_box(model.infer(canonical).expect("legacy inference"));
        });

        assert_eq!(
            adapter_allocations, direct_allocations,
            "trait dispatch, shape cast, and adapter must add no allocation beyond direct EyeNet"
        );
    }
}
