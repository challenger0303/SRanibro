use serde::{Deserialize, Serialize};

use crate::luminance::{self, LuminancePlan, MeanMatchRecord, PreparedPoint};
use crate::renderer::{EyeComponents, PhotometricTransform, StereoPolicy, SyntheticEyeSpec};

pub const SUITE_VERSION: &str = "synthetic-eye-phase0-v1";
pub const MILESTONE1_VERSION: &str = "synthetic-eye-milestone1-v1";
pub const LUMINANCE_MATCH_VERSION: &str = luminance::VERSION;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaseDefinition {
    pub experiment: String,
    pub case_name: String,
    pub interpretation_scope: String,
    pub stereo_policy: StereoPolicy,
    pub photometric: PhotometricTransform,
    pub factor_x_name: Option<String>,
    pub factor_x: Option<f32>,
    pub factor_y_name: Option<String>,
    pub factor_y: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_match: Option<MeanMatchRecord>,
    pub left: SyntheticEyeSpec,
    pub right: SyntheticEyeSpec,
}

pub fn phase0_cases() -> Vec<CaseDefinition> {
    let anchor = SyntheticEyeSpec::default();
    let mut cases = Vec::new();

    for (name, aperture) in [
        ("anchor_near_095", 0.95),
        ("anchor_center_100", 1.0),
        ("anchor_near_105", 1.05),
    ] {
        let mut spec = anchor.clone();
        spec.aperture = aperture;
        cases.push(case(
            "anchor_family",
            name,
            "phase0_go_no_go_only",
            spec,
            StereoPolicy::AnatomicalMirror,
        ));
        let last = cases.last_mut().unwrap();
        last.factor_x_name = Some("aperture".into());
        last.factor_x = Some(aperture);
    }

    let ladder = [
        ("p0_uniform", EyeComponents::none()),
        (
            "p1_skin_region",
            EyeComponents {
                skin_region: true,
                ..EyeComponents::none()
            },
        ),
        (
            "p2_sclera_opening",
            EyeComponents {
                skin_region: true,
                sclera: true,
                ..EyeComponents::none()
            },
        ),
        (
            "p3_sclera_pupil",
            EyeComponents {
                skin_region: true,
                sclera: true,
                pupil: true,
                ..EyeComponents::none()
            },
        ),
        (
            "p4_sclera_iris_pupil",
            EyeComponents {
                skin_region: true,
                sclera: true,
                iris: true,
                pupil: true,
                ..EyeComponents::none()
            },
        ),
        (
            "p5_add_upper_lid",
            EyeComponents {
                skin_region: true,
                sclera: true,
                iris: true,
                pupil: true,
                upper_lid: true,
                ..EyeComponents::none()
            },
        ),
        (
            "p6_add_lower_lid",
            EyeComponents {
                skin_region: true,
                sclera: true,
                iris: true,
                pupil: true,
                upper_lid: true,
                lower_lid: true,
                canthi: false,
            },
        ),
        ("p7_add_canthi", EyeComponents::full()),
    ];
    for (name, components) in ladder {
        let mut spec = anchor.clone();
        spec.components = components;
        cases.push(case(
            "presence_ladder",
            name,
            "exploratory_only_not_feature_attribution",
            spec,
            StereoPolicy::AnatomicalMirror,
        ));
    }

    let ablations = [
        ("full_anchor", EyeComponents::full()),
        (
            "remove_sclera",
            EyeComponents {
                sclera: false,
                ..EyeComponents::full()
            },
        ),
        (
            "remove_iris",
            EyeComponents {
                iris: false,
                ..EyeComponents::full()
            },
        ),
        (
            "remove_pupil",
            EyeComponents {
                pupil: false,
                ..EyeComponents::full()
            },
        ),
        (
            "remove_upper_lid",
            EyeComponents {
                upper_lid: false,
                ..EyeComponents::full()
            },
        ),
        (
            "remove_lower_lid",
            EyeComponents {
                lower_lid: false,
                ..EyeComponents::full()
            },
        ),
        (
            "remove_canthi",
            EyeComponents {
                canthi: false,
                ..EyeComponents::full()
            },
        ),
    ];
    for (name, components) in ablations {
        let mut spec = anchor.clone();
        spec.components = components;
        cases.push(case(
            "subtractive_ablation",
            name,
            "raw_unmatched_ablation_report_covariates_no_feature_claim",
            spec,
            StereoPolicy::AnatomicalMirror,
        ));
    }

    cases
}

pub fn milestone1_cases() -> Vec<CaseDefinition> {
    let anchor = SyntheticEyeSpec::default();
    let mut cases = Vec::new();

    for step in 0..=40 {
        let aperture = 1.3 * step as f32 / 40.0;
        cases.push(aperture_geometry_case(step));

        let mut edge_spec = anchor.clone();
        edge_spec.aperture = aperture;
        edge_spec.components = EyeComponents {
            skin_region: true,
            upper_lid: true,
            lower_lid: true,
            canthi: true,
            ..EyeComponents::none()
        };
        let mut edge = case(
            "aperture_edge_only_control",
            &format!("edge_{step:02}"),
            "non_eye_like_control_moving_lid_edges_only",
            edge_spec,
            StereoPolicy::AnatomicalMirror,
        );
        set_x(&mut edge, "aperture", aperture);
        cases.push(edge);

        let mut fill_spec = anchor.clone();
        fill_spec.aperture = 1.0;
        fill_spec.sclera_level = 0.55 + 0.40 * step as f32 / 40.0;
        let mut fill = case(
            "aperture_fill_intensity_control",
            &format!("fill_{step:02}"),
            "fixed_geometry_sclera_intensity_control_not_area_matched",
            fill_spec,
            StereoPolicy::AnatomicalMirror,
        );
        set_x(&mut fill, "sclera_level", 0.55 + 0.40 * step as f32 / 40.0);
        cases.push(fill);
    }

    for step in 0..=10 {
        let delta = -0.25 + 0.05 * step as f32;
        let mut brightness = case(
            "global_brightness_offset",
            &format!("offset_{step:02}"),
            "eye_like_global_offset_record_and_flag_clipping",
            anchor.clone(),
            StereoPolicy::AnatomicalMirror,
        );
        brightness.photometric = PhotometricTransform::Offset { delta };
        set_x(&mut brightness, "offset", delta);
        cases.push(brightness);

        let gain = 0.50 + 0.10 * step as f32;
        let mut contrast = case(
            "global_contrast_about_mean",
            &format!("contrast_{step:02}"),
            "eye_like_mean_preserving_contrast_unless_clipped",
            anchor.clone(),
            StereoPolicy::AnatomicalMirror,
        );
        contrast.photometric = PhotometricTransform::ContrastAroundOwnMean { gain };
        set_x(&mut contrast, "contrast_gain", gain);
        cases.push(contrast);
    }

    for step in 0..=18 {
        let rotation = -45.0 + 5.0 * step as f32;
        let mut spec = anchor.clone();
        spec.rotation_deg = rotation;
        let mut rotation_case = case(
            "rotation",
            &format!("rotation_{step:02}"),
            "eye_like_canonical_rotation_sweep",
            spec,
            StereoPolicy::AnatomicalMirror,
        );
        set_x(&mut rotation_case, "rotation_deg", rotation);
        cases.push(rotation_case);
    }

    let scales = [0.80, 0.90, 1.00, 1.10, 1.20];
    for (y_index, scale_y) in scales.iter().copied().enumerate() {
        for (x_index, scale_x) in scales.iter().copied().enumerate() {
            let mut spec = anchor.clone();
            spec.scale_x = scale_x;
            spec.scale_y = scale_y;
            let mut stretch = case(
                "stretch_grid",
                &format!("stretch_y{y_index}_x{x_index}"),
                "eye_like_bounded_two_dimensional_scale_grid",
                spec,
                StereoPolicy::AnatomicalMirror,
            );
            set_x(&mut stretch, "scale_x", scale_x);
            stretch.factor_y_name = Some("scale_y".into());
            stretch.factor_y = Some(scale_y);
            cases.push(stretch);
        }
    }

    cases
}

pub fn luminance_match_cases() -> Result<(Vec<CaseDefinition>, LuminancePlan), String> {
    let prepared = luminance::prepare()?;
    let mut cases = Vec::with_capacity(
        41 + prepared.constant_mean.len()
            + prepared.same_sclera_control.len()
            + prepared.original_mean_control.len(),
    );
    for step in 0..=40 {
        cases.push(aperture_geometry_case(step));
    }
    append_prepared(
        &mut cases,
        prepared.constant_mean,
        "aperture_constant_mean",
        "geometry_plus_compensating_sclera_at_constant_whole_image_mean",
        false,
    );
    append_prepared(
        &mut cases,
        prepared.same_sclera_control,
        "fixed_geometry_same_sclera_control",
        "fixed_reference_geometry_replaying_constant_mean_suite_sclera_sequence",
        true,
    );
    append_prepared(
        &mut cases,
        prepared.original_mean_control,
        "fixed_geometry_original_mean_control",
        "fixed_reference_geometry_matching_original_aperture_whole_image_mean_via_sclera",
        true,
    );
    Ok((cases, prepared.plan))
}

fn aperture_geometry_case(step: usize) -> CaseDefinition {
    let aperture = luminance::aperture(step);
    let mut spec = SyntheticEyeSpec::default();
    spec.aperture = aperture;
    let mut geometry = case(
        "aperture_geometry",
        &format!("aperture_{step:02}"),
        "recognized_eye_like_geometry_sweep",
        spec,
        StereoPolicy::AnatomicalMirror,
    );
    set_x(&mut geometry, "aperture", aperture);
    geometry
}

fn append_prepared(
    cases: &mut Vec<CaseDefinition>,
    points: Vec<PreparedPoint>,
    experiment: &str,
    interpretation_scope: &str,
    source_label_only: bool,
) {
    for point in points {
        let index = point.record.source_aperture_index;
        let source_aperture = point.record.source_aperture;
        let mut definition = case(
            experiment,
            &format!("source_aperture_{index:02}"),
            interpretation_scope,
            point.spec,
            StereoPolicy::AnatomicalMirror,
        );
        set_x(
            &mut definition,
            if source_label_only {
                "source_aperture"
            } else {
                "aperture"
            },
            source_aperture,
        );
        definition.mean_match = Some(point.record);
        cases.push(definition);
    }
}

fn case(
    experiment: &str,
    case_name: &str,
    interpretation_scope: &str,
    left: SyntheticEyeSpec,
    stereo_policy: StereoPolicy,
) -> CaseDefinition {
    CaseDefinition {
        experiment: experiment.into(),
        case_name: case_name.into(),
        interpretation_scope: interpretation_scope.into(),
        stereo_policy,
        photometric: PhotometricTransform::Identity,
        factor_x_name: None,
        factor_x: None,
        factor_y_name: None,
        factor_y: None,
        mean_match: None,
        right: left.clone(),
        left,
    }
}

fn set_x(case: &mut CaseDefinition, name: &str, value: f32) {
    case.factor_x_name = Some(name.into());
    case.factor_x = Some(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase0_has_three_preregistered_neighboring_anchors() {
        let anchors: Vec<_> = phase0_cases()
            .into_iter()
            .filter(|case| case.experiment == "anchor_family")
            .collect();
        assert_eq!(anchors.len(), 3);
        assert_eq!(
            anchors
                .iter()
                .map(|case| case.left.aperture)
                .collect::<Vec<_>>(),
            vec![0.95, 1.0, 1.05]
        );
        for case in &anchors {
            let mut expected = SyntheticEyeSpec::default();
            expected.aperture = case.left.aperture;
            assert_eq!(case.left, expected, "anchor varies only aperture");
        }
    }

    #[test]
    fn case_names_are_unique_and_stable() {
        let mut cases = phase0_cases();
        cases.extend(milestone1_cases());
        let mut names: Vec<_> = cases
            .iter()
            .map(|case| format!("{}/{}", case.experiment, case.case_name))
            .collect();
        let original = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), original);
    }

    #[test]
    fn milestone_one_has_preregistered_bounded_suite_sizes() {
        let cases = milestone1_cases();
        assert_eq!(
            cases
                .iter()
                .filter(|case| case.experiment == "aperture_geometry")
                .count(),
            41
        );
        assert_eq!(
            cases
                .iter()
                .filter(|case| case.experiment == "global_brightness_offset")
                .count(),
            11
        );
        assert_eq!(
            cases
                .iter()
                .filter(|case| case.experiment == "rotation")
                .count(),
            19
        );
        assert_eq!(
            cases
                .iter()
                .filter(|case| case.experiment == "stretch_grid")
                .count(),
            25
        );
    }

    #[test]
    fn milestone_one_primary_sweeps_vary_only_the_declared_factors() {
        let anchor = SyntheticEyeSpec::default();
        for definition in milestone1_cases() {
            match definition.experiment.as_str() {
                "aperture_geometry" => {
                    let mut expected = anchor.clone();
                    expected.aperture = definition.factor_x.unwrap();
                    assert_eq!(definition.left, expected);
                    assert_eq!(definition.photometric, PhotometricTransform::Identity);
                }
                "global_brightness_offset" => {
                    assert_eq!(definition.left, anchor);
                    assert_eq!(
                        definition.photometric,
                        PhotometricTransform::Offset {
                            delta: definition.factor_x.unwrap()
                        }
                    );
                }
                "rotation" => {
                    let mut expected = anchor.clone();
                    expected.rotation_deg = definition.factor_x.unwrap();
                    assert_eq!(definition.left, expected);
                }
                "stretch_grid" => {
                    let mut expected = anchor.clone();
                    expected.scale_x = definition.factor_x.unwrap();
                    expected.scale_y = definition.factor_y.unwrap();
                    assert_eq!(definition.left, expected);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn luminance_cases_preserve_pairing_and_original_reference_sweep() {
        let (cases, plan) = luminance_match_cases().unwrap();
        assert_eq!(
            cases
                .iter()
                .filter(|case| case.experiment == "aperture_geometry")
                .count(),
            41
        );
        for experiment in [
            "aperture_constant_mean",
            "fixed_geometry_same_sclera_control",
        ] {
            assert_eq!(
                cases
                    .iter()
                    .filter(|case| case.experiment == experiment)
                    .count(),
                plan.selected_count
            );
        }
        let b: Vec<_> = cases
            .iter()
            .filter(|case| case.experiment == "aperture_constant_mean")
            .collect();
        let c: Vec<_> = cases
            .iter()
            .filter(|case| case.experiment == "fixed_geometry_same_sclera_control")
            .collect();
        for (b, c) in b.into_iter().zip(c) {
            assert_eq!(b.left.sclera_level.to_bits(), c.left.sclera_level.to_bits());
            assert_eq!(c.left.aperture.to_bits(), plan.reference_aperture.to_bits());
        }
    }
}
