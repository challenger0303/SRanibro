//! Purpose-labelled calibration contracts.
//!
//! Capture, scoring and committing are intentionally separate concepts.  A
//! descriptor tells the UI exactly which evidence is being recorded and which
//! parameter domain the eventual objective is allowed to change.  The marker
//! types below make that change domain part of an objective's Rust type instead
//! of relying on button text or a free-form protocol string.

use crate::geometry_calib::CapturePlan;
use std::marker::PhantomData;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParameterDomain {
    ImageGeometry = 0,
    Photometric = 1,
    EyelidEndpoints = 2,
    GazeEyelidCompensation = 3,
    WinkResponse = 4,
    BlinkTiming = 5,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DomainSet(u16);

impl DomainSet {
    pub const ALL: Self = Self((1 << 6) - 1);
    pub const NONE: Self = Self(0);

    pub const fn one(domain: ParameterDomain) -> Self {
        Self(1 << domain as u8)
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn complement(self) -> Self {
        Self(Self::ALL.0 & !self.0)
    }

    pub const fn contains(self, domain: ParameterDomain) -> bool {
        self.0 & Self::one(domain).0 != 0
    }

    pub const fn is_disjoint(self, other: Self) -> bool {
        self.0 & other.0 == 0
    }

    pub fn user_text(self) -> String {
        const DOMAINS: [(ParameterDomain, &str); 6] = [
            (ParameterDomain::ImageGeometry, "image crop and angle"),
            (ParameterDomain::Photometric, "brightness and illumination"),
            (
                ParameterDomain::EyelidEndpoints,
                "eyelid open / closed range",
            ),
            (
                ParameterDomain::GazeEyelidCompensation,
                "eyelids while looking around",
            ),
            (ParameterDomain::WinkResponse, "left / right wink"),
            (ParameterDomain::BlinkTiming, "blink timing"),
        ];
        let labels = DOMAINS
            .iter()
            .filter_map(|(domain, label)| self.contains(*domain).then_some(*label))
            .collect::<Vec<_>>();
        if labels.is_empty() {
            "nothing (diagnostic only)".to_owned()
        } else {
            labels.join(", ")
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Marker implemented only inside this module.  An objective can receive a
/// commit permit only for its declared domain.
pub trait ChangeDomain: sealed::Sealed {
    const SET: DomainSet;
}

pub enum GeometryChange {}
pub enum PhotometricChange {}
pub enum EndpointChange {}
pub enum GazeEyelidChange {}
pub enum WinkChange {}
pub enum BlinkTimingChange {}

impl sealed::Sealed for GeometryChange {}
impl sealed::Sealed for PhotometricChange {}
impl sealed::Sealed for EndpointChange {}
impl sealed::Sealed for GazeEyelidChange {}
impl sealed::Sealed for WinkChange {}
impl sealed::Sealed for BlinkTimingChange {}

impl ChangeDomain for GeometryChange {
    const SET: DomainSet = DomainSet::one(ParameterDomain::ImageGeometry);
}
impl ChangeDomain for PhotometricChange {
    const SET: DomainSet = DomainSet::one(ParameterDomain::Photometric);
}
impl ChangeDomain for EndpointChange {
    const SET: DomainSet = DomainSet::one(ParameterDomain::EyelidEndpoints);
}
impl ChangeDomain for GazeEyelidChange {
    const SET: DomainSet = DomainSet::one(ParameterDomain::GazeEyelidCompensation);
}
impl ChangeDomain for WinkChange {
    const SET: DomainSet = DomainSet::one(ParameterDomain::WinkResponse);
}
impl ChangeDomain for BlinkTimingChange {
    const SET: DomainSet = DomainSet::one(ParameterDomain::BlinkTiming);
}

/// Non-forgeable outside this module.  Later objective implementations stage a
/// value through this permit before the common backup/save/apply path accepts it.
pub struct CommitPermit<D: ChangeDomain> {
    session_id: u64,
    _domain: PhantomData<fn() -> D>,
}

impl<D: ChangeDomain> CommitPermit<D> {
    pub(crate) fn new(session_id: u64) -> Self {
        Self {
            session_id,
            _domain: PhantomData,
        }
    }

    pub fn session_id(&self) -> u64 {
        self.session_id
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SessionKind {
    SafeGeometry,
    Photometric,
    EyelidEndpoints,
    GazeDirections,
    Winks,
    NaturalBlinks,
}

impl SessionKind {
    pub const ALL: [Self; 6] = [
        Self::SafeGeometry,
        Self::Photometric,
        Self::EyelidEndpoints,
        Self::GazeDirections,
        Self::Winks,
        Self::NaturalBlinks,
    ];

    pub const fn descriptor(self) -> &'static SessionDescriptor {
        match self {
            Self::SafeGeometry => &SAFE_GEOMETRY,
            Self::Photometric => &PHOTOMETRIC,
            Self::EyelidEndpoints => &EYELID_ENDPOINTS,
            Self::GazeDirections => &GAZE_DIRECTIONS,
            Self::Winks => &WINKS,
            Self::NaturalBlinks => &NATURAL_BLINKS,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceApplicability {
    AllEyeImageDevices,
    DreamAirXr5,
    FrontalHotMirror,
}

impl DeviceApplicability {
    pub fn supports(self, device: &str) -> bool {
        let device = crate::config::canonical_device_key(device);
        match self {
            Self::AllEyeImageDevices => true,
            Self::DreamAirXr5 => device == "pimax_xr5",
            Self::FrontalHotMirror => crate::config::supports_photometric_fit(&device),
        }
    }

    pub const fn user_text(self) -> &'static str {
        match self {
            Self::AllEyeImageDevices => "all supported eye-camera HMDs",
            Self::DreamAirXr5 => "Pimax Dream Air / SE (XR5)",
            Self::FrontalHotMirror => "Pimax VR4, Varjo and other frontal Hotmirror profiles",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SessionDescriptor {
    pub kind: SessionKind,
    pub title: &'static str,
    pub checks: &'static str,
    pub after_apply: &'static str,
    pub may_change: DomainSet,
    pub never_changes: DomainSet,
    pub applicability: DeviceApplicability,
    pub estimated_seconds: f32,
    pub capture_plan: CapturePlan,
    pub protocol: &'static str,
}

const SAFE_GEOMETRY: SessionDescriptor = SessionDescriptor {
    kind: SessionKind::SafeGeometry,
    title: "Image alignment",
    checks: "Finds a crop and angle that keep eyelid motion readable.",
    after_apply: "Saves the validated XR5 crop, angle, scale and distortion for this HMD.",
    may_change: GeometryChange::SET,
    never_changes: GeometryChange::SET.complement(),
    applicability: DeviceApplicability::DreamAirXr5,
    estimated_seconds: 119.0,
    capture_plan: CapturePlan::Full,
    protocol: "safe_geometry_fit_v1",
};

const PHOTOMETRIC: SessionDescriptor = SessionDescriptor {
    kind: SessionKind::Photometric,
    title: "Lighting correction",
    checks: "Balances brightness, contrast, shadows and left/right illumination.",
    after_apply: "Saves the validated brightness and illumination correction for this HMD.",
    may_change: PhotometricChange::SET,
    never_changes: PhotometricChange::SET.complement(),
    applicability: DeviceApplicability::FrontalHotMirror,
    estimated_seconds: 119.0,
    capture_plan: CapturePlan::Full,
    protocol: "photometric_fit_v1",
};

const EYELID_ENDPOINTS: SessionDescriptor = SessionDescriptor {
    kind: SessionKind::EyelidEndpoints,
    title: "Eyelid open / closed range",
    checks: "Sets relaxed open to 100% and gentle full close to 0% for each eye.",
    after_apply:
        "Locks each validated eye's 100% / 0% range; Recenter can still update relaxed-open.",
    may_change: EndpointChange::SET,
    never_changes: EndpointChange::SET.complement(),
    applicability: DeviceApplicability::AllEyeImageDevices,
    estimated_seconds: 60.0,
    capture_plan: CapturePlan::EyelidEndpoints,
    protocol: "eyelid_endpoints_v1",
};

const GAZE_DIRECTIONS: SessionDescriptor = SessionDescriptor {
    kind: SessionKind::GazeDirections,
    title: "XR5 SRanipal side-gaze compensation",
    checks:
        "Corrects SRanipal eyelid false-closing when an XR5 user looks left, right, up or down.",
    after_apply: "Adds a validated XR5 SRanipal eyelid lift without changing Tobii or avatar gaze.",
    may_change: GazeEyelidChange::SET,
    never_changes: GazeEyelidChange::SET.complement(),
    applicability: DeviceApplicability::DreamAirXr5,
    estimated_seconds: 187.4,
    capture_plan: CapturePlan::GazeDirections,
    protocol: "gaze_eyelid_evidence_v1",
};

const WINKS: SessionDescriptor = SessionDescriptor {
    kind: SessionKind::Winks,
    title: "Left / right wink",
    checks: "Calibrates each wink and its squeeze signature while keeping the other eye open.",
    after_apply:
        "Adds a per-eye wink response, using validated squeeze corroboration when available.",
    may_change: WinkChange::SET,
    never_changes: WinkChange::SET.complement(),
    applicability: DeviceApplicability::AllEyeImageDevices,
    estimated_seconds: 49.0,
    capture_plan: CapturePlan::Winks,
    protocol: "wink_response_v1",
};

const NATURAL_BLINKS: SessionDescriptor = SessionDescriptor {
    kind: SessionKind::NaturalBlinks,
    title: "Blink timing",
    checks: "Makes fast natural blinks reach 0% before reopening.",
    after_apply: "Holds fast two-eye blinks at 0% for the validated minimum time; winks and slow closes stay unchanged.",
    may_change: BlinkTimingChange::SET,
    never_changes: BlinkTimingChange::SET.complement(),
    applicability: DeviceApplicability::AllEyeImageDevices,
    estimated_seconds: 22.0,
    capture_plan: CapturePlan::NaturalBlinks,
    protocol: "natural_blink_timing_v1",
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CalibrationView {
    #[default]
    Problems,
    FitAssist,
    Individual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecipeId {
    Initial,
    SideGazeClosesEye,
    FullCloseDoesNotReachZero,
    WinkStopsHalfOpen,
    BlinkTooShort,
}

#[derive(Clone, Copy, Debug)]
pub struct RecipeDescriptor {
    pub id: RecipeId,
    pub title: &'static str,
    pub symptom: &'static str,
    pub steps: &'static [SessionKind],
}

const INITIAL_STEPS: &[SessionKind] = &[SessionKind::EyelidEndpoints];
const SIDE_GAZE_STEPS: &[SessionKind] = &[SessionKind::GazeDirections];
const ENDPOINT_STEPS: &[SessionKind] = &[SessionKind::EyelidEndpoints];
const WINK_STEPS: &[SessionKind] = &[SessionKind::Winks];
const BLINK_STEPS: &[SessionKind] = &[SessionKind::NaturalBlinks];

pub const RECIPES: [RecipeDescriptor; 5] = [
    RecipeDescriptor {
        id: RecipeId::Initial,
        title: "Quick eyelid setup",
        symptom: "Checks only relaxed-open and gentle full-close for each eye.",
        steps: INITIAL_STEPS,
    },
    RecipeDescriptor {
        id: RecipeId::SideGazeClosesEye,
        title: "Eyes close while looking sideways",
        symptom: "Checks only gaze-dependent eyelid correction.",
        steps: SIDE_GAZE_STEPS,
    },
    RecipeDescriptor {
        id: RecipeId::FullCloseDoesNotReachZero,
        title: "Closed eyes do not reach 0%",
        symptom: "Recalibrates the open and closed eyelid range.",
        steps: ENDPOINT_STEPS,
    },
    RecipeDescriptor {
        id: RecipeId::WinkStopsHalfOpen,
        title: "Wink remains half-open",
        symptom: "Recalibrates left and right winks separately.",
        steps: WINK_STEPS,
    },
    RecipeDescriptor {
        id: RecipeId::BlinkTooShort,
        title: "Blink reopens too early",
        symptom: "Measures the minimum visible closed time.",
        steps: BLINK_STEPS,
    },
];

pub fn applicable_steps(recipe: RecipeDescriptor, device: &str) -> Vec<SessionKind> {
    recipe
        .steps
        .iter()
        .copied()
        .filter(|kind| kind.descriptor().applicability.supports(device))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_change_domains_are_isolated() {
        for kind in SessionKind::ALL {
            let descriptor = kind.descriptor();
            assert!(descriptor.may_change.is_disjoint(descriptor.never_changes));
            assert_eq!(
                descriptor.may_change.union(descriptor.never_changes),
                DomainSet::ALL
            );
        }
        assert_eq!(SAFE_GEOMETRY.may_change, GeometryChange::SET);
        assert_eq!(PHOTOMETRIC.may_change, PhotometricChange::SET);
        assert_eq!(EYELID_ENDPOINTS.may_change, EndpointChange::SET);
        assert_eq!(GAZE_DIRECTIONS.may_change, GazeEyelidChange::SET);
        assert_eq!(WINKS.may_change, WinkChange::SET);
        assert_eq!(NATURAL_BLINKS.may_change, BlinkTimingChange::SET);
    }

    #[test]
    fn initial_recipe_contains_only_open_closed_endpoints() {
        assert_eq!(INITIAL_STEPS, &[SessionKind::EyelidEndpoints]);
        let xr5 = applicable_steps(RECIPES[0], "pimax_xr5");
        let vr4 = applicable_steps(RECIPES[0], "pimax_vr4");
        assert_eq!(xr5, vec![SessionKind::EyelidEndpoints]);
        assert_eq!(vr4, vec![SessionKind::EyelidEndpoints]);
    }

    #[test]
    fn sranipal_side_gaze_compensation_is_xr5_only() {
        assert_eq!(
            applicable_steps(RECIPES[1], "pimax_xr5"),
            vec![SessionKind::GazeDirections]
        );
        assert!(applicable_steps(RECIPES[1], "pimax_vr4").is_empty());
        assert!(applicable_steps(RECIPES[1], "varjo").is_empty());
        assert!(applicable_steps(RECIPES[1], "psvr2").is_empty());
    }

    #[test]
    fn descriptors_derive_duration_and_plan_identity() {
        for kind in SessionKind::ALL {
            let descriptor = kind.descriptor();
            if kind == SessionKind::GazeDirections {
                assert_eq!(descriptor.estimated_seconds, 187.4);
                assert!(descriptor.estimated_seconds > descriptor.capture_plan.total_seconds());
            } else {
                assert_eq!(
                    descriptor.estimated_seconds,
                    descriptor.capture_plan.total_seconds()
                );
            }
            assert!(!descriptor.protocol.is_empty());
            assert!(!descriptor.after_apply.trim().is_empty());
        }
    }

    #[test]
    fn commit_permit_is_bound_to_one_marker_domain() {
        let permit = CommitPermit::<EndpointChange>::new(42);
        assert_eq!(permit.session_id(), 42);
        assert_eq!(
            EndpointChange::SET,
            DomainSet::one(ParameterDomain::EyelidEndpoints)
        );
    }
}
