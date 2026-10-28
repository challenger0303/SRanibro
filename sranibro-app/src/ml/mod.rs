//! ML: load SRanipal's EyePrediction eyelid model from the user's SRanipal dir.
pub mod brightness;
pub mod brow_calfit;
pub mod brow_fit;
pub mod brow_net;
pub mod eye_net;
#[cfg(any(windows, target_os = "linux"))]
pub(crate) mod eye_net_gpu;
#[cfg(any(windows, target_os = "linux"))]
pub(crate) mod gpu_worker;
#[cfg(any(windows, target_os = "linux"))]
#[doc(hidden)]
pub use gpu_worker::{enable_gpu_worker_host, gpu_worker_entrypoint};
pub(crate) mod eyelid_model;
pub mod heatmap;
pub mod preprocess;
pub mod tvm_params;
pub mod wide_calfit;
pub mod wide_net;
#[cfg(any(feature = "xr5-research-model", feature = "external-eyelid-model"))]
pub mod xr5_research;
#[cfg(feature = "external-eyelid-model")]
pub mod external_eyelid;
#[cfg(feature = "external-eyelid-model")]
pub mod xr5_gaze;
#[cfg(feature = "external-eyelid-model")]
pub mod xr5_gaze_hybrid;
#[cfg(feature = "external-eyelid-model")]
pub mod xr5_gaze_origin;
