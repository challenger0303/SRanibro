//! DX12/WebGPU compute backend for the fixed SRanipal EyePrediction network.
//!
//! The user's parsed weights remain the source of truth. They are uploaded once and
//! kept resident; each live update uploads only one or two `[2,100,100]` tensors and
//! reads back five floats per tensor. The two-input path evaluates the normal tensor
//! and the mirrored/swapped right-eye tensor in one command submission.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;

use super::eye_net::{EyeNet, OUT_DIM};

const INPUT_LEN: usize = 2 * 100 * 100;
const MAX_BATCH: usize = 2;
const WORKGROUP: u32 = 64;
const SYNC_READBACK_TIMEOUT: Duration = Duration::from_secs(2);
const LIVE_READBACK_TIMEOUT: Duration = Duration::from_millis(250);
const CALLBACK_POLL_INTERVAL: Duration = Duration::from_millis(2);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum GpuFailure {
    DeviceLost = 1,
    DeviceDestroyed,
    CallbackReplaced,
    DeviceInvalid,
    OutOfMemory,
    Validation,
    ReadbackTimeout,
    ReadbackDisconnected,
    ReadbackFailed,
    InvalidOutput,
    InvalidInput,
    Internal,
}

/// Callbacks can run under wgpu 22's internal error-sink lock. Record one bounded
/// signal only: no logging, channel waits, device calls, or recovery in a callback.
#[derive(Default)]
struct GpuHealth {
    first_failure: AtomicU8,
}

impl GpuHealth {
    fn fail(&self, failure: GpuFailure) {
        let _ = self.first_failure.compare_exchange(
            0,
            failure as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    fn check(&self) -> Result<(), String> {
        let reason = match self.first_failure.load(Ordering::Acquire) {
            0 => return Ok(()),
            n if n == GpuFailure::DeviceLost as u8 => "device lost",
            n if n == GpuFailure::DeviceDestroyed as u8 => "device destroyed",
            n if n == GpuFailure::CallbackReplaced as u8 => "device health callback replaced",
            n if n == GpuFailure::DeviceInvalid as u8 => "device already invalid",
            n if n == GpuFailure::OutOfMemory as u8 => "out of memory",
            n if n == GpuFailure::Validation as u8 => "uncaptured validation error",
            n if n == GpuFailure::ReadbackTimeout as u8 => "readback deadline exceeded",
            n if n == GpuFailure::ReadbackDisconnected as u8 => "readback callback dropped",
            n if n == GpuFailure::ReadbackFailed as u8 => "readback mapping failed",
            n if n == GpuFailure::InvalidOutput as u8 => "invalid or non-finite output",
            n if n == GpuFailure::InvalidInput as u8 => "invalid or non-finite input",
            n if n == GpuFailure::Internal as u8 => "internal GPU error",
            _ => "unknown device failure",
        };
        Err(format!("GPU EyeNet disabled: {reason}"))
    }
}

/// Absolute, monotonic deadline. Repeated polls and delayed callbacks never renew
/// the budget. An expired result is rejected even if its callback is already queued.
#[derive(Clone, Copy)]
struct ReadbackDeadline(Instant);

impl ReadbackDeadline {
    fn remaining(self, now: Instant) -> Result<Duration, GpuFailure> {
        self.0
            .checked_duration_since(now)
            .filter(|remaining| !remaining.is_zero())
            .ok_or(GpuFailure::ReadbackTimeout)
    }

    fn capped(self, other: Option<Self>) -> Self {
        Self(other.map_or(self.0, |other| self.0.min(other.0)))
    }
}

struct PendingInference {
    batch: usize,
    output_bytes: u64,
    deadline: ReadbackDeadline,
    receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

const CONV1_ELEMS: usize = 20 * 96 * 96;
const POOL1_ELEMS: usize = 20 * 48 * 48;
const CONV2_ELEMS: usize = 48 * 44 * 44;
const POOL2_ELEMS: usize = 48 * 22 * 22;
const CONV3_ELEMS: usize = 64 * 20 * 20;
const FC1_ELEMS: usize = 500;

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_data: array<f32>;

var<workgroup> weight_tile: array<f32, 256>;
var<workgroup> input_tile: array<f32, 256>;
var<workgroup> partial_sum: array<f32, 64>;

@compute @workgroup_size(16, 16, 1)
fn conv1(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let batch = wid.z;
    let oc = wid.y * 16u + lid.y;
    let pos = wid.x * 16u + lid.x;
    var sum = 0.0;
    for (var tile = 0u; tile < 4u; tile = tile + 1u) {
        let weight_k = tile * 16u + lid.x;
        let input_k = tile * 16u + lid.y;
        let local = lid.y * 16u + lid.x;
        if (oc < 20u && weight_k < 50u) {
            weight_tile[local] = weights[oc * 50u + weight_k];
        } else {
            weight_tile[local] = 0.0;
        }
        if (pos < 9216u && input_k < 50u) {
            let ic = input_k / 25u;
            let kr = input_k % 25u;
            let ky = kr / 5u;
            let kx = kr % 5u;
            let oy = pos / 96u;
            let ox = pos % 96u;
            let ii = ((batch * 2u + ic) * 100u + oy + ky) * 100u + ox + kx;
            input_tile[local] = input_data[ii];
        } else {
            input_tile[local] = 0.0;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            sum = sum + weight_tile[lid.y * 16u + k] * input_tile[k * 16u + lid.x];
        }
        workgroupBarrier();
    }
    if (oc < 20u && pos < 9216u) {
        output_data[(batch * 20u + oc) * 9216u + pos] = max(sum + biases[oc], 0.0);
    }
}

@compute @workgroup_size(64)
fn pool1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let per_batch = 20u * 48u * 48u;
    let total = 2u * per_batch;
    if (idx >= total) { return; }
    let batch = idx / per_batch;
    let rem0 = idx % per_batch;
    let ch = rem0 / (48u * 48u);
    let rem1 = rem0 % (48u * 48u);
    let oy = rem1 / 48u;
    let ox = rem1 % 48u;
    let base = (batch * 20u + ch) * 96u * 96u + oy * 2u * 96u + ox * 2u;
    var value = input_data[base];
    value = max(value, input_data[base + 1u]);
    value = max(value, input_data[base + 96u]);
    value = max(value, input_data[base + 97u]);
    output_data[idx] = value;
}

@compute @workgroup_size(16, 16, 1)
fn conv2(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let batch = wid.z;
    let oc = wid.y * 16u + lid.y;
    let pos = wid.x * 16u + lid.x;
    var sum = 0.0;
    for (var tile = 0u; tile < 32u; tile = tile + 1u) {
        let weight_k = tile * 16u + lid.x;
        let input_k = tile * 16u + lid.y;
        let local = lid.y * 16u + lid.x;
        if (oc < 48u && weight_k < 500u) {
            weight_tile[local] = weights[oc * 500u + weight_k];
        } else {
            weight_tile[local] = 0.0;
        }
        if (pos < 1936u && input_k < 500u) {
            let ic = input_k / 25u;
            let kr = input_k % 25u;
            let ky = kr / 5u;
            let kx = kr % 5u;
            let oy = pos / 44u;
            let ox = pos % 44u;
            let ii = ((batch * 20u + ic) * 48u + oy + ky) * 48u + ox + kx;
            input_tile[local] = input_data[ii];
        } else {
            input_tile[local] = 0.0;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            sum = sum + weight_tile[lid.y * 16u + k] * input_tile[k * 16u + lid.x];
        }
        workgroupBarrier();
    }
    if (oc < 48u && pos < 1936u) {
        output_data[(batch * 48u + oc) * 1936u + pos] = max(sum + biases[oc], 0.0);
    }
}

@compute @workgroup_size(64)
fn pool2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let per_batch = 48u * 22u * 22u;
    let total = 2u * per_batch;
    if (idx >= total) { return; }
    let batch = idx / per_batch;
    let rem0 = idx % per_batch;
    let ch = rem0 / (22u * 22u);
    let rem1 = rem0 % (22u * 22u);
    let oy = rem1 / 22u;
    let ox = rem1 % 22u;
    let base = (batch * 48u + ch) * 44u * 44u + oy * 2u * 44u + ox * 2u;
    var value = input_data[base];
    value = max(value, input_data[base + 1u]);
    value = max(value, input_data[base + 44u]);
    value = max(value, input_data[base + 45u]);
    output_data[idx] = value;
}

@compute @workgroup_size(16, 16, 1)
fn conv3(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let batch = wid.z;
    let oc = wid.y * 16u + lid.y;
    let pos = wid.x * 16u + lid.x;
    var sum = 0.0;
    for (var tile = 0u; tile < 27u; tile = tile + 1u) {
        let weight_k = tile * 16u + lid.x;
        let input_k = tile * 16u + lid.y;
        let local = lid.y * 16u + lid.x;
        if (oc < 64u && weight_k < 432u) {
            weight_tile[local] = weights[oc * 432u + weight_k];
        } else {
            weight_tile[local] = 0.0;
        }
        if (pos < 400u && input_k < 432u) {
            let ic = input_k / 9u;
            let kr = input_k % 9u;
            let ky = kr / 3u;
            let kx = kr % 3u;
            let oy = pos / 20u;
            let ox = pos % 20u;
            let ii = ((batch * 48u + ic) * 22u + oy + ky) * 22u + ox + kx;
            input_tile[local] = input_data[ii];
        } else {
            input_tile[local] = 0.0;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            sum = sum + weight_tile[lid.y * 16u + k] * input_tile[k * 16u + lid.x];
        }
        workgroupBarrier();
    }
    if (oc < 64u && pos < 400u) {
        output_data[(batch * 64u + oc) * 400u + pos] = max(sum + biases[oc], 0.0);
    }
}

@compute @workgroup_size(64)
fn fc1(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = wid.x;
    let total = 2u * 500u;
    if (idx >= total) { return; }
    let batch = idx / 500u;
    let oc = idx % 500u;
    var sum = 0.0;
    for (var i = lid.x; i < 25600u; i = i + 64u) {
        sum = sum + input_data[batch * 25600u + i] * weights[oc * 25600u + i];
    }
    partial_sum[lid.x] = sum;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) {
            partial_sum[lid.x] = partial_sum[lid.x] + partial_sum[lid.x + stride];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        output_data[idx] = max(partial_sum[0] + biases[oc], 0.0);
    }
}

@compute @workgroup_size(64)
fn fc2(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = wid.x;
    let total = 2u * 5u;
    if (idx >= total) { return; }
    let batch = idx / 5u;
    let oc = idx % 5u;
    var sum = 0.0;
    for (var i = lid.x; i < 500u; i = i + 64u) {
        sum = sum + input_data[batch * 500u + i] * weights[oc * 500u + i];
    }
    partial_sum[lid.x] = sum;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) {
            partial_sum[lid.x] = partial_sum[lid.x] + partial_sum[lid.x + stride];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        output_data[idx] = max(partial_sum[0] + biases[oc], 0.0);
    }
}
"#;

#[derive(Clone, Copy)]
enum Dispatch {
    Linear(usize),
    Conv { positions: usize, channels: usize },
    Dense(usize),
}

struct Stage {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    dispatch: Dispatch,
}

/// Fixed-network GPU executor. It is used only behind the validated hybrid model in
/// `eyelid_model`; errors are returned so the caller can keep tracking on CPU.
pub(crate) struct GpuEyeNet {
    // Production GPU work is isolated from the GUI process. Arc also permits the
    // diagnostic from_device path without duplicating resource-building code.
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
    stages: Vec<Stage>,
    upload: Vec<f32>,
    adapter_name: String,
    pending: Option<PendingInference>,
    health: Arc<GpuHealth>,
}

impl GpuEyeNet {
    pub(crate) fn new(net: &EyeNet) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: if cfg!(target_os = "linux") {
                wgpu::Backends::VULKAN
            } else {
                wgpu::Backends::DX12
            },
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or_else(|| "no supported GPU compute adapter available".to_string())?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let required_weight_bytes = (500usize * 25_600 * 4) as u64;
        if (limits.max_storage_buffer_binding_size as u64) < required_weight_bytes {
            return Err(format!(
                "GPU storage-buffer limit is {} MiB; EyeNet needs {} MiB",
                limits.max_storage_buffer_binding_size as u64 / (1024 * 1024),
                required_weight_bytes / (1024 * 1024)
            ));
        }
        let mut requested_limits = wgpu::Limits::downlevel_defaults();
        requested_limits.max_storage_buffer_binding_size = limits
            .max_storage_buffer_binding_size
            .min(128 * 1024 * 1024);
        requested_limits.max_buffer_size = limits.max_buffer_size.min(256 * 1024 * 1024);
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("SRanibro EyeNet compute device"),
                required_features: wgpu::Features::empty(),
                required_limits: requested_limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|error| format!("GPU compute device creation failed: {error}"))?;
        Self::from_device(net, Arc::new(device), Arc::new(queue), info.name)
    }

    /// Build EyeNet resources on an externally supplied device/queue for diagnostics.
    /// This executor takes ownership of the device's health callbacks. Do not pass a
    /// GUI device whose callback policy must remain independent. Production uses a
    /// private device inside the supervised GPU worker, not the renderer's device.
    pub(crate) fn from_device(
        net: &EyeNet,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        adapter_name: String,
    ) -> Result<Self, String> {
        let health = Arc::new(GpuHealth::default());
        let error_health = Arc::clone(&health);
        device.on_uncaptured_error(Box::new(move |error| {
            error_health.fail(match error {
                wgpu::Error::OutOfMemory { .. } => GpuFailure::OutOfMemory,
                wgpu::Error::Validation { .. } => GpuFailure::Validation,
                wgpu::Error::Internal { .. } => GpuFailure::Internal,
            });
        }));
        let lost_health = Arc::clone(&health);
        device.set_device_lost_callback(move |reason, _message| {
            lost_health.fail(match reason {
                wgpu::DeviceLostReason::Unknown => GpuFailure::DeviceLost,
                wgpu::DeviceLostReason::Destroyed | wgpu::DeviceLostReason::Dropped => {
                    GpuFailure::DeviceDestroyed
                }
                wgpu::DeviceLostReason::ReplacedCallback => GpuFailure::CallbackReplaced,
                wgpu::DeviceLostReason::DeviceInvalid => GpuFailure::DeviceInvalid,
            });
        });
        health.check()?;
        let limits = device.limits();
        let required_weight_bytes = (500usize * 25_600 * 4) as u64;
        if (limits.max_storage_buffer_binding_size as u64) < required_weight_bytes {
            return Err(format!(
                "shared GPU storage-buffer limit is {} MiB; EyeNet needs {} MiB",
                limits.max_storage_buffer_binding_size as u64 / (1024 * 1024),
                required_weight_bytes / (1024 * 1024)
            ));
        }

        let weights = net.weights();
        let input = empty_buffer(
            &device,
            "EyeNet input batch",
            MAX_BATCH * INPUT_LEN,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let scratch_a = empty_buffer(
            &device,
            "EyeNet activation A",
            MAX_BATCH * CONV1_ELEMS.max(CONV2_ELEMS).max(CONV3_ELEMS),
            wgpu::BufferUsages::STORAGE,
        );
        let scratch_b = empty_buffer(
            &device,
            "EyeNet activation B",
            MAX_BATCH * POOL1_ELEMS.max(POOL2_ELEMS).max(FC1_ELEMS),
            wgpu::BufferUsages::STORAGE,
        );
        let output = empty_buffer(
            &device,
            "EyeNet output",
            MAX_BATCH * OUT_DIM,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let readback = empty_buffer(
            &device,
            "EyeNet output readback",
            MAX_BATCH * OUT_DIM,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let dummy = init_buffer(&device, "EyeNet unused binding", &[0.0]);

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("EyeNet stage bindings"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, true),
                storage_layout_entry(2, true),
                storage_layout_entry(3, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("EyeNet compute pipeline layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("EyeNet fixed compute shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let c1w = init_buffer(&device, "EyeNet conv1 weights", weights.conv1_w);
        let c1b = init_buffer(&device, "EyeNet conv1 bias", weights.conv1_b);
        let c2w = init_buffer(&device, "EyeNet conv2 weights", weights.conv2_w);
        let c2b = init_buffer(&device, "EyeNet conv2 bias", weights.conv2_b);
        let c3w = init_buffer(&device, "EyeNet conv3 weights", weights.conv3_w);
        let c3b = init_buffer(&device, "EyeNet conv3 bias", weights.conv3_b);
        let f1w = init_buffer(&device, "EyeNet fc1 weights", weights.fc1_w);
        let f1b = init_buffer(&device, "EyeNet fc1 bias", weights.fc1_b);
        let f2w = init_buffer(&device, "EyeNet fc2 weights", weights.fc2_w);
        let f2b = init_buffer(&device, "EyeNet fc2 bias", weights.fc2_b);

        let specs = [
            (
                "conv1",
                &input,
                &c1w,
                &c1b,
                &scratch_a,
                Dispatch::Conv {
                    positions: 96 * 96,
                    channels: 20,
                },
            ),
            (
                "pool1",
                &scratch_a,
                &dummy,
                &dummy,
                &scratch_b,
                Dispatch::Linear(POOL1_ELEMS),
            ),
            (
                "conv2",
                &scratch_b,
                &c2w,
                &c2b,
                &scratch_a,
                Dispatch::Conv {
                    positions: 44 * 44,
                    channels: 48,
                },
            ),
            (
                "pool2",
                &scratch_a,
                &dummy,
                &dummy,
                &scratch_b,
                Dispatch::Linear(POOL2_ELEMS),
            ),
            (
                "conv3",
                &scratch_b,
                &c3w,
                &c3b,
                &scratch_a,
                Dispatch::Conv {
                    positions: 20 * 20,
                    channels: 64,
                },
            ),
            (
                "fc1",
                &scratch_a,
                &f1w,
                &f1b,
                &scratch_b,
                Dispatch::Dense(FC1_ELEMS),
            ),
            (
                "fc2",
                &scratch_b,
                &f2w,
                &f2b,
                &output,
                Dispatch::Dense(OUT_DIM),
            ),
        ];
        let mut stages = Vec::with_capacity(specs.len());
        for (entry, source, stage_weights, bias, destination, dispatch) in specs {
            health.check()?;
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: entry,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(entry),
                layout: &bind_layout,
                entries: &[
                    whole_buffer_entry(0, source),
                    whole_buffer_entry(1, stage_weights),
                    whole_buffer_entry(2, bias),
                    whole_buffer_entry(3, destination),
                ],
            });
            stages.push(Stage {
                pipeline,
                bind_group,
                dispatch,
            });
        }
        health.check()?;

        Ok(Self {
            device,
            queue,
            input,
            output,
            readback,
            stages,
            upload: vec![0.0; MAX_BATCH * INPUT_LEN],
            adapter_name,
            pending: None,
            health,
        })
    }

    pub(crate) fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub(crate) fn infer_batch(
        &mut self,
        primary: &[f32],
        secondary: Option<&[f32]>,
    ) -> Result<[[f32; OUT_DIM]; MAX_BATCH], String> {
        self.check_health()?;
        let deadline = ReadbackDeadline(Instant::now() + SYNC_READBACK_TIMEOUT);
        // Synchronous diagnostics may interrupt the live pipelined path. Finish and
        // discard its older sample before reusing the single readback buffer. Both
        // draining and this submission share one deadline, not two renewed waits.
        if self.pending.is_some() {
            let _ = self.finish_pending(true, Some(deadline))?;
        }
        self.submit_batch(primary, secondary, deadline)?;
        self.finish_pending(true, Some(deadline))?
            .map(|(_, result)| result)
            .ok_or_else(|| "GPU EyeNet synchronous readback did not complete".to_string())
    }

    /// Queue diagnostic live inference with one bounded readback in flight.
    ///
    /// The first call enqueues work. Later calls publish the preceding result and
    /// immediately enqueue the newest input. This adds one model tick of latency but
    /// bounds callback waiting on each tick. Production uses the supervised worker
    /// instead, so fatal native-driver failures cannot abort the tracking process.
    pub(crate) fn infer_live_batch(
        &mut self,
        primary: &[f32],
        secondary: Option<&[f32]>,
    ) -> Result<Option<[[f32; OUT_DIM]; MAX_BATCH]>, String> {
        self.check_health()?;
        if validate_input_batch(primary, secondary).is_err() {
            return self.fail(GpuFailure::InvalidInput);
        }
        let requested_batch = if secondary.is_some() { 2 } else { 1 };
        if self.pending.is_none() {
            self.submit_batch(
                primary,
                secondary,
                ReadbackDeadline(Instant::now() + LIVE_READBACK_TIMEOUT),
            )?;
            return Ok(None);
        }
        let Some((completed_batch, result)) = self.finish_pending(false, None)? else {
            return Ok(None);
        };
        self.submit_batch(
            primary,
            secondary,
            ReadbackDeadline(Instant::now() + LIVE_READBACK_TIMEOUT),
        )?;
        if completed_batch == requested_batch {
            Ok(Some(result))
        } else {
            // A live routing toggle changed the batch shape. Do not reinterpret an old
            // single result as an A/B pair, or an old pair as the new single sample.
            Ok(None)
        }
    }

    fn submit_batch(
        &mut self,
        primary: &[f32],
        secondary: Option<&[f32]>,
        deadline: ReadbackDeadline,
    ) -> Result<(), String> {
        self.check_health()?;
        if let Err(failure) = deadline.remaining(Instant::now()) {
            return self.fail(failure);
        }
        if self.pending.is_some() {
            return Err("GPU EyeNet already has an inference in flight".into());
        }
        let batch = match validate_input_batch(primary, secondary) {
            Ok(batch) => batch,
            Err(failure) => return self.fail(failure),
        };
        self.upload[..INPUT_LEN].copy_from_slice(primary);
        if let Some(secondary) = secondary {
            self.upload[INPUT_LEN..2 * INPUT_LEN].copy_from_slice(secondary);
        }
        self.queue.write_buffer(
            &self.input,
            0,
            bytemuck::cast_slice(&self.upload[..batch * INPUT_LEN]),
        );
        self.check_health()?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("EyeNet inference"),
            });
        for stage in &self.stages {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("EyeNet stage"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&stage.pipeline);
            pass.set_bind_group(0, &stage.bind_group, &[]);
            match stage.dispatch {
                Dispatch::Linear(elements_per_batch) => {
                    let elements = batch * elements_per_batch;
                    pass.dispatch_workgroups((elements as u32).div_ceil(WORKGROUP), 1, 1);
                }
                Dispatch::Conv {
                    positions,
                    channels,
                } => pass.dispatch_workgroups(
                    (positions as u32).div_ceil(16),
                    (channels as u32).div_ceil(16),
                    batch as u32,
                ),
                Dispatch::Dense(outputs_per_batch) => {
                    pass.dispatch_workgroups((batch * outputs_per_batch) as u32, 1, 1);
                }
            }
        }
        let output_bytes = (batch * OUT_DIM * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.readback, 0, output_bytes);
        self.check_health()?;
        if let Err(failure) = deadline.remaining(Instant::now()) {
            return self.fail(failure);
        }
        self.queue.submit(Some(encoder.finish()));
        self.check_health()?;
        if let Err(failure) = deadline.remaining(Instant::now()) {
            return self.fail(failure);
        }

        let slice = self.readback.slice(..output_bytes);
        let (sender, receiver) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // A retired request owns no receiver. A late callback is discarded and
            // never waits on the tracking thread or retains a device/queue handle.
            let _ = sender.try_send(result);
        });
        self.pending = Some(PendingInference {
            batch,
            output_bytes,
            deadline,
            receiver,
        });
        self.check_health()
    }

    fn finish_pending(
        &mut self,
        wait: bool,
        operation_deadline: Option<ReadbackDeadline>,
    ) -> Result<Option<(usize, [[f32; OUT_DIM]; MAX_BATCH])>, String> {
        self.check_health()?;
        let Some(pending) = self.pending.as_ref() else {
            return Ok(None);
        };
        let deadline = pending.deadline.capped(operation_deadline);
        loop {
            self.check_health()?;
            if let Err(failure) = deadline.remaining(Instant::now()) {
                return self.fail(failure);
            }
            // Never ask the driver to wait for a submission. The external worker
            // supervisor is still needed: even Poll/submit/drop can stall inside a
            // broken native driver and cannot be interrupted by this Rust deadline.
            self.device.poll(wgpu::Maintain::Poll);
            self.check_health()?;
            let remaining = match deadline.remaining(Instant::now()) {
                Ok(remaining) => remaining,
                Err(failure) => return self.fail(failure),
            };
            let Some(pending) = self.pending.as_ref() else {
                return self.fail(GpuFailure::ReadbackDisconnected);
            };
            let callback = pending
                .receiver
                .recv_timeout(CALLBACK_POLL_INTERVAL.min(remaining));
            self.check_health()?;
            // Reject a completion delivered after the total deadline; it must not
            // become a fresh eye sample simply because the channel became readable.
            if let Err(failure) = deadline.remaining(Instant::now()) {
                return self.fail(failure);
            }
            match callback {
                Ok(Ok(())) => break,
                Ok(Err(_)) => return self.fail(GpuFailure::ReadbackFailed),
                Err(mpsc::RecvTimeoutError::Timeout) if !wait => return Ok(None),
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return self.fail(GpuFailure::ReadbackDisconnected)
                }
            }
        }

        let Some(pending) = self.pending.take() else {
            return self.fail(GpuFailure::ReadbackDisconnected);
        };
        let slice = self.readback.slice(..pending.output_bytes);
        let mapped = slice.get_mapped_range();
        let result = bytemuck::try_cast_slice::<u8, f32>(&mapped)
            .map_err(|_| GpuFailure::InvalidOutput)
            .and_then(|values| decode_output(values, pending.batch));
        drop(mapped);
        self.readback.unmap();
        self.check_health()?;
        if let Err(failure) = deadline.remaining(Instant::now()) {
            return self.fail(failure);
        }
        match result {
            Ok(result) => Ok(Some((pending.batch, result))),
            Err(failure) => self.fail(failure),
        }
    }

    fn check_health(&mut self) -> Result<(), String> {
        let result = self.health.check();
        if result.is_err() {
            // Invalidate the receiver without calling back into an unhealthy driver.
            // At most one map, one channel slot and the fixed buffers remain until
            // teardown. This executor can never accept another submission afterward.
            self.pending.take();
        }
        result
    }

    fn fail<T>(&mut self, failure: GpuFailure) -> Result<T, String> {
        self.health.fail(failure);
        self.pending.take();
        Err(self
            .health
            .check()
            .err()
            .unwrap_or_else(|| "GPU EyeNet disabled after a failure".to_string()))
    }
}

fn validate_input_batch(primary: &[f32], secondary: Option<&[f32]>) -> Result<usize, GpuFailure> {
    let valid =
        |input: &[f32]| input.len() == INPUT_LEN && input.iter().all(|value| value.is_finite());
    if !valid(primary) || secondary.is_some_and(|input| !valid(input)) {
        return Err(GpuFailure::InvalidInput);
    }
    Ok(if secondary.is_some() { 2 } else { 1 })
}

fn decode_output(values: &[f32], batch: usize) -> Result<[[f32; OUT_DIM]; MAX_BATCH], GpuFailure> {
    if !(1..=MAX_BATCH).contains(&batch)
        || values.len() != batch * OUT_DIM
        || !values.iter().all(|value| value.is_finite())
    {
        return Err(GpuFailure::InvalidOutput);
    }
    let mut result = [[0.0; OUT_DIM]; MAX_BATCH];
    for (output, values) in result.iter_mut().zip(values.chunks_exact(OUT_DIM)) {
        output.copy_from_slice(values);
    }
    Ok(result)
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn whole_buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn init_buffer(device: &wgpu::Device, label: &str, values: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn empty_buffer(
    device: &wgpu::Device,
    label: &str,
    elements: usize,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (elements * std::mem::size_of::<f32>()) as u64,
        usage,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_live_polls_do_not_renew_total_deadline() {
        let start = Instant::now();
        let deadline = ReadbackDeadline(start + LIVE_READBACK_TIMEOUT);
        for tick in 0..250 {
            assert_eq!(
                deadline.remaining(start + Duration::from_millis(tick)),
                Ok(Duration::from_millis(250 - tick))
            );
        }
        assert_eq!(
            deadline.remaining(start + LIVE_READBACK_TIMEOUT),
            Err(GpuFailure::ReadbackTimeout)
        );
        assert_eq!(
            deadline.remaining(start + Duration::from_secs(60)),
            Err(GpuFailure::ReadbackTimeout)
        );
    }

    #[test]
    fn synchronous_drain_and_submission_share_one_budget() {
        let start = Instant::now();
        let operation = ReadbackDeadline(start + SYNC_READBACK_TIMEOUT);
        let pending = ReadbackDeadline(start + Duration::from_secs(20));
        assert_eq!(pending.capped(Some(operation)).0, operation.0);
        let earlier = ReadbackDeadline(start + LIVE_READBACK_TIMEOUT);
        assert_eq!(earlier.capped(Some(operation)).0, earlier.0);
        // Spending time draining does not grant the next submission another 2 s.
        assert_eq!(
            operation.remaining(start + Duration::from_millis(1950)),
            Ok(Duration::from_millis(50))
        );
        assert_eq!(
            operation.remaining(start + SYNC_READBACK_TIMEOUT),
            Err(GpuFailure::ReadbackTimeout)
        );
    }

    #[test]
    fn callback_error_signal_is_sticky_and_keeps_first_cause() {
        let health = GpuHealth::default();
        assert!(health.check().is_ok());
        health.fail(GpuFailure::ReadbackTimeout);
        for _ in 0..10_000 {
            health.fail(GpuFailure::DeviceLost);
            health.fail(GpuFailure::DeviceDestroyed);
        }
        assert_eq!(
            health.first_failure.load(Ordering::Acquire),
            GpuFailure::ReadbackTimeout as u8
        );
        assert!(health.check().unwrap_err().contains("deadline exceeded"));
        assert_eq!(std::mem::size_of::<GpuHealth>(), 1);
    }

    #[test]
    fn callback_has_bounded_nonblocking_delivery_and_late_result_is_discarded() {
        let (sender, receiver) = mpsc::sync_channel(1);
        assert!(sender.try_send(Ok::<(), ()>(())).is_ok());
        assert!(matches!(
            sender.try_send(Ok(())),
            Err(mpsc::TrySendError::Full(_))
        ));
        drop(receiver);
        assert!(matches!(
            sender.try_send(Ok(())),
            Err(mpsc::TrySendError::Disconnected(_))
        ));
    }

    #[test]
    fn a_queued_callback_does_not_override_an_expired_deadline() {
        let now = Instant::now();
        let deadline = ReadbackDeadline(now);
        let (sender, receiver) = mpsc::sync_channel(1);
        sender.try_send(Ok::<(), ()>(())).unwrap();
        // Production checks the total deadline even after receiving a completion.
        assert!(receiver.try_recv().unwrap().is_ok());
        assert_eq!(deadline.remaining(now), Err(GpuFailure::ReadbackTimeout));
    }

    #[test]
    fn lost_validation_and_normal_destruction_have_distinct_reasons() {
        for (failure, expected) in [
            (GpuFailure::DeviceLost, "device lost"),
            (GpuFailure::DeviceDestroyed, "device destroyed"),
            (GpuFailure::CallbackReplaced, "callback replaced"),
            (GpuFailure::DeviceInvalid, "already invalid"),
            (GpuFailure::OutOfMemory, "out of memory"),
            (GpuFailure::Validation, "validation error"),
            (GpuFailure::Internal, "internal GPU error"),
            (GpuFailure::ReadbackDisconnected, "callback dropped"),
            (GpuFailure::ReadbackFailed, "mapping failed"),
        ] {
            let health = GpuHealth::default();
            health.fail(failure);
            assert!(health.check().unwrap_err().contains(expected));
        }
    }

    #[test]
    fn canonical_gpu_inputs_require_correct_shape_and_finite_values() {
        let valid = vec![0.3; INPUT_LEN];
        assert_eq!(validate_input_batch(&valid, None), Ok(1));
        assert_eq!(validate_input_batch(&valid, Some(&valid)), Ok(2));
        assert_eq!(
            validate_input_batch(&valid[..INPUT_LEN - 1], None),
            Err(GpuFailure::InvalidInput)
        );
        assert_eq!(
            validate_input_batch(&valid, Some(&valid[..INPUT_LEN - 1])),
            Err(GpuFailure::InvalidInput)
        );
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad = valid.clone();
            bad[INPUT_LEN - 1] = value;
            assert_eq!(
                validate_input_batch(&bad, None),
                Err(GpuFailure::InvalidInput)
            );
            assert_eq!(
                validate_input_batch(&valid, Some(&bad)),
                Err(GpuFailure::InvalidInput)
            );
        }
    }

    #[test]
    fn gpu_output_validation_preserves_asymmetric_mapping_and_raw_finite_bits() {
        let values = [-0.0, 0.21, 0.32, 0.43, 0.54, 0.65, 0.76, 0.87, 0.98, 1.09];
        let output = decode_output(&values, 2).unwrap();
        for (actual, expected) in output.iter().flatten().zip(values) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        let single = decode_output(&values[..OUT_DIM], 1).unwrap();
        assert_eq!(single[1], [0.0; OUT_DIM]);
        assert_eq!(single[0][0].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn invalid_gpu_output_is_rejected_without_slicing_panic() {
        for batch in [0, 3, usize::MAX] {
            assert_eq!(decode_output(&[], batch), Err(GpuFailure::InvalidOutput));
        }
        assert_eq!(
            decode_output(&[0.0; OUT_DIM - 1], 1),
            Err(GpuFailure::InvalidOutput)
        );
        assert_eq!(
            decode_output(&[0.0; 2 * OUT_DIM], 1),
            Err(GpuFailure::InvalidOutput)
        );
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad = [0.0; 2 * OUT_DIM];
            bad[2 * OUT_DIM - 1] = value;
            assert_eq!(decode_output(&bad, 2), Err(GpuFailure::InvalidOutput));
        }
    }
}
