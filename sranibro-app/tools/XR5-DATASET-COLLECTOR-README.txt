SRanibro XR5 Dataset Collector
==============================

Recorder v0.2.0 update
---------------------
The standalone recorder now waits for Connect / reconnect cameras before any
hardware handoff, uses isolated default mapping (no normal app settings), and
uses the Glow renderer. Only a detected Dream Air / XR5 is accepted.

Every accepted stereo image has two native_evidence.csv rows, keyed by index
and eye. Numeric openness, exact f32 bits, reported/valid flags, receive ages
and per-openness timestamps are independent fields. Disable/invalid is never
silently converted into numeric zero in this sidecar. The original wearable
data message (stream 1285 only) is also attached when available so its decoder
can be audited later. These messages contain no authentication/control traffic.
The existing adapter's validity interpretation is not itself ground truth.

This distribution does not enable the optional upload service. Share ZIPs
manually. Images and native signals are not guaranteed exposure-synchronized.
Static phases record about 20Hz; motion phases at up to about 67Hz.
Normal app settings and executable are not modified.

Purpose
-------
This standalone application records labelled Dream Air / XR5 stereo eye-camera
frames for SRanibro's native eyelid model. It does not run VRCFT/OSC output,
train a model, change gaze calibration, or apply any SRanibro calibration.

Before recording
----------------
1. Connect and wear a Pimax Dream Air / SE (XR5).
2. Complete the normal Pimax/Tobii gaze calibration if it is available.
3. Start SteamVR so the large in-headset guide can be displayed.
4. Close normal SRanibro before opening the collector. Only one application can
   own the eye camera at a time.

Recording
---------
1. Confirm both LEFT and RIGHT previews are live.
2. Choose the current fit condition.
3. Read and accept the biometric-data notice.
4. Press Prepare recording, then Begin.
5. Follow each instruction. Manual explanation screens wait for Continue/Space;
   they do not advance while you are reading.
6. Do not move the HMD during one session. Move only your eyes for gaze targets.
7. Wait for the ZIP save to finish before closing the application.

Output and privacy
------------------
The completed ZIP is saved under:

  Desktop\SRanibro XR5 Dataset

It contains raw infrared images of both eyes and is biometric data. Nothing is
uploaded automatically. An official service-enabled build shows a separate
Send recording action only after the local ZIP is safely written. That action
requires a second explicit consent and uses HTTPS. A private upload receipt is
saved next to the ZIP so the contribution can be tracked or deleted later.

The collector contains no local training, Python training environment, model
fitting, or model activation code. Uploading places evidence in quarantine;
only the maintainer can review it and use it in an offline training run.

If online contribution is not configured, share the ZIP manually only when the
wearer agreed that it may be used for SRanibro eyelid-model development.

The service-enabled collector creates one random pseudonymous contributor ID
on this PC. It is not derived from a real name or EyeChip serial, and lets the
maintainer keep every person's train/validation/test split separate.

For generalization research, a useful pair is:

- one session in the best normal fit;
- one later session after removing and reseating the HMD.

Developer commands
------------------

  cargo run --features xr5-only --bin sranibro-xr5-dataset-collector -- --mock
  cargo test --features xr5-only --bin sranibro-xr5-dataset-collector
  cargo build --release --features xr5-only --bin sranibro-xr5-dataset-collector

Hardware-free validation of the packaged executable:

  SRanibro-XR5-Recorder.exe --self-test-output <empty-test-directory>
  SRanibro-XR5-Recorder.exe --ui-smoke-test <test-report-path>

The first writes a SIMULATION_DO_NOT_IMPORT ZIP using the complete protocol.
The second draws the unconnected UI for two seconds and closes itself.

Service-client validation build:

  cargo test --features xr5-only,xr5-model-service-client --bin sranibro-xr5-dataset-collector

An official release embeds SRANIBRO_MODEL_SERVICE_URL at compile time. Only a
debug build accepts --service-url, and insecure HTTP is restricted to loopback.

The --mock mode exists only for UI testing. Its filenames contain
SIMULATION_DO_NOT_IMPORT, and the Python importer rejects its metadata.
