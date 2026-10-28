# Standalone recorder v0.3.0

Four participant-selected groups: Wide open, Normal, Half open, Blinks. Gaze is free in every group. Each item starts explicitly, with a five-second countdown, a configurable 10–120 second capture, and its own automatic ZIP export. Stop and save retains a partial session. Pause does not advance the timer; minimizing does not pause it. The old multi-step guided sequence is no longer used by this binary. Existing normal-app calibration protocols are unchanged.

`xr5-free-capture.rs` is recorder-only. The `xr5_free_gaze_pose_recording_v1` / metadata schema 4 archive contains:

- Exact mapped Gray8 stereo camera images before ML transforms.
- `prompted_action` as recording context only; `kind=unlabelled`.
- Empty `expected_open`, `commanded_target`, and `holdout` fields.
- `split=unassigned`, not automatic training/evaluation assignment.
- `collection_group` shared by item recordings during an app run. Keep this group together when designing independent evaluations; also group by contributor across runs.
- Active time and wall elapsed time, separate generation numbers, native values/flags/receive ages, and bounded wearable data packets.

Never use the requested item as per-frame ground truth: blinking and resting are allowed in every item. Half open does not mean an exact 0.5 regression target. Existing guided-data importers must not silently reinterpret this format; use an explicit importer with unknown-state handling.

Capture buffers are bounded to 8,000 stereo samples at up to about 67/s (640 MB raw-image ceiling plus native evidence and application overhead). Both eyes must be fresh and advance before a pair is accepted. A stopped camera produces gaps, not duplicate frames or an endless wait. Low coverage is reported without discarding the available images. Failed saves retain buffers for Retry; successful saves release them. Close is blocked while capturing/saving until explicit stop/discard/completion.

The desktop UI is rebuilt around four selectable items with a compact two-column camera/setup layout and a narrow-window fallback. Recording hides the camera preview. SteamVR is optional and disabled initially. `VrResearchOverlay::new_recorder()` uses a separate smooth-font timer layout; `new()` keeps the original normal-app guide renderer. No avatar inference/output is started.

Validation performed:

- 14 recorder/native-evidence tests passed.
- 5 existing overlay regression tests passed.
- Optional service-client feature compiles; distributed build does not enable upload.
- Synthetic ten-second export retained 625 fresh stereo pairs / 1,250 native rows; invalid native values and flags preserved.
- Desktop idle and recording pages rendered and captured without claiming hardware, using `--ui-smoke-test` (and `--ui-smoke-recording` for the synthetic active view).
- Exact SteamVR texture pixels inspected via `--self-test-output`; real headset display not independently verified.

No changes to normal SRanibro settings, no application replacement, no upload, and no model deployment. Local distribution folder: `outputs/XR5-Recorder-v0.3.0` with English/Japanese instructions, restoration helper and third-party notices.

## Distribution validation

Before packaging, run the finished EXE with `--check-runtime <report-path>` under an empty temporary `LOCALAPPDATA`. Require exit code 0 and a PASS report. Also run the recorder tests, synthetic export and desktop smoke check. Runtime validation does not establish a live camera connection or tracking accuracy.
