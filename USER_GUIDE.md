# SRanibro User Guide

For **SRanibro v0.1.5-beta**. The interface may change while the app is in beta.

[日本語版 / Japanese guide](USER_GUIDE.ja.md)

SRanibro converts eye-camera data from hot-mirror VR HMDs into eye tracking for
VRCFaceTracking (VRCFT). Processing stays on your PC. The app does not include
SRanipal model weights or Tobii runtime files; you must supply copies you are
authorized to use.

## 1. Before you start

You need:

- Windows 10 or 11 x64.
- A supported HMD:
  - Pimax Crystal / Crystal Super (VR4)
  - Pimax Dream Air / SE (XR5)
  - StarVR One
  - A supported Varjo camera path
- Your own SRanipal installation or EyePrediction model weights.
- Your own compatible Tobii stream-engine runtime DLL.
- VRCFaceTracking if you want to drive a VRChat avatar.

For Pimax or Varjo, install the normal vendor software first. Complete the
vendor's gaze calibration before using SRanibro's finishing correction. This is
especially important on Dream Air / SE (XR5): SRanibro can correct a small residual
offset, but it cannot replace the headset's full gaze calibration.

## 2. Download SRanibro

Download only from the official
[SRanibro Releases page](https://github.com/challenger0303/SRanibro/releases).

The recommended download is `SRanibro-v0.1.5-beta-bundle.zip`, which contains:

- `SRanibro.exe`
- `SRanibro-VRCFT-module.zip`

The executable is not code-signed yet, so Windows SmartScreen may warn. Do not
disable Windows security. Verify the SHA-256 values shown on the Release page.

Extract the app to a folder your account can write to, such as a folder on the
Desktop. Do not put it under `Program Files`. SRanibro is portable and has no
installer. Its configuration, calibration and logs are stored under:

```text
%APPDATA%\SRanibro\
```

## 3. Install the VRCFaceTracking module

Unzip `SRanibro-VRCFT-module.zip`. It contains `SRanibro.dll`, `module.json`,
`config.json`, and a small module README.

Create this exact folder:

```text
%APPDATA%\VRCFaceTracking\CustomLibs\4d4b786f-e496-4df9-9421-dae811edff06\
```

Copy these three files into it:

```text
SRanibro.dll
module.json
config.json
```

Start SRanibro before VRCFaceTracking. The active eye module should be named
`SRanibro`.

This is an **eye-only** VRCFT module. It provides gaze, openness, pupil,
EyeWide and EyeSquint without taking the face/expression provider slot, so a
separate facial tracker can remain active.

## 4. Configure SRanibro

Run `SRanibro.exe`, open **Settings**, then expand **Connection & models**.

Set:

- **SRanipal model folder** — select the SRanipal installation folder. SRanibro
  finds the EyePrediction weights inside it.
- **Tobii runtime DLL** — select your compatible Tobii stream-engine DLL.

Then expand **Tracking & device** and select the headset path:

| Setting | Use it for |
| --- | --- |
| `auto` | Recommended for Pimax. Detects a connected VR4 or XR5 EyeChip. |
| `pimax_vr4` | Explicit Pimax Crystal / Crystal Super VR4 path. |
| `pimax_xr5` | Explicit Pimax Dream Air / SE (XR5) angled-camera path. |
| `starvr` | StarVR One through its Tobii runtime. |
| `varjo` | Native Varjo Base camera path. |
| `varjo_mjpeg` | External Varjo Eye Streamer MJPEG path. |

`auto` is for Pimax detection; choose StarVR or Varjo explicitly.

Click **Apply & reload**. Device, model, file-path and OSC changes do not take
effect until this button is pressed. Live switches and most tuning controls are
saved immediately.

Pimax direct access may show a UAC prompt while SRanibro releases the EyeChip
from the Tobii platform service. Allow it if you want SRanibro to own the camera.

## 5. Check the dashboard

The pipeline should turn green from **DEVICE** through **OUTPUT**, and both eye
images should appear.

The number such as `120/s` on a camera image is the observed camera-frame arrival
rate. The eyelid model can run at a lower rate than the camera while gaze and
output continue at their own rates.

If a stage is red, expand the pipeline or open **Console**. The first failed
stage and its current reason are more useful than changing unrelated tuning.

## 6. Connect VRCFaceTracking and VRChat

Recommended startup order:

1. Start the headset vendor software and connect the HMD.
2. Start SRanibro and wait for the pipeline to go green.
3. Start VRCFaceTracking and enable the `SRanibro` eye module.
4. Start VRChat.

SRanibro serves its local eye stream on `127.0.0.1:5555`. If VRCFT does not
connect, make sure another app is not already using TCP port 5555.

The **VRCFT openness low-pass** control in Settings is live:

- `0` or `1` sample: pass-through, least latency.
- Larger values: smoother, but add delay.

SRanibro already performs eyelid post-processing, so start with pass-through
unless your avatar still needs extra smoothing.

## 7. First eyelid calibration

Wear the HMD normally before calibrating. Do not press the headset into an
unusual position just for calibration.

1. Open **Calibration**.
2. Look straight ahead with both eyes naturally open.
3. Press **Recenter**.
4. Wait about a second, then perform two or three deliberate slow blinks.
5. Test normal blinks, a slow close, a held wink and a relaxed reopen.

Recenter learns the current relaxed-open baseline. **Adaptive blink bounds**
learns the real full-close depth separately. It is enabled by default and is
normally the better choice. If tracking becomes biased after the headset moves
on your face, reseat it normally and press Recenter again.

Calibration is stored separately for each HMD. After updating from an older
version, press Recenter once on every headset you use.

### Eye mapping

Use **Settings → Eye mapping** only when the symptom matches:

- Avatar looks right when you look left → **Flip gaze left / right**.
- The avatar's left and right eyes are swapped → **Swap left / right**.
- The eye-camera image itself is mirrored → **Flip image horizontally**.

These settings are saved per HMD.

## 8. Dream Air / SE (XR5)

The following controls are shown only while Dream Air / SE (XR5) is active.

### Gaze source and correction

Calibrate Tobii gaze in Pimax software first. If that base calibration is wrong,
SRanibro may receive plausible gaze data while the left/right alignment is still
incorrect.

The optional **EyeChip combined gaze** source is usually steadier when one eye
jitters or temporarily loses tracking. It trades some natural near-focus
vergence for stability. Changing the source requires **Apply & reload**; run
the gaze **Center** step again afterwards.

Use **Dream Air / XR5 gaze correction** only for the remaining center, range or
vergence error after the vendor calibration.

### Automatic eye image alignment / Safe Geometry Fit

The built-in XR5 geometry is already a valid fallback. Run the fit only if the
eyelid model does not respond well for the current wearer or optical position.

1. Start the guided recording and follow every OPEN, HALF, CLOSED, gaze, slow
   close and blink prompt.
2. Keep the HMD seated consistently for the full recording.
3. Run the fit after capture completes.
4. Use **Preview candidate live** and check normal blinks and slow closes.
5. If the result is `HOLDOUT PASS`, save it with **Apply validated candidate**.

`KEEP CURRENT GEOMETRY` is the automatic recommendation to retain the current
settings. You can still preview a rejected candidate. If it works better in the
headset, acknowledge the warning and choose **Apply unvalidated candidate**.
SRanibro creates a backup first, and **Rollback last applied geometry** restores
the previous settings during the same run.

Safe Geometry Fit is inside `SRanibro.exe` and does not require Python. It uses
the configured SRanipal eyelid model and does not train a new model.

The completed recording stays in memory for fitting. It is saved to disk only
when you explicitly export it. An exported ZIP contains biometric eye images;
inspect it and share it only with people you trust.

### XR5 image EyeWide

`SRanipal` uses the legacy Wide response. `Auto` uses a valid, fresh custom XR5
Wide model when available and otherwise falls back safely. `Custom` requires a
working custom model.

**Fit in app (no Python)** adapts an existing compatible Wide base model. It
cannot create a useful model without that base model.

## 9. Optional eyebrow tracking

Eyebrow tracking is optional and requires a compatible `brow.bin` base model.

- **Fit in app (no Python)** refits the existing model head from your captured
  eyebrow dataset. This is the quick per-user route.
- **Train & bake** creates a model through the external `vr_eyebrow` project and
  requires Python, PyTorch and a configured environment.

Use **Enable eyebrow tracking** as the master switch. The bundled VRCFT module is
eye-only, so enable **Send eyebrows directly to VRChat OSC** if VRCFT should keep
handling the eyes while SRanibro sends FT/v2 eyebrow parameters.

## 10. Daily use and shutdown

Minimizing the SRanibro window does not stop tracking. The UI reduces its redraw
work, while the camera, model and VRCFT output continue in the background.

Close SRanibro when you no longer need it. If a Pimax/Tobii vendor runtime cannot
reacquire the EyeChip afterwards, run:

```text
SRanibro.exe restore
```

Rebooting also restores the normal vendor-service state.

## 11. Troubleshooting and feedback

### Window closes or never appears

Read:

```text
%APPDATA%\SRanibro\sranibro.log
```

### Camera images are missing

- Confirm the correct HMD path is selected.
- Confirm the HMD is connected and awake.
- Confirm the Tobii DLL path is valid.
- Apply again and accept the UAC handoff if requested.

### Pupil arrives but gaze does not

Complete the vendor's Tobii/Pimax gaze calibration, then use **Apply & reload**.
Check Console for the wearable/gaze subscription state before changing eyelid
geometry.

### One eyelid becomes biased over time

Reseat the HMD normally and press Recenter. Leave **Adaptive blink bounds** on,
then perform two or three slow blinks. If the problem returns, record a short
diagnostic CSV with **REC** on the Calibration page.

### VRCFT does not connect

- Start SRanibro first.
- Check that the custom module folder uses the exact GUID above.
- Confirm `SRanibro.dll`, `module.json` and `config.json` are in that folder.
- Check that TCP port 5555 is free.

### Sending a report

Useful files include the relevant log tail, a short diagnostic CSV, and—only
when needed—an exported XR5 calibration recording. Logs may contain your Windows
username and local file paths. Calibration recordings contain eye imagery.
Review all files before sharing them.
