# SRanibro User Guide

For **v0.1.10-beta on Windows: Hotmirror, PSVR2 and Dream Air / XR5**.

[日本語版 / Japanese guide](USER_GUIDE.ja.md) · [Download](https://github.com/challenger0303/SRanibro/releases/tag/v0.1.10-beta)

SRanibro processes eye-camera images from supported VR headsets and sends eyelid openness, EyeWide and, where supported, EyeSquint to VRCFaceTracking (VRCFT). It also handles gaze and pupil data. Eyelid inference runs locally on your PC.

You do not need a long calibration sequence to get started. Wear the headset normally, check tracking, and adjust image brightness and the open/closed handles only if needed. Wearing Memory is optional: use it after you have a setup that works well.

## 1. Before you start

- Windows 10 or 11 x64.
- A supported headset and its normal vendor software:
  - Pimax Crystal / Crystal Super (VR4)
  - StarVR One
  - Varjo, through a supported camera path
  - PlayStation VR2 through PSVR2Toolkit
  - Pimax Dream Air / SE (XR5, beta)
- A SRanipal installation and EyePrediction model you are entitled to use, unless using the bundled XR5 native model.
- VRCFaceTracking and a compatible avatar for VRChat output.

**Dream Air / SE (XR5):** choose **XR5 native model** under **Tracking & device → XR5 tracking**. It provides openness and EyeWide; Squeeze is not supported. No Python is needed. The older **SRanipal + image transform** path remains an option and requires SRanipal weights. Safe Geometry Fit is not needed for the native model.

Complete the headset vendor's gaze calibration first. SRanibro's Recenter sets an eyelid reference; it is not the same as calibrating gaze in the vendor software.

PSVR2 requires PSVR2Toolkit and a running SteamVR session. Installing the PlayStation VR2 App alone does not install the Toolkit functionality.

## 2. Download and extract

Download **`SRanibro-v0.1.10-beta-Windows.zip`** from the official release and extract it to a writable folder.

- `SRanibro.exe` — application
- `eyebrow.bin` — generic eyebrow model
- `SRanibro-VRCFT-module.zip` — VRCFT module
- `models/` — XR5 native eyelid model
- `model-runtime/` — native model inference dependencies
- `licenses/` — dependency notices
- `README.txt` — quick instructions

Keep `eyebrow.bin` beside the executable. If you have already selected a personal eyebrow model, that model takes priority.

Keep `models` and `model-runtime` beside the EXE, including when updating. A blank XR5 model selection uses the bundled model; a previously selected custom file is kept. Other eyelid paths need the SRanipal model separately. See "First launch" for setup instructions.

The executable is unsigned, so SmartScreen may warn. Download from the official release and do not disable Windows protection.

Before upgrading, keep a copy of your settings folder:

```text
%APPDATA%\SRanibro\
```

Settings and logs normally live there. If a writable `sranibro.toml` exists beside the executable, SRanibro uses that folder instead (portable mode).

## 3. Install the VRCFaceTracking module

Extract `SRanibro-VRCFT-module.zip` and create this folder:

```text
%APPDATA%\VRCFaceTracking\CustomLibs\4d4b786f-e496-4df9-9421-dae811edff06\
```

Copy `SRanibro.dll`, `module.json` and `config.json` into it. Copy the contents, not the unopened ZIP.

Start SRanibro, then start VRCFT. Its Eye Module should report `SRanibro`. The bundled module uses the eye-provider slot, leaving a separate facial tracker such as Vive Facial Tracker available.

## 4. First launch

1. Connect the headset and start its vendor software. For PSVR2, also start SteamVR with PSVR2Toolkit installed.
2. Run `SRanibro.exe` and open the gear-shaped **Settings** page.
3. For SRanipal tracking, under **SRanipal runtime**, try **Find automatically**. If that fails, select `sr_runtime.exe` or specify the folder containing it. Skip this for XR5 native tracking.
4. Choose your headset under **Tracking & device**. Select it explicitly if automatic detection chooses the wrong device.
5. For Dream Air, choose **XR5 native model** under **XR5 tracking**, then press **Apply & reload**. For other headsets, press **Apply & reload** after selecting the device.

When using SRanipal, its folder must contain:

```text
model\EyePrediction\00-0000.params_opencl.params
```

Use Apply & reload for device, model and path changes. The lower-left Reload button also applies settings and reconnects. During reload, the current page dims and a centered progress indicator appears. Settings are inactive until it finishes.

## 5. Check the dashboard and load

Expand Pipeline to check the device, camera, model and output stages. If a stage has failed, check its reason before changing unrelated settings.

Turn **PREVIEW** on to display the two camera images. Turning it off hides the images without stopping tracking.

A number such as `120/s` on an image is the camera-frame arrival rate. It is not your monitor's refresh rate or the eyelid inference rate. The model may run more slowly than the cameras.

If load is a problem:

- Leave PREVIEW off during normal use, or minimize the window.
- Start with **GPU** under **Settings → Tracking & device → Eyelid processing**.
- If problems occur only with GPU inference, compare **CPU** mode. Changing the inference mode reconnects tracking.

GPU processing can fall back to CPU if initialization or execution fails. GPU is not guaranteed to be faster on every PC.

## 6. Adjust eyelids, Wide and Squeeze

Use **Recenter** and **Live eyelid response** on the left sidebar's tuning page.

### Start with the fit and image

Wear the headset in your usual position. Look straight ahead with both eyes naturally open; do not widen your eyes for Recenter.

If the image seems to be causing poor model response, open the camera gear on the dashboard and adjust **Filter → Eye-image brightness** in small steps. This is a fixed slider: brightness does not keep adapting automatically. Brighter is not always better.

This changes the image passed to the model, not Tobii gaze or raw camera output. Recheck your eyelid references after changing image settings.

### Open and closed handles

1. Press **Recenter** and wait for the relaxed-open reference to settle.
2. Watch the green marker and **Avatar openness** while adjusting the open and closed handles for each eye.
3. Check normal blinks, slow closing and winks.
4. Confirm that relaxed open reaches 100% and gentle full closure reaches 0%.

Changes apply live. They save automatically when you release the drag; there is no separate Apply step.

- **Not closing fully:** move the closed handle toward the green marker's value while that eye is gently closed.
- **Closing before your eye is actually shut:** move the closed handle back toward a value that requires more closure, then check with a slow close.
- **Open side feels wrong:** Recenter first, then adjust the open handle.

The two eyes can produce different model values. LINK links adjustment values; it does not force both detected outputs to be identical. Unlink the controls when only one eye needs a different setting.

### EyeWide and Squeeze

Press **Set Wide neutral** with your eyes relaxed, not widened. Then widen your eyes and adjust the Wide start/full range while watching the orange marker and output. Green shows normal eyelid movement; orange shows Wide.

Squeeze has its own rail when using a model that supports it (the XR5 native model does not). Compare ordinary closure with gently adding tension after closing, then adjust its range. You do not need to squeeze forcefully.

**Eyelid response** changes the response between the endpoints. Set the endpoints first, then use this control if the movement in between still feels wrong.

## 7. Wearing Memory (optional)

Wearing Memory saves a wearing position and eyelid settings that you have confirmed work well. It recalls a correction when the eye-camera appearance is similar. It does not automatically learn its way out of a bad setup.

1. Open **Wearing memory...**.
2. Choose **Adjust without recovery** to pause automatic correction.
3. Recenter with relaxed open eyes and adjust the eyelid handles.
4. If useful, use **Set L closed / Set R closed**. Close the selected eye and hold it closed through the two tones.
5. Check both eyes' opening, full closure and natural blinks.
6. Press **Save current good state**. Keep both eyes relaxed and open, looking straight ahead, until saving finishes.
7. Turn on **Automatic wearing-position recovery**.

Recenter and slider edits alone do not save a memory. Do not save straight from a recovered or Try state: use Adjust without recovery and verify the uncorrected response first.

- **Try:** temporarily try a saved state. Use Adjust without recovery to end the trial.
- **Delete:** remove an unwanted state.
- **Undo threshold edits:** undo the current eyelid-threshold edits without rolling back later Wide, Squeeze or response-curve changes.
- **Finish without saving memory:** leave adjustment mode without adding a memory. This does not undo your settings.

Up to eight states can be stored. Saving nearly the same state updates the existing entry, so the count may not increase. Memories are separated by device, unit identity and image settings. Changing brightness, crop or other image settings can make previous memories unavailable for display or matching in the new setup.

Brief blinks retain the correction; prolonged missing evidence releases it. Turning recovery off keeps the saved memories. If recovery does not help, turn it off and return to manual settings.

A memory stores a small eye-image reference and adjustment values locally. Normal operation does not upload these, but inspect your settings folder before sharing it.

## 8. Eyebrows (optional)

Choose a mode under **BROW** on the dashboard:

- **LEGACY:** the bundled VRCFT module derives eyebrow motion from EyeWide / EyeSquint. No personal training is required.
- **ESTIMATE:** use an independent eyebrow model. This is unavailable without a compatible loaded model. Python is not run during normal tracking.
- **BROW L/R SYNC:** link or unlink the independent eyebrows.

Try the included `eyebrow.bin` first. To adapt a model to yourself, record data in the Eyebrow section and use **Fit in app (no Python)**. This adapts an existing model.

**Train & bake** uses an external `vr_eyebrow` project and Python environment for retraining. It is not required just to run the application or use the bundled model.

For independent eyebrows in VRChat, check **VRChat eyebrow OSC → Send eyebrows directly to VRChat OSC** and its destination in the Eyebrow section. Avoid sending the same eyebrow parameters from multiple applications at once. The avatar also needs compatible parameters.

## 9. Gaze, output and everyday use

Use **Gaze centre & movement range...** for gaze centering and range. Avatars can display the same gaze differently, so check the avatar's range before changing image brightness or eyelid handles.

**Eye mapping** changes eye identity or direction. Reversing gaze direction and swapping eye streams are different operations; a stream swap also swaps the camera images. If the problem is harder to identify, such as near-focus convergence going outward, report the headset, camera view and current mapping instead of stacking several flips.

VRCFT normally connects to `127.0.0.1:5555`. Increasing **VRCFT openness low-pass** adds smoothing and delay. Start with 0 or 1 sample.

Minimizing keeps camera, model and VRCFT output running. Close the app when finished. Eye image output is a separate streaming feature; you do not need to enable it to see the dashboard PREVIEW.

## 10. Troubleshooting

| Symptom | Check first |
| --- | --- |
| The app will not open or immediately exits | Read `sranibro.log` in the settings folder. Copy your settings before removing anything. |
| No camera preview | Check PREVIEW, the headset selection, whether the headset is awake, and the camera stage in Pipeline. |
| PSVR2 reports Reload failed | Check that PSVR2Toolkit is installed and running in SteamVR. The PSVR2 App alone is not enough. |
| An eye will not close, or closes too early | Pause recovery; check wearing position, brightness, Recenter and that eye's closed handle. |
| EyeWide does not respond | Use Set Wide neutral with relaxed eyes, then check the orange marker and Wide range. |
| Tracking changes after putting the headset back on | Turn Memory off and verify manual settings. Save only after tracking works well. |
| The saved count does not increase | Check whether a similar entry was updated. The limit is eight. Also check for a save error. |
| VRCFT will not connect | Check module placement, that SRanibro is running, and whether another app uses TCP port 5555. |
| UI interaction is heavy | Turn PREVIEW off; compare minimized operation and CPU inference. |

For a report, include the application version, headset, reproduction steps and relevant log tail. Use **REC** for a short diagnostic CSV if needed. Logs and CSVs can contain local paths and tracking data; memories and recordings can contain eye images. Review files before sharing them.
