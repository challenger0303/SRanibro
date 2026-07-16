# SRanibro — Setup

VR eye-tracking → VRCFaceTracking bridge for Tobii IS4 hotmirror HMDs
(Pimax Crystal / Crystal Super, StarVR One).

SRanibro ships nothing proprietary. You point it at the eye-tracking assets you already
own; it stays inert until you do.

---

## 1. What you need

| Item | What | Where it goes |
|------|------|----------------|
| **SRanibro.exe** | the app (this pack) | extract anywhere you can write (a normal folder, your Desktop, etc. — NOT Program Files) |
| **SRanipal install** | the SRanipal runtime (you point SRanibro at its **folder**; SRanibro reads the EyePrediction eye-model weights from inside) | just point at the SRanipal folder |
| **Tobii DLL** | the Tobii stream-engine DLL | place it next to the exe or anywhere; you'll point SRanibro at it |
| **VRChat + VRCFaceTracking** | the consumer | VRCFT's Tobii/Advanced module connects to SRanibro |

> You just give SRanibro your **SRanipal folder** — it finds the eye-model weights inside.
> (SRanibro uses only those weights, not SRanipal's DLLs.) SRanibro bundles neither the
> weights nor the Tobii DLL — supply your own. *(Advanced: you can instead point at the
> single weights file `…\model\EyePrediction\00-0000.params_opencl.params` directly.)*

---

## 2. First run

1. Run **SRanibro.exe**.
2. It creates its config + log under `%APPDATA%\SRanibro\`
   (`sranibro.toml`, `sranibro.log`).
3. The dashboard opens. With nothing configured yet, the **DEVICE** line shows
   **"Tobii DLL required — set it in Settings, then reload"** — that's expected.

---

## 3. Configure (Settings tab)

Open **Settings** (gear icon) → **Assets**:

- **SRanipal folder** → Browse to your SRanipal install directory (SRanibro reads the
  eye model from inside). *(Advanced: leave this and set a direct `.params` file instead.)*
- **Tobii DLL (required to connect)** → Browse to your Tobii stream-engine DLL.
- **Device**:
  - `pimax_vr4` — Pimax Crystal/Super over WinUSB.
  - `pimax_dll` — Pimax via the Tobii stream engine (use if `pimax_vr4` doesn't stream).
  - `starvr` — StarVR One.
- Click **Apply & reload** (no app restart).

On connect, SRanibro frees the EyeChip from the Tobii runtime — **allow the UAC
prompt** (it needs admin once to stop/disable the Tobii service).

The dashboard's **PIPELINE** should go green (DEVICE → CAMERA/GAZE → ML → CORE →
OUTPUT) and the eye-camera images should appear.

---

## 4. Fix-ups (Settings → Eye mapping)

- **Gaze looks the wrong way left/right** → tick **"Flip gaze left / right"**
  (common on `pimax_dll`). Saved automatically.
- **Left/right eyes swapped** → tick **"Swap left / right"**.
- **Eye image mirrored** → tick **"Flip image horizontally"**.

Tuning (open/close/wide/squeeze feel) lives in the **Calibration** tab.

---

## 5. Connect to VRChat

- SRanibro serves the **BrokenEye protocol on TCP 5555**.
- In **VRCFaceTracking**, enable the **Tobii/Advanced** module — it connects to
  `127.0.0.1:5555` and drives your avatar's eye parameters via OSC.
- (Optional) enable direct VRChat OSC in `sranibro.toml` (`[output] osc = true`).

---

## 6. Troubleshooting

- **Nothing happens / window flashes and closes** → check
  `%APPDATA%\SRanibro\sranibro.log` for the reason (graphics init, panic, etc.).
- **"Tobii DLL required"** → set the Tobii DLL in Settings and Apply & reload.
- **DEVICE red, "no camera stream"** → the headset isn't connected, or the Tobii
  runtime still holds the device (the UAC handoff was declined). Re-Apply and allow UAC.
- **Pipeline stuck before CORE** → the fault summary at the bottom of the dashboard
  names the first broken stage and the next action.
- **Window too big for your screen** → it auto-fits the monitor on launch.

The bottom log panel shows a live FAULT SUMMARY (what's broken + the fix) and the
recent event history.
