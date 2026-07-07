SRanibro  (v0.1.3-beta)
=======================

VR eye-tracking -> VRCFaceTracking bridge for Tobii-based eye-tracking VR
headsets: Pimax Crystal / Crystal Super, StarVR One and Varjo.

SRanibro reads your headset's eye cameras, runs eyelid + gaze inference
locally, and serves the result over the BrokenEye protocol so VRCFaceTracking
can drive your avatar's eyes -- openness, wide, squeeze, and gaze.

  BETA. Expect rough edges. Eyebrow tracking is not in this beta.

The SRanibro APP is a closed-source, binary-only beta -- it's on the Releases
page. Its CORE is open, though: the HMD-agnostic eye-tracking post-processor
(openness / wide / squeeze / gaze calibration + smoothing) and the ML front-end
are MIT-licensed in the sranibro-core/ folder here. The device-access layer
(camera capture, connection handling) stays in the app.

SRanibro includes no third-party assets. The eye-model weights (from a SRanipal
install) and the Tobii stream-engine DLL are NOT bundled and NOT distributed
here -- supply your own copies, from sources you are authorized to use under
their licenses (see "What you supply"). Until you do, SRanibro runs but stays
inert.


DOWNLOAD
--------

Grab the latest SRanibro.exe from the Releases page:
  https://github.com/challenger0303/SRanibro/releases

Each release lists a SHA-256 checksum -- verify your download against it. Only
download from the link above, and don't disable Windows security to run it (the
exe is not code-signed yet, so SmartScreen may warn).

SRanibro is a single portable executable -- no installer. Put SRanibro.exe in a
folder you can write to (your Desktop, etc. -- NOT Program Files). On first run
it creates its config + log under %APPDATA%\SRanibro\.

All processing is local: SRanibro makes no internet connections and sends no
telemetry. Its only outputs are the local BrokenEye / OSC sockets described
below.


REQUIREMENTS
------------

  OS        Windows 10 / 11 (x64), a DX12- or Vulkan-capable GPU
  Headset   Pimax Crystal / Crystal Super, or StarVR One
            (the StarVR One uses Tobii IS4)
  Consumer  VRCFaceTracking -- the BrokenEye module (Tobii Advanced)
            https://docs.vrcft.io/


WHAT YOU SUPPLY
---------------

SRanibro does not include or distribute these. Point it at your own copies,
obtained from sources you are authorized to use under their licenses:

  SRanipal install
      Point SRanibro at the install FOLDER; it reads the eye-model weights
      from inside.

  Tobii stream-engine DLL
      Point SRanibro at the DLL file. Required to open the eye-camera device.


QUICK START
-----------

1. Run SRanibro.exe. The dashboard opens. With nothing configured, the DEVICE
   line reads "Tobii DLL required -- set it in Settings, then reload." -- that
   is expected.

2. Settings -> Assets:
     - SRanipal folder       Browse to your SRanipal install directory.
     - Tobii DLL (required)  Browse to your Tobii stream-engine DLL.
     - Device                auto (default; auto-detects a connected Pimax
                             eye-chip), pimax_vr4 (Pimax over WinUSB),
                             pimax_dll (Pimax via the Tobii stream engine),
                             or starvr (StarVR One). StarVR One users: pick
                             "starvr" explicitly -- auto covers Pimax only.
     - Apply & reload (no app restart).

3. On connect, SRanibro stops the Tobii runtime service to open the eye-camera
   device -- ALLOW THE UAC PROMPT (one elevation for that handoff). The PIPELINE
   should go green (DEVICE -> CAMERA/GAZE -> ML -> CORE -> OUTPUT) and the eye
   images appear. While SRanibro owns the device, the headset's own eye tracking
   is paused; to hand it back, run "SRanibro.exe restore" (or reboot).

4. VRChat: SRanibro serves the BrokenEye protocol on TCP port 5555. In
   VRCFaceTracking, install/enable the BrokenEye module and select Tobii
   Advanced -- it connects to 127.0.0.1:5555 and drives your avatar via OSC.
   A dedicated SRanibro VRCFaceTracking module is planned for a later release.
   (Optional: SRanibro can also send VRChat OSC directly -- set [output]
   osc = true in sranibro.toml. Use one output path or the other, not both.)

Your avatar needs VRCFaceTracking eye parameters. SRanibro drives: eye gaze
(EyeLeftX/Y, EyeRightX/Y, and combined EyeX/EyeY), eyelid openness
(EyeLidLeft/Right), widen (EyeWideLeft/Right), squint (EyeSquintLeft/Right), and
pupil dilation (PupilDilation).

Eye mapping (Settings -> Eye mapping) is per-device and remembered
automatically. Fix it by symptom:
     - Looking left moves the avatar right  -> tick "Flip gaze left / right"
     - Left / right eyes look swapped        -> tick "Swap left / right"
     - The eye image is mirrored             -> tick "Flip image horizontally"

Eye openness / wide / squeeze response is tuned in the Calibration tab. The gear at
the top-left of the eye cameras opens ML-input tools: crop / rotate / stretch the
image the model sees, a reflection filter, an optional Brightness match, and a
response heatmap that shows how the model reacts to each part of the eye image.


TROUBLESHOOTING
---------------

  Window opens then closes
      Check %APPDATA%\SRanibro\sranibro.log for the startup error.
  "Tobii DLL required"
      Set the Tobii DLL in Settings -> Apply & reload.
  DEVICE red / "no camera stream"
      Headset not connected, or the Tobii runtime still holds the device (the
      UAC handoff was declined). Re-Apply and allow UAC.
  Pipeline stuck before CORE
      The fault summary at the bottom names the first broken stage + the fix.
  VRCFaceTracking won't connect
      Confirm the BrokenEye module is installed (set to Tobii Advanced) and that
      nothing else is using port 5555; start SRanibro before VRCFaceTracking.

The bottom log panel shows a live fault summary (what's broken + the fix) and
the recent event history. Logs can contain local paths and your Windows
username -- review them before sharing.


UPDATE HISTORY
--------------

v0.1.3-beta
  - Calibration overhaul: expressions (long squints, held wide eyes) can no longer
    corrupt the auto-calibration, anything that does drift self-recovers within
    seconds, and Recenter now reliably fixes everything instantly. This also fixes
    setups whose camera levels sit lower than usual (one eye could stop calibrating
    entirely).
  - Slow-blink calibration now STICKS: two or three deliberate slow blinks teach each
    eye's true closed point and also equalize the left / right response while slowly
    closing the eyes. Remembered across restarts.
  - Fast-blink detection rebuilt: sensitivity is now per-eye (a weaker eye camera no
    longer misses quick blinks), blinks no longer leave one eye half-open, and the
    lids no longer "bounce" back open right after closing -- held winks stay fully
    shut.
  - Per-eye image settings: the gear's crop / rotate / stretch can target both /
    left / right, with the sliders syncing to whichever eye is selected.
  - NET view: a toggle on the eye-cameras card shows the exact image the model sees
    (all filters applied, correct orientation).
  - Blink recovery slider (Tuning): sets a minimum reopen time after a blink so
    the receiving avatar's animation has time to render it (0 = instant, as
    before).
  - Gaze-yoke fix: gaze no longer trembles with the yoke enabled.
  - Diagnostic recorder: a REC button (Calibration tab) captures a CSV of raw +
    processed tracking values -- attach it when reporting tracking issues.

v0.1.2-beta
  - Steadier openness: fixed a case where an eye could read stuck part-open at rest
    (common on a noisier / dimmer eye camera).
  - Reflection filter (on by default): removes the bright IR / glasses reflection
    dots that were destabilizing the openness reading. Glasses wearers -- this is the
    big one.
  - A squinting / winking eye now follows your open eye instead of freezing where it
    was.
  - New ML-input tools on a gear at the top-left of the eye cameras: crop / rotate /
    stretch the model input, an experimental Brightness match (opt-in, still being
    tuned), and a response heatmap that shows how the model reacts to each region of
    the eye image (a diagnostic).
  - Optional wide / squeeze link (padlock in the ML parameters) so they don't both
    fire at once.
  - Assorted reliability fixes.

v0.1.1-beta
  - New Console tab -- a live view of SRanibro's own log output, so you can watch
    what the app is doing (device handoff, connection, ML) without opening the log
    file.
  - Device selection now defaults to "auto," which picks the right path for a
    connected Pimax eye-chip. (StarVR One users: still select "starvr" -- see
    Quick Start.)
  - Smoother dashboard: fixed the stutter when dragging the window; the UI now
    repaints more responsively.
  - Assorted reliability and device-handling improvements under the hood.

v0.1.0-beta
  - First public beta. Local eyelid + gaze inference for Pimax Crystal / Crystal
    Super and StarVR One, served over the BrokenEye protocol to VRCFaceTracking.
    Portable single-exe; the asset paths (SRanipal weights, Tobii DLL) are yours
    to supply and are editable live in Settings.


LICENSE
-------

The APP is a closed-source, binary-only beta. (c) 2026. No license is granted
beyond running this beta as provided; don't redistribute the exe -- link to the
Releases page instead. (Full app terms to follow.)

The sranibro-core/ SOURCE folder is separate and MIT-licensed -- see
sranibro-core/LICENSE. Use, modify, and redistribute that under the MIT terms.

SRanibro is an independent project, not affiliated with or endorsed by Tobii,
HTC, Pimax, StarVR, Varjo, VRChat, or VRCFaceTracking. All trademarks belong to
their respective owners.
