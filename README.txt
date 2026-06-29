SRanibro  (v0.1.0-beta)
=======================

VR eye-tracking -> VRCFaceTracking bridge for Tobii-based eye-tracking VR
headsets: Pimax Crystal / Crystal Super, StarVR One and Varjo.

SRanibro reads your headset's eye cameras, runs eyelid + gaze inference
locally, and serves the result over the BrokenEye protocol so VRCFaceTracking
can drive your avatar's eyes -- openness, wide, squeeze, and gaze.

  BETA. Expect rough edges. Eyebrow tracking is not in this beta.

This repository is documentation only. SRanibro is a closed-source, binary-only
beta -- the app is on the Releases page; the source is not published.

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
     - Device                pimax_vr4 (Pimax over WinUSB),
                             pimax_dll (Pimax via the Tobii stream engine),
                             or starvr (StarVR One).
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

Eye openness / wide / squeeze response is tuned in the Calibration tab.


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


LICENSE
-------

Closed-source, binary-only beta. (c) 2026. No license is granted yet beyond
running this beta as provided; redistribution is not permitted -- link to the
Releases page instead. (Full terms to follow.)

SRanibro is an independent project, not affiliated with or endorsed by Tobii,
HTC, Pimax, StarVR, Varjo, VRChat, or VRCFaceTracking. All trademarks belong to
their respective owners.
