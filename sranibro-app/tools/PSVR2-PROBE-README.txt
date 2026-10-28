SRanibro PSVR2 Hardware Probe
=============================

Purpose
-------
This read-only test checks the current PSVR2Toolkit CAPI before PSVR2 support is
enabled in SRanibro. It records native gaze/pupil/blink values and the raw
400x200 eye-camera preview. It does not change calibration, SteamVR settings,
USB state, haptics, or the PSVR2Toolkit driver.

Requirements
------------
1. Windows, SteamVR, the official PlayStation VR2 PC driver/app, and the updated
   PSVR2Toolkit must already be installed.
2. Connect PSVR2 and start SteamVR first.
3. Complete the normal PSVR2 eye-tracking calibration.
4. The PSVR2Toolkit eye-camera feature has its own headset/jailbreak
   requirements. Gaze can still work when the image stream is unavailable.

How to test
-----------
1. Run sranibro-psvr2-probe.exe.
2. Wait for "PSVR2Toolkit CAPI connected".
3. Confirm that native gaze numbers move. If available, Camera A and Camera B
   should show two 200x200 eye images.
4. Press "Start recording".
5. Each item has a four-second preparation interval. Read the instruction and
   move into the requested pose. A second tone starts the eight-second recording.
6. Follow the seven instructions:
   - relaxed eyes / straight ahead
   - look left
   - look right
   - look up and down
   - close both eyes and blink
   - hold a LEFT wink
   - hold a RIGHT wink
7. When finished, send the generated ZIP to the SRanibro developer.

The ZIP is saved in:
  Desktop\SRanibro PSVR2 reports

Please send the ZIP unchanged. Camera A/B are intentionally not called
left/right until the wink recordings prove the anatomical order and mirroring.

Troubleshooting
---------------
"PSVR2Toolkit CAPI not found"
  Start SteamVR and verify that the updated PSVR2Toolkit is installed. The
  Toolkit should create %TEMP%\psvr2tk_capi_path.txt while its driver is active.

"Toolkit CAPI loaded, but its SteamVR driver is inactive"
  Keep SteamVR and the PSVR2 driver running, then wait for automatic retry.

"All Toolkit client slots are occupied"
  Close another application that is currently reading PSVR2Toolkit.

Gaze works but Camera A/B stay blank
  Still run and send the report. This distinguishes a camera/jailbreak setup
  problem from a CAPI or gaze problem.

Data and license note
---------------------
The report contains eye-camera images and eye-tracking telemetry. Share it only
with the person you intend to send it to. PSVR2Toolkit states that its
eye-tracking data must not be used commercially.
