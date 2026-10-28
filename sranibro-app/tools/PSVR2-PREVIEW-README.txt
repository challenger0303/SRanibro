SRanibro 0.1.6 Beta - PSVR2-only
=================================

This is a separate public beta for PlayStation VR2 hardware validation. It is
not the normal multi-headset build: the headset route is fixed to PSVR2 and it
uses its own sranibro-psvr2.toml settings file.

What is included
----------------
- SRanibro eyelid, gaze, output and calibration pipeline
- Read-only connection to the CAPI installed by PSVR2Toolkit
- A PSVR2-only runtime guard; copied settings cannot select another HMD

What is not included
--------------------
- Tobii DLLs, a Tobii runtime loader, or Tobii restoration code
- PSVR2Toolkit or Sony software
- SRanipal EyePrediction model weights

Requirements
------------
1. Windows, SteamVR, the official PlayStation VR2 PC app/driver, and a current
   PSVR2Toolkit installation.
2. PSVR2Toolkit eye-camera access. Native gaze alone is not enough for
   SRanibro's continuous eyelid model.
3. SRanipal EyePrediction weights selected in SRanibro Settings.

First test
----------
1. Close the separate SRanibro PSVR2 Hardware Probe and other PSVR2Toolkit gaze
   clients.
2. Connect PSVR2, start SteamVR, and complete the normal PSVR2 eye calibration.
3. Start SRanibro-PSVR2-Beta.exe. The Headset field is already fixed to PSVR2.
4. In Settings, select the SRanipal folder or EyePrediction weights, then press
   Apply & reload.
5. Confirm that Camera L/R match the anatomical eyes, gaze moves in the expected
   direction, and relaxed open / both-eye close / left wink / right wink behave
   correctly.

Troubleshooting and feedback
----------------------------
The Device status line gives the direct connection state. If it does not connect,
confirm SteamVR and PSVR2Toolkit are active, close other CAPI clients, and use
Apply & reload.

Log:
  %APPDATA%\SRanibro\sranibro.log

Settings:
  %APPDATA%\SRanibro\sranibro-psvr2.toml

For a useful hardware report, include the log, the PSVR2Toolkit version, and a
short recording of the live eye cameras and output parameters. This executable
is unsigned, so Windows SmartScreen may display a warning.
