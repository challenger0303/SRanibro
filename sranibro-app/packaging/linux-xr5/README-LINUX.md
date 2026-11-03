# SRanibro XR5 for Ubuntu 24.04 — experimental preview

Native **x86_64 Ubuntu 24.04** application for the **Pimax Dream Air / Dream Air SE (XR5)** eye cameras. Includes the existing XR5 eyelid model and Linux CPU runtime. Python and SRanipal model weights are not required for the default XR5 model.

This is a separate XR5 build. The existing PSVR2 Linux build remains available. This package is an eye-tracking application; it does not provide an HMD display driver.

## Start

1. Extract the whole archive into a writable folder. Keep `models` and `model-runtime` beside `sranibro-rs`.
2. Install the USB access rule once: `sudo ./install-udev-rules.sh`. Reconnect the headset USB afterward. The rule applies to XR5 EyeChips only and grants access to the active local desktop user.
3. Close other applications using the eye cameras. Run **`./sranibro-xr5` as your normal user**, then check the connection and both eye previews.

If runtime libraries are missing on a minimal Ubuntu installation:

```sh
sudo apt install libusb-1.0-0 libxkbcommon0 libx11-6 libgl1 libegl1 libwayland-client0
```

The model runs on CPU in this preview. Native gaze is the default. Linux SteamVR in-headset research overlays and recording notification sounds are outside this build's verified scope. Optional experimental model-gaze packages are not included.

Native VRChat eye OSC is enabled by default, using localhost port 9000. It provides VRChat's shared blink channel, not independent left/right winks. A TCP output is also available for compatible clients; a Windows VRCFaceTracking module is not a native Linux client. Configure outputs for your own avatar/runtime setup.

Settings use **`sranibro-xr5.toml`**, separate from PSVR2 and ordinary SRanibro settings. An existing portable file beside the executable wins; otherwise settings use the user's XDG configuration directory.

## Connection check

```sh
./sranibro-xr5 xr5-check
```

This explicitly opens the eye cameras and checks for both 200×200 image streams and usable per-eye gaze references, for up to 20 seconds. It prints counts and a PASS/FAIL result. It does not save eye images, run the eyelid model, send OSC or upload anything. Close the GUI before running it. PASS confirms received streams, not eye assignment, gaze accuracy or expression accuracy.

`./sranibro-xr5 status` reports configured model assets without starting tracking. Missing USB access, an occupied interface and a missing headset are reported separately. The application does not detach kernel drivers, reset the whole USB device or change headset firmware.

## Verification limits

Build, offline tests, packaged model replay and desktop checks do not certify Linux hardware streaming. The local development machine keeps its headset attached to Windows; **Linux XR5 authentication, live cameras/gaze and end-to-end VRChat operation still require a real Ubuntu/XR5 test**. Treat this release as an experimental preview. No model accuracy improvement is claimed.

## 日本語

Ubuntu 24.04／x86_64向けのXR5専用試験版です。既存のPSVR2 Linux版とは別です。初期設定では同梱のXR5まぶたモデルをCPUで動かします。

全ファイルを展開し、初回だけ `sudo ./install-udev-rules.sh` でUSBアクセスを設定してから、HMDのUSBを接続し直してください。その後は通常ユーザーで `./sranibro-xr5` を実行します。アプリ全体をrootで起動する必要はありません。

GUIを閉じて `./sranibro-xr5 xr5-check` を実行すると、最大20秒間、左右画像と参考視線の受信数を確認できます。画像保存・モデル推論・OSC送信・自動アップロードは行いません。

Linuxに接続したXR5実機での動作は未確認です。ビルドやオフライン検証の成功を、実機接続・視線精度の保証とは扱っていません。
