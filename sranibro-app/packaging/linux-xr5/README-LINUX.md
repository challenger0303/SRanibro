# SRanibro XR5 for Linux

Native **Linux x86_64** application for the **Pimax Dream Air / Dream Air SE (XR5)** eye cameras. Built on Ubuntu 24.04; requires **glibc 2.39+**. Includes the XR5 eyelid model and Linux CPU runtime. Python and SRanipal model weights are not required for the default XR5 model.

This is a separate XR5 build. The existing PSVR2 Linux build remains available. This package is an eye-tracking application; it does not provide an HMD display driver.

## Start

1. Extract the whole archive into a writable folder. Keep `models` and `model-runtime` beside `sranibro-rs`.
2. Install the USB access rule once: `sudo ./install-udev-rules.sh`. Reconnect the headset USB afterward. The rule applies to XR5 EyeChips only and grants access to the active local desktop user.
3. Close other applications using the eye cameras. Run **`./sranibro-xr5` as your normal user**, then check the connection and both eye previews.

If runtime libraries are missing on a minimal Ubuntu installation:

```sh
sudo apt install libusb-1.0-0 libxkbcommon0 libx11-6 libgl1 libegl1 libwayland-client0
```

The bundled model runs on CPU. Native gaze is the default. Optional model-gaze packages are not included.

Native VRChat eye OSC is enabled by default, using localhost port 9000. It provides VRChat's shared blink channel, not independent left/right winks.

## VRCFaceTracking

The SRanibro VRCFT module has been reported working on Linux. Extract the bundled `SRanibro-VRCFT-module.zip` and install it in your VRCFT setup's custom-module directory. Start SRanibro first, then VRCFT; its Eye Module should show **SRanibro**. The module connects to SRanibro over TCP at **127.0.0.1:5555**. Allow that output in SRanibro and keep the port free.

Settings use **`sranibro-xr5.toml`**, separate from PSVR2 and ordinary SRanibro settings. An existing portable file beside the executable wins; otherwise settings use the user's XDG configuration directory.

## Connection check

```sh
./sranibro-xr5 xr5-check
```

This explicitly opens the eye cameras and checks for both 200×200 image streams and usable per-eye gaze references, for up to 20 seconds. It prints counts and a PASS/FAIL result. It does not save eye images, run the eyelid model, send OSC or upload anything. Close the GUI before running it. PASS confirms received streams, not eye assignment, gaze accuracy or expression accuracy.

`./sranibro-xr5 status` reports configured model assets without starting tracking. Missing USB access, an occupied interface and a missing headset are reported separately. The application does not detach kernel drivers, reset the whole USB device or change headset firmware.

## Background windows

The Linux OpenGL UI uses non-blocking VSync. Tracking and output continue when the window is minimized or moved to another workspace.

## 日本語

Linux x86_64向けのXR5専用版です（glibc 2.39以上）。既存のPSVR2 Linux版とは別です。初期設定では同梱のXR5まぶたモデルをCPUで動かします。

全ファイルを展開し、初回だけ `sudo ./install-udev-rules.sh` でUSBアクセスを設定してから、HMDのUSBを接続し直してください。その後は通常ユーザーで `./sranibro-xr5` を実行します。アプリ全体をrootで起動する必要はありません。

GUIを閉じて `./sranibro-xr5 xr5-check` を実行すると、最大20秒間、左右画像と参考視線の受信数を確認できます。画像保存・モデル推論・OSC送信・自動アップロードは行いません。

VRCFTモジュールはLinuxでの動作報告があります。同梱の `SRanibro-VRCFT-module.zip` を展開し、利用中のVRCFT環境のカスタムモジュール用フォルダーへ配置してください。SRanibro、VRCFTの順に起動し、Eye Moduleに **SRanibro** が表示されることを確認します。接続先はTCP **127.0.0.1:5555** です。

バックグラウンド表示対策として、LinuxのOpenGL UIでブロッキングVSyncを無効にしました。
