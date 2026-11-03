# XR5 Recorder v0.4.2

The standalone Windows recorder collects raw stereo eye images, fixed-dot/head-pose context and native gaze references for offline research. [Download the Windows ZIP](https://github.com/challenger0303/SRanibro/releases/tag/xr5-recorder-v0.4.2).

1. Close apps using the eye cameras, open the recorder, and connect both cameras.
2. Choose Fixed dot + head movement (requires SteamVR) or Free eye movement.
3. Choose an expression and duration, read the instruction, accept the recording notice, then press Start. A five-second countdown precedes capture. Each item is recorded separately; rest and blink whenever needed.
4. Wait for automatic ZIP saving, or use Stop and save for a shorter take. The app shows the output path. Close the recorder before returning to normal tracking; the Windows package includes Restore Pimax tracking.cmd.

| Item | Instruction |
| --- | --- |
| Wide open | Open your eyes wide without straining. |
| Normal | Keep eyes and eyebrows relaxed; blink normally. |
| Half open | Relax the lids roughly halfway closed without squeezing; no exact halfway position is required. |
| Blinks | Blink naturally and include a few slow complete closes and reopens. |
| Squint | Gently narrow the eyes while keeping the eyebrows relaxed. |
| Eyebrow up | Raise only the eyebrows; keep eye opening unchanged, or use slow relaxed blinks. Do not open the eyes wider. |
| Eyebrow down | Lower only the eyebrows without squeezing the eyes. |

In fixed-dot mode, the dot is placed about 2 m ahead of the initial tracked pose and remains fixed in the room while the head moves. VR shows only the dot; read the expression cue before starting. Tracking loss pauses capture time. Free movement has no fixed gaze target.

The ZIP contains `samples.csv`, raw images, `fixation.csv`/events, `native_evidence.csv`, `tobii_gaze.csv` and their referenced raw packets. `prompted_action` distinguishes all seven items; `metadata.txt` retains `prompted_instruction`. Samples remain unlabelled with an unassigned split. Blinks, rests and transitions are valid within every item. Prompt names are not per-frame ground truth, and brow visibility depends on camera placement and fit.

Images contain biometric data. Storage is local, with no automatic upload or training. Gaze availability reflects the adapter's vector/freshness checks; it does not certify attention, fixation or native confidence. Camera/reference pairing uses receive time, not synchronized exposures. Coordinate and physical-eye assignment need separate verification. Keep recordings from one contributor/session together when assigning research splits.

From `sranibro-app`, the recorder target is:

```powershell
cargo test --locked --release --features xr5-only --bin sranibro-xr5-dataset-collector
cargo build --locked --release --features xr5-only --bin sranibro-xr5-dataset-collector
```

Use the release ZIP for the ready-to-run Windows distribution. Test fixtures and UI-smoke previews use synthetic data and must not be imported as participant recordings.

## 日本語

眉上げでは、目の開きを保ったまま眉だけ上げます。ゆっくりした自然な瞬きは可能ですが、目を見開かないでください。Squintは眉を楽にして軽く目を細め、Half openとは別項目で録画します。

固定点モードは、空間の同じ点を見続けながら頭をゆっくり動かします。VR内は点だけなので、開始前に表情の指示を読んでください。項目ごとにStartを押して録画し、終了後の保存を待ってください。ZIPは自動送信されません。項目名は動作の指定であり、画像ごとの正解ラベルではありません。
