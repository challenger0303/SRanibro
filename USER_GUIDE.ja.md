# SRanibro 使い方ガイド

**SRanibro v0.1.5-beta**向けです。ベータ期間中は画面や項目名が変わることがあります。

[English guide](USER_GUIDE.md)

SRanibroは、Hotmirror系VR HMDのアイカメラ映像を処理し、VRCFaceTracking
（VRCFT）で利用できる視線・まぶた・EyeWide・EyeSquintへ変換します。処理は
PC内で完結します。SRanipalのモデルやTobiiのランタイムファイルは同梱されない
ため、利用許諾を持つ自分のファイルを用意してください。

## 1. 必要なもの

- Windows 10 / 11 x64
- 対応HMD
  - Pimax Crystal / Crystal Super（VR4）
  - Pimax Dream Air / SE (XR5)
  - StarVR One
  - 対応するVarjoカメラ経路
- 自分が利用権を持つSRanipalインストール、またはEyePredictionモデル
- 互換性のあるTobii stream-engineランタイムDLL
- VRChatアバターを動かす場合はVRCFaceTracking

PimaxやVarjoの通常ソフトウェアを先に導入してください。SRanibroの補正を使う
前に、メーカー側の視線キャリブレーションを完了します。特にDream Air / SE (XR5)
では重要です。SRanibroは小さな残差を後から補正できますが、HMD本来の多点
キャリブレーションを置き換えるものではありません。

## 2. ダウンロード

必ず公式の
[SRanibro Releasesページ](https://github.com/challenger0303/SRanibro/releases)
からダウンロードしてください。

推奨は `SRanibro-v0.1.5-beta-bundle.zip` です。次の2つが入っています。

- `SRanibro.exe`
- `SRanibro-VRCFT-module.zip`

現在のexeはコード署名されていないため、Windows SmartScreenが警告することが
あります。Windowsの保護機能を無効にせず、Releaseページに記載されたSHA-256
と一致することを確認してください。

デスクトップ上のフォルダーなど、書き込み可能な場所へ展開します。
`Program Files`の下には置かないでください。インストーラーはなく、設定・
キャリブレーション・ログは次の場所へ保存されます。

```text
%APPDATA%\SRanibro\
```

## 3. VRCFaceTrackingモジュールの導入

`SRanibro-VRCFT-module.zip`を展開します。中には `SRanibro.dll`、
`module.json`、`config.json`、簡単なREADMEが入っています。

次のフォルダーを正確に作成します。

```text
%APPDATA%\VRCFaceTracking\CustomLibs\4d4b786f-e496-4df9-9421-dae811edff06\
```

その中へ次の3ファイルをコピーします。

```text
SRanibro.dll
module.json
config.json
```

SRanibroを先に起動し、その後VRCFaceTrackingを起動してください。有効な
Eye Moduleの名前が `SRanibro` になれば導入成功です。

このモジュールは**目専用**です。視線、まぶた、瞳孔、EyeWide、EyeSquintを
提供しますが、Face / Expression Providerの枠は取得しません。そのため、
Vive Facial Trackerなど別の顔トラッカーと併用できます。

## 4. SRanibroの初期設定

`SRanibro.exe`を起動し、**Settings**を開いて、
**Connection & models**を展開します。

- **SRanipal model folder** — SRanipalのインストールフォルダーを選択します。
  SRanibroが中のEyePredictionモデルを探します。
- **Tobii runtime DLL** — 互換性のあるTobii stream-engine DLLを選択します。

次に**Tracking & device**を展開し、HMDを選びます。

| 設定 | 用途 |
| --- | --- |
| `auto` | Pimax向けの推奨設定。接続されたVR4 / XR5 EyeChipを自動判別します。 |
| `pimax_vr4` | Pimax Crystal / Crystal SuperのVR4経路を明示選択します。 |
| `pimax_xr5` | Pimax Dream Air / SE (XR5)の斜めアイカメラ経路を明示選択します。 |
| `starvr` | StarVR OneをTobiiランタイム経由で使用します。 |
| `varjo` | Varjo Baseのネイティブカメラ経路です。 |
| `varjo_mjpeg` | 外部Varjo Eye StreamerのMJPEG経路です。 |

`auto`はPimax用です。StarVRとVarjoは明示的に選択してください。

最後に**Apply & reload**を押します。HMD、モデル、ファイルパス、OSC送信先の
変更は、このボタンを押すまで反映されません。ライブトグルや多くのTuning項目
は変更時に自動保存されます。

Pimaxを直接開くときは、Tobii Platform ServiceからEyeChipを引き渡すために
UACが表示されることがあります。SRanibroでカメラを使う場合は許可してください。

## 5. Dashboardの確認

**DEVICE**から**OUTPUT**までのPipelineが緑色になり、左右のアイカメラ映像が
表示されれば正常です。

映像右下の `120/s` などの表示は、実際に届いているカメラフレームのレートです。
まぶたMLはカメラより低いレートで動く場合があります。視線とVRCFT出力も、それぞれ
別のレートで動作します。

赤い段階があるときはPipelineを展開するか、**Console**ページを開いてください。
無関係なTuningを動かす前に、最初に失敗している段階と理由を確認します。

## 6. VRCFaceTracking / VRChatとの接続

推奨起動順は次の通りです。

1. HMDのメーカーソフトを起動し、HMDを接続する
2. SRanibroを起動し、Pipelineが緑になるまで待つ
3. VRCFaceTrackingを起動し、`SRanibro` Eye Moduleを有効にする
4. VRChatを起動する

SRanibroは `127.0.0.1:5555` でローカルの目データを配信します。VRCFTが接続
できない場合は、別のアプリがTCP 5555番ポートを使用していないか確認してください。

Settingsの**VRCFT openness low-pass**は即時反映されます。

- `0` または `1` samples — パススルー。遅延が最小です。
- 大きな値 — 滑らかになりますが、遅延が増えます。

SRanibro側ですでにまぶたの後処理をしているため、まずはパススルーから試し、
アバター上で追加の平滑化が必要な場合だけ増やしてください。

## 7. 最初のまぶたキャリブレーション

普段と同じ位置にHMDを装着してから行います。キャリブレーションのためだけに
強く押し付けたり、不自然な位置にずらしたりしないでください。

1. **Calibration**を開く
2. 両目を自然に開き、正面を見る
3. **Recenter**を押す
4. 約1秒待ち、意識的なゆっくりした瞬きを2～3回行う
5. 普通の瞬き、ゆっくり閉じる動作、片目を閉じた保持、自然な再開眼を確認する

Recenterは現在の自然な開眼状態を基準として学習します。**Adaptive blink bounds**
は完全に閉じた深さを別に学習します。既定でONであり、通常はONのほうが良い結果に
なります。時間がたって偏りが出た場合は、HMDを普段の位置へ戻してRecenterします。

キャリブレーションはHMDごとに別保存されます。古いバージョンから更新した後は、
使用するHMDごとに一度Recenterしてください。

### Eye mapping

**Settings → Eye mapping**は、症状が一致するときだけ変更します。

- 左を見たのにアバターが右を見る → **Flip gaze left / right**
- アバター上で左右の目が入れ替わっている → **Swap left / right**
- アイカメラ映像そのものが左右反転している → **Flip image horizontally**

これらはHMDごとに保存されます。

## 8. Dream Air / SE (XR5)専用機能

この章の項目は、Dream Air / SE (XR5)が実際に動作中のときだけ表示されます。

### Gaze sourceと補正

最初にPimaxソフト側でTobiiの視線キャリブレーションを完了してください。基礎
キャリブレーションが間違っていると、SRanibroへ視線値は届いていても、左右の視線が
不自然にずれることがあります。

オプションの**EyeChip combined gaze**は、片目の揺れや一時的な認識喪失がある
ときに安定しやすい経路です。一方で、近距離を見たときの自然な左右眼の寄りを一部
失う代わりに安定性を得ます。切り替え後は**Apply & reload**を押し、Gaze correction
の**Center**をやり直してください。

**Dream Air / XR5 gaze correction**は、メーカー側キャリブレーション後に残った
中心・範囲・左右眼の寄りの誤差だけを調整するために使います。

### Automatic eye image alignment / Safe Geometry Fit

内蔵XR5 geometryは、それ自体が有効なフォールバックです。現在の装着者や光学位置で
まぶたMLの反応が悪い場合にだけFitを試してください。

1. ガイド付き記録を開始し、OPEN、HALF、CLOSED、視線移動、Slow Close、Blinkの
   指示をすべて行う
2. 記録中はHMDの装着位置を一定に保つ
3. 記録完了後にFitを実行する
4. untouched holdout検証を通った候補だけPreview / Applyする

`KEEP CURRENT GEOMETRY`は失敗ではなく、安全側の正常な結果です。候補が現在値より
他の動きにも適用できると証明できなかったため、現在値を維持したという意味です。

Safe Geometry Fitは`SRanibro.exe`内で完結し、Pythonは不要です。設定済みの
SRanipalまぶたモデルを評価しますが、新しいMLモデルを学習する機能ではありません。

完了した記録はFitのためにメモリ上へ保持されます。明示的にExportを押したときだけ
ディスクへ保存されます。出力ZIPには生体情報に当たる目の映像が含まれるため、内容を
確認し、信頼できる相手にだけ共有してください。

### XR5 image EyeWide

`SRanipal`は従来のWide出力を使用します。`Auto`は、有効で新鮮なカスタムXR5
Wideモデルがあるときだけそれを使い、それ以外では安全に従来経路へ戻ります。
`Custom`は正常なカスタムモデルを必須にします。

**Fit in app (no Python)**は、互換性のある既存Wideベースモデルを個人向けに
合わせ直します。ベースモデルがなければ、有用なモデルを一から作ることはできません。

## 9. 眉トラッキング（任意）

眉トラッキングには互換性のある `brow.bin` ベースモデルが必要です。

- **Fit in app (no Python)** — 記録した眉データを使い、既存モデルのHeadを個人向け
  に合わせ直します。短時間で行う通常の個人調整向けです。
- **Train & bake** — 外部の `vr_eyebrow` プロジェクトでモデルを作ります。
  Python、PyTorch、設定済み環境が必要です。

**Enable eyebrow tracking**がマスタートグルです。同梱VRCFTモジュールは目専用なので、
VRCFTに目を任せたままSRanibroからFT/v2眉パラメーターだけを送る場合は、
**Send eyebrows directly to VRChat OSC**を有効にします。

## 10. 普段の使い方と終了

SRanibroを最小化してもトラッキングは停止しません。UIの再描画負荷だけを落とし、
カメラ、ML、VRCFT出力はバックグラウンドで継続します。

使用しないときはSRanibroを終了してください。その後Pimax / Tobiiの通常ランタイムが
EyeChipを再取得できない場合は、次を実行します。

```text
SRanibro.exe restore
```

PCの再起動でも通常のメーカーサービス状態へ戻ります。

## 11. トラブルシューティングとフィードバック

### ウィンドウが閉じる、または表示されない

次のログを確認します。

```text
%APPDATA%\SRanibro\sranibro.log
```

### アイカメラ映像が来ない

- HMD選択が正しいか確認する
- HMDが接続済み・起動済みか確認する
- Tobii DLLのパスを確認する
- Apply & reloadを再実行し、必要ならUACの引き渡しを許可する

### Pupilは来るがGazeが来ない

メーカー側のTobii / Pimax視線キャリブレーションを完了してから、
**Apply & reload**を実行します。まぶたgeometryを変更する前に、Consoleで
Wearable / Gaze subscriptionの状態を確認してください。

### 時間がたつと片方のまぶたが偏る

HMDを普段の位置へ装着し直してRecenterします。**Adaptive blink bounds**をONのまま、
ゆっくりした瞬きを2～3回行います。再発する場合はCalibrationページの**REC**で
短い診断CSVを記録してください。

### VRCFTが接続しない

- SRanibroを先に起動する
- CustomLibsのフォルダー名が上記GUIDと完全に一致するか確認する
- `SRanibro.dll`、`module.json`、`config.json`の3つが入っているか確認する
- TCP 5555番ポートが他のアプリに使われていないか確認する

### 問題報告を送る

関係するログ末尾、短い診断CSV、必要な場合のみXR5 calibration recordingを用意します。
ログにはWindowsユーザー名やローカルパスが含まれることがあります。Calibration
recordingには目の映像が含まれます。共有前に必ず内容を確認してください。
