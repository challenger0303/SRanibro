# SRanibro 使い方ガイド

**Windows版 v0.1.10-beta（Hotmirror / PSVR2 / Dream Air・XR5）**向けです。

[English guide](USER_GUIDE.md) · [ダウンロード](https://github.com/challenger0303/SRanibro/releases/tag/v0.1.10-beta)

SRanibroは、対応VR HMDのアイカメラ映像を使い、まぶた・見開き・対応モデルでは目を強く閉じる動きをVRCFaceTracking（VRCFT）へ送るソフトです。視線や瞳孔のデータも扱います。まぶたモデルの推論はPC内で行います。

最初から長いキャリブレーションをする必要はありません。まず普通に装着して動きを確認し、必要な場合だけ明るさと開閉のバーを調整してください。Wearing Memoryは、良い設定ができた後に使う任意の機能です。

## 1. 必要なもの

- Windows 10 / 11 x64
- 対応HMDとメーカーの通常ソフトウェア
  - Pimax Crystal / Crystal Super（VR4）
  - StarVR One
  - Varjo（対応カメラ経路）
  - PlayStation VR2（PSVR2Toolkit経由）
  - Pimax Dream Air / SE（XR5、ベータ対応）
- 利用できるSRanipalインストールとEyePredictionモデル（同梱のXR5専用モデルを使う場合は不要）
- VRChatへ送る場合は、VRCFaceTrackingと対応アバター

**Dream Air / SE（XR5）**の新規設定は、**Tracking & device → XR5 tracking → XR5 native model**が標準です。既存のSRanipal選択は保持するので、変更したい場合はここで切り替えてください。開閉とEyeWideに対応し、Squeezeは未対応です。Pythonは不要です。従来の**SRanipal + image transform**も選べますが、そちらはSRanipalモデルが必要です。旧XR5 EyeWideの学習項目は、そのフォールバック使用時だけ表示します。専用モデルではSafe Geometry Fitは不要です。

先にメーカー側の視線キャリブレーションを済ませてください。SRanibroのRecenterはまぶたの基準を合わせる操作で、メーカー側の視線キャリブレーションとは別です。

PSVR2は、PSVR2Toolkitを導入してSteamVRを起動しておく必要があります。PlayStation VR2 Appだけをインストールした状態では、Toolkitの機能は利用できません。

## 2. ダウンロードと配置

公式リリースから **`SRanibro-v0.1.10-beta-Windows.zip`** をダウンロードし、書き込み可能なフォルダーへ展開します。

- `SRanibro.exe` — アプリ本体
- `eyebrow.bin` — 汎用眉モデル
- `SRanibro-VRCFT-module.zip` — VRCFT用モジュール
- `models/` — XR5専用まぶたモデル
- `model-runtime/` — 専用モデルの推論に必要なファイル
- `licenses/` — 依存ライブラリのライセンス表記
- `README.txt` — 簡単な案内

`eyebrow.bin`はexeと同じフォルダーに置いてください。個人用の眉モデルが選択済みなら、そちらが優先されます。

更新時も`models`と`model-runtime`をexeと一緒に配置してください。XR5モデルの選択が空欄なら同梱モデルを使い、選択済みの個人ファイルはそのまま保持します。他のまぶた処理ではSRanipalモデルが別途必要です。指定方法は「最初の起動」を参照してください。

exeは未署名のため、SmartScreenの警告が出る場合があります。公式リリースから入手し、Windowsの保護機能を無効にしないでください。

更新前は設定フォルダーをコピーして保管すると、元に戻しやすくなります。

```text
%APPDATA%\SRanibro\
```

通常の設定・ログはここに保存されます。ただし、exeの横に書き込み可能な`sranibro.toml`がある場合は、そのフォルダーを使うポータブル動作になります。

## 3. VRCFaceTrackingモジュールの導入

`SRanibro-VRCFT-module.zip`を展開し、次のフォルダーを作ります。

```text
%APPDATA%\VRCFaceTracking\CustomLibs\4d4b786f-e496-4df9-9421-dae811edff06\
```

その中へ`SRanibro.dll`、`module.json`、`config.json`の3ファイルを入れます。ZIPをそのまま置くのではなく、中身をコピーしてください。

SRanibroを起動してからVRCFTを起動し、Eye Moduleに`SRanibro`が表示されることを確認します。同梱モジュールは目用の枠を使うため、Vive Facial Trackerなど別の顔トラッカーと併用できます。

## 4. 最初の起動

1. HMDを接続し、メーカーのソフトウェアを起動する。PSVR2はToolkitを導入したSteamVRも起動する。
2. `SRanibro.exe`を起動し、歯車の**Settings**を開く。
3. SRanipalを使う場合は、**SRanipal runtime**で**Find automatically**を試す。見つからなければ`sr_runtime.exe`を選ぶか、それが入っているフォルダーを指定する。XR5専用モデルならこの操作は不要。
4. **Tracking & device**のHeadsetで使用する機種を選ぶ。自動判別が意図と違う場合は明示的に選ぶ。
5. Dream Airは**XR5 tracking → XR5 native model**を選んで**Apply & reload**を押す。他の機種は機種選択後に**Apply & reload**を押す。

SRanipalを使う場合、指定したフォルダーには次のモデルが必要です。

```text
model\EyePrediction\00-0000.params_opencl.params
```

機種・モデル・パスなどの変更にはApply & reloadを使います。左下のReloadも設定を適用して再接続する操作です。リロード中は画面が暗くなり、中央に読み込み表示が出ます。完了までは設定を操作できません。

## 5. Dashboardと負荷の確認

Pipelineを展開すると、機器・カメラ・モデル・出力の状態を確認できます。失敗している段階があれば、先にその理由を確認してください。

**PREVIEW**をONにすると左右のアイカメラ映像が表示されます。OFFは映像表示だけを止める操作で、トラッキングを止めるものではありません。

映像の`120/s`などはカメラフレームの到着レートです。モニター上の表示Hzや、まぶたMLの推論レートとは別です。MLはカメラより低いレートで動く場合があります。

負荷が気になるときは：

- 普段はPREVIEWをOFFにする、またはウィンドウを最小化する。
- **Settings → Tracking & device → Eyelid processing**で、まず**GPU**を使う。
- GPU利用時だけ問題が出るなら**CPU**に切り替えて比較する。切り替え時には再接続されます。

GPUの初期化や実行に失敗した場合はCPUへフォールバックします。GPUを選べば必ず速くなるとは限りません。

## 6. まぶた・Wide・Squeezeを調整する

左サイドバーの調整ページで、**Recenter**と**Live eyelid response**を使います。

### まず装着と映像を整える

普段使う位置にHMDを装着し、両目を自然に開いて正面を見ます。見開いたままRecenterしないでください。

映像の見え方が原因で反応が悪そうな場合は、Dashboardのアイカメラの歯車から画像設定を開き、**Filter → Eye-image brightness**を少しずつ調整します。現在の明るさ調整は固定スライダーで、自動では変わりません。明るくしすぎれば良いわけでもありません。

これはモデルへ渡す映像の調整です。Tobiiの視線データや、生のアイカメラ出力は変更しません。画像設定を変えた後は、まぶたの基準も確認し直してください。

### 開閉のバー

1. **Recenter**を押し、自然に開いた状態の基準が落ち着くまで待つ。
2. 緑のマーカーと**Avatar openness**を見ながら、左右それぞれの開眼側・閉眼側のハンドルを調整する。
3. 普通の瞬き、ゆっくり閉じる動作、ウィンクを確認する。
4. 自然に開いたときに100%、無理に力を入れず閉じたときに0%になるか確認する。

変更はリアルタイムで反映されます。ドラッグを離すと自動保存されるため、別のApply操作は不要です。

- **閉じ切らない**：目を閉じた状態の緑マーカーに、閉眼側のハンドルを近づける。
- **まだ開いているのに閉じる**：閉眼側を「さらに閉じないと届かない」位置へ戻し、ゆっくり閉じて確認する。
- **開眼側が合わない**：まずRecenterし、その後で開眼側を調整する。

左右でモデルの数値が違うことはあります。LINKは調整値を連動させるためのもので、左右の検出値を必ず同じにする機能ではありません。片側だけ合わない場合は連動を外して調整します。

### EyeWideとSqueeze

**Set Wide neutral**は、見開いていない自然な開眼状態で押します。その後で見開き、オレンジのWideマーカーと出力を見ながらWideの開始・最大側を調整します。緑は通常の開閉、オレンジはWideの確認用です。

Squeezeは対応モデル用の別のバーです（XR5専用モデルは未対応）。普通の閉眼と、閉じた状態から少し力を入れる動作を比べて範囲を調整してください。無理に強く力を入れる必要はありません。

**Eyelid response**は途中の開閉の反応を変える調整です。まず上下限を合わせ、それでも途中の動きが合わない場合に使います。

## 7. Wearing Memory（任意）

「この装着位置・この設定なら良く動く」という状態を保存し、似た映像になったときに自動で補正する機能です。悪い状態を自動で学習し直す機能ではありません。

1. **Wearing memory...**を開く。
2. **Adjust without recovery**を押して自動補正を止める。
3. 自然な開眼状態でRecenterし、開閉のバーを合わせる。
4. 必要なら**Set L closed / Set R closed**を使う。選んだ目を閉じ、2回の通知音の間は閉じたままにする。
5. 両目の開閉と普通の瞬きが正しく動くことを確認する。
6. **Save current good state**を押し、両目を自然に開いて正面を見たまま保存完了を待つ。
7. **Automatic wearing-position recovery**をONにする。

Recenterやバーの調整だけではMemoryは保存されません。補正中・Try中は、そのまま保存せず、Adjust without recoveryで補正を外した後の動作を確認してください。

- **Try**：保存済みの状態を試す。試用を終えるにはAdjust without recoveryを使う。
- **Delete**：不要な状態を削除する。
- **Undo threshold edits**：今回のまぶた閾値の編集を戻す。後から変更したWide・Squeeze・反応カーブは戻さない。
- **Finish without saving memory**：Memoryへ追加せず調整モードを終える。設定を元に戻す操作ではない。

最大8件を保存します。ほぼ同じ状態を再保存すると既存の記録を更新するため、件数が増えない場合があります。保存先は機器・個体識別・画像設定で分かれます。明るさやクロップなどを変更すると、以前のMemoryが表示・適用されなくなる場合があります。

短い瞬きでは補正を保ちますが、長く確認できない状態が続けば補正を解除します。スイッチをOFFにしても保存済みMemoryは消えません。合わない場合はOFFにして手動設定へ戻してください。

Memoryには目の映像から作った小さな参照画像と調整値を保存します。通常動作で外部へ送信するものではありませんが、設定フォルダーを共有するときは内容を確認してください。

## 8. 眉（任意）

Dashboardの**BROW**で切り替えます。

- **LEGACY**：EyeWide / EyeSquintに連動した眉を、同梱VRCFTモジュールで動かす。個人モデルの学習は不要。
- **ESTIMATE**：独立した眉モデルを使う。互換モデルが読み込まれていないと選択できない。通常のトラッキング中にPythonは起動しない。
- **BROW L/R SYNC**：独立眉の左右連動を切り替える。

まず同梱の`eyebrow.bin`で試してください。自分向けに合わせる場合は、Eyebrow項目でデータを記録し、**Fit in app (no Python)**を使えます。これは既存モデルを個人向けに合わせる機能です。

**Train & bake**は外部の`vr_eyebrow`プロジェクトとPython環境を使う再学習です。通常利用や同梱モデルを使うだけなら不要です。

独立眉をVRChatへ送る場合は、Eyebrow内の**VRChat eyebrow OSC → Send eyebrows directly to VRChat OSC**と送信先を確認します。同じ眉パラメーターを複数のアプリから同時に送らないでください。アバター側にも対応パラメーターが必要です。

## 9. 視線・出力・普段の使い方

視線の中心や動く幅は**Gaze centre & movement range...**で確認します。アバターによって見え方が違うため、モデルの明るさやまぶたのバーを動かす前に、アバター側の範囲も確認してください。

**Eye mapping**は左右眼の対応や方向を直すための設定です。視線方向の反転と左右ストリームの交換は別の操作で、交換すると映像の左右も変わります。より目だけ逆になるなど判断が難しい場合は、反転を重ねず、機種・映像・現在のmapping設定を添えて報告してください。

VRCFTへは通常`127.0.0.1:5555`で接続します。**VRCFT openness low-pass**を増やすと滑らかになりますが、遅延も増えます。まず0または1 samplesで試してください。

最小化してもカメラ・ML・VRCFT出力は続きます。使い終わったらアプリを終了してください。Eye image outputは映像配信用の別機能で、DashboardのPREVIEWに映像を表示するためにONにする必要はありません。

## 10. 困ったとき

| 症状 | 最初に確認すること |
| --- | --- |
| 起動しない／すぐ終了する | 設定フォルダーの`sranibro.log`。設定を消す前にコピーを保管する。 |
| 映像が表示されない | PREVIEWがONか、機種が正しいか、HMDが起動しているか。Pipelineのカメラ段階も確認する。 |
| PSVR2でReload failed | PSVR2Toolkitが導入済みか、SteamVR上で動いているか。PSVR2 Appだけでは不足。 |
| 閉じ切らない／片側だけ早く閉じる | 自動復帰を止め、装着位置・明るさ・Recenter・左右それぞれの閉眼側を確認する。 |
| EyeWideが出ない | 自然な開眼状態でSet Wide neutralを行い、オレンジのマーカーとWide範囲を確認する。 |
| 装着し直すと合わない | MemoryをOFFにして手動設定を確認する。良く動く状態になってから保存する。 |
| 保存件数が増えない | 似た状態の更新か確認する。最大8件。保存エラーの表示も確認する。 |
| VRCFTがつながらない | モジュールの配置、SRanibroの起動、TCP 5555番ポートの競合を確認する。 |
| 操作が重い | PREVIEWをOFFにし、最小化やCPU推論との比較を行う。 |

報告にはアプリのバージョン、機種、再現手順、関係するログ末尾を添えてください。必要なら**REC**で短い診断CSVを記録します。ログやCSVにはローカルパス・追跡データが、Memoryや映像記録には目の画像が含まれる場合があります。共有前に確認してください。
