# Jetson Nano 60 FPS Face Tracker (MediaPipe & 2-Axis Servo)

NVIDIA Jetson Nano と CSI カメラ、PCA9685 サーボドライバを使用し、**60 FPS の完全リアルタイム**で人間の顔を追尾する自律2軸（Pan / Tilt）カメラシステムです。

---

## 🌟 主な特徴

- **⚡ 60 FPS 超高速エッジAI顔検出:**
  MediaPipe（BlazeFace）を OpenGL ES 3.2 GPU デリゲートで駆動し、Jetson Nano の Tegra Maxwell GPU 上でわずか ~8.5ms の推論速度を実現。
- **🎥 ハードウェアアクセラレーション GStreamer パイプライン:**
  `nvarguscamerasrc` ➔ `nvvidconv`（ハードウェア双線形補間）により、CPU負荷ゼロで高品位な 60 FPS キャプチャを処理。
- **🎯 滑らかな PD フィードバック制御:**
  比例（P）および微分（D）制御による急制動ダンピングにより、慣性オーバーシュートやハンチング（振動）を防止。
- **🛡️ 3重のハードウェア保護設計:**
  物理可動域リミット（Pan: 20°〜160°, Tilt: 35°〜145°）、最大変位角制限（1.0°/step）、中央不感帯（±3.5%）でモータのギア破損や唸りを防止。
- **🔄 自律 2軸パトロール（顔捜査モード）:**
  顔を見失った際は直近位置を1秒間ホールド後、人の着座・立位目線高さ（Tilt: 55°〜95°）を優先して自動捜査（Wave / Raster スキャン）。
- **📊 2D レーダーミニマップ & HUD:**
  探索中のカメラ向きをリアルタイムで視覚化する 2D レーダー HUD を画面右下に描画。
- **🧵 非同期パイプライン設計:**
  重い X11 画面描画を `queue.Queue(maxsize=1)` で別スレッドに完全分離。描画負荷による制御遅延（レイテンシ爆発）を遮断。

---

## 📁 ディレクトリ構成

```text
.
├── face_tracker_mediapipe.py    # ★メイン顔追尾プログラム (60 FPS)
├── requirements.txt             # 依存 Python パッケージ一覧
├── .gitignore                   # Git 除外設定
├── README.md                    # 本ドキュメント
│
├── tools/                       # 動作確認・診断用ツール群
│   ├── test_servo.py            # PCA9685 サーボ動作・I2C 診断スクリプト
│   └── run_face_detection.py    # カメラ・顔検出 FPS ベンチマーク
│
└── docs/                        # ドキュメント・技術解説資料
    ├── face_tracker_slides.html # ブラウザ閲覧用プレゼンテーションスライド
    └── face_tracker_mediapipe_slides.md # Marp 互換 Markdown スライド原稿
```

---

## 🔌 ハードウェア構成と配線

### 必要機材
1. **NVIDIA Jetson Nano**（開発者キット / 4GB または 2GB）
2. **CSI カメラモジュール**（Raspberry Pi Camera Module V2 / Sony IMX219 推奨）
3. **PCA9685 16ch PWM サーボドライバ基板**（I2C アドレス: `0x40`）
4. **2軸サーボマウント & RCサーボ**（SG90, MG996R 等）
   - **CH0:** Pan（水平回転サーボ）
   - **CH1:** Tilt（垂直仰角サーボ）
5. **サーボ駆動用 5V 外部電源**（※サーボ用電源は Jetson Nano の端子からではなく、必ず外部電源から供給してください）

### 配線ピンアサイン (Jetson Nano J41 40ピンヘッダ ➔ PCA9685)
| Jetson Nano 40pin | ピン機能 | PCA9685 側端子 |
| :---: | :---: | :---: |
| **Pin 3** | I2C1 SDA | **SDA** |
| **Pin 5** | I2C1 SCL | **SCL** |
| **Pin 1** | 3.3V 出力 | **VCC**（ロジック電源） |
| **Pin 6** | GND | **GND** |
| - | 外部電源 5V | **V+**（サーボ駆動電源） |

---

## 🚀 セットアップ

### 1. 依存ライブラリのインストール
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev i2c-tools
pip3 install -r requirements.txt
```

### 2. MediaPipe (GPU デリゲート対応ビルド) のインストール
本システムは Jetson Nano の Maxwell GPU (OpenGL ES 3.2) で BlazeFace を推論（約 8.5ms）するため、GPU デリゲートを有効化して aarch64 向けにビルドした Python 3.6 用 wheel を使用します。

環境に合わせて以下のいずれかの方法でインストールしてください：

- **方法 A: GitHub Releases から直接ワンライナーでインストール (推奨)**
  ```bash
  pip3 install https://github.com/mtkw1976/face_tracker_mediapipe/releases/download/v1.0.0/mediapipe-dev-cp36-cp36m-linux_aarch64.whl
  ```
  *(※ リポジトリの Releases に wheel ファイルをアセットとして登録して使用します)*

- **方法 B: ローカルのビルド済み wheel を指定してインストール**
  ```bash
  pip3 install mediapipe/dist/mediapipe-dev-cp36-cp36m-linux_aarch64.whl
  ```

### 3. I2C アクセス権限の付与
```bash
sudo usermod -aG i2c $USER
# 反映のため一度ログアウトして再ログインするか、再起動してください
```

### 4. I2C 接続の確認
```bash
i2cdetect -y -r 1
# アドレス 0x40 が表示されれば接続成功です
```

---

## 🎮 使用方法

### メイン顔追尾システムの起動
```bash
# 推奨デフォルト (Preset: large, キャプチャ 640x360, 画面 960x540, しきい値 60%)
python3 face_tracker_mediapipe.py
```

### 主なオプション引数
| オプション | デフォルト値 | 説明 |
| :--- | :---: | :--- |
| `--preset` | `large` | 動作プリセット (`large`, `smooth`, `hd`, `distance`) |
| `--conf` | `0.6` | 顔検出信頼度しきい値 (0.0 〜 1.0) |
| `--init-pan` | `90.0` | Pan サーボ初期角度 (度) |
| `--init-tilt` | `75.0` | Tilt サーボ初期角度 (着座目線高さ: 75.0°) |
| `--search-pattern` | `wave` | 顔ロスト時の探索パターン (`wave`: 連続波状, `raster`: 段階走査) |
| `--search-tilt-min` | `55.0` | 探索時の最上仰角 (度) |
| `--search-tilt-max` | `95.0` | 探索時の最下仰角 (度) |
| `--no-servo` | - | サーボ出力を無効化（カメラと検出のみテスト） |
| `--fullscreen` | - | 起動時に全画面表示 |

#### 実行例
```bash
# 1m〜2m 離れた遠距離の顔を重点的に追従したい場合
python3 face_tracker_mediapipe.py --preset distance

# サーボを繋がずにカメラと検出性能のみをテストしたい場合
python3 face_tracker_mediapipe.py --no-servo

# 段階的ラスタースキャンで捜索させたい場合
python3 face_tracker_mediapipe.py --search-pattern raster
```

### 画面操作キー
- **`f`**: フルスクリーン表示の切り替え
- **`q`** または **`ESC`**: 安全停止（サーボを中央復帰させて正常終了）

---

## 🛠️ テスト・診断ツール (`tools/`)

### サーボ単体動作テスト (`tools/test_servo.py`)
モータの結線、回転方向、初期位置への緩やかな復帰を確認できます：
```bash
python3 tools/test_servo.py
```

### カメラ・顔検出 FPS ベンチマーク (`tools/run_face_detection.py`)
GUI 表示や解像度別の推論スループットを計測できます：
```bash
python3 tools/run_face_detection.py --preset large
```

---

## 💡 トラブルシューティング

- **`Failed to create CaptureSession` が発生する場合:**
  過去の強制終了等でカメラデーモンがセッションを保持している状態です。以下のコマンドでリセットしてください：
  ```bash
  sudo systemctl restart nvargus-daemon
  ```
- **サーボが動かない / I2C 初期化エラー:**
  1. `i2cdetect -y -r 1` で `0x40` が見えているか確認してください。
  2. PCA9685 の `V+` 端子にサーボ用の 5V 外部電源が給電されているか確認してください。
