#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
face_tracker_mediapipe.py
Jetson Nano 上で CSI カメラと MediaPipe (GPU デリゲート) を使用し、
60 FPS で顔検出および 2軸 (Pan / Tilt) サーボ追従を行うスクリプト。

face_tracker_caffe.py および face_tracker_yunet.py の長所を統合:
- 60 FPS 完全リアルタイム GPU 顔検出 (MediaPipe OpenGL ES デリゲート)
- PCA9685 I2C サーボ制御 (パルス幅 500-2400us、物理リミット保護)
- PD 制御による滑らかで振動のない追従 (不感帯・変位角リミッター完備)
- 非同期 GUI 描画スレッド (X11 描画負荷による 60 FPS 追従ループ遅延の完全防止)
- 顔ロスト時の位置保持 & 一定時間経過後のスムーズな自動中央復帰 (ソフトリターン)
- 終了時 (Ctrl+C / 'q') の安全なサーボ初期位置 (90度) 復帰
"""

import os
import sys

# Python 2 での実行を防止
if sys.version_info[0] < 3:
    sys.exit("エラー: Python 3 で実行してください (例: python3 face_tracker_mediapipe.py)")

# OpenCV / MediaPipe / glog の不要なログ・Warningを抑制
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import signal
import argparse
import threading
import queue
import traceback
import cv2
import numpy as np
import mediapipe as mp

# サーボ制御用ライブラリ
try:
    from adafruit_servokit import ServoKit
    SERVOKIT_AVAILABLE = True
except ImportError:
    SERVOKIT_AVAILABLE = False
    print("Warning: adafruit_servokit が見つかりません。サーボ制御はモック動作になります。")

# -------------------------------------------------------------
# 1. 引数設定
# -------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="MediaPipe Face Tracker at 60 FPS on Jetson Nano")
    parser.add_argument("--preset", type=str, default="large", choices=["large", "smooth", "hd", "distance"],
                        help="Display preset: 'large' (640x360 cap, 960x540 win, ~55-60 FPS), 'smooth' (480x270 cap, 640x360 win, 60 FPS), 'hd' (960x540 cap, 1280x720 win), 'distance' (960x540 cap for 1m-2m)")
    parser.add_argument("--width", type=int, default=None, help="Camera capture width")
    parser.add_argument("--height", type=int, default=None, help="Camera capture height")
    parser.add_argument("--win-width", type=int, default=None, help="Window display width")
    parser.add_argument("--win-height", type=int, default=None, help="Window display height")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode (press 'f' to toggle)")
    parser.add_argument("--conf", type=float, default=0.6, help="Face detection confidence threshold (default: 0.6)")
    parser.add_argument("--no-servo", action="store_true", help="Disable physical servo output (test camera & detection only)")
    parser.add_argument("--pan-ch", type=int, default=0, help="PCA9685 Pan channel (default: 0)")
    parser.add_argument("--tilt-ch", type=int, default=1, help="PCA9685 Tilt channel (default: 1)")
    parser.add_argument("--init-pan", type=float, default=90.0, help="Initial Pan angle (default: 90.0)")
    parser.add_argument("--init-tilt", type=float, default=75.0, help="Initial Tilt angle facing human eye level (default: 75.0)")
    parser.add_argument("--search-pattern", type=str, default="wave", choices=["wave", "raster"],
                        help="Search pattern: 'wave' (continuous 2D wave), 'raster' (multi-level stepped sweep)")
    parser.add_argument("--search-pan-min", "--search-min", dest="search_pan_min", type=float, default=45.0, help="Search sweep minimum Pan angle (default: 45.0)")
    parser.add_argument("--search-pan-max", "--search-max", dest="search_pan_max", type=float, default=135.0, help="Search sweep maximum Pan angle (default: 135.0)")
    parser.add_argument("--search-pan-speed", "--search-speed", dest="search_pan_speed", type=float, default=0.5, help="Search sweep Pan speed deg/step (default: 0.5)")
    parser.add_argument("--search-tilt-min", type=float, default=55.0, help="Search sweep minimum Tilt angle (default: 55.0, upward)")
    parser.add_argument("--search-tilt-max", type=float, default=95.0, help="Search sweep maximum Tilt angle (default: 95.0, horizontal/slightly down)")
    parser.add_argument("--search-tilt-speed", type=float, default=0.3, help="Search sweep Tilt speed deg/step (default: 0.3)")
    parser.add_argument("--hold-time", type=float, default=1.0, help="Hold duration before searching in seconds (default: 1.0)")
    parser.add_argument("--no-search", action="store_true", help="Disable active search (return to center on face lost)")
    args = parser.parse_args()

    presets = {
        "large":    {"width": 640, "height": 360, "win_width": 960,  "win_height": 540},
        "smooth":   {"width": 480, "height": 270, "win_width": 640,  "win_height": 360},
        "hd":       {"width": 960, "height": 540, "win_width": 1280, "win_height": 720},
        "distance": {"width": 960, "height": 540, "win_width": 960,  "win_height": 540},
    }
    cfg = presets[args.preset]
    if args.width is None:
        args.width = cfg["width"]
    if args.height is None:
        args.height = cfg["height"]
    if args.win_width is None:
        args.win_width = cfg["win_width"]
    if args.win_height is None:
        args.win_height = cfg["win_height"]

    return args

# -------------------------------------------------------------
# 2. 安全サーボコントローラ
# -------------------------------------------------------------
class SafeServoController:
    def __init__(self, pan_ch=0, tilt_ch=1, enabled=True,
                 init_pan=90.0, init_tilt=75.0,
                 search_enabled=True, search_pattern="wave",
                 search_pan_min=45.0, search_pan_max=135.0, search_pan_speed=0.5,
                 search_tilt_min=55.0, search_tilt_max=95.0, search_tilt_speed=0.3,
                 hold_time=1.0):
        self.enabled = enabled and SERVOKIT_AVAILABLE
        self.pan_ch = pan_ch
        self.tilt_ch = tilt_ch

        # 物理安全角リミット (face_tracker_caffe.py 準拠)
        self.PAN_MIN, self.PAN_MAX = 20.0, 160.0
        self.TILT_MIN, self.TILT_MAX = 35.0, 145.0

        # 方向係数 (画面右に顔があるとき右を向き、画面上に顔があるとき上を向く)
        self.PAN_DIR = -1
        self.TILT_DIR = 1

        # PD 制御ゲイン (60 FPS 追従向けに最適化)
        self.KP_PAN = 2.0
        self.KP_TILT = 1.8
        self.KD_PAN = 0.12
        self.KD_TILT = 0.10

        # 急激な振り切れを防止する最大ステップ角 (度 / 更新)
        self.MAX_DELTA = 1.0
        # 振動防止のための不感帯 (画面中央 ±3.5%)
        self.DEAD_ZONE = 0.035

        # 2軸 (Pan & Tilt) 捜査 (パトロール・探索) パラメータ
        self.init_pan = init_pan
        self.init_tilt = init_tilt
        self.search_enabled = search_enabled
        self.search_pattern = search_pattern

        self.search_pan_min = max(self.PAN_MIN, min(self.PAN_MAX, search_pan_min))
        self.search_pan_max = max(self.PAN_MIN, min(self.PAN_MAX, search_pan_max))
        self.search_pan_speed = search_pan_speed
        self.search_pan_dir = -1  # -1: 右(角度小), +1: 左(角度大)

        self.search_tilt_min = max(self.TILT_MIN, min(self.TILT_MAX, search_tilt_min))
        self.search_tilt_max = max(self.TILT_MIN, min(self.TILT_MAX, search_tilt_max))
        self.search_tilt_speed = search_tilt_speed
        self.search_tilt_dir = -1  # -1: 上(角度小、顔探索優先), +1: 下

        self.search_tilt_levels = [self.init_tilt, self.search_tilt_min, self.init_tilt, self.search_tilt_max]
        self.search_tilt_level_idx = 0
        self.search_tilt_target = self.init_tilt

        self.hold_time = hold_time        # ロスト後に位置保持する猶予時間 (秒)

        self.pan_angle = self.init_pan
        self.tilt_angle = self.init_tilt
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.last_update_time = time.time()
        self.last_target_time = 0.0
        self.kit = None

        if self.enabled:
            print("I2C アドレス 0x40 の PCA9685 を初期化中...")
            try:
                self.kit = ServoKit(channels=16, i2c=None, address=0x40)
                self.kit.servo[self.pan_ch].set_pulse_width_range(500, 2400)
                self.kit.servo[self.tilt_ch].set_pulse_width_range(500, 2400)
                self.kit.servo[self.pan_ch].angle = self.pan_angle
                self.kit.servo[self.tilt_ch].angle = self.tilt_angle
                print(f"PCA9685 の初期化に成功しました (初期位置: Pan={self.pan_angle}°, Tilt={self.tilt_angle}°)。")
            except Exception as e:
                print(f"\n【I2C初期化エラー】: {e}")
                print("以下の項目を確認してください:")
                print("1. I2Cバスにデバイスが認識されているか: 'i2cdetect -y -r 1' で 0x40 が表示されるか")
                print("2. 権限があるか: ユーザーが i2c グループに所属しているか ('groups')")
                print("3. 配線 (SDA/SCL/VCC/GND) が緩んでいないか")
                traceback.print_exc()
                print("サーボ制御を無効化し、検出・画面表示のみで継続します (--no-servo モード)。\n")
                self.enabled = False
                self.kit = None

    def update(self, target_norm_x, target_norm_y):
        """
        顔の中心 (または鼻先) の正規化座標 (0.0 〜 1.0) を受け取り、PD制御でサーボ角を更新
        """
        now = time.time()
        self.last_target_time = now

        # 中心からの偏差 (-0.5 〜 +0.5)
        err_x = target_norm_x - 0.5
        err_y = target_norm_y - 0.5

        dt = now - self.last_update_time
        if dt < 0.015:  # サーボ更新は最大 ~50-60 Hz にレート制御
            return self.pan_angle, self.tilt_angle, err_x, err_y

        self.last_update_time = now

        # Pan (水平) 軸制御
        if abs(err_x) > self.DEAD_ZONE:
            d_err_x = (err_x - self.prev_err_x) / dt
            step_pan = self.PAN_DIR * (err_x * self.KP_PAN + d_err_x * self.KD_PAN)
            step_pan = max(-self.MAX_DELTA, min(self.MAX_DELTA, step_pan))
            self.pan_angle = max(self.PAN_MIN, min(self.PAN_MAX, self.pan_angle + step_pan))
        self.prev_err_x = err_x

        # Tilt (垂直) 軸制御
        if abs(err_y) > self.DEAD_ZONE:
            d_err_y = (err_y - self.prev_err_y) / dt
            step_tilt = self.TILT_DIR * (err_y * self.KP_TILT + d_err_y * self.KD_TILT)
            step_tilt = max(-self.MAX_DELTA, min(self.MAX_DELTA, step_tilt))
            self.tilt_angle = max(self.TILT_MIN, min(self.TILT_MAX, self.tilt_angle + step_tilt))
        self.prev_err_y = err_y

        # ハードウェア出力
        if self.enabled and self.kit is not None:
            self.kit.servo[self.pan_ch].angle = self.pan_angle
            self.kit.servo[self.tilt_ch].angle = self.tilt_angle

        return self.pan_angle, self.tilt_angle, err_x, err_y

    def handle_face_lost(self):
        """
        顔未検出時の動作:
        1. 直近 hold_time 秒以内: 一時的な遮蔽や瞬きを考慮し、現在の角度をホールド (HOLD)
        2. 猶予時間経過後:
           - 捜査有効時 (デフォルト): 2軸 (Pan & Tilt) の往復走査で顔を探す (SEARCHING)
           - 捜査無効時 (--no-search): 緩やかに中央復帰 (RETURNING)
        """
        now = time.time()
        lost_duration = now - self.last_target_time

        if lost_duration < self.hold_time:
            return "HOLD"

        if not self.search_enabled:
            # 捜査無効時は初期角度 (init_pan, init_tilt) へ復帰
            dt = now - self.last_update_time
            if dt >= 0.03:
                step = 0.5
                if abs(self.pan_angle - self.init_pan) > 0.5:
                    self.pan_angle += step if self.pan_angle < self.init_pan else -step
                if abs(self.tilt_angle - self.init_tilt) > 0.5:
                    self.tilt_angle += step if self.tilt_angle < self.init_tilt else -step
                if self.enabled and self.kit is not None:
                    try:
                        self.kit.servo[self.pan_ch].angle = self.pan_angle
                        self.kit.servo[self.tilt_ch].angle = self.tilt_angle
                    except Exception:
                        pass
                self.last_update_time = now
                self.prev_err_x = 0.0
                self.prev_err_y = 0.0
            return "RETURNING"

        # アクティブ 2軸 (Pan & Tilt) 捜査
        dt = now - self.last_update_time
        if dt >= 0.02:  # 約50Hz 周期で滑らかに走査
            self.last_update_time = now

            if self.search_pattern == "raster":
                # ラスタースキャン: 水平走査端に達するごとに Tilt 高さを段階切り替え (上・中・下)
                self.pan_angle += self.search_pan_dir * self.search_pan_speed
                pan_hit_boundary = False
                if self.pan_angle >= self.search_pan_max:
                    self.pan_angle = self.search_pan_max
                    self.search_pan_dir = -1
                    pan_hit_boundary = True
                elif self.pan_angle <= self.search_pan_min:
                    self.pan_angle = self.search_pan_min
                    self.search_pan_dir = 1
                    pan_hit_boundary = True

                if pan_hit_boundary:
                    self.search_tilt_level_idx = (self.search_tilt_level_idx + 1) % len(self.search_tilt_levels)
                    self.search_tilt_target = self.search_tilt_levels[self.search_tilt_level_idx]

                # Tilt を目標レベルへスムーズに追従移動
                if abs(self.tilt_angle - self.search_tilt_target) > 0.5:
                    tilt_step = 0.4
                    self.tilt_angle += tilt_step if self.tilt_angle < self.search_tilt_target else -tilt_step

            else:
                # 連続波状 (Wave) スキャン: Pan と Tilt を異なる周期で同時走査し 2D 空間全域を網羅
                # 1. Pan 軸走査 (水平)
                self.pan_angle += self.search_pan_dir * self.search_pan_speed
                if self.pan_angle >= self.search_pan_max:
                    self.pan_angle = self.search_pan_max
                    self.search_pan_dir = -1
                elif self.pan_angle <= self.search_pan_min:
                    self.pan_angle = self.search_pan_min
                    self.search_pan_dir = 1

                # 2. Tilt 軸走査 (垂直: 上下に緩やかに連続往復、初期は上方走査優先)
                self.tilt_angle += self.search_tilt_dir * self.search_tilt_speed
                if self.tilt_angle >= self.search_tilt_max:
                    self.tilt_angle = self.search_tilt_max
                    self.search_tilt_dir = -1
                elif self.tilt_angle <= self.search_tilt_min:
                    self.tilt_angle = self.search_tilt_min
                    self.search_tilt_dir = 1

            # ハードウェア出力
            if self.enabled and self.kit is not None:
                try:
                    self.kit.servo[self.pan_ch].angle = self.pan_angle
                    self.kit.servo[self.tilt_ch].angle = self.tilt_angle
                except Exception:
                    pass

            self.prev_err_x = 0.0
            self.prev_err_y = 0.0

        return "SEARCHING"

    def smooth_reset_to_center(self, target_pan=None, target_tilt=None, step=0.8, delay=0.015):
        """終了時のソフトリターン (急激な衝撃を与えず中央へ戻す)"""
        if not self.enabled or self.kit is None:
            return
        if target_pan is None:
            target_pan = self.init_pan
        if target_tilt is None:
            target_tilt = self.init_tilt
        p, t = self.pan_angle, self.tilt_angle
        try:
            while abs(p - target_pan) > 0.5 or abs(t - target_tilt) > 0.5:
                if abs(p - target_pan) > 0.5:
                    p += step if p < target_pan else -step
                    p = max(self.PAN_MIN, min(self.PAN_MAX, p))
                    self.kit.servo[self.pan_ch].angle = p
                if abs(t - target_tilt) > 0.5:
                    t += step if t < target_tilt else -step
                    t = max(self.TILT_MIN, min(self.TILT_MAX, t))
                    self.kit.servo[self.tilt_ch].angle = t
                time.sleep(delay)
            self.kit.servo[self.pan_ch].angle = target_pan
            self.kit.servo[self.tilt_ch].angle = target_tilt
        except Exception as e:
            print(f"サーボ復帰中の通信エラー (無視): {e}")

# -------------------------------------------------------------
# 3. カメラ初期化 (CSI 60 FPS GStreamer パイプライン)
# -------------------------------------------------------------
def get_camera_capture(width=640, height=360, framerate=60):
    csi_pipeline = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv flip-method=0 interpolation-method=1 ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
    )

    cap = cv2.VideoCapture(csi_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        # 実際に1フレーム読めるか検証 (nvargus-daemon のハング状態を検出)
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print(f"CSI Camera (nvarguscamerasrc {framerate} FPS, {width}x{height}) opened successfully.")
            return cap
        else:
            print("\n【注意】CSI カメラのセッション作成に失敗しました (CaptureSession Error)。")
            print("カメラデーモンが前回のセッションを保持している可能性があります。")
            print("解決方法: 別ターミナルで 'sudo systemctl restart nvargus-daemon' を実行してください。\n")
            cap.release()

    print("CSI camera not found or busy. Falling back to USB Camera (/dev/video0)...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, framerate)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

# -------------------------------------------------------------
# 4. 非同期 GUI 描画ワーカー (推論ループのブロックを防止)
# -------------------------------------------------------------
def display_worker(disp_queue, stop_event, win_width, win_height, start_fullscreen=False,
                   radar_bounds=(45.0, 135.0, 55.0, 95.0)):
    window_name = "Face Tracker 60FPS (MediaPipe & Jetson Nano)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, win_width, win_height)

    is_fullscreen = start_fullscreen
    if is_fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    disp_fps_smooth = 60.0
    prev_disp_time = time.time()
    pan_min, pan_max, tilt_min, tilt_max = radar_bounds

    while not stop_event.is_set():
        try:
            item = disp_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        frame, face_info, tracking_status, pan_ang, tilt_ang, err_vals, infer_fps = item

        now = time.time()
        disp_fps = 1.0 / (now - prev_disp_time + 1e-6)
        prev_disp_time = now
        disp_fps_smooth = 0.9 * disp_fps_smooth + 0.1 * disp_fps

        # 指定の表示解像度にリサイズ
        if frame.shape[1] != win_width or frame.shape[0] != win_height:
            disp_frame = cv2.resize(frame, (win_width, win_height), interpolation=cv2.INTER_LINEAR)
        else:
            disp_frame = frame

        h, w = disp_frame.shape[:2]
        cx_frame, cy_frame = w // 2, h // 2

        # 画面中央の不感帯ゾーン (薄いクロスヘアと矩形)
        dz_w = int(0.035 * w)
        dz_h = int(0.035 * h)
        cv2.rectangle(disp_frame, (cx_frame - dz_w, cy_frame - dz_h),
                      (cx_frame + dz_w, cy_frame + dz_h), (80, 80, 80), 1, cv2.LINE_AA)
        cv2.drawMarker(disp_frame, (cx_frame, cy_frame), (120, 120, 120),
                       markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1, line_type=cv2.LINE_AA)

        # 顔および追跡ターゲットの描画
        if face_info is not None:
            box, nose_pt, score = face_info
            x1 = max(0, int(box[0] * w))
            y1 = max(0, int(box[1] * h))
            x2 = min(w, int(box[2] * w))
            y2 = min(h, int(box[3] * h))

            # 顔バウンディングボックス (すっきりした細線幅 1)
            cv2.rectangle(disp_frame, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

            # 信頼度
            cv2.putText(disp_frame, f"{score*100:.0f}%", (x1, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            # 追従ターゲット (鼻先または顔中心) のレティクル
            tx, ty = int(nose_pt[0] * w), int(nose_pt[1] * h)
            cv2.circle(disp_frame, (tx, ty), 5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(disp_frame, (tx, ty), 1, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.line(disp_frame, (tx - 8, ty), (tx + 8, ty), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(disp_frame, (tx, ty - 8), (tx, ty + 8), (0, 0, 255), 1, cv2.LINE_AA)

            # 画面中心と追従ターゲットを結ぶベクトル線
            cv2.line(disp_frame, (cx_frame, cy_frame), (tx, ty), (0, 255, 255), 1, cv2.LINE_AA)

        # ステータスバッジ (色分け)
        status_colors = {
            "TRACKING": (0, 255, 0),       # 緑: 追尾中
            "HOLD": (0, 215, 255),         # 黄: 一時ロスト(位置保持)
            "SEARCHING": (0, 200, 255),    # アンバー/オレンジ: 顔を捜査中
            "RETURNING": (200, 150, 50),   # 青系: 中央復帰中
        }
        badge_color = status_colors.get(tracking_status, (200, 200, 200))

        # HUD テレメトリ情報 (左上)
        err_x, err_y = err_vals
        cv2.putText(disp_frame, f"STATUS: {tracking_status}", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, badge_color, 1, cv2.LINE_AA)
        cv2.putText(disp_frame, f"FPS: {infer_fps:.1f} (Disp: {disp_fps_smooth:.1f})", (12, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(disp_frame, f"Pan: {pan_ang:5.1f} deg | Tilt: {tilt_ang:5.1f} deg", (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(disp_frame, f"Offset: dX={err_x:+.3f}, dY={err_y:+.3f}", (12, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # 下部ステータス表示 & 2D 捜査レーダー
        if tracking_status == "SEARCHING":
            search_text = f"[ SEARCHING: PAN {pan_ang:5.1f} deg | TILT {tilt_ang:5.1f} deg ]"
            cv2.putText(disp_frame, search_text, (cx_frame - 180, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

            # 2D レーダーミニマップ (Pan x Tilt 探索状況)
            radar_w, radar_h = 100, 50
            radar_x = w - radar_w - 15
            radar_y = h - radar_h - 15
            cv2.rectangle(disp_frame, (radar_x, radar_y), (radar_x + radar_w, radar_y + radar_h), (40, 40, 40), -1)
            cv2.rectangle(disp_frame, (radar_x, radar_y), (radar_x + radar_w, radar_y + radar_h), (80, 80, 80), 1)
            # 十字中心線
            cv2.line(disp_frame, (radar_x + radar_w // 2, radar_y), (radar_x + radar_w // 2, radar_y + radar_h), (60, 60, 60), 1)
            cv2.line(disp_frame, (radar_x, radar_y + radar_h // 2), (radar_x + radar_w, radar_y + radar_h // 2), (60, 60, 60), 1)
            # 現在のカメラ向き (光るアンバードット)
            norm_pan = max(0.0, min(1.0, (pan_ang - pan_min) / (pan_max - pan_min + 1e-6)))
            norm_tilt = max(0.0, min(1.0, (tilt_ang - tilt_min) / (tilt_max - tilt_min + 1e-6)))
            cur_dot_x = int(radar_x + norm_pan * radar_w)
            cur_dot_y = int(radar_y + norm_tilt * radar_h)
            cv2.circle(disp_frame, (cur_dot_x, cur_dot_y), 4, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.putText(disp_frame, "2D RADAR", (radar_x, radar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
        elif tracking_status == "HOLD":
            hold_text = "[ FACE LOST: HOLDING POSITION ]"
            cv2.putText(disp_frame, hold_text, (cx_frame - 130, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)
        elif tracking_status == "TRACKING":
            track_text = "[ TARGET LOCKED ]"
            cv2.putText(disp_frame, track_text, (cx_frame - 65, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(window_name, disp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            stop_event.set()
            break
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, win_width, win_height)

    cv2.destroyAllWindows()

# -------------------------------------------------------------
# 5. メイン追従ループ
# -------------------------------------------------------------
def main():
    args = parse_args()
    cap = get_camera_capture(args.width, args.height, framerate=60)
    if not cap.isOpened():
        print("Error: カメラをオープンできませんでした。")
        sys.exit(1)

    servo = None
    disp_thread = None
    stop_event = threading.Event()

    def sig_handler(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    try:
        servo = SafeServoController(
            pan_ch=args.pan_ch,
            tilt_ch=args.tilt_ch,
            enabled=not args.no_servo,
            init_pan=args.init_pan,
            init_tilt=args.init_tilt,
            search_enabled=not args.no_search,
            search_pattern=args.search_pattern,
            search_pan_min=args.search_pan_min,
            search_pan_max=args.search_pan_max,
            search_pan_speed=args.search_pan_speed,
            search_tilt_min=args.search_tilt_min,
            search_tilt_max=args.search_tilt_max,
            search_tilt_speed=args.search_tilt_speed,
            hold_time=args.hold_time
        )

        mp_face = mp.solutions.face_detection
        disp_queue = queue.Queue(maxsize=1)

        # 非同期表示スレッドの起動
        radar_bounds = (args.search_pan_min, args.search_pan_max, args.search_tilt_min, args.search_tilt_max)
        disp_thread = threading.Thread(
            target=display_worker,
            args=(disp_queue, stop_event, args.win_width, args.win_height, args.fullscreen, radar_bounds),
            daemon=True
        )
        disp_thread.start()

        print("=== MediaPipe 60 FPS 顔追尾システムを開始 ===")
        print(f"モード: Preset={args.preset}, 解像度: {args.width}x{args.height}, 画面: {args.win_width}x{args.win_height}, 検出信頼度: {args.conf}")
        print(f"初期角度: Pan={args.init_pan}°, Tilt={args.init_tilt}° (着座・対面アイレベル)")
        search_desc = f"ON ({args.search_pattern.upper()} 2D走査: Pan {args.search_pan_min}°〜{args.search_pan_max}°, Tilt {args.search_tilt_min}°〜{args.search_tilt_max}°)" if not args.no_search else "OFF (中央復帰)"
        print(f"捜査モード: {search_desc}")
        print("操作: 'f' でフルスクリーン切替, 'q' または ESC で安全停止")

        with mp_face.FaceDetection(model_selection=0, min_detection_confidence=args.conf) as detector:
            # GPU delegate のウォームアップ (初回起動スパイクの回避)
            print("GPU デリゲートをウォームアップ中...")
            for _ in range(15):
                ret, frame = cap.read()
                if ret and frame is not None:
                    detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            prev_time = time.time()
            last_print_time = time.time()
            fps_smooth = 60.0
            err_x, err_y = 0.0, 0.0

            while cap.isOpened() and not stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(frame_rgb)

                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time + 1e-6)
                prev_time = curr_time
                fps_smooth = 0.9 * fps_smooth + 0.1 * fps

                face_info = None
                tracking_status = "SEARCHING"

                if results.detections:
                    # 最も信頼度の高い顔を選択
                    best_detection = max(results.detections, key=lambda d: d.score[0] if d.score else 0)
                    score = best_detection.score[0] if best_detection.score else 0.0
                    box = best_detection.location_data.relative_bounding_box
                    bbox = [box.xmin, box.ymin, box.xmin + box.width, box.ymin + box.height]

                    # 追尾ターゲット点: 鼻先キーポイント (keypoints[2]) があればそれを使用、なければ顔中心
                    kps = best_detection.location_data.relative_keypoints
                    if len(kps) > 2:
                        target_x = kps[2].x
                        target_y = kps[2].y
                    else:
                        target_x = box.xmin + box.width * 0.5
                        target_y = box.ymin + box.height * 0.5

                    face_info = (bbox, (target_x, target_y), score)

                    # サーボ制御の更新
                    pan_ang, tilt_ang, err_x, err_y = servo.update(target_x, target_y)
                    tracking_status = "TRACKING"
                else:
                    # 顔未検出時のハンドリング (位置保持または中央復帰)
                    tracking_status = servo.handle_face_lost()
                    pan_ang, tilt_ang = servo.pan_angle, servo.tilt_angle

                # 定期コンソールログ
                if curr_time - last_print_time >= 2.0:
                    print(f"[{tracking_status}] FPS: {fps_smooth:4.1f} | Pan: {pan_ang:5.1f}° | Tilt: {tilt_ang:5.1f}° | dX={err_x:+.3f}, dY={err_y:+.3f}")
                    last_print_time = curr_time

                # 描画スレッドへフレームとテレメトリを送信 (最新フレームのみ維持)
                if disp_queue.full():
                    try:
                        disp_queue.get_nowait()
                    except queue.Empty:
                        pass
                disp_queue.put((frame, face_info, tracking_status, pan_ang, tilt_ang, (err_x, err_y), fps_smooth))

    except KeyboardInterrupt:
        print("\n[中断] Ctrl+C が押されました。安全に終了処理へ移行します...")
    except Exception as e:
        print(f"\n【例外発生】: {e}")
        traceback.print_exc()
    finally:
        # 終了処理 (例外・中断・正常終了いずれも確実に実行)
        stop_event.set()
        if disp_thread is not None and disp_thread.is_alive():
            disp_thread.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()

        if servo is not None:
            print(f"サーボを安全に初期位置 (Pan={servo.init_pan}°, Tilt={servo.init_tilt}°) へ復帰しています...")
            servo.smooth_reset_to_center(step=0.8, delay=0.015)
        print("顔追尾システムを正常終了しました。")

if __name__ == "__main__":
    main()
