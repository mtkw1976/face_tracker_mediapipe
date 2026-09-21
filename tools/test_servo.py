#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_servo.py
PCA9685 16チャンネル PWM サーボドライバ 動作確認テストスクリプト
- Pan (CH0) および Tilt (CH1) の安全なスロー動作テスト
- 詳細な自己診断 & エラーハンドリング (I2C 接続確認、権限確認、例外追跡)
- Ctrl+C (中断) やエラー時でも必ず安全に 90 度 (センター) へ復帰
"""

import sys
import time
import traceback

# 1. ライブラリ読み込み
try:
    from adafruit_servokit import ServoKit
except ImportError:
    print("【エラー】adafruit_servokit が見つかりません。")
    print("Python 3 環境で実行されているか確認してください。")
    print("  実行コマンド: python3 test_servo.py")
    sys.exit(1)

PAN_CH = 0
TILT_CH = 1

# パルス幅および物理安全リミット
PULSE_MIN = 500
PULSE_MAX = 2400
PAN_MIN, PAN_MAX = 20.0, 160.0
TILT_MIN, TILT_MAX = 35.0, 145.0

def move_servo_smooth(kit, channel, start_angle, target_angle, step=1.0, delay=0.02):
    """
    指定した角度まで徐々に動かす安全制御関数
    step: 1回あたりの変化量 (度)
    delay: ステップ間の待機時間 (秒)
    """
    angle = float(start_angle)
    target = float(target_angle)

    if angle < target:
        while angle < target:
            angle = min(angle + step, target)
            kit.servo[channel].angle = angle
            time.sleep(delay)
    else:
        while angle > target:
            angle = max(angle - step, target)
            kit.servo[channel].angle = angle
            time.sleep(delay)
    return target

def main():
    print("=== サーボ安全動作テスト開始 ===")

    # 2. PCA9685 I2C 接続初期化
    print(f"I2C アドレス 0x40 の PCA9685 を初期化中...")
    try:
        kit = ServoKit(channels=16, i2c=None, address=0x40)
        kit.servo[PAN_CH].set_pulse_width_range(PULSE_MIN, PULSE_MAX)
        kit.servo[TILT_CH].set_pulse_width_range(PULSE_MIN, PULSE_MAX)
        print("PCA9685 の初期化に成功しました。")
    except Exception as e:
        print(f"\n【I2C初期化エラー】: {e}")
        print("以下の項目を確認してください:")
        print("1. I2Cバスにデバイスが認識されているか: 'i2cdetect -y -r 1' で 0x40 が表示されるか")
        print("2. 権限があるか: ユーザーが i2c グループに所属しているか ('groups')")
        print("3. 配線 (SDA/SCL/VCC/GND) が緩んでいないか")
        traceback.print_exc()
        sys.exit(1)

    current_pan = 90.0
    current_tilt = 90.0

    try:
        # 1. センター (90度) に設定
        print("\n1. センター位置 (Pan=90.0°, Tilt=90.0°) に設定中...")
        kit.servo[PAN_CH].angle = 90.0
        kit.servo[TILT_CH].angle = 90.0
        time.sleep(1.0)
        print("   センター位置完了")

        # 2. 水平 (Pan) スロー移動テスト: 90 -> 45 -> 135 -> 90
        print("\n2. Pan (水平: CH0) スロー移動テスト開始")
        print("   -> 45度 へ移動中...")
        current_pan = move_servo_smooth(kit, PAN_CH, current_pan, 45.0, step=1.0, delay=0.02)
        time.sleep(0.5)

        print("   -> 135度 へ移動中...")
        current_pan = move_servo_smooth(kit, PAN_CH, current_pan, 135.0, step=1.0, delay=0.02)
        time.sleep(0.5)

        print("   -> 90度 (中央) へ移動中...")
        current_pan = move_servo_smooth(kit, PAN_CH, current_pan, 90.0, step=1.0, delay=0.02)
        time.sleep(0.5)
        print("   Pan テスト完了")

        # 3. 垂直 (Tilt) スロー移動テスト: 90 -> 60 -> 120 -> 90
        print("\n3. Tilt (垂直: CH1) スロー移動テスト開始")
        print("   -> 60度 へ移動中...")
        current_tilt = move_servo_smooth(kit, TILT_CH, current_tilt, 60.0, step=1.0, delay=0.02)
        time.sleep(0.5)

        print("   -> 120度 へ移動中...")
        current_tilt = move_servo_smooth(kit, TILT_CH, current_tilt, 120.0, step=1.0, delay=0.02)
        time.sleep(0.5)

        print("   -> 90度 (中央) へ移動中...")
        current_tilt = move_servo_smooth(kit, TILT_CH, current_tilt, 90.0, step=1.0, delay=0.02)
        time.sleep(0.5)
        print("   Tilt テスト完了")

        print("\n=== すべてのサーボ動作テストが正常に完了しました ===")

    except KeyboardInterrupt:
        print("\n[中断] Ctrl+C が押されました。安全に中央へ戻します...")
    except Exception as e:
        print(f"\n【テスト実行中エラー】: {e}")
        traceback.print_exc()
    finally:
        print("サーボを 90度 (中央) に安全復帰しています...")
        try:
            kit.servo[PAN_CH].angle = 90.0
            kit.servo[TILT_CH].angle = 90.0
        except Exception:
            pass
        print("テスト終了。")

if __name__ == "__main__":
    main()