import os
import sys

# Suppress OpenCV and MediaPipe / glog warning logs
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import signal
import argparse
import threading
import queue
import cv2
import mediapipe as mp

def parse_args():
    parser = argparse.ArgumentParser(description="MediaPipe Face Detection on Jetson Nano (Large Display & High FPS)")
    parser.add_argument("--preset", type=str, default="large", choices=["large", "smooth", "hd"],
                        help="Preset mode: 'large' (960x540, ~45-50 FPS), 'smooth' (640x360, solid 60 FPS), 'hd' (1280x720, ~32 FPS)")
    parser.add_argument("--width", type=int, default=None, help="Camera capture width (overrides preset)")
    parser.add_argument("--height", type=int, default=None, help="Camera capture height (overrides preset)")
    parser.add_argument("--win-width", type=int, default=None, help="Display window width (overrides preset)")
    parser.add_argument("--win-height", type=int, default=None, help="Display window height (overrides preset)")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode (press 'f' to toggle)")
    parser.add_argument("--conf", type=float, default=0.6, help="Minimum detection confidence (default: 0.6)")
    parser.add_argument("--model", type=int, default=0, choices=[0, 1], help="Model selection: 0=short-range, 1=full-range")
    args = parser.parse_args()

    presets = {
        "large":  {"width": 480, "height": 270, "win_width": 960,  "win_height": 540},
        "smooth": {"width": 320, "height": 180, "win_width": 640,  "win_height": 360},
        "hd":     {"width": 640, "height": 360, "win_width": 1280, "win_height": 720},
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

def get_camera_capture(width=480, height=270, framerate=60):
    # CSI カメラ (Raspberry Pi Camera Module V2 など) 用 GStreamer パイプライン (60 FPS & 16:9)
    csi_pipeline = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
    )

    # 1. まず CSI カメラの起動を試行
    cap = cv2.VideoCapture(csi_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print(f"CSI Camera (nvarguscamerasrc {framerate} FPS, {width}x{height}) opened successfully.")
            return cap
        else:
            print("\n【注意】CSI カメラのセッション作成に失敗しました (CaptureSession Error)。")
            print("カメラデーモンが前回のセッションを保持している可能性があります。")
            print("解決方法: 別ターミナルで 'sudo systemctl restart nvargus-daemon' を実行してください。\n")
            cap.release()

    # 2. CSI が開けない場合は USB カメラ (/dev/video0) をオープン
    print("CSI camera not found. Falling back to USB Camera (/dev/video0)...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, framerate)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def display_worker(disp_queue, stop_event, win_width, win_height, start_fullscreen=False):
    window_name = "MediaPipe Face Detection (Jetson Nano)"
    # WINDOW_NORMAL を使用してユーザーがウィンドウ端をドラッグして自由に拡大縮小可能にする
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, win_width, win_height)

    is_fullscreen = start_fullscreen
    if is_fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    disp_fps_smooth = 60.0
    prev_disp_time = time.time()

    while not stop_event.is_set():
        try:
            item = disp_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        frame, detections, infer_fps = item

        now = time.time()
        disp_fps = 1.0 / (now - prev_disp_time + 1e-6)
        prev_disp_time = now
        disp_fps_smooth = 0.9 * disp_fps_smooth + 0.1 * disp_fps

        # 指定の表示解像度にリサイズして描画
        if frame.shape[1] != win_width or frame.shape[0] != win_height:
            disp_frame = cv2.resize(frame, (win_width, win_height), interpolation=cv2.INTER_LINEAR)
        else:
            disp_frame = frame

        h, w = disp_frame.shape[:2]

        if detections:
            for d in detections:
                box = d.location_data.relative_bounding_box
                x1 = max(0, int(box.xmin * w))
                y1 = max(0, int(box.ymin * h))
                x2 = min(w, int((box.xmin + box.width) * w))
                y2 = min(h, int((box.ymin + box.height) * h))

                # 顔バウンディングボックス (すっきりした細い線幅 1)
                cv2.rectangle(disp_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # 信頼度スコア表示 (控えめなフォントサイズ 0.4)
                score = d.score[0] if d.score else 0.0
                cv2.putText(
                    disp_frame, f"{score*100:.0f}%", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA
                )

                # 顔のキーポイント (両目・鼻・口・両耳)
                for kp in d.location_data.relative_keypoints:
                    kx = int(kp.x * w)
                    ky = int(kp.y * h)
                    cv2.circle(disp_frame, (kx, ky), 2, (0, 0, 255), -1, cv2.LINE_AA)

        # FPS 表示 (推論FPS / 描画FPS 両方を分かりやすく表示、アンチエイリアス描画)
        info_text = f"FPS: {infer_fps:.1f} (Disp: {disp_fps_smooth:.1f})"
        cv2.putText(
            disp_frame, info_text, (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA
        )

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

def main():
    args = parse_args()
    cap = get_camera_capture(args.width, args.height, framerate=60)
    if not cap.isOpened():
        print("Error: Could not open any video source.")
        sys.exit(1)

    mp_face = mp.solutions.face_detection
    disp_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    def sig_handler(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # GUI 描画スレッドをバックグラウンド起動 (推論・キャプチャループのブロックを防止)
    disp_thread = threading.Thread(
        target=display_worker,
        args=(disp_queue, stop_event, args.win_width, args.win_height, args.fullscreen),
        daemon=True
    )
    disp_thread.start()

    with mp_face.FaceDetection(model_selection=args.model, min_detection_confidence=args.conf) as detector:
        # GPU / パイプラインのウォームアップ
        print("Warming up GPU delegate...")
        for _ in range(15):
            ret, frame = cap.read()
            if ret and frame is not None:
                detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        print(f"Face detection started (Preset: {args.preset}, Window: {args.win_width}x{args.win_height}, Capture: {args.width}x{args.height}).")
        print("Controls: 'f' to toggle fullscreen, 'q' or 'ESC' to quit.")
        prev_time = time.time()
        last_print_time = time.time()
        fps_smooth = 60.0

        try:
            while cap.isOpened() and not stop_event.is_set():
                success, frame = cap.read()
                if not success or frame is None:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(frame_rgb)

                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time + 1e-6)
                prev_time = curr_time
                fps_smooth = 0.9 * fps_smooth + 0.1 * fps

                # 定期的にコンソールにFPSを出力
                if curr_time - last_print_time >= 2.0:
                    print(f"Current Face Detection FPS: {fps_smooth:.1f}")
                    last_print_time = curr_time

                # 最新フレームを描画キューに投入 (未処理フレームは即時ドロップして最新性を維持)
                if disp_queue.full():
                    try:
                        disp_queue.get_nowait()
                    except queue.Empty:
                        pass
                disp_queue.put((frame, results.detections, fps_smooth))

        except Exception as e:
            print(f"Exception: {e}")

    stop_event.set()
    disp_thread.join(timeout=1.0)
    cap.release()
    print("Stream closed successfully.")

if __name__ == "__main__":
    main()
