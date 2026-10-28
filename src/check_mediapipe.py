"""
MediaPipe Pose Landmarker CPU Smoke Test for Cricket Video Frames.
Initializes MediaPipe Tasks PoseLandmarker with official pose_landmarker_lite.task,
runs CPU-only inference on a selected frame from outputs/sample_clips/,
extracts 33 body landmarks with visibility scores, and releases all resources cleanly.
"""

from pathlib import Path
import time
import json
import sys

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV (cv2) is not installed.")
    sys.exit(1)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"[ERROR] MediaPipe Tasks API import failed: {e}")
    sys.exit(1)


def run_mediapipe_smoke_test():
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "models" / "pose_landmarker_lite.task"
    sample_video = base_dir / "outputs" / "sample_clips" / "Cover Drive" / "vid100_41.avi"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("MEDIAPIPE POSE LANDMARKER SMOKE TEST (CPU)")
    print("=" * 76)
    print(f"MediaPipe Version: {mp.__version__}")
    print(f"Model Path: {model_path.relative_to(base_dir)}")
    print(f"Sample Video: {sample_video.relative_to(base_dir)}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model asset missing: {model_path}")
    if not sample_video.exists():
        raise FileNotFoundError(f"Sample video clip missing: {sample_video}")

    # Read the first frame
    cap = cv2.VideoCapture(str(sample_video))
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        raise RuntimeError("Failed to decode test frame from sample video.")

    h, w, c = frame_bgr.shape
    print(f"Loaded Test Frame: {w}x{h} (3 channels)")

    # Prepare MediaPipe Landmarker options
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        num_poses=1,
    )

    # Initialize detector
    init_start = time.perf_counter()
    detector = vision.PoseLandmarker.create_from_options(options)
    init_time_ms = round((time.perf_counter() - init_start) * 1000, 2)
    print(f"Detector Initialized (CPU XNNPACK) in: {init_time_ms} ms")

    # Convert frame BGR to RGB and construct mp.Image
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Run inference
    infer_start = time.perf_counter()
    detection_result = detector.detect(mp_image)
    infer_time_ms = round((time.perf_counter() - infer_start) * 1000, 2)
    print(f"Inference Completed in: {infer_time_ms} ms")

    # Release resources
    detector.close()
    print("PoseLandmarker instance cleanly closed.")

    # Parse results
    num_poses = len(detection_result.pose_landmarks) if detection_result.pose_landmarks else 0
    pose_detected = num_poses > 0

    keypoints_summary = []
    if pose_detected:
        landmarks = detection_result.pose_landmarks[0]
        print(f"\n[SUCCESS] Detected {num_poses} pose with {len(landmarks)} landmarks!")

        # Key joint indices: Nose(0), L/R Shoulders(11,12), L/R Elbows(13,14), L/R Wrists(15,16), L/R Hips(23,24), L/R Knees(25,26), L/R Ankles(27,28)
        joints_to_log = [
            ("nose", 0),
            ("left_shoulder", 11),
            ("right_shoulder", 12),
            ("left_elbow", 13),
            ("right_elbow", 14),
            ("left_wrist", 15),
            ("right_wrist", 16),
            ("left_hip", 23),
            ("right_hip", 24),
            ("left_knee", 25),
            ("right_knee", 26),
            ("left_ankle", 27),
            ("right_ankle", 28),
        ]
        print("-" * 76)
        print(f"{'Joint Name':<16} {'Index':<6} {'X (norm)':<10} {'Y (norm)':<10} {'Z (depth)':<10} {'Visibility':<10}")
        print("-" * 76)
        for name, idx in joints_to_log:
            pt = landmarks[idx]
            vis = getattr(pt, "visibility", 0.0)
            print(f"{name:<16} {idx:<6} {pt.x:<10.4f} {pt.y:<10.4f} {pt.z:<10.4f} {vis:<10.4f}")
            keypoints_summary.append({
                "joint_name": name,
                "index": idx,
                "x": round(float(pt.x), 4),
                "y": round(float(pt.y), 4),
                "z": round(float(pt.z), 4),
                "visibility": round(float(vis), 4),
            })
        print("-" * 76)
    else:
        print("[WARNING] No human pose landmarks detected in test frame.")

    print("=" * 76)
    return {
        "mediapipe_version": mp.__version__,
        "model_file": str(model_path.relative_to(base_dir)).replace("\\", "/"),
        "sample_video": str(sample_video.relative_to(base_dir)).replace("\\", "/"),
        "frame_dimensions": {"width": w, "height": h},
        "pose_detected": pose_detected,
        "landmarks_count": len(detection_result.pose_landmarks[0]) if pose_detected else 0,
        "initialization_time_ms": init_time_ms,
        "inference_time_ms": infer_time_ms,
        "key joints_sample": keypoints_summary,
    }


def main():
    report = run_mediapipe_smoke_test()
    return report


if __name__ == "__main__":
    main()
