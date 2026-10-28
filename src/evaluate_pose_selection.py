"""
Evaluate and compare 2D pose detection on Uniform vs. Motion-Aware selected frames
across all 12 cricket video clips using MediaPipe Tasks PoseLandmarker (CPU-only).

Generates:
1. reports/pose_selection_evaluation.json (Detailed per-frame evaluations)
2. reports/pose_selection_summary.json (Concise aggregate and per-class comparison)
3. outputs/pose_samples/ (At most 3 representative annotated sample images)
"""

from pathlib import Path
import json
import time
import sys
import numpy as np

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
    print(f"[ERROR] MediaPipe Tasks import failed: {e}")
    sys.exit(1)


# MediaPipe Pose connections for drawing skeletal edges
POSE_CONNECTIONS = [
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left face
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right face
    (9, 10),  # Mouth
    (15, 17), (15, 19), (15, 21),  # Left hand
    (16, 18), (16, 20), (16, 22),  # Right hand
    (27, 29), (27, 31),  # Left foot
    (28, 30), (28, 32),  # Right foot
]


def load_selected_frames_dict(video_path: Path, target_indices: set):
    """
    Decodes only the frames whose indices are in target_indices to avoid unnecessary decoding.
    Returns a dict: {frame_idx: bgr_numpy_array}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frames_by_idx = {}
    current_idx = 0
    max_target = max(target_indices) if target_indices else -1

    while cap.isOpened() and current_idx <= max_target:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if current_idx in target_indices:
            frames_by_idx[current_idx] = frame
        current_idx += 1

    cap.release()
    return frames_by_idx


def draw_pose_overlay(frame_bgr, landmarks, clip_label, method_label, frame_idx):
    """Draws skeletal lines, joint circles, and an informative header card on a frame copy."""
    annotated = frame_bgr.copy()
    h, w, _ = annotated.shape

    # Draw connection lines
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            p1, p2 = landmarks[start_idx], landmarks[end_idx]
            v1 = getattr(p1, "visibility", 1.0)
            v2 = getattr(p2, "visibility", 1.0)
            if v1 > 0.3 and v2 > 0.3:
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))
                color = (0, 230, 255) if (v1 > 0.5 and v2 > 0.5) else (100, 160, 200)
                cv2.line(annotated, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw joints
    for idx, lm in enumerate(landmarks):
        vis = getattr(lm, "visibility", 1.0)
        cx, cy = int(lm.x * w), int(lm.y * h)
        if 0 <= cx < w and 0 <= cy < h:
            dot_color = (0, 255, 128) if vis > 0.5 else (0, 140, 255)
            cv2.circle(annotated, (cx, cy), 4, dot_color, -1, cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 5, (20, 20, 20), 1, cv2.LINE_AA)

    # Header banner
    banner_h = 38
    cv2.rectangle(annotated, (0, 0), (w, banner_h), (24, 24, 24), -1)
    tag = f"{clip_label} | {method_label.upper()} | Frame #{frame_idx} | 33 Landmarks"
    cv2.putText(annotated, tag, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    return annotated


def run_evaluation():
    base_dir = Path(__file__).resolve().parents[1]
    motion_json_path = base_dir / "reports" / "motion_analysis.json"
    model_path = base_dir / "models" / "pose_landmarker_lite.task"
    samples_dir = base_dir / "outputs" / "pose_samples"
    reports_dir = base_dir / "reports"

    samples_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not motion_json_path.exists():
        raise FileNotFoundError(f"Motion analysis file not found: {motion_json_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(motion_json_path, "r", encoding="utf-8") as f:
        clips_data = json.load(f)

    print("=" * 86)
    print("2D POSE ESTIMATION COMPARISON: UNIFORM VS MOTION-AWARE (CPU)")
    print(f"Total Clips: {len(clips_data)} | Frames per Method: 6 | Total Evaluations: {len(clips_data)*12}")
    print("=" * 86)

    # Initialize MediaPipe PoseLandmarker
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        num_poses=1,
    )

    detector = vision.PoseLandmarker.create_from_options(options)

    eval_records = []
    saved_overlay_count = 0
    max_overlays = 3

    # Distinct target sample images: spanning different shot classes and both selection methods
    overlay_targets = [
        ("Cover Drive", "motion_aware", 7),
        ("Defensive", "uniform", 0),
        ("Scoop", "motion_aware", 15),
    ]

    try:
        for clip in clips_data:
            clip_name = clip["clip_name"]
            clip_class = clip["class"]
            rel_path = clip["relative_video_path"]
            video_file = base_dir / rel_path

            u_indices = clip["uniform_selected_indices"]
            m_indices = clip["motion_selected_indices"]
            needed_indices = set(u_indices).union(set(m_indices))

            # Decode only selected frames into memory
            frames_cache = load_selected_frames_dict(video_file, needed_indices)

            for method_name, idx_list in [("uniform", u_indices), ("motion_aware", m_indices)]:
                for f_idx in idx_list:
                    if f_idx not in frames_cache:
                        eval_records.append({
                            "clip_name": clip_name,
                            "class": clip_class,
                            "method": method_name,
                            "frame_index": f_idx,
                            "pose_detected": False,
                            "error": "Frame decoding failed",
                            "inference_time_ms": 0.0,
                            "mean_visibility": 0.0,
                            "visible_landmarks_count": 0,
                            "total_landmarks": 0,
                        })
                        continue

                    frame_bgr = frames_cache[f_idx]
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                    t_start = time.perf_counter()
                    result = detector.detect(mp_image)
                    inf_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

                    pose_found = bool(result.pose_landmarks and len(result.pose_landmarks) > 0)
                    mean_vis = 0.0
                    vis_above_05_count = 0
                    total_lm = 0
                    landmarks = []

                    if pose_found:
                        landmarks = result.pose_landmarks[0]
                        total_lm = len(landmarks)
                        visibilities = [float(getattr(lm, "visibility", 0.0)) for lm in landmarks]
                        mean_vis = round(float(np.mean(visibilities)), 4)
                        vis_above_05_count = sum(1 for v in visibilities if v >= 0.5)

                        # Save illustrative overlay if matching target criteria and under quota
                        if saved_overlay_count < max_overlays:
                            for target_class, target_method, target_frame in overlay_targets:
                                if clip_class == target_class and method_name == target_method and f_idx == target_frame:
                                    overlay_filename = f"pose_{clip_class.replace(' ', '_')}_{method_name}_frame{f_idx}.jpg"
                                    out_img_path = samples_dir / overlay_filename
                                    annotated_img = draw_pose_overlay(
                                        frame_bgr, landmarks, f"{clip_class}/{clip_name}", method_name, f_idx
                                    )
                                    cv2.imwrite(str(out_img_path), annotated_img)
                                    saved_overlay_count += 1
                                    print(f"  [SAVED OVERLAY] {out_img_path.name}")
                                    break

                    eval_records.append({
                        "clip_name": clip_name,
                        "class": clip_class,
                        "method": method_name,
                        "frame_index": f_idx,
                        "pose_detected": pose_found,
                        "inference_time_ms": inf_time_ms,
                        "mean_visibility": mean_vis,
                        "visible_landmarks_count": vis_above_05_count,
                        "total_landmarks": total_lm,
                    })

    finally:
        # Guarantee detector closure and resource cleanup
        detector.close()
        print("MediaPipe PoseLandmarker closed cleanly.")

    # Calculate aggregate and per-method statistics
    methods = ["uniform", "motion_aware"]
    summary = {
        "pilot_scope": {
            "total_clips": len(clips_data),
            "shot_classes": len(set(c["class"] for c in clips_data)),
            "frames_per_method_per_clip": 6,
            "total_evaluations_per_method": len(clips_data) * 6,
            "total_evaluations": len(eval_records),
            "disclaimer": "This is a 12-clip pilot benchmark. Differences illustrate empirical trends on this sample set and do not constitute formal statistical significance or validate 3D reconstruction accuracy."
        },
        "aggregate_comparison": {},
        "per_class_comparison": {},
    }

    for m in methods:
        recs = [r for r in eval_records if r["method"] == m]
        detected = [r for r in recs if r["pose_detected"]]
        det_rate = round((len(detected) / len(recs)) * 100, 2) if recs else 0.0
        avg_infer = round(float(np.mean([r["inference_time_ms"] for r in recs])), 2) if recs else 0.0
        total_infer = round(float(np.sum([r["inference_time_ms"] for r in recs])), 2) if recs else 0.0

        mean_vis = round(float(np.mean([r["mean_visibility"] for r in detected])), 4) if detected else 0.0
        mean_vis_count = round(float(np.mean([r["visible_landmarks_count"] for r in detected])), 2) if detected else 0.0

        summary["aggregate_comparison"][m] = {
            "total_frames_evaluated": len(recs),
            "successful_pose_detections": len(detected),
            "detection_rate_pct": det_rate,
            "mean_inference_time_ms": avg_infer,
            "total_inference_time_ms": total_infer,
            "mean_landmark_visibility": mean_vis,
            "mean_visible_landmarks_above_05": mean_vis_count,
        }

    # Per-class breakdowns
    classes = sorted(list(set(r["class"] for r in eval_records)))
    for c in classes:
        summary["per_class_comparison"][c] = {}
        for m in methods:
            c_recs = [r for r in eval_records if r["class"] == c and r["method"] == m]
            c_det = [r for r in c_recs if r["pose_detected"]]
            det_rate = round((len(c_det) / len(c_recs)) * 100, 1) if c_recs else 0.0
            avg_vis = round(float(np.mean([r["mean_visibility"] for r in c_det])), 3) if c_det else 0.0
            avg_vis_count = round(float(np.mean([r["visible_landmarks_count"] for r in c_det])), 1) if c_det else 0.0

            summary["per_class_comparison"][c][m] = {
                "evaluated": len(c_recs),
                "detected": len(c_det),
                "detection_rate_pct": det_rate,
                "mean_visibility": avg_vis,
                "mean_visible_landmarks": avg_vis_count,
            }

    # Write JSON reports
    det_json_path = reports_dir / "pose_selection_evaluation.json"
    sum_json_path = reports_dir / "pose_selection_summary.json"

    with open(det_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=2)
    with open(sum_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 86)
    print("AGGREGATE POSE EVALUATION SUMMARY (12 CLIPS, 144 EVALUATIONS)")
    print("=" * 86)
    print(f"{'Metric':<36} {'Uniform Selection':<24} {'Motion-Aware Selection'}")
    print("-" * 86)
    u_agg = summary["aggregate_comparison"]["uniform"]
    m_agg = summary["aggregate_comparison"]["motion_aware"]

    print(f"{'Evaluated Frames':<36} {u_agg['total_frames_evaluated']:<24} {m_agg['total_frames_evaluated']}")
    print(f"{'Successful Detections':<36} {u_agg['successful_pose_detections']:<24} {m_agg['successful_pose_detections']}")
    print(f"{'Detection Rate (%)':<36} {u_agg['detection_rate_pct']:<24.2f} {m_agg['detection_rate_pct']:.2f}%")
    print(f"{'Mean Landmark Visibility':<36} {u_agg['mean_landmark_visibility']:<24.4f} {m_agg['mean_landmark_visibility']:.4f}")
    print(f"{'Mean Visible Landmarks (>0.5)':<36} {u_agg['mean_visible_landmarks_above_05']:<24.2f} {m_agg['mean_visible_landmarks_above_05']:.2f}")
    print(f"{'Mean Inference Latency (CPU)':<36} {u_agg['mean_inference_time_ms']} ms{'':<17} {m_agg['mean_inference_time_ms']} ms")
    print(f"{'Total Inference Time (CPU)':<36} {u_agg['total_inference_time_ms']} ms{'':<16} {m_agg['total_inference_time_ms']} ms")
    print("=" * 86)

    # Print per-class detection rate table
    print("\nPER-CLASS DETECTION & VISIBILITY BREAKDOWN:")
    print("-" * 86)
    print(f"{'Class':<20} {'Uniform Det %':<16} {'Motion Det %':<16} {'Uniform Vis':<16} {'Motion Vis'}")
    print("-" * 86)
    for c in classes:
        u_c = summary["per_class_comparison"][c]["uniform"]
        m_c = summary["per_class_comparison"][c]["motion_aware"]
        u_det_str = f"{u_c['detected']}/{u_c['evaluated']} ({u_c['detection_rate_pct']}%)"
        m_det_str = f"{m_c['detected']}/{m_c['evaluated']} ({m_c['detection_rate_pct']}%)"
        print(f"{c:<20} {u_det_str:<16} {m_det_str:<16} {u_c['mean_visibility']:<16.3f} {m_c['mean_visibility']:.3f}")
    print("=" * 86)

    print(f"\n[OK] Detailed evaluations: {det_json_path.relative_to(base_dir)}")
    print(f"[OK] Summary comparison: {sum_json_path.relative_to(base_dir)}")
    print(f"[OK] Saved {saved_overlay_count} illustrative pose sample images to: {samples_dir.relative_to(base_dir)}/")


if __name__ == "__main__":
    run_evaluation()
