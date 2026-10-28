"""
Hybrid Frame Selection Strategy & Multi-Method Evaluation.

Implements a 6-frame hybrid selection method:
1. Anchor Frames (3 frames):
   Evenly distributed across the sequence to guarantee 100% temporal span and boundary coverage:
   indices [0, 11, 23] (start, mid, end of 24 frames).
2. Motion Peaks (3 frames):
   Partitions the remaining intermediate frame ranges into 3 temporal bands
   ([1..7], [8..15], [16..22]).
   In each band, selects the frame with the maximum motion energy score that is not already chosen.
3. Final 6 frames:
   Combined, deduplicated, and sorted in chronological order.

Compares across all 12 clips:
- Uniform Selection (6 frames)
- Motion-Aware Selection (6 frames)
- Hybrid Selection (6 frames)
against the full 24-frame reference sequence.

Metrics:
- Pose detection coverage (%)
- Mean landmark visibility
- Temporal span ratio & maximum frame gap
- Nearest-Selected-Pose Error (NSPE) with explicit tracking of reference frames
  lacking a valid selected pose or mutually visible joints.

Outputs:
- reports/hybrid_selection_evaluation.json
- reports/hybrid_selection_summary.json
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


def select_hybrid_frames(frame_energies: list, total_frames: int = 24, total_budget: int = 6) -> list:
    """
    Hybrid 6-frame selection:
    - 3 anchors: boundary frames [0, total_frames - 1] and midpoint [round((total_frames - 1) / 2)].
      For 24 frames: [0, 11, 23].
    - 3 motion peaks: from 3 non-overlapping intermediate temporal intervals:
      Band 1: [1 .. 7]
      Band 2: [8 .. 15] excluding midpoint
      Band 3: [16 .. 22]
    Ensures zero duplicate indices and chronological sorting.
    """
    if total_frames <= total_budget:
        return list(range(total_frames))

    # 1. Fixed anchor frames (guarantees full temporal boundary span)
    anchors = [0, int(round((total_frames - 1) / 2)), total_frames - 1]  # [0, 11, 23]
    selected_set = set(anchors)

    # 2. Define 3 intermediate temporal bands to distribute remaining 3 motion picks
    bands = [
        [i for i in range(1, 8) if i not in selected_set],
        [i for i in range(8, 16) if i not in selected_set],
        [i for i in range(16, total_frames - 1) if i not in selected_set],
    ]

    for band in bands:
        if not band:
            continue
        # Pick index with maximum motion energy in this band
        best_idx = max(band, key=lambda idx: frame_energies[idx])
        selected_set.add(best_idx)

    # Safeguard if any band was exhausted (reach exactly budget)
    if len(selected_set) < total_budget:
        remaining = [i for i in range(total_frames) if i not in selected_set]
        remaining.sort(key=lambda idx: frame_energies[idx], reverse=True)
        needed = total_budget - len(selected_set)
        for r_idx in remaining[:needed]:
            selected_set.add(r_idx)

    return sorted(list(selected_set))


def decode_all_clip_frames(video_path: Path):
    """Decode all frames into BGR list."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()
    return frames


def extract_poses_for_all_frames(frames, detector):
    """
    Runs MediaPipe Tasks PoseLandmarker across all frames in sequence.
    Returns:
        dict: {frame_idx: {"detected": bool, "landmarks": list of dicts, "inference_time_ms": float}}
    """
    poses = {}
    for idx, frame_bgr in enumerate(frames):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        t_start = time.perf_counter()
        result = detector.detect(mp_image)
        inf_ms = round((time.perf_counter() - t_start) * 1000, 2)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm_list = []
            for pt in result.pose_landmarks[0]:
                lm_list.append({
                    "x": float(pt.x),
                    "y": float(pt.y),
                    "z": float(pt.z),
                    "visibility": float(getattr(pt, "visibility", 0.0)),
                    "presence": float(getattr(pt, "presence", 0.0)),
                })
            poses[idx] = {
                "detected": True,
                "landmarks": lm_list,
                "inference_time_ms": inf_ms,
            }
        else:
            poses[idx] = {
                "detected": False,
                "landmarks": [],
                "inference_time_ms": inf_ms,
            }
    return poses


def compute_temporal_span_and_gaps(selected_indices, total_frames=24):
    """Computes temporal span ratio and gap stats."""
    if not selected_indices:
        return {"temporal_span_ratio": 0.0, "max_gap": total_frames, "mean_gap": float(total_frames)}

    s_idx = sorted(selected_indices)
    span = (s_idx[-1] - s_idx[0]) / max(total_frames - 1, 1)

    if len(s_idx) > 1:
        gaps = [s_idx[i + 1] - s_idx[i] for i in range(len(s_idx) - 1)]
        max_gap = int(max(gaps))
        mean_gap = round(float(np.mean(gaps)), 2)
    else:
        max_gap = total_frames
        mean_gap = float(total_frames)

    return {
        "temporal_span_ratio": round(span, 4),
        "min_frame": s_idx[0],
        "max_frame": s_idx[-1],
        "max_gap": max_gap,
        "mean_gap": mean_gap,
    }


def compute_nspe_with_audit(clip_poses, selected_indices, total_frames=24):
    """
    Computes Nearest-Selected-Pose Error (NSPE) with explicit accounting of reference frames
    lacking a valid selected pose or lacking mutually visible joints.
    Consistent across uniform, motion-aware, and hybrid.
    """
    valid_selected = [s for s in selected_indices if clip_poses.get(s, {}).get("detected", False)]

    total_ref_frames_with_pose = sum(1 for t in range(total_frames) if clip_poses.get(t, {}).get("detected"))

    if not valid_selected:
        return {
            "computable": False,
            "mean_nspe": None,
            "median_nspe": None,
            "valid_selected_count": 0,
            "matched_ref_frames_count": 0,
            "unmatched_ref_frames_count": total_ref_frames_with_pose,
            "unmatched_ref_frames": [t for t in range(total_frames) if clip_poses.get(t, {}).get("detected")],
            "per_frame_details": [],
        }

    per_frame_details = []
    distances = []
    unmatched_ref = []

    for t in range(total_frames):
        ref_p = clip_poses.get(t, {})
        if not ref_p.get("detected", False):
            continue  # Reference frame had no human pose

        ref_lm = ref_p["landmarks"]

        # Nearest selected index with valid pose
        nearest_s = min(valid_selected, key=lambda s: abs(t - s))
        sel_lm = clip_poses[nearest_s]["landmarks"]

        # Mutually visible joints >= 0.5
        joint_dists = []
        for j in range(min(len(ref_lm), len(sel_lm))):
            r_pt = ref_lm[j]
            s_pt = sel_lm[j]
            if r_pt["visibility"] >= 0.5 and s_pt["visibility"] >= 0.5:
                d = np.sqrt((r_pt["x"] - s_pt["x"]) ** 2 + (r_pt["y"] - s_pt["y"]) ** 2)
                joint_dists.append(float(d))

        if joint_dists:
            frame_err = float(np.mean(joint_dists))
            distances.append(frame_err)
            per_frame_details.append({
                "ref_frame": t,
                "matched_selected_frame": nearest_s,
                "temporal_distance": abs(t - nearest_s),
                "num_visible_joints": len(joint_dists),
                "error": round(frame_err, 4),
            })
        else:
            unmatched_ref.append(t)
            per_frame_details.append({
                "ref_frame": t,
                "matched_selected_frame": nearest_s,
                "temporal_distance": abs(t - nearest_s),
                "num_visible_joints": 0,
                "error": None,
                "note": "No mutually visible joints >= 0.5"
            })

    mean_err = round(float(np.mean(distances)), 4) if distances else None
    med_err = round(float(np.median(distances)), 4) if distances else None

    return {
        "computable": bool(distances),
        "mean_nspe": mean_err,
        "median_nspe": med_err,
        "valid_selected_count": len(valid_selected),
        "matched_ref_frames_count": len(distances),
        "unmatched_ref_frames_count": len(unmatched_ref),
        "unmatched_ref_frames": unmatched_ref,
        "per_frame_details": per_frame_details,
    }


def main():
    base_dir = Path(__file__).resolve().parents[1]
    motion_json_path = base_dir / "reports" / "motion_analysis.json"
    model_path = base_dir / "models" / "pose_landmarker_lite.task"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not motion_json_path.exists():
        raise FileNotFoundError(f"Missing motion analysis file: {motion_json_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    with open(motion_json_path, "r", encoding="utf-8") as f:
        clips_data = json.load(f)

    print("=" * 102)
    print("HYBRID FRAME SELECTION EVALUATION: UNIFORM vs MOTION-AWARE vs HYBRID (CPU)")
    print(f"Total Clips: {len(clips_data)} | Methods: 3 (Budget=6 frames/clip) | Reference: 24 frames/clip")
    print("=" * 102)

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

    clip_evaluations = []

    try:
        for idx, clip in enumerate(clips_data):
            clip_name = clip["clip_name"]
            clip_class = clip["class"]
            rel_path = clip["relative_video_path"]
            video_file = base_dir / rel_path
            frame_energies = clip["frame_motion_energies"]

            # 1. Existing Uniform & Motion Indices
            u_indices = clip["uniform_selected_indices"]
            m_indices = clip["motion_selected_indices"]

            # 2. Compute Hybrid Indices (3 anchors + 3 motion peaks)
            h_indices = select_hybrid_frames(frame_energies, total_frames=24, total_budget=6)

            # 3. Decode all 24 frames and extract full pose sequence
            all_frames = decode_all_clip_frames(video_file)
            total_frames = len(all_frames)
            clip_poses = extract_poses_for_all_frames(all_frames, detector)

            # Reference stats
            ref_detected = [f for f in range(total_frames) if clip_poses[f]["detected"]]
            ref_det_pct = round((len(ref_detected) / total_frames) * 100, 2)
            ref_vis_vals = [pt["visibility"] for f in ref_detected for pt in clip_poses[f]["landmarks"]]
            ref_mean_vis = round(float(np.mean(ref_vis_vals)), 4) if ref_vis_vals else 0.0

            method_results = {}
            for m_key, indices in [("uniform", u_indices), ("motion_aware", m_indices), ("hybrid", h_indices)]:
                det_frames = [f for f in indices if clip_poses[f]["detected"]]
                det_pct = round((len(det_frames) / len(indices)) * 100, 2)
                vis_vals = [pt["visibility"] for f in det_frames for pt in clip_poses[f]["landmarks"]]
                mean_vis = round(float(np.mean(vis_vals)), 4) if vis_vals else 0.0

                temporal_meta = compute_temporal_span_and_gaps(indices, total_frames)
                nspe_meta = compute_nspe_with_audit(clip_poses, indices, total_frames)

                method_results[m_key] = {
                    "indices": indices,
                    "detected_frames_count": len(det_frames),
                    "detection_rate_pct": det_pct,
                    "mean_visibility": mean_vis,
                    "temporal_coverage": temporal_meta,
                    "nearest_pose_error": {
                        "computable": nspe_meta["computable"],
                        "mean_nspe": nspe_meta["mean_nspe"],
                        "median_nspe": nspe_meta["median_nspe"],
                        "matched_ref_frames": nspe_meta["matched_ref_frames_count"],
                        "unmatched_ref_frames": nspe_meta["unmatched_ref_frames_count"],
                    },
                    "missing_detection_indices": [f for f in indices if not clip_poses[f]["detected"]],
                    "nspe_details": nspe_meta["per_frame_details"],
                }

            clip_evaluations.append({
                "clip_name": clip_name,
                "class": clip_class,
                "total_frames": total_frames,
                "reference_sequence": {
                    "total_frames": total_frames,
                    "detected_frames_count": len(ref_detected),
                    "detection_rate_pct": ref_det_pct,
                    "mean_visibility": ref_mean_vis,
                    "missed_frames": [f for f in range(total_frames) if not clip_poses[f]["detected"]],
                },
                "methods": method_results,
            })

            label = f"{clip_class}/{clip_name}"
            u_err = method_results["uniform"]["nearest_pose_error"]["mean_nspe"]
            m_err = method_results["motion_aware"]["nearest_pose_error"]["mean_nspe"]
            h_err = method_results["hybrid"]["nearest_pose_error"]["mean_nspe"]

            u_str = f"{u_err:.4f}" if u_err is not None else "N/A"
            m_str = f"{m_err:.4f}" if m_err is not None else "N/A"
            h_str = f"{h_err:.4f}" if h_err is not None else "N/A"
            h_idx_str = str(h_indices)

            print(f"[{idx+1:02d}/12] {label:<28} Hybrid Idx: {h_idx_str:<22} NSPE [U: {u_str} | M: {m_str} | H: {h_str}]")

    finally:
        detector.close()
        print("MediaPipe PoseLandmarker closed cleanly.")

    # Aggregate summaries across methods
    methods = ["uniform", "motion_aware", "hybrid"]
    summary = {
        "pilot_scope": {
            "total_clips": len(clip_evaluations),
            "total_reference_frames": len(clip_evaluations) * 24,
            "disclaimer": "Evaluated on CPU using MediaPipe estimated landmarks across 12 short video clips. Not a claim of formal statistical significance or 3D biomechanical accuracy. Nearest-Selected-Pose Error (NSPE) measures representativeness against markerless video detections."
        },
        "aggregate_comparison": {},
        "per_class_comparison": {},
    }

    # Reference sequence summary
    summary["aggregate_comparison"]["reference_sequence"] = {
        "total_frames": len(clip_evaluations) * 24,
        "total_detected_frames": sum(c["reference_sequence"]["detected_frames_count"] for c in clip_evaluations),
        "mean_detection_rate_pct": round(float(np.mean([c["reference_sequence"]["detection_rate_pct"] for c in clip_evaluations])), 2),
        "mean_landmark_visibility": round(float(np.mean([c["reference_sequence"]["mean_visibility"] for c in clip_evaluations])), 4),
    }

    for m in methods:
        det_pcts = [c["methods"][m]["detection_rate_pct"] for c in clip_evaluations]
        vis_vals = [c["methods"][m]["mean_visibility"] for c in clip_evaluations if c["methods"][m]["mean_visibility"] > 0]
        spans = [c["methods"][m]["temporal_coverage"]["temporal_span_ratio"] for c in clip_evaluations]
        max_gaps = [c["methods"][m]["temporal_coverage"]["max_gap"] for c in clip_evaluations]
        nspes = [c["methods"][m]["nearest_pose_error"]["mean_nspe"] for c in clip_evaluations if c["methods"][m]["nearest_pose_error"]["mean_nspe"] is not None]

        summary["aggregate_comparison"][m] = {
            "mean_detection_rate_pct": round(float(np.mean(det_pcts)), 2),
            "mean_landmark_visibility": round(float(np.mean(vis_vals)), 4),
            "mean_temporal_span_ratio": round(float(np.mean(spans)), 4),
            "mean_max_temporal_gap_frames": round(float(np.mean(max_gaps)), 2),
            "mean_nearest_pose_error_nspe": round(float(np.mean(nspes)), 4) if nspes else None,
            "total_unmatched_ref_frames": sum(c["methods"][m]["nearest_pose_error"]["unmatched_ref_frames"] for c in clip_evaluations),
        }

    # Per-class comparison
    classes = sorted(list(set(c["class"] for c in clip_evaluations)))
    for c in classes:
        c_clips = [clip for clip in clip_evaluations if clip["class"] == c]
        summary["per_class_comparison"][c] = {"num_clips": len(c_clips)}

        for m in methods:
            m_det = round(float(np.mean([clip["methods"][m]["detection_rate_pct"] for clip in c_clips])), 1)
            m_vis = round(float(np.mean([clip["methods"][m]["mean_visibility"] for clip in c_clips])), 3)
            m_nspe_list = [clip["methods"][m]["nearest_pose_error"]["mean_nspe"] for clip in c_clips if clip["methods"][m]["nearest_pose_error"]["mean_nspe"] is not None]
            m_nspe = round(float(np.mean(m_nspe_list)), 4) if m_nspe_list else None

            summary["per_class_comparison"][c][m] = {
                "detection_rate_pct": m_det,
                "mean_visibility": m_vis,
                "mean_nspe": m_nspe,
            }

    # Save reports
    eval_json_path = reports_dir / "hybrid_selection_evaluation.json"
    sum_json_path = reports_dir / "hybrid_selection_summary.json"

    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(clip_evaluations, f, indent=2)
    with open(sum_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 102)
    print("AGGREGATE COMPARISON: UNIFORM vs MOTION-AWARE vs HYBRID (12 CLIPS, BUDGET=6 FRAMES)")
    print("=" * 102)
    u_agg = summary["aggregate_comparison"]["uniform"]
    m_agg = summary["aggregate_comparison"]["motion_aware"]
    h_agg = summary["aggregate_comparison"]["hybrid"]

    print(f"{'Evaluation Metric':<38} {'Uniform':<18} {'Motion-Aware':<18} {'Hybrid':<18}")
    print("-" * 102)
    print(f"{'Pose Detection Rate (%)':<38} {u_agg['mean_detection_rate_pct']:<18.2f} {m_agg['mean_detection_rate_pct']:<18.2f} {h_agg['mean_detection_rate_pct']:.2f}%")
    print(f"{'Mean Landmark Visibility':<38} {u_agg['mean_landmark_visibility']:<18.4f} {m_agg['mean_landmark_visibility']:<18.4f} {h_agg['mean_landmark_visibility']:.4f}")
    print(f"{'Temporal Span Ratio (1.0 = full)':<38} {u_agg['mean_temporal_span_ratio']:<18.4f} {m_agg['mean_temporal_span_ratio']:<18.4f} {h_agg['mean_temporal_span_ratio']:.4f}")
    print(f"{'Mean Max Temporal Gap (frames)':<38} {u_agg['mean_max_temporal_gap_frames']:<18.2f} {m_agg['mean_max_temporal_gap_frames']:<18.2f} {h_agg['mean_max_temporal_gap_frames']:.2f}")
    print(f"{'Nearest-Selected-Pose Error (NSPE)':<38} {u_agg['mean_nearest_pose_error_nspe']:<18.4f} {m_agg['mean_nearest_pose_error_nspe']:<18.4f} {h_agg['mean_nearest_pose_error_nspe']:.4f}")
    print("=" * 102)

    # Per-class NSPE Table
    print("\nPER-CLASS NEAREST-POSE ERROR (NSPE) COMPARISON:")
    print("-" * 102)
    print(f"{'Class':<20} {'Clips':<8} {'Uniform NSPE':<18} {'Motion NSPE':<18} {'Hybrid NSPE':<18} {'Best Method'}")
    print("-" * 102)
    for c in classes:
        p_c = summary["per_class_comparison"][c]
        u_e = p_c["uniform"]["mean_nspe"]
        m_e = p_c["motion_aware"]["mean_nspe"]
        h_e = p_c["hybrid"]["mean_nspe"]

        u_s = f"{u_e:.4f}" if u_e is not None else "N/A"
        m_s = f"{m_e:.4f}" if m_e is not None else "N/A"
        h_s = f"{h_e:.4f}" if h_e is not None else "N/A"

        scores = [("Uniform", u_e), ("Motion-Aware", m_e), ("Hybrid", h_e)]
        valid_scores = [(name, val) for name, val in scores if val is not None]
        best_name = min(valid_scores, key=lambda x: x[1])[0] if valid_scores else "Inconclusive"

        print(f"{c:<20} {p_c['num_clips']:<8} {u_s:<18} {m_s:<18} {h_s:<18} {best_name}")
    print("=" * 102)

    print(f"\n[OK] Detailed evaluations: {eval_json_path.relative_to(base_dir)}")
    print(f"[OK] Summary comparison: {sum_json_path.relative_to(base_dir)}")


if __name__ == "__main__":
    main()
