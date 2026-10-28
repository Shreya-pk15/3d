"""
Inspect extracted cricket shot video clips using OpenCV.
Reports duration, FPS, frame count, resolution, and decodability.
Saves JSON inspection results to reports/sample_video_inspection.json.
"""

from pathlib import Path
import json
import sys

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV (cv2) is not installed in the current Python environment.")
    sys.exit(1)


def inspect_video(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "file_name": video_path.name,
            "relative_path": str(video_path).replace("\\", "/"),
            "can_decode": False,
            "error": "Failed to open video file",
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Test reading actual frames to verify decoding
    decoded_frames = 0
    can_decode = False
    first_frame_read = False

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if not first_frame_read:
            first_frame_read = True
            can_decode = True
        decoded_frames += 1

    cap.release()

    effective_fps = fps if fps > 0 else 25.0
    duration_sec = round(decoded_frames / effective_fps, 3) if effective_fps > 0 else 0.0

    return {
        "file_name": video_path.name,
        "relative_path": str(video_path.relative_to(video_path.parents[2])).replace("\\", "/"),
        "class": video_path.parent.name,
        "can_decode": can_decode,
        "reported_fps": round(fps, 2),
        "reported_frame_count": reported_frame_count,
        "actual_decoded_frames": decoded_frames,
        "width": width,
        "height": height,
        "duration_seconds": duration_sec,
        "file_size_bytes": video_path.stat().st_size,
    }


def main():
    base_dir = Path(__file__).resolve().parents[1]
    sample_dir = base_dir / "outputs" / "sample_clips"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_json = reports_dir / "sample_video_inspection.json"

    video_files = sorted(sample_dir.rglob("*.avi"))
    if not video_files:
        print(f"[WARNING] No .avi video files found in {sample_dir}")
        return

    results = []
    print("=" * 72)
    print("CRICKET VIDEO CLIPS INSPECTION SUMMARY")
    print("=" * 72)
    print(f"{'Class / Clip':<32} {'Res':<10} {'FPS':<6} {'Frames':<8} {'Duration':<9} {'Decodable'}")
    print("-" * 72)

    for vf in video_files:
        info = inspect_video(vf)
        results.append(info)
        label = f"{info['class']}/{info['file_name']}"
        res = f"{info['width']}x{info['height']}"
        fps = f"{info['reported_fps']}"
        frames = f"{info['actual_decoded_frames']}"
        dur = f"{info['duration_seconds']}s"
        dec = "YES" if info["can_decode"] else "NO"
        print(f"{label:<32} {res:<10} {fps:<6} {frames:<8} {dur:<9} {dec}")

    print("=" * 72)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved detailed inspection report to: {out_json.relative_to(base_dir)}")


if __name__ == "__main__":
    main()
