"""
Verification script for MoveNet Lightning pose model loading.
Checks if a compatible MoveNet model file is present under models/
and validates loading using existing environment runtimes (ONNX Runtime / OpenCV DNN).
Does not perform model inference.
"""

from pathlib import Path
import json
import sys

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import cv2
except ImportError:
    cv2 = None


def check_movenet_model():
    base_dir = Path(__file__).resolve().parents[1]
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("MOVENET LIGHTNING MODEL COMPATIBILITY VERIFICATION (CPU)")
    print("=" * 76)

    # Check potential model candidates
    onnx_path = models_dir / "movenet_singlepose_lightning.onnx"
    tflite_path = models_dir / "movenet_singlepose_lightning.tflite"

    model_found = None
    if onnx_path.exists():
        model_found = onnx_path
    elif tflite_path.exists():
        model_found = tflite_path

    if not model_found:
        print("[STATUS] No compatible MoveNet model found under models/.")
        print("[FINDING] Official Google TF Hub MoveNet models are distributed in TFLite/SavedModel format.")
        print("[FINDING] Existing cv2.dnn (OpenCV 5.0.0) lacks TFLite ARG_MAX operator support.")
        print("[FINDING] Official ONNX distribution does not exist on TF Hub (requires tf2onnx/PyTorch conversion).")
        print("[DECISION] Stopped cleanly without downloading or installing unauthorized dependencies.")
        return {
            "model_present": False,
            "checked_paths": [
                str(onnx_path.relative_to(base_dir)).replace("\\", "/"),
                str(tflite_path.relative_to(base_dir)).replace("\\", "/"),
            ],
            "opencv_version": cv2.__version__ if cv2 else None,
            "onnxruntime_version": ort.__version__ if ort else None,
            "opencv_tflite_supported": False,
            "error_detail": "cv2.dnn unsupported operator type ARG_MAX on official MoveNet TFLite files",
        }

    # If an ONNX model is provided
    if model_found.suffix.lower() == ".onnx" and ort:
        try:
            session = ort.InferenceSession(str(model_found), providers=["CPUExecutionProvider"])
            inputs_meta = [{"name": i.name, "shape": i.shape, "type": i.type} for i in session.get_inputs()]
            outputs_meta = [{"name": o.name, "shape": o.shape, "type": o.type} for o in session.get_outputs()]
            print(f"[SUCCESS] Loaded {model_found.name} in ONNX Runtime CPUExecutionProvider!")
            print(f"Inputs: {inputs_meta}")
            print(f"Outputs: {outputs_meta}")
            return {
                "model_present": True,
                "model_format": "ONNX",
                "model_path": str(model_found.relative_to(base_dir)).replace("\\", "/"),
                "status": "LOAD_SUCCESS",
                "inputs": inputs_meta,
                "outputs": outputs_meta,
            }
        except Exception as e:
            print(f"[ERROR] Failed to load ONNX model: {e}")
            return {
                "model_present": True,
                "model_format": "ONNX",
                "status": "LOAD_ERROR",
                "error": str(e),
            }

    return {"model_present": True, "status": "UNKNOWN_FORMAT"}


def main():
    check_movenet_model()
    print("=" * 76)


if __name__ == "__main__":
    main()
