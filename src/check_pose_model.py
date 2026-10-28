"""
Verification script for ONNX Runtime CPU execution and pose model loading.
Checks if models/yolov8n-pose.onnx is present, and inspects its input/output shapes.
Does not perform model inference.
"""

from pathlib import Path
import json
import sys

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime is not installed in the current Python environment.")
    sys.exit(1)


def check_pose_model():
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "models" / "yolov8n-pose.onnx"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ONNX RUNTIME POSE MODEL VERIFICATION (CPU)")
    print("=" * 72)

    available_providers = ort.get_available_providers()
    print(f"Available Providers: {available_providers}")
    has_cpu_provider = "CPUExecutionProvider" in available_providers
    print(f"CPUExecutionProvider Ready: {has_cpu_provider}")

    if not model_path.exists():
        print(f"\n[STATUS] Model file not found at: {model_path.relative_to(base_dir)}")
        print("[INFO] Official Ultralytics releases distribute PyTorch (.pt) weights only.")
        print("[INFO] Direct ONNX export requires PyTorch, which is blocked by Windows App Control on this host.")
        return {
            "model_present": False,
            "model_path": str(model_path.relative_to(base_dir)).replace("\\", "/"),
            "cpu_provider_available": has_cpu_provider,
            "session_load_status": "SKIPPED_MODEL_ABSENT",
        }

    print(f"\nModel File Found: {model_path.relative_to(base_dir)} ({model_path.stat().st_size} bytes)")

    try:
        session_options = ort.SessionOptions()
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"]
        )

        inputs_meta = [
            {"name": inp.name, "shape": inp.shape, "type": inp.type}
            for inp in session.get_inputs()
        ]
        outputs_meta = [
            {"name": out.name, "shape": out.shape, "type": out.type}
            for out in session.get_outputs()
        ]

        print("[SUCCESS] Model loaded successfully with CPUExecutionProvider!")
        print(f"Inputs: {inputs_meta}")
        print(f"Outputs: {outputs_meta}")

        return {
            "model_present": True,
            "model_path": str(model_path.relative_to(base_dir)).replace("\\", "/"),
            "cpu_provider_available": has_cpu_provider,
            "session_load_status": "SUCCESS",
            "inputs": inputs_meta,
            "outputs": outputs_meta,
        }
    except Exception as e:
        print(f"[ERROR] Failed to load ONNX model session: {e}")
        return {
            "model_present": True,
            "model_path": str(model_path.relative_to(base_dir)).replace("\\", "/"),
            "cpu_provider_available": has_cpu_provider,
            "session_load_status": "ERROR",
            "error_message": str(e),
        }


def main():
    result = check_pose_model()
    print("=" * 72)


if __name__ == "__main__":
    main()
