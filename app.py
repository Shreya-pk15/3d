"""
Cricket3D_Motion: Web Analytics & 3D Pose Estimation Dashboard.
Built using Streamlit & Plotly (CPU-only).
Features:
- Dual-panel synchronized 2D Video Frame & Interactive Plotly 3D Skeleton (mouse rotate/zoom).
- Real-time frame slider & automated playback controls.
- Automatic zero-persistence handling for undetected pose frames.
- Multi-Method Evaluation page comparing Uniform, Motion-Aware, and Hybrid selection using existing reports.
"""

from pathlib import Path
import json
import time
import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Set page configuration
st.set_page_config(
    page_title="Cricket3D Motion | Sports Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark-mode sports telemetry design
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .metric-card {
        background-color: #1A1F2C;
        border: 1px solid #2E384D;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .status-badge-ok {
        background-color: #113824;
        color: #00FF88;
        border: 1px solid #00FF88;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.82rem;
    }
    .status-badge-warn {
        background-color: #381515;
        color: #FF5555;
        border: 1px solid #FF5555;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.82rem;
    }
    .disclaimer-box {
        background-color: #1E232F;
        border-left: 4px solid #00ADB5;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.80rem;
        color: #A0AAB8;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"
SAMPLE_DIR = BASE_DIR / "outputs" / "sample_clips"
REPORTS_DIR = BASE_DIR / "reports"

# Anatomical Connections for 3D Skeletal Graph
SKELETON_BONES = [
    (11, 12, "torso"), (11, 23, "torso"), (12, 24, "torso"), (23, 24, "torso"),
    (11, 13, "left_arm"), (13, 15, "left_arm"), (15, 17, "left_arm"), (15, 19, "left_arm"), (15, 21, "left_arm"),
    (12, 14, "right_arm"), (14, 16, "right_arm"), (16, 18, "right_arm"), (16, 20, "right_arm"), (16, 22, "right_arm"),
    (23, 25, "left_leg"), (25, 27, "left_leg"), (27, 29, "left_leg"), (27, 31, "left_leg"),
    (24, 26, "right_leg"), (26, 28, "right_leg"), (28, 30, "right_leg"), (28, 32, "right_leg"),
    (0, 1, "head"), (1, 2, "head"), (2, 3, "head"), (3, 7, "head"),
    (0, 4, "head"), (4, 5, "head"), (5, 6, "head"), (6, 8, "head"), (9, 10, "head")
]

BONE_COLORS = {
    "torso": "#00ADB5",
    "left_arm": "#FF5722",
    "right_arm": "#4CAF50",
    "left_leg": "#E91E63",
    "right_leg": "#2196F3",
    "head": "#FFC107"
}

@st.cache_resource
def get_pose_detector():
    """Initializes and caches the MediaPipe Tasks PoseLandmarker detector."""
    if not MODEL_PATH.exists():
        st.error(f"MediaPipe model file missing: {MODEL_PATH}")
        st.stop()
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        num_poses=1
    )
    return vision.PoseLandmarker.create_from_options(options)


def step_pose_frame(delta: int, total_frames: int) -> None:
    current_frame = st.session_state.timeline_frame
    st.session_state.timeline_frame = max(
        0, min(total_frames - 1, current_frame + delta)
    )


@st.cache_data
def load_and_cache_clip_data(clip_rel_path: str):
    """
    Decodes all 24 frames of a clip and extracts 3D world landmarks into memory.
    Cached by Streamlit to guarantee zero-latency browser scrubbing.
    """
    video_path = BASE_DIR / clip_rel_path
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = get_pose_detector()
    frames_bgr = []
    poses_wlm = []
    detections = []

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            frames_bgr.append(None)
            poses_wlm.append([])
            detections.append(False)
            continue

        frames_bgr.append(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = detector.detect(mp_img)

        if res.pose_world_landmarks and len(res.pose_world_landmarks) > 0:
            pts = [
                {
                    "x": float(p.x),
                    "y": -float(p.y),  # Invert Y so upright elevation is positive
                    "z": float(p.z),
                    "vis": float(getattr(p, "visibility", 0.0))
                }
                for p in res.pose_world_landmarks[0]
            ]
            poses_wlm.append(pts)
            detections.append(True)
        else:
            poses_wlm.append([])
            detections.append(False)

    cap.release()
    return frames_bgr, poses_wlm, detections, total_frames


def create_plotly_3d_skeleton(landmarks, detected: bool, frame_idx: int):
    """Generates an interactive Plotly 3D skeletal figure with camera rotation and zooming."""
    fig = go.Figure()

    if not detected or not landmarks:
        # Empty space with clear warning annotation
        fig.add_annotation(
            text=f"Frame #{frame_idx}: No Pose Detected in Video Stream<br><sub>(Zero-Persistence Mode Active)</sub>",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#FF5555", size=14, family="Arial")
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="#12151E"
            ),
            paper_bgcolor="#12151E",
            margin=dict(l=0, r=0, t=30, b=0),
            height=460
        )
        return fig

    # 1. Add skeletal connection bones
    for start_idx, end_idx, limb in SKELETON_BONES:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            p1, p2 = landmarks[start_idx], landmarks[end_idx]
            if p1["vis"] >= 0.3 and p2["vis"] >= 0.3:
                color = BONE_COLORS.get(limb, "#00ADB5")
                width = 5 if (p1["vis"] >= 0.5 and p2["vis"] >= 0.5) else 2
                fig.add_trace(go.Scatter3d(
                    x=[p1["x"], p2["x"]],
                    y=[p1["z"], p2["z"]],
                    z=[p1["y"], p2["y"]],
                    mode="lines",
                    line=dict(color=color, width=width),
                    hoverinfo="none",
                    showlegend=False
                ))

    # 2. Add joint landmark nodes
    xs = [pt["x"] for pt in landmarks]
    ys = [pt["z"] for pt in landmarks]
    zs = [pt["y"] for pt in landmarks]
    vis_scores = [pt["vis"] for pt in landmarks]
    node_colors = ["#00FF66" if v >= 0.5 else ("#FFAA00" if v >= 0.3 else "#777777") for v in vis_scores]

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(
            size=4.5,
            color=node_colors,
            opacity=0.9,
            line=dict(color="#111111", width=1)
        ),
        text=[f"Joint {i}<br>Vis: {v:.2f}" for i, v in enumerate(vis_scores)],
        hoverinfo="text",
        showlegend=False
    ))

    # Equal scaling bounding cube
    max_range = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) / 2.0
    mid_x, mid_y, mid_z = np.mean(xs), np.mean(ys), np.mean(zs)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Lateral X (m)", range=[mid_x - max_range, mid_x + max_range], backgroundcolor="#12151E", gridcolor="#2E384D", color="#A0AAB8"),
            yaxis=dict(title="Depth Z (m)", range=[mid_y - max_range, mid_y + max_range], backgroundcolor="#12151E", gridcolor="#2E384D", color="#A0AAB8"),
            zaxis=dict(title="Upright Y (m)", range=[mid_z - max_range, mid_z + max_range], backgroundcolor="#12151E", gridcolor="#2E384D", color="#A0AAB8"),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.35, y=-1.35, z=0.75),
                up=dict(x=0, y=0, z=1)
            )
        ),
        paper_bgcolor="#12151E",
        margin=dict(l=0, r=0, t=10, b=0),
        height=480
    )
    return fig


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/isometric/100/cricket.png", width=56)
st.sidebar.title("Cricket3D Motion")
st.sidebar.caption("Motion-Aware Keyframing & Lightweight 3D Pose Analytics")

nav_choice = st.sidebar.radio(
    "Navigation",
    ["🎬 Interactive 3D Viewer", "📊 Methodology & Benchmark Results", "ℹ️ Project Scope & Setup"]
)

# Discover available clips
available_clips = sorted(list(SAMPLE_DIR.rglob("*.avi")))
clip_dict = {f"{c.parent.name} - {c.name}": str(c.relative_to(BASE_DIR)).replace("\\", "/") for c in available_clips}

# --- PAGE 1: INTERACTIVE 3D VIEWER ---
if nav_choice == "🎬 Interactive 3D Viewer":
    st.title("Interactive 3D Cricket Player Pose Analytics")
    st.caption("Play a sample clip and inspect its video frames alongside monocular 3D body landmarks.")

    # Controls Bar in Sidebar
    st.sidebar.subheader("Clip Selection")
    default_clip = list(clip_dict.keys())[7] if len(clip_dict) > 7 else list(clip_dict.keys())[0] # Pull shot default
    selected_clip_label = st.sidebar.selectbox("Choose Sample Clip", list(clip_dict.keys()), index=list(clip_dict.keys()).index(default_clip))
    selected_clip_path = clip_dict[selected_clip_label]

    # Pre-cache and load clip frames
    with st.spinner("Decoding video & extracting 3D world landmarks..."):
        frames_bgr, poses_wlm, detections, total_frames = load_and_cache_clip_data(selected_clip_path)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Pose Frame Scrubbing")

    # Session state for the pose-frame slider
    if "timeline_frame" not in st.session_state:
        st.session_state.timeline_frame = 12 if total_frames > 12 else 0
    else:
        st.session_state.timeline_frame = min(
            st.session_state.timeline_frame, total_frames - 1
        )
    if "is_pose_playing" not in st.session_state:
        st.session_state.is_pose_playing = False

    # Slider control
    selected_frame = st.sidebar.slider(
        "Timeline Frame Index",
        min_value=0,
        max_value=total_frames - 1,
        value=st.session_state.timeline_frame,
        step=1
    )
    st.session_state.timeline_frame = selected_frame

    # These controls animate the pose visualization independently of the video player.
    col_btn1, col_btn2, col_btn3 = st.sidebar.columns([1, 1.3, 1])
    with col_btn1:
        st.button(
            "◀ Step",
            on_click=step_pose_frame,
            args=(-1, total_frames),
            use_container_width=True
        )
    with col_btn2:
        play_label = "❚❚ Pause 3D" if st.session_state.is_pose_playing else "▶ Play 3D"
        if st.button(play_label, use_container_width=True):
            st.session_state.is_pose_playing = not st.session_state.is_pose_playing
            st.rerun()
    with col_btn3:
        st.button(
            "Step ▶",
            on_click=step_pose_frame,
            args=(1, total_frames),
            use_container_width=True
        )

    # Main Visualizer Dual Panel
    f_idx = selected_frame
    has_pose = detections[f_idx]
    current_landmarks = poses_wlm[f_idx]
    current_frame_bgr = frames_bgr[f_idx]

    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        st.markdown(f"**Shot Action:** `{selected_clip_label.split(' - ')[0]}`")
    with col_meta2:
        st.markdown(f"**Current Frame:** `{f_idx} / {total_frames - 1}`")
    with col_meta3:
        if has_pose:
            st.markdown('<span class="status-badge-ok">● POSE DETECTED (33 Landmarks)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge-warn">▲ NO POSE DETECTED (Zero Persistence)</span>', unsafe_allow_html=True)

    col_video, col_3d = st.columns([1.1, 1.2])

    with col_video:
        st.subheader("Source Video")
        video_path = (BASE_DIR / selected_clip_path).with_suffix(".mp4")
        if video_path.is_file():
            st.video(video_path.read_bytes(), format="video/mp4")
            st.caption("Video playback is independent of the pose-frame scrubber.")
        else:
            st.error(f"Sample video file missing: {video_path}")

        st.subheader("Selected Pose Frame")
        if current_frame_bgr is not None:
            rgb_img = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2RGB)
            st.image(rgb_img, caption=f"Frame #{f_idx} ({rgb_img.shape[1]}x{rgb_img.shape[0]})", use_container_width=True)
        else:
            st.error(f"Failed to read frame #{f_idx}")

    with col_3d:
        st.subheader("Interactive 3D Skeletal Graph")
        fig_3d = create_plotly_3d_skeleton(current_landmarks, has_pose, f_idx)
        st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("""
    <div class="disclaimer-box">
        <b>Scientific & Methodological Notice:</b> 3D coordinates represent estimated monocular poses output by Google's MediaPipe <code>pose_world_landmarks</code> neural network, measured in meters relative to the batsman's hip root. They do <b>not</b> constitute calibrated optical motion-capture ground truth.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.is_pose_playing:
        time.sleep(0.2)
        st.session_state.timeline_frame = (f_idx + 1) % total_frames
        st.rerun()

# --- PAGE 2: METHODOLOGY & BENCHMARK RESULTS ---
elif nav_choice == "📊 Methodology & Benchmark Results":
    st.title("Frame Selection Evaluation & Comparative Benchmarks")
    st.caption("Quantitative comparison of Uniform, Motion-Aware, and Hybrid selection across all 12 clips (Budget = 6 frames/clip).")

    sum_file = REPORTS_DIR / "hybrid_selection_summary.json"
    audit_file = REPORTS_DIR / "hybrid_metric_audit.json"

    if sum_file.exists():
        with open(sum_file, "r") as f:
            summary_data = json.load(f)
        with open(audit_file, "r") as f:
            audit_data = json.load(f)

        agg = summary_data["aggregate_comparison"]
        u_agg, m_agg, h_agg = agg["uniform"], agg["motion_aware"], agg["hybrid"]

        st.subheader("1. Aggregate Evaluation (12 Clips, 288 Reference Frames)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Detection Rate", f"{h_agg['mean_detection_rate_pct']:.1f}%", f"+{h_agg['mean_detection_rate_pct'] - u_agg['mean_detection_rate_pct']:.1f}% vs Uniform")
        with col2:
            st.metric("Landmark Visibility", f"{h_agg['mean_landmark_visibility']:.4f}", f"+{h_agg['mean_landmark_visibility'] - u_agg['mean_landmark_visibility']:.4f}")
        with col3:
            st.metric("Temporal Span Ratio", f"{h_agg['mean_temporal_span_ratio']:.2f}", "100% Full Clip Coverage")
        with col4:
            st.metric("Nearest-Pose Error (NSPE)", f"{h_agg['mean_nearest_pose_error_nspe']:.4f}", f"{h_agg['mean_nearest_pose_error_nspe'] - u_agg['mean_nearest_pose_error_nspe']:+.4f} vs Uniform")

        st.markdown("---")
        st.subheader("2. Comparative Methodology Metrics Table")

        comp_table = [
            {"Method": "Uniform (Baseline)", "Frames": 6, "Detection Rate": f"{u_agg['mean_detection_rate_pct']:.2f}%", "Visibility": f"{u_agg['mean_landmark_visibility']:.4f}", "Temporal Span": f"{u_agg['mean_temporal_span_ratio']:.2f}", "Max Gap": f"{u_agg['mean_max_temporal_gap_frames']:.1f} frames", "NSPE": f"{u_agg['mean_nearest_pose_error_nspe']:.4f}"},
            {"Method": "Motion-Aware (Peaks)", "Frames": 6, "Detection Rate": f"{m_agg['mean_detection_rate_pct']:.2f}%", "Visibility": f"{m_agg['mean_landmark_visibility']:.4f}", "Temporal Span": f"{m_agg['mean_temporal_span_ratio']:.2f}", "Max Gap": f"{m_agg['mean_max_temporal_gap_frames']:.1f} frames", "NSPE": f"{m_agg['mean_nearest_pose_error_nspe']:.4f}"},
            {"Method": "Hybrid (Anchored)", "Frames": 6, "Detection Rate": f"{h_agg['mean_detection_rate_pct']:.2f}%", "Visibility": f"{h_agg['mean_landmark_visibility']:.4f}", "Temporal Span": f"{h_agg['mean_temporal_span_ratio']:.2f}", "Max Gap": f"{h_agg['mean_max_temporal_gap_frames']:.1f} frames", "NSPE": f"{h_agg['mean_nearest_pose_error_nspe']:.4f}"},
        ]
        st.dataframe(comp_table, use_container_width=True)

        st.markdown("---")
        st.subheader("3. Per-Class Optimal Selection Performance")
        per_class = summary_data["per_class_comparison"]
        class_table = []
        for c_name, c_data in per_class.items():
            u_err = c_data['uniform']['mean_nspe']
            m_err = c_data['motion_aware']['mean_nspe']
            h_err = c_data['hybrid']['mean_nspe']
            scores = [("Uniform", u_err), ("Motion-Aware", m_err), ("Hybrid", h_err)]
            best = min(scores, key=lambda x: x[1])[0]
            class_table.append({
                "Cricket Shot Class": c_name,
                "Clips": c_data["num_clips"],
                "Uniform NSPE": f"{u_err:.4f}",
                "Motion NSPE": f"{m_err:.4f}",
                "Hybrid NSPE": f"{h_err:.4f}",
                "Top Method": best
            })
        st.dataframe(class_table, use_container_width=True)

    else:
        st.warning("Evaluation report files not found in reports/")

# --- PAGE 3: PROJECT SCOPE & SETUP ---
else:
    st.title("Project Scope, Hardware Budget & Architecture")
    st.markdown("""
    ### Research Objective
    To explore **motion-aware frame selection and lightweight 3D cricket-player pose estimation** from short broadcast action clips under consumer hardware constraints.

    ### Hardware Budget & Environment
    - **Operating System:** Windows 64-bit
    - **Target GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB / 4 GB VRAM)
    - **Active Execution Mode:** Pure CPU Execution (TensorFlow Lite XNNPACK delegate, 0 MiB VRAM used)
    - **Host Python:** Python 3.11.9 64-bit (Pip 24.0)

    ### Pipeline Architecture
    1. **Selective Decoupled Ingestion:** Target action `.avi` clips sampled from RAR archives via native `tar.exe` without full dataset extraction.
    2. **Motion Analysis Engine:** Consecutive frame Mean Absolute Difference (MAD) motion scoring.
    3. **Hybrid Frame Selection (6-frame budget):** 3 full-span boundary anchors + 3 intra-stroke motion energy peaks.
    4. **Pose Estimation:** MediaPipe Tasks PoseLandmarker extracting 33 2D/3D world landmark coordinates.
    5. **Telemetry Dashboard:** Streamlit & Plotly interactive browser visualization.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Cricket3D_Motion • Research Methodology Project")
