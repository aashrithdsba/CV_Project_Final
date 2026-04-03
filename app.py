import streamlit as st
import cv2
import os
import time
import numpy as np
from cv_pipeline import ClassicalCVPipeline
from grozi_pipeline import GroziPipeline

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, 'data')
IS_DEMO_MODE = False

if not os.path.exists(DATA_ROOT):
    DATA_ROOT = os.path.join(BASE_DIR, 'demo_data')
    IS_DEMO_MODE = True

OTB_DIR = os.path.join(DATA_ROOT, 'OTB-dataset', 'sequences')

# Page Config
st.set_page_config(
    layout="wide", 
    page_title="Classical CV Dashboard" + (" (Demo Mode)" if IS_DEMO_MODE else ""), 
    page_icon="🔬", 
    initial_sidebar_state="expanded"
)

# ─── Shared HTML / CSS ──────────────────────────────────────────────
EMBEDDED_STYLES = """
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: transparent; font-family: 'Inter', -apple-system, sans-serif; }
    .inventory-grid { display: flex; flex-direction: column; gap: 8px; }
    .status-card {
        background: #161b22; border: 1px solid #21262d; border-radius: 8px;
        padding: 10px 12px; border-left: 4px solid #21262d; transition: all 0.3s ease;
    }
    .status-card.on-shelf { border-left-color: #00ff88; }
    .status-card.picked { border-left-color: #ff6b6b; }
    .status-card.restocked { border-left-color: #ffd93d; }
    .status-card.unknown { border-left-color: #484f58; }
    .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
    .card-pid { color: #e6edf3; font-weight: 600; font-size: 0.85rem; }
    .card-badge { padding: 2px 8px; border-radius: 12px; font-size: 0.62rem; font-weight: 600; letter-spacing: 0.5px; }
    .badge-on-shelf { background: #00ff8822; color: #00ff88; }
    .badge-picked { background: #ff6b6b22; color: #ff6b6b; }
    .badge-restocked { background: #ffd93d22; color: #ffd93d; }
    .badge-unknown { background: #484f5822; color: #8b949e; }
    .card-detail { color: #8b949e; font-size: 0.7rem; }
    .cv-info-bar {
        background: #161b22; border: 1px solid #21262d; border-radius: 8px;
        padding: 10px 14px; display: flex; justify-content: space-between;
        align-items: center; flex-wrap: wrap; gap: 8px; font-size: 0.75rem;
    }
    .cv-info-item { display: flex; align-items: center; gap: 5px; }
    .cv-info-label { color: #484f58; text-transform: uppercase; letter-spacing: 0.5px; font-size: 0.65rem; }
    .cv-info-value { color: #c9d1d9; font-weight: 600; }
    .metric-container {
        background: #161b22; border: 1px solid #21262d; border-radius: 10px;
        padding: 14px 16px; text-align: center;
    }
    .metric-container.cyan { border-top: 3px solid #00d4ff; }
    .metric-container.green { border-top: 3px solid #00ff88; }
    .metric-container.red { border-top: 3px solid #ff6b6b; }
    .metric-container.orange { border-top: 3px solid #ffa657; }
    .metric-label { color: #8b949e; font-size: 0.68rem; text-transform: uppercase; margin-bottom: 4px; }
    .metric-value { color: #e6edf3; font-size: 1.7rem; font-weight: 700; line-height: 1.2; }
    .empty-state { text-align: center; padding: 50px 20px; color: #484f58; }
    .empty-state .icon { font-size: 2.5rem; margin-bottom: 12px; }
    .empty-state h3 { color: #8b949e; margin-bottom: 6px; font-size: 1rem; }
</style>
"""

def inject_custom_css():
    st.markdown("""
    <style>
    .stApp { background-color: #0a0e14; font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} [data-testid="stToolbar"] {display: none;} header[data-testid="stHeader"] {display: none;} footer {visibility: hidden;}
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); border-right: 1px solid #21262d; }
    .dashboard-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid #21262d; border-radius: 12px; padding: 18px 30px; margin-bottom: 16px; text-align: center;
    }
    .dashboard-header h1 { color: #e6edf3; font-size: 1.6rem; font-weight: 700; margin: 0; }
    .dashboard-header p { color: #8b949e; font-size: 0.85rem; margin: 4px 0 0 0; }
    .section-header { color: #e6edf3; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; padding-bottom: 8px; margin-bottom: 10px; margin-top: 6px; border-bottom: 1px solid #21262d; }
    .live-badge { background: #da3633; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.6rem; font-weight: 700; margin-left: 8px; }
    .disclaimer-box {
        background: #1c1c0e; border: 1px solid #3d3d00; border-radius: 8px;
        padding: 10px 14px; margin-bottom: 12px; font-size: 0.78rem; color: #c9a227;
    }
    .stPlotlyChart { border-radius: 10px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────
def _wrap_html(body): return EMBEDDED_STYLES + body

def render_metric_html(label, value, accent_class):
    return _wrap_html(f'<div class="metric-container {accent_class}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>')

def render_cv_info_html(method, templates_loaded, frame_num, fps):
    return _wrap_html(f"""
    <div class="cv-info-bar">
        <div class="cv-info-item"><span class="cv-info-label">Method</span><span class="cv-info-value">{method}</span></div>
        <div class="cv-info-item"><span class="cv-info-label">Template</span><span class="cv-info-value">{"Acquired" if templates_loaded else "Failed"}</span></div>
        <div class="cv-info-item"><span class="cv-info-label">Frame</span><span class="cv-info-value">{frame_num}</span></div>
        <div class="cv-info-item"><span class="cv-info-label">FPS</span><span class="cv-info-value">{fps:.1f}</span></div>
    </div>""")

def render_inventory_card_html(pid, current, overall):
    card_cls = 'on-shelf' if current == 'In View' else ('picked' if current == 'Lost' else 'restocked')
    return f"""
    <div class="status-card {card_cls}">
        <div class="card-header"><span class="card-pid">🎯 {pid}</span><span class="card-badge {card_cls}">{current.upper()}</span></div>
        <div class="card-detail">{overall}</div>
    </div>"""

def render_inventory_cards_html(items):
    """items: list of (pid_label, current, overall)"""
    cards = ''.join(render_inventory_card_html(p, c, o) for p, c, o in items)
    return _wrap_html(f'<div class="inventory-grid">{cards}</div>')


# ═══════════════════════════════════════════════════════════════════
#  OTB TAB — process_video
# ═══════════════════════════════════════════════════════════════════
def process_otb_video(sequence_path, placeholders, metric_cols):
    pipeline = ClassicalCVPipeline(sequence_path)
    templates_loaded = 1 if pipeline.template_data.get('img') is not None else 0
    target_name = pipeline.target_name

    img_dir = os.path.join(sequence_path, 'img')
    img_format = os.path.join(img_dir, '%04d.jpg')
    if not os.path.exists(os.path.join(img_dir, '0001.jpg')):
        img_format = os.path.join(img_dir, '%05d.jpg')

    cap = cv2.VideoCapture(img_format)
    if not cap.isOpened():
        st.error(f"Failed to open image sequence in {img_dir}")
        return

    total_frames = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    frame_num = 0
    fps_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_num += 1

        annotated_frame, statuses = pipeline.process_frame(frame)
        current = statuses[target_name]['current']
        overall = statuses[target_name]['overall']

        elapsed = time.time() - fps_start
        fps_value = frame_num / elapsed if elapsed > 0 else 0

        placeholders['video_frame'].image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        placeholders['cv_info'].html(render_cv_info_html('SIFT', templates_loaded, frame_num, fps_value))
        placeholders['progress_bar'].progress(min(frame_num / total_frames, 1.0))

        metric_cols[0].html(render_metric_html("Target", target_name, "cyan"))
        metric_cols[1].html(render_metric_html("Status", current, "green" if current == "In View" else "red"))
        metric_cols[2].html(render_metric_html("Frames", str(frame_num), "cyan"))

        if frame_num % 2 == 0:
            cards = [(target_name, current, overall)]
            placeholders['inventory_panel'].html(render_inventory_cards_html(cards))

    cap.release()


# ═══════════════════════════════════════════════════════════════════
#  GROZI TAB — process_video
# ═══════════════════════════════════════════════════════════════════
def process_grozi_video(video_name, product_ids, placeholders, metric_cols):
    pipeline = GroziPipeline(video_name, product_ids, data_root=DATA_ROOT)
    vid_path = pipeline.get_video_path()

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        st.error(f"Failed to open video: {vid_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 500
    frame_num = 0
    fps_start = time.time()
    detected_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_num += 1

        annotated_frame, statuses = pipeline.process_frame(frame)

        in_view = sum(1 for s in statuses.values() if s['current'] == 'In View')
        if in_view > detected_count:
            detected_count = in_view

        elapsed = time.time() - fps_start
        fps_value = frame_num / elapsed if elapsed > 0 else 0

        placeholders['video_frame'].image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        placeholders['cv_info'].html(render_cv_info_html('SIFT', len(pipeline.templates), frame_num, fps_value))
        placeholders['progress_bar'].progress(min(frame_num / total_frames, 1.0))

        metric_cols[0].html(render_metric_html("Products", f"{len(product_ids)}", "cyan"))
        metric_cols[1].html(render_metric_html("In View", str(in_view), "green" if in_view > 0 else "red"))
        metric_cols[2].html(render_metric_html("Frame", str(frame_num), "cyan"))

        if frame_num % 2 == 0:
            items = [(s['name'], s['current'], s['overall']) for s in statuses.values()]
            placeholders['inventory_panel'].html(render_inventory_cards_html(items))

    cap.release()


# ═══════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════
inject_custom_css()

tab_otb, tab_grozi = st.tabs(["🎯 OTB Object Tracking", "🛒 Retail Detection (Grozi-120)"])

# ───────────── TAB 1: OTB ─────────────
with tab_otb:
    st.markdown('<div class="dashboard-header"><h1>OTB Visual Object Tracking</h1><p>Template extraction via ground-truth bounding boxes · SIFT feature matching</p></div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    otb_metric_cols = [m1.empty(), m2.empty(), m3.empty()]
    otb_metric_cols[0].html(render_metric_html("Target", "—", "cyan"))
    otb_metric_cols[1].html(render_metric_html("Status", "—", "green"))
    otb_metric_cols[2].html(render_metric_html("Frames", "0", "cyan"))
    st.markdown('<div style="margin-top:18px"></div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([7, 3])
    with left_col:
        st.markdown('<div class="section-header"><span class="icon">📹</span>Sequence Feed<span class="live-badge">LIVE</span></div>', unsafe_allow_html=True)
        otb_cv_info = st.empty()
        otb_video = st.empty()
        otb_progress = st.empty()
    with right_col:
        st.markdown('<div class="section-header"><span class="icon">📦</span>Target State</div>', unsafe_allow_html=True)
        otb_inventory = st.empty()

with st.sidebar:
    st.markdown("## 🔬 Classical CV Pipeline")
    if IS_DEMO_MODE:
        st.info("🚀 **Running in Demo Mode**\nUsing a lightweight subset for cloud deployment.")
    st.markdown("---")

    # Detect which tab is active via a hidden selectbox synced with tabs
    active_section = st.radio("Section", ["OTB Tracking", "Grozi Retail"], horizontal=True, label_visibility="collapsed")

    if active_section == "OTB Tracking":
        st.markdown("### 🎯 OTB Settings")
        otb_sequences = []
        if os.path.exists(OTB_DIR):
            otb_sequences = sorted([d for d in os.listdir(OTB_DIR)
                                    if os.path.isdir(os.path.join(OTB_DIR, d)) and not d.startswith('.')])
        selected_otb_seq = st.selectbox("Sequence", otb_sequences, key="otb_seq")

        if selected_otb_seq:
            img_path = os.path.join(OTB_DIR, selected_otb_seq, 'img', '0001.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(OTB_DIR, selected_otb_seq, 'img', '00001.jpg')
            if os.path.exists(img_path):
                st.image(img_path, caption=f"Frame 1: {selected_otb_seq}", use_container_width=True)

        otb_start = st.button("▶ Start Tracking", use_container_width=True, type="primary", key="otb_go")

    else:
        st.markdown("### 🛒 Grozi Settings")
        grozi_videos = GroziPipeline.get_available_videos(data_root=DATA_ROOT)
        selected_grozi_vid = st.selectbox("Shelf Video", grozi_videos, key="grozi_vid")

        available_pids = []
        if selected_grozi_vid:
            available_pids = GroziPipeline.get_products_for_video(selected_grozi_vid, data_root=DATA_ROOT)

        selected_pids = st.multiselect(
            "Products to track",
            options=available_pids,
            default=available_pids[:3] if len(available_pids) >= 3 else available_pids,
            format_func=lambda x: f"Product {x}",
            key="grozi_pids"
        )

        if selected_grozi_vid and selected_pids:
            # Show first template preview
            first_tpl = os.path.join(GroziPipeline.TEMPLATE_DIR, str(selected_pids[0]), 'web', 'JPEG', 'web1.jpg')
            if os.path.exists(first_tpl):
                st.image(first_tpl, caption=f"Template: Product {selected_pids[0]}", use_container_width=True)

        grozi_start = st.button("▶ Start Detection", use_container_width=True, type="primary", key="grozi_go")

# ───────────── TAB 2: GROZI ─────────────
with tab_grozi:
    st.markdown('<div class="dashboard-header"><h1>Retail Product Detection <span style="font-size:0.7em;color:#8b949e;">(Grozi-120)</span></h1><p>Real-world shelf monitoring demo · SIFT feature matching</p></div>', unsafe_allow_html=True)
    if IS_DEMO_MODE:
        st.warning("⚠️ **Note:** Currently running in **Demo Mode** with a limited subset. For the full dataset, refer to the [README.md](https://github.com/aashrithdsba/CV_Project) local setup instructions.")
    st.markdown('<div class="disclaimer-box">⚠️ <strong>Proof of concept</strong> — Classical CV methods (SIFT matching) have inherent limitations for cluttered retail environments. This tab demonstrates the potential application; accuracy improves significantly with deep-learning approaches.</div>', unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    grozi_metric_cols = [g1.empty(), g2.empty(), g3.empty()]
    grozi_metric_cols[0].html(render_metric_html("Products", "—", "cyan"))
    grozi_metric_cols[1].html(render_metric_html("In View", "—", "green"))
    grozi_metric_cols[2].html(render_metric_html("Frame", "0", "cyan"))
    st.markdown('<div style="margin-top:18px"></div>', unsafe_allow_html=True)

    gleft, gright = st.columns([7, 3])
    with gleft:
        st.markdown('<div class="section-header"><span class="icon">📹</span>Shelf Camera<span class="live-badge">LIVE</span></div>', unsafe_allow_html=True)
        grozi_cv_info = st.empty()
        grozi_video = st.empty()
        grozi_progress = st.empty()
    with gright:
        st.markdown('<div class="section-header"><span class="icon">📦</span>Product Status</div>', unsafe_allow_html=True)
        grozi_inventory = st.empty()

# ─── Run logic ───────────────────────────────────────────────────
if active_section == "OTB Tracking" and otb_start and selected_otb_seq:
    seq_path = os.path.join(OTB_DIR, selected_otb_seq)
    phs = {
        'video_frame': otb_video,
        'cv_info': otb_cv_info,
        'progress_bar': otb_progress,
        'inventory_panel': otb_inventory,
    }
    with st.spinner("Initializing OTB pipeline…"):
        process_otb_video(seq_path, phs, otb_metric_cols)
    st.success("Sequence processing complete.")

elif active_section == "Grozi Retail" and grozi_start and selected_grozi_vid and selected_pids:
    phs = {
        'video_frame': grozi_video,
        'cv_info': grozi_cv_info,
        'progress_bar': grozi_progress,
        'inventory_panel': grozi_inventory,
    }
    with st.spinner("Initializing Grozi pipeline…"):
        process_grozi_video(selected_grozi_vid, selected_pids, phs, grozi_metric_cols)
    st.success("Video processing complete.")
