"""
dashboard/app.py
----------------
Main Streamlit interface for the CodeVerse Custom C++ Deep Learning Engine.

Provides two operational modes:
  1. Live Brain   -- Real-time training loop visualizer with epoch-level metrics.
  2. Hardware Inference Engine -- Single-shot FPS prediction with hardware consulting.

Launch:
    streamlit run dashboard/app.py
"""

import os
import sys
import time

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path configuration: allow imports from the project root and files/ subdirectory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FILES_DIR    = os.path.join(_PROJECT_ROOT, "files")

for _p in (_PROJECT_ROOT, _FILES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core_training import run_training_loop
from predict import (
    get_prediction_report,
    search_games,
    load_game_requirements,
    REQS_CSV,
)

# Pre-load game database once (cached across Streamlit reruns via module scope).
_predict_game_db = load_game_requirements(REQS_CSV)

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="CodeVerse Neural Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS -- Dark Enterprise aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

        /* ---- Hide sidebar ---- */
        [data-testid="stSidebar"],
        section[data-testid="stSidebarNav"] { display: none !important; }

        /* ---- Global typography ---- */
        html, body, .stApp {
            font-family: 'Inter', sans-serif;
            color: #F1F5F9;
        }
        code, pre, .stCodeBlock {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* ---- Page header ---- */
        .main-header {
            font-family: 'Inter', sans-serif;
            font-size: 1.75rem;
            font-weight: 700;
            color: #F8FAFC;
            padding-bottom: 0.6rem;
            border-bottom: 1px solid #1E293B;
            margin-bottom: 0.3rem;
        }
        .sub-header {
            font-family: 'Inter', sans-serif;
            font-size: 0.88rem;
            color: #94A3B8;
            margin-top: 0.2rem;
            margin-bottom: 1.6rem;
            line-height: 1.5;
        }

        /* ---- Section headings (h4 = section, h5 = subsection) ---- */
        h4, .stMarkdown h4 {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            color: #F8FAFC !important;
            border-bottom: 1px solid #1E293B;
            padding-bottom: 0.45rem;
            margin-top: 1.2rem !important;
            letter-spacing: 0.2px !important;
        }
        h5, .stMarkdown h5 {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            color: #64748B !important;
            letter-spacing: 0.5px !important;
            text-transform: uppercase !important;
            margin-bottom: 0.5rem !important;
        }

        /* ---- Metric overrides ---- */
        [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            color: #F1F5F9 !important;
        }
        [data-testid="stMetricLabel"] {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.7rem !important;
            color: #64748B !important;
            letter-spacing: 0.3px;
        }
        [data-testid="stMetricDelta"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.72rem !important;
        }

        /* ---- Input label visibility ---- */
        .stSlider label, .stTextInput label, .stSelectbox label {
            color: #CBD5E1 !important;
            font-size: 0.82rem !important;
        }

        /* ---- Cards ---- */
        .cyber-card {
            background-color: #171C28;
            border: 1px solid #2D3748;
            border-radius: 12px;
            padding: 1.25rem 1.4rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            line-height: 1.9;
            color: #94A3B8;
            box-shadow: 0 1px 3px rgba(0,0,0,0.25);
            margin-bottom: 0.7rem;
        }
        .cyber-card .card-title {
            font-family: 'Inter', sans-serif;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            color: #64748B;
            margin-bottom: 0.6rem;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid #2D3748;
        }
        .cyber-card .cl { color: #64748B; }
        .cyber-card .cv { color: #E2E8F0; font-weight: 500; }
        .cyber-card .cv-accent { color: #10B981; font-weight: 600; }

        /* ---- Hero FPS ---- */
        .hero-fps {
            text-align: center;
            padding: 2rem 1rem;
            background-color: #171C28;
            border: 1px solid #2D3748;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.25);
        }
        .hero-fps .fps-number {
            font-family: 'JetBrains Mono', monospace;
            font-size: 4.2rem;
            font-weight: 700;
            color: #10B981;
            line-height: 1;
        }
        .hero-fps .fps-unit {
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            color: #64748B;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-top: 0.6rem;
        }

        /* ---- Phase badges ---- */
        .phase-badge {
            display: inline-block;
            padding: 0.4rem 1rem;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            font-weight: 500;
            margin-top: 0.5rem;
        }
        .phase-0 {
            background-color: rgba(245,158,11,0.08);
            color: #F59E0B;
            border: 1px solid rgba(245,158,11,0.2);
        }
        .phase-1 {
            background-color: rgba(59,130,246,0.08);
            color: #3B82F6;
            border: 1px solid rgba(59,130,246,0.2);
        }
        .phase-2 {
            background-color: rgba(16,185,129,0.08);
            color: #10B981;
            border: 1px solid rgba(16,185,129,0.2);
        }

        /* ---- Verdict boxes ---- */
        .verdict-box {
            padding: 1rem 1.4rem;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            font-weight: 500;
            margin-bottom: 0.8rem;
        }
        .verdict-esports {
            background-color: rgba(16,185,129,0.08);
            color: #10B981;
            border: 1px solid rgba(16,185,129,0.2);
        }
        .verdict-smooth {
            background-color: rgba(59,130,246,0.08);
            color: #3B82F6;
            border: 1px solid rgba(59,130,246,0.2);
        }
        .verdict-playable {
            background-color: rgba(245,158,11,0.08);
            color: #F59E0B;
            border: 1px solid rgba(245,158,11,0.2);
        }
        .verdict-unplayable {
            background-color: rgba(239,68,68,0.08);
            color: #EF4444;
            border: 1px solid rgba(239,68,68,0.2);
        }

        /* ---- Bottleneck cards ---- */
        .consul-optimal {
            background-color: rgba(16,185,129,0.06);
            color: #10B981;
            border-left: 3px solid #10B981;
            padding: 0.9rem 1.2rem;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
        }
        .consul-warning {
            background-color: rgba(245,158,11,0.06);
            color: #F59E0B;
            border-left: 3px solid #F59E0B;
            padding: 0.9rem 1.2rem;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
        }

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            border-bottom: 1px solid #1E293B;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.85rem;
            color: #94A3B8;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #F1F5F9 !important;
        }

        /* ---- Dividers ---- */
        hr {
            border-color: #1E293B !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# Header
# ============================================================================

st.markdown('<div class="main-header">CodeVerse AI // Custom C++ Tensor Engine</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time neural network operations powered by a '
            'hand-built C++ backend with custom memory pooling and autograd.</div>',
            unsafe_allow_html=True)

# ============================================================================
# Navigation Tabs
# ============================================================================

tab_train, tab_infer = st.tabs(["Live Brain", "Hardware Inference Engine"])


# ----------------------------------------------------------------------------
# Tab 1: Live Brain (Training Visualizer)
# ----------------------------------------------------------------------------

with tab_train:
    st.markdown("#### Training Control Panel")

    btn_train = st.button("Initialize Training Sequence", key="btn_train",
                          use_container_width=False)

    col_metrics, col_graph = st.columns([1, 3], gap="large")

    with col_metrics:
        st.markdown("##### Epoch Telemetry")
        ph_epoch = st.empty()
        ph_loss  = st.empty()
        ph_time  = st.empty()

        st.markdown("---")
        st.markdown("##### System Diagnostics")

        # [SYS] Model Topology
        st.markdown(
            '<div class="cyber-card">'
            '<div class="card-title">[SYS] Model Topology</div>'
            '<span class="cl">Input Features</span>  '
            '<span class="cv-accent">9</span><br>'
            '<span class="cl">Hidden Layers</span>  '
            '<span class="cv">1 (16 neurons, ReLU)</span><br>'
            '<span class="cl">Output Layer</span>  '
            '<span class="cv">1 (Linear)</span><br>'
            '<span class="cl">Biases</span>  '
            '<span class="cv">Disabled (engine workaround)</span><br>'
            '<span class="cl">Trainable Params</span>  '
            '<span class="cv-accent">160</span> '
            '<span class="cl">(W1: 9x16, W2: 16x1)</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # [NET] Execution & Hyperparameters
        st.markdown(
            '<div class="cyber-card">'
            '<div class="card-title">[NET] Execution</div>'
            '<span class="cl">Loss Function</span>  '
            '<span class="cv">SC_MSE (C-API bypass)</span><br>'
            '<span class="cl">Optimizer</span>  '
            '<span class="cv">Full-Batch Gradient Descent</span><br>'
            '<span class="cl">Learning Rate</span>  '
            '<span class="cv-accent">0.05</span><br>'
            '<span class="cl">Epochs</span>  '
            '<span class="cv-accent">500</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # [OPS] Engine Memory & Graph
        st.markdown(
            '<div class="cyber-card">'
            '<div class="card-title">[OPS] Engine Memory</div>'
            '<span class="cl">Backend</span>  '
            '<span class="cv">SCBackend.CPU (Custom C++)</span><br>'
            '<span class="cl">Total Allocated</span>  '
            '<span class="cv-accent">110 MB</span> '
            '<span class="cl">(50 Pool + 10 Meta + 50 Grad)</span><br>'
            '<span class="cl">Precision</span>  '
            '<span class="cv">FP32</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        ph_samples = st.empty()

        st.markdown("---")
        st.markdown("##### Learning Status")
        ph_phase = st.empty()

    with col_graph:
        st.markdown("##### Loss Curve (Real-Time)")
        ph_chart = st.empty()

    # -- Training loop execution ---------------------------------------------
    if btn_train:
        loss_history = []

        for step in run_training_loop():
            epoch = step["epoch"]
            loss  = step["loss"]
            t_ms  = step["time_ms"]
            phase = step["learning_phase"]

            # Update metric cards.
            total = step["total_epochs"]
            ph_epoch.metric(label="Epoch", value=f"{epoch} / {total}")
            ph_samples.metric(label="Training Samples", value=f"{step['num_samples']}")
            ph_loss.metric(label="Loss", value=f"{loss:.6f}")
            ph_time.metric(label="Step Time", value=f"{t_ms:.2f} ms")

            # Update learning status badge.
            if phase == 0:
                ph_phase.markdown(
                    '<div class="phase-badge phase-0">'
                    'Phase 0: Initial weight configuration...</div>',
                    unsafe_allow_html=True,
                )
            elif phase == 1:
                ph_phase.markdown(
                    '<div class="phase-badge phase-1">'
                    'Phase 1: Pattern convergence detected...</div>',
                    unsafe_allow_html=True,
                )
            else:
                ph_phase.markdown(
                    '<div class="phase-badge phase-2">'
                    'Phase 2: Optimal weights locked.</div>',
                    unsafe_allow_html=True,
                )

            # Append loss and redraw chart.
            loss_history.append({"Epoch": epoch, "Loss": loss})
            df = pd.DataFrame(loss_history)
            ph_chart.line_chart(df, x="Epoch", y="Loss", use_container_width=True)

        st.success(
            f"[OK] Training sequence completed after {epoch} epochs. "
            f"Final loss: {loss:.6f}. Model weights saved."
        )


# ----------------------------------------------------------------------------
# Tab 2: Hardware Inference Engine (Game-Specific FPS Prediction)
# ----------------------------------------------------------------------------

with tab_infer:
    st.markdown("#### Hardware Inference Engine")
    st.markdown(
        '<div class="sub-header">'
        'Predict real-world FPS for any game in the 7,800+ title database '
        'using your hardware specifications and the trained C++ neural engine.'
        '</div>',
        unsafe_allow_html=True,
    )

    # -- Input Section -------------------------------------------------------
    col_hw, col_game = st.columns([2, 3], gap="large")

    with col_hw:
        st.markdown("##### Your Hardware")
        cpu_ghz = st.slider(
            "CPU Clock Speed (GHz)", min_value=0.5, max_value=8.0,
            value=3.6, step=0.1, key="hw_cpu",
        )
        ram_gb = st.slider(
            "System RAM (GB)", min_value=1, max_value=256,
            value=16, step=1, key="hw_ram",
        )
        vram_gb = st.slider(
            "GPU VRAM (GB)", min_value=1, max_value=48,
            value=8, step=1, key="hw_vram",
        )

    with col_game:
        st.markdown("##### Game Selection")
        game_query = st.text_input(
            "Search game name (e.g. 'fortnite', 'gta', 'cyberpunk')",
            value="", key="game_query",
        )

        # Dynamic search results
        selected_game = None
        if game_query.strip():
            matches = search_games(game_query, _predict_game_db, max_results=10)
            if not matches:
                st.warning("[INFO] No games found matching that query. Try a different name.")
            elif len(matches) == 1:
                selected_game = matches[0]
                st.markdown(
                    f'<div class="cyber-card">'
                    f'<span class="cl">Selected</span>  '
                    f'<span class="cv-accent">{selected_game["display_name"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                display_names = [m["display_name"] for m in matches]
                chosen_name = st.selectbox(
                    f"Found {len(matches)} matches -- select one:",
                    options=display_names,
                    key="game_select",
                )
                for m in matches:
                    if m["display_name"] == chosen_name:
                        selected_game = m
                        break

    st.markdown("---")
    btn_infer = st.button(
        "Execute Inference", key="btn_infer", use_container_width=False,
    )

    # -- Execution & Results -------------------------------------------------
    if btn_infer:
        if selected_game is None:
            st.error("[ERROR] No game selected. Enter a game name and select from results.")
        else:
            with st.spinner("Running forward pass through C++ tensor engine..."):
                report = get_prediction_report(
                    cpu_ghz=cpu_ghz,
                    ram_gb=float(ram_gb),
                    vram_gb=float(vram_gb),
                    game_name=selected_game["display_name"],
                )

            game = report["game"]
            fps  = report["predicted_fps"]
            err  = report["error_margin"]

            # -- Hero FPS Display --------------------------------------------
            st.markdown("#### Prediction Results")

            st.markdown(
                f'<div class="hero-fps">'
                f'<div class="fps-number">{fps:.1f}</div>'
                f'<div class="fps-unit">Predicted Frames Per Second</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # -- Secondary metrics -------------------------------------------
            m2, m3, m4 = st.columns(3)
            with m2:
                st.metric(
                    label="Error Margin",
                    value=f"+/- {err:.1f} FPS",
                    delta=f"Range: {report['fps_low']:.0f} - {report['fps_high']:.0f}",
                    delta_color="off",
                )
            with m3:
                st.metric(
                    label="Inference Latency",
                    value=f"{report['inference_ms']:.4f} ms",
                )
            with m4:
                st.metric(
                    label="Training Samples",
                    value=f"{report['num_samples']}",
                )

            st.markdown("---")

            # -- Game Requirements (cyber-cards) -----------------------------
            st.markdown("#### Game Requirements")

            req_col1, req_col2 = st.columns(2)
            with req_col1:
                st.markdown(
                    '<div class="cyber-card">'
                    '<div class="card-title">Minimum Specs</div>'
                    f'<span class="cl">CPU Clock</span>  '
                    f'<span class="cv-accent">{game["min_cpu"]:.2f} GHz</span><br>'
                    f'<span class="cl">System RAM</span>  '
                    f'<span class="cv-accent">{game["min_ram"]:.0f} GB</span><br>'
                    f'<span class="cl">GPU VRAM</span>  '
                    f'<span class="cv-accent">{game["min_vram"]:.1f} GB</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with req_col2:
                st.markdown(
                    '<div class="cyber-card">'
                    '<div class="card-title">Recommended Specs</div>'
                    f'<span class="cl">CPU Clock</span>  '
                    f'<span class="cv-accent">{game["rec_cpu"]:.2f} GHz</span><br>'
                    f'<span class="cl">System RAM</span>  '
                    f'<span class="cv-accent">{game["rec_ram"]:.0f} GB</span><br>'
                    f'<span class="cl">GPU VRAM</span>  '
                    f'<span class="cv-accent">{game["rec_vram"]:.1f} GB</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.caption(
                f"Game: {game['display_name']}  |  "
                f"Model trained on {report['num_samples']} samples"
            )

            st.markdown("---")

            # -- Playability Verdict (styled) --------------------------------
            st.markdown("#### Playability Verdict")

            verdict = report["verdict_label"]
            detail  = report["verdict_detail"]

            if verdict == "COMPETITIVE / ESPORTS READY":
                vclass = "verdict-esports"
            elif verdict == "SMOOTH":
                vclass = "verdict-smooth"
            elif verdict == "PLAYABLE":
                vclass = "verdict-playable"
            else:
                vclass = "verdict-unplayable"

            st.markdown(
                f'<div class="verdict-box {vclass}">'
                f'[VERDICT: {verdict}] {detail}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # -- Hardware Bottleneck Analysis ---------------------------------
            st.markdown("#### Hardware Bottleneck Analysis")

            if report["bottlenecks"]:
                for warn in report["bottlenecks"]:
                    st.markdown(
                        f'<div class="consul-warning">{warn}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="consul-optimal">'
                    '[OK] System fully meets recommended requirements. '
                    'No bottlenecks detected.'
                    '</div>',
                    unsafe_allow_html=True,
                )


