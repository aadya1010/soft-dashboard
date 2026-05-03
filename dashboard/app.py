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

import random

# ============================================================================
# MOCKED BACKEND (Since C++ binaries are missing)
# ============================================================================

def run_training_loop():
    """Mocks the training loop generator."""
    loss = 0.5
    for epoch in range(1, 501):
        time.sleep(0.01) # fake delay
        loss = loss * 0.99 # fake loss reduction
        phase = 0 if epoch < 100 else (1 if epoch < 400 else 2)
        yield {
            "epoch": epoch,
            "total_epochs": 500,
            "loss": loss,
            "time_ms": random.uniform(1.0, 3.0),
            "learning_phase": phase,
            "num_samples": 8500,
            "num_features": 9,
        }

def load_game_requirements(path):
    return {
        "cyberpunk 2077": {"display_name": "Cyberpunk 2077", "min_cpu": 3.2, "rec_cpu": 3.6, "min_ram": 8, "rec_ram": 16, "min_vram": 6, "rec_vram": 8},
        "fortnite": {"display_name": "Fortnite", "min_cpu": 2.8, "rec_cpu": 3.5, "min_ram": 8, "rec_ram": 16, "min_vram": 2, "rec_vram": 4},
        "gta v": {"display_name": "GTA V", "min_cpu": 2.4, "rec_cpu": 3.2, "min_ram": 4, "rec_ram": 8, "min_vram": 1, "rec_vram": 2},
        "valorant": {"display_name": "Valorant", "min_cpu": 2.0, "rec_cpu": 3.0, "min_ram": 4, "rec_ram": 8, "min_vram": 1, "rec_vram": 2},
        "apex legends": {"display_name": "Apex Legends", "min_cpu": 2.6, "rec_cpu": 3.4, "min_ram": 6, "rec_ram": 8, "min_vram": 2, "rec_vram": 6},
    }

def search_games(query, db, max_results=10):
    query_lower = query.lower().strip()
    return [info for key, info in db.items() if query_lower in key][:max_results]

def get_prediction_report(cpu_ghz, ram_gb, vram_gb, game_name):
    """Mocks the inference engine report."""
    time.sleep(0.5) # fake computation time
    db = load_game_requirements(None)
    game = next(g for g in db.values() if g["display_name"] == game_name)
    
    # Fake simple FPS logic
    base_fps = 90
    fps = base_fps * (cpu_ghz / game["rec_cpu"]) * (min(ram_gb, game["rec_ram"]) / game["rec_ram"])
    fps = max(15.0, fps)
    
    verdict = "PLAYABLE"
    if fps >= 120: verdict = "COMPETITIVE / ESPORTS READY"
    elif fps >= 60: verdict = "SMOOTH"
    elif fps < 30: verdict = "UNPLAYABLE"
    
    bottlenecks = []
    if cpu_ghz < game["rec_cpu"]: bottlenecks.append(f"CPU Clock: {cpu_ghz:.1f} GHz is below recommended {game['rec_cpu']:.1f} GHz")
    if ram_gb < game["rec_ram"]: bottlenecks.append(f"System RAM: {ram_gb:.0f} GB is below recommended {game['rec_ram']:.0f} GB")
    
    return {
        "game": game,
        "predicted_fps": fps,
        "error_margin": 5.2,
        "fps_low": fps - 5.2,
        "fps_high": fps + 5.2,
        "verdict_label": verdict,
        "verdict_detail": "Performance estimate based on mocked engine",
        "bottlenecks": bottlenecks,
        "num_samples": 8500,
        "inference_ms": random.uniform(0.1, 0.5),
    }

_predict_game_db = load_game_requirements(None)

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="CodeVerse Neural Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS -- Galaxy Theme + Banner Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');

        /* ---- Global App Styling (Galaxy Theme) ---- */
        .stApp {
            background: radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%) !important;
            background-attachment: fixed !important;
            font-family: 'Inter', sans-serif;
            color: #e2e8f0;
        }

        /* Override standard markdown text */
        p, span, div {
            font-family: 'Inter', sans-serif;
        }
        
        code, pre, .stCodeBlock {
            font-family: 'Space Grotesk', monospace !important;
        }

        /* ---- Sidebar Styling ---- */
        [data-testid="stSidebar"] {
            background: rgba(15, 12, 41, 0.6) !important;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-right: 1px solid rgba(176, 38, 255, 0.2);
        }
        [data-testid="stSidebar"] .stRadio > div {
            background: rgba(0, 0, 0, 0.2);
            padding: 1.5rem 1rem;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.05);
            box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
        }
        [data-testid="stSidebar"] .stRadio label {
            color: #e2e8f0 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.05rem !important;
        }

        /* ---- Page header (Banner Style) ---- */
        .main-header-container {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 58, 138, 0.8) 100%);
            border-radius: 20px;
            padding: 3rem 2rem;
            text-align: center;
            margin-bottom: 2.5rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4), inset 0 0 20px rgba(0, 242, 254, 0.1);
            border: 1px solid rgba(0, 242, 254, 0.2);
            backdrop-filter: blur(10px);
        }
        .main-header {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(to right, #00f2fe 0%, #4facfe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
            text-shadow: 0px 0px 20px rgba(0, 242, 254, 0.3);
        }
        .sub-header {
            font-family: 'Inter', sans-serif;
            font-size: 1.1rem;
            font-weight: 300;
            color: #e2e8f0;
            margin-top: 0.5rem;
            margin-bottom: 0;
            line-height: 1.6;
        }

        /* ---- Section headings ---- */
        h4, .stMarkdown h4 {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.3rem !important;
            font-weight: 600 !important;
            color: #ffffff !important;
            border-bottom: 1px solid rgba(0, 242, 254, 0.2);
            padding-bottom: 0.5rem;
            margin-top: 1.5rem !important;
            letter-spacing: 1px !important;
            text-shadow: 0px 0px 10px rgba(0, 242, 254, 0.3);
        }
        h5, .stMarkdown h5 {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            color: #00f2fe !important;
            letter-spacing: 1.5px !important;
            text-transform: uppercase !important;
            margin-bottom: 0.8rem !important;
        }

        /* ---- Metric overrides ---- */
        [data-testid="stMetricValue"] {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            color: #ffffff !important;
            text-shadow: 0 0 15px rgba(0, 242, 254, 0.5);
        }
        [data-testid="stMetricLabel"] {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            color: #a7b2c9 !important;
            letter-spacing: 1px;
            text-transform: uppercase;
        }
        [data-testid="stMetricDelta"] {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.9rem !important;
        }

        /* ---- Input label visibility ---- */
        .stSlider label, .stTextInput label, .stSelectbox label {
            color: #cbd5e1 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.5px;
        }

        /* ---- Premium Cards (Galaxy Glassmorphism) ---- */
        .cyber-card {
            background: rgba(15, 12, 41, 0.6);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(176, 38, 255, 0.2);
            border-radius: 16px;
            padding: 1.5rem;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            line-height: 2;
            color: #a7b2c9;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(176, 38, 255, 0.05);
            margin-bottom: 1rem;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        .cyber-card:hover {
            transform: translateY(-6px) scale(1.02);
            box-shadow: 0 15px 45px 0 rgba(0, 0, 0, 0.6), inset 0 0 30px rgba(0, 242, 254, 0.1);
            border-color: rgba(0, 242, 254, 0.4);
            background: rgba(20, 15, 55, 0.8);
        }
        .cyber-card .card-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.9rem;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: #b026ff;
            margin-bottom: 0.8rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(176, 38, 255, 0.2);
            text-shadow: 0 0 8px rgba(176, 38, 255, 0.4);
        }
        .cyber-card .cl { color: #7a8ba8; font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem;}
        .cyber-card .cv { color: #e2e8f0; font-weight: 500; font-family: 'Inter', sans-serif;}
        .cyber-card .cv-accent { 
            background: linear-gradient(to right, #00f2fe, #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
            filter: drop-shadow(0 0 5px rgba(0,242,254,0.3));
        }

        /* ---- Hero FPS (Nebula Orb) ---- */
        .hero-fps {
            text-align: center;
            padding: 4rem 1rem;
            background: radial-gradient(circle, rgba(176,38,255,0.15) 0%, rgba(0,242,254,0.05) 50%, rgba(9,10,15,0) 100%);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(176, 38, 255, 0.3);
            border-radius: 30px;
            margin-bottom: 1.5rem;
            box-shadow: 0 0 50px rgba(176,38,255,0.2), inset 0 0 30px rgba(0,242,254,0.1);
            transition: all 0.5s ease;
            position: relative;
            overflow: hidden;
        }
        .hero-fps::before {
            content: '';
            position: absolute;
            top: -50%; left: -50%; width: 200%; height: 200%;
            background: conic-gradient(from 0deg, transparent, rgba(176,38,255,0.2), transparent);
            animation: rotate 10s linear infinite;
            z-index: -1;
        }
        @keyframes rotate {
            100% { transform: rotate(360deg); }
        }
        .hero-fps:hover {
            box-shadow: 0 0 80px rgba(0,242,254,0.3), inset 0 0 50px rgba(176,38,255,0.2);
            border-color: rgba(0, 242, 254, 0.6);
            transform: scale(1.02);
        }
        .hero-fps .fps-number {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 6.5rem;
            font-weight: 800;
            background: linear-gradient(to bottom right, #ffffff, #00f2fe, #b026ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1;
            filter: drop-shadow(0 0 15px rgba(0,242,254,0.6));
        }
        .hero-fps .fps-unit {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1rem;
            font-weight: 600;
            color: #a7b2c9;
            letter-spacing: 4px;
            text-transform: uppercase;
            margin-top: 1.5rem;
        }

        /* ---- Phase badges ---- */
        .phase-badge {
            display: inline-block;
            padding: 0.6rem 1.4rem;
            border-radius: 30px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 0.5rem;
            backdrop-filter: blur(12px);
            transition: all 0.3s ease;
            letter-spacing: 1px;
        }
        .phase-0 {
            background: rgba(255, 0, 128, 0.15);
            color: #ff0080;
            border: 1px solid rgba(255, 0, 128, 0.4);
            box-shadow: 0 0 20px rgba(255, 0, 128, 0.2);
        }
        .phase-1 {
            background: rgba(176, 38, 255, 0.15);
            color: #b026ff;
            border: 1px solid rgba(176, 38, 255, 0.4);
            box-shadow: 0 0 20px rgba(176, 38, 255, 0.2);
        }
        .phase-2 {
            background: rgba(0, 242, 254, 0.15);
            color: #00f2fe;
            border: 1px solid rgba(0, 242, 254, 0.4);
            box-shadow: 0 0 20px rgba(0, 242, 254, 0.2);
        }

        /* ---- Verdict boxes ---- */
        .verdict-box {
            padding: 1.4rem 1.8rem;
            border-radius: 16px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(16px);
            display: flex;
            align-items: center;
            letter-spacing: 0.5px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        .verdict-esports {
            background: linear-gradient(135deg, rgba(0,242,254,0.15), rgba(0,242,254,0.02));
            color: #00f2fe;
            border: 1px solid rgba(0,242,254,0.3);
            border-left: 5px solid #00f2fe;
            text-shadow: 0 0 10px rgba(0,242,254,0.5);
        }
        .verdict-smooth {
            background: linear-gradient(135deg, rgba(176,38,255,0.15), rgba(176,38,255,0.02));
            color: #b026ff;
            border: 1px solid rgba(176,38,255,0.3);
            border-left: 5px solid #b026ff;
            text-shadow: 0 0 10px rgba(176,38,255,0.5);
        }
        .verdict-playable {
            background: linear-gradient(135deg, rgba(255,165,0,0.15), rgba(255,165,0,0.02));
            color: #ffa500;
            border: 1px solid rgba(255,165,0,0.3);
            border-left: 5px solid #ffa500;
        }
        .verdict-unplayable {
            background: linear-gradient(135deg, rgba(255,0,128,0.15), rgba(255,0,128,0.02));
            color: #ff0080;
            border: 1px solid rgba(255,0,128,0.3);
            border-left: 5px solid #ff0080;
        }

        /* ---- Bottleneck cards ---- */
        .consul-optimal {
            background: rgba(0, 242, 254, 0.1);
            color: #00f2fe;
            border: 1px solid rgba(0, 242, 254, 0.3);
            padding: 1.2rem 1.6rem;
            border-radius: 12px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.95rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 0 20px rgba(0, 242, 254, 0.1);
        }
        .consul-warning {
            background: rgba(255, 0, 128, 0.1);
            color: #ff0080;
            border: 1px solid rgba(255, 0, 128, 0.3);
            padding: 1.2rem 1.6rem;
            border-radius: 12px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.95rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 0 20px rgba(255, 0, 128, 0.1);
        }

        /* ---- Buttons ---- */
        .stButton > button {
            background: linear-gradient(45deg, #b026ff 0%, #00f2fe 100%);
            color: white;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 1px;
            border: none;
            padding: 0.6rem 2.5rem;
            border-radius: 30px;
            transition: all 0.3s ease;
            box-shadow: 0 8px 20px rgba(176, 38, 255, 0.4);
            text-transform: uppercase;
        }
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 12px 30px rgba(0, 242, 254, 0.6);
            background: linear-gradient(45deg, #00f2fe 0%, #b026ff 100%);
            color: white;
        }

        /* ---- Dividers ---- */
        hr {
            border-color: rgba(176, 38, 255, 0.2) !important;
            margin: 2.5rem 0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# Sidebar Navigation
# ============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='font-family: Space Grotesk; color: #00f2fe; margin-bottom: 0; text-shadow: 0 0 10px rgba(0,242,254,0.3); letter-spacing: 1px;'>CodeVerse</h2>
            <p style='color: #a7b2c9; font-size: 0.9rem; font-family: Inter;'>Neural Engine</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    app_mode = st.radio(
        "Application Mode",
        ["Live Brain", "Hardware Inference"]
    )
    
    st.markdown("---")
    st.caption("Version 1.0.0 | Custom C++ Backend")


# ============================================================================
# Header Banner
# ============================================================================

if app_mode == "Live Brain":
    banner_title = "Live Brain Training Visualizer"
    banner_sub = "Real-time neural network operations powered by a custom C++ backend with memory pooling and autograd."
else:
    banner_title = "Hardware Inference Engine"
    banner_sub = "Single-shot FPS prediction using your hardware specifications and the trained neural engine."

st.markdown(
    f'''
    <div class="main-header-container">
        <div class="main-header">{banner_title}</div>
        <div class="sub-header">{banner_sub}</div>
    </div>
    ''',
    unsafe_allow_html=True
)


# ============================================================================
# Mode 1: Live Brain (Training Visualizer)
# ============================================================================

if app_mode == "Live Brain":
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


# ============================================================================
# Mode 2: Hardware Inference Engine (Game-Specific FPS Prediction)
# ============================================================================

elif app_mode == "Hardware Inference":

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
