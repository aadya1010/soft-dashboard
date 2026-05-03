"""
dashboard/app.py
----------------
Main Streamlit interface for the soft-cuda Custom C++ Deep Learning Engine.

Provides two operational modes:
  1. Live Brain   -- Real-time training loop visualizer with epoch-level metrics.
  2. Hardware Inference -- Target hardware FPS prediction using trained weights.
"""

import streamlit as st
import pandas as pd
import time
import random
import math

# ============================================================================
# MOCK BACKEND GENERATORS (Replaces ctypes calls for UI Testing)
# ============================================================================

def run_training_loop(epochs=500, base_loss=1.2):
    """Generates mock training epoch data to simulate the C++ engine."""
    current_loss = base_loss
    for epoch in range(1, epochs + 1):
        # Simulate C++ processing time (faster over time as weights settle)
        time_ms = random.uniform(2.5, 5.0) - (epoch / epochs) * 1.5
        
        # Simulate loss curve with noise and asymptotic convergence
        noise = random.uniform(-0.02, 0.02) * (1 - epoch/epochs)
        current_loss = current_loss * 0.98 + noise
        if current_loss < 0.01:
            current_loss = 0.01 + random.uniform(0, 0.005)
            
        # Determine learning phase
        if epoch < epochs * 0.3:
            phase = 0
        elif epoch < epochs * 0.8:
            phase = 1
        else:
            phase = 2
            
        yield {
            "epoch": epoch,
            "loss": current_loss,
            "time_ms": max(0.5, time_ms),
            "learning_phase": phase,
            "total_epochs": epochs,
            "num_samples": 8500
        }
        time.sleep(0.01) # UI refresh rate

def load_game_requirements(db_path):
    """Mock game requirements database."""
    return [
        {"id": "cs2", "display_name": "Counter-Strike 2", "min_cpu": 2.8, "min_ram": 8, "min_vram": 2, "rec_cpu": 3.6, "rec_ram": 16, "rec_vram": 8},
        {"id": "cyberpunk", "display_name": "Cyberpunk 2077", "min_cpu": 3.2, "min_ram": 12, "min_vram": 6, "rec_cpu": 4.0, "rec_ram": 16, "rec_vram": 12},
        {"id": "valorant", "display_name": "Valorant", "min_cpu": 2.5, "min_ram": 4, "min_vram": 1, "rec_cpu": 3.2, "rec_ram": 8, "rec_vram": 4},
        {"id": "gta5", "display_name": "Grand Theft Auto V", "min_cpu": 2.4, "min_ram": 4, "min_vram": 1, "rec_cpu": 3.2, "rec_ram": 8, "rec_vram": 4},
        {"id": "fortnite", "display_name": "Fortnite", "min_cpu": 2.8, "min_ram": 8, "min_vram": 2, "rec_cpu": 3.5, "rec_ram": 16, "rec_vram": 8},
    ]

def search_games(query, db, max_results=5):
    """Filter mock database by query string."""
    q = query.lower()
    results = [g for g in db if q in g["display_name"].lower()]
    return results[:max_results]

def get_prediction_report(cpu_ghz, ram_gb, vram_gb, game_name):
    """Mocks the C++ inference forward pass."""
    db = load_game_requirements(None)
    game = next((g for g in db if g["display_name"] == game_name), db[0])
    
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
    page_title="soft-cuda Neural Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS -- Catppuccin Starry Galaxy Theme (High Contrast)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        /* ---- Global App Styling (Starry Background) ---- */
        .stApp {
            background-color: #0d0f18 !important;
            background-image: 
                radial-gradient(1px 1px at 25px 5px, white, rgba(255,255,255,0)),
                radial-gradient(1px 1px at 50px 25px, white, rgba(255,255,255,0)),
                radial-gradient(1px 1px at 125px 20px, white, rgba(255,255,255,0)),
                radial-gradient(1.5px 1.5px at 50px 75px, white, rgba(255,255,255,0)),
                radial-gradient(2px 2px at 15px 125px, white, rgba(255,255,255,0)),
                radial-gradient(2.5px 2.5px at 110px 140px, white, rgba(255,255,255,0)),
                radial-gradient(1.5px 1.5px at 180px 10px, white, rgba(255,255,255,0)),
                radial-gradient(1px 1px at 220px 60px, white, rgba(255,255,255,0)),
                radial-gradient(2px 2px at 280px 120px, white, rgba(255,255,255,0)),
                radial-gradient(1px 1px at 320px 40px, white, rgba(255,255,255,0)),
                radial-gradient(1.5px 1.5px at 380px 90px, white, rgba(255,255,255,0)),
                radial-gradient(2px 2px at 420px 10px, white, rgba(255,255,255,0)),
                radial-gradient(1px 1px at 480px 140px, white, rgba(255,255,255,0)),
                radial-gradient(1.5px 1.5px at 550px 50px, white, rgba(255,255,255,0)),
                radial-gradient(1px 1px at 600px 100px, white, rgba(255,255,255,0)),
                radial-gradient(ellipse at bottom, rgba(137,180,250,0.1) 0%, rgba(17,17,27,0) 100%) !important;
            background-size: 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 100% 100% !important;
            background-attachment: fixed !important;
            font-family: 'Inter', sans-serif;
            color: #ffffff;
        }

        /* Override standard markdown text */
        p, span, div {
            font-family: 'Inter', sans-serif;
            color: #ffffff; /* Max contrast */
        }
        
        code, pre, .stCodeBlock {
            font-family: 'JetBrains Mono', monospace !important;
            background: rgba(17, 17, 27, 0.9) !important;
            border: 1px solid rgba(137, 180, 250, 0.4);
            border-radius: 0px !important;
            color: #cdd6f4 !important;
        }

        /* ---- Sidebar Styling ---- */
        [data-testid="stSidebar"] {
            background: rgba(17, 17, 27, 0.8) !important; /* Darker for contrast */
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-right: 1px solid rgba(203, 166, 247, 0.3);
        }
        [data-testid="stSidebar"] .stRadio > div {
            background: rgba(0, 0, 0, 0.4);
            padding: 1.5rem 1rem;
            border-radius: 0px;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: inset 0 0 15px rgba(0,0,0,0.8);
        }
        [data-testid="stSidebar"] .stRadio label {
            color: #ffffff !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.05rem !important;
        }

        /* ---- Page header (Banner Style) ---- */
        .main-header-container {
            background: linear-gradient(135deg, rgba(30, 30, 46, 0.95) 0%, rgba(17, 17, 27, 0.9) 100%);
            border-radius: 0px;
            padding: 3rem 2rem;
            text-align: center;
            margin-bottom: 2.5rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.6), inset 0 0 20px rgba(137, 180, 250, 0.2);
            border: 1px solid rgba(137, 180, 250, 0.4);
            backdrop-filter: blur(16px);
        }
        .main-header {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(to right, #cba6f7 0%, #89b4fa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
            text-shadow: 0px 0px 25px rgba(203, 166, 247, 0.5);
        }
        .sub-header {
            font-family: 'Inter', sans-serif;
            font-size: 1.1rem;
            font-weight: 400;
            color: #cdd6f4;
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
            border-bottom: 1px solid rgba(137, 180, 250, 0.4);
            padding-bottom: 0.5rem;
            margin-top: 1.5rem !important;
            letter-spacing: 1px !important;
            text-shadow: 0px 0px 10px rgba(137, 180, 250, 0.5);
        }
        h5, .stMarkdown h5 {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            color: #89b4fa !important;
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
            text-shadow: 0 0 15px rgba(137, 180, 250, 0.8);
        }
        [data-testid="stMetricLabel"] {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            color: #bac2de !important;
            letter-spacing: 1px;
            text-transform: uppercase;
        }
        [data-testid="stMetricDelta"] {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.9rem !important;
        }

        /* ---- Input label visibility ---- */
        .stSlider label, .stTextInput label, .stSelectbox label {
            color: #ffffff !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.5px;
        }

        /* ---- Premium Cards (Galaxy Glassmorphism + High Contrast) ---- */
        .cyber-card {
            background: rgba(17, 17, 27, 0.85); /* Darker base for contrast */
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(203, 166, 247, 0.3);
            border-radius: 0px;
            padding: 1.5rem;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            line-height: 2;
            color: #ffffff;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6), inset 0 0 20px rgba(203, 166, 247, 0.1);
            margin-bottom: 1rem;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        .cyber-card:hover {
            transform: translateY(-6px) scale(1.02);
            box-shadow: 0 15px 45px 0 rgba(0, 0, 0, 0.8), inset 0 0 30px rgba(137, 180, 250, 0.2);
            border-color: rgba(137, 180, 250, 0.6);
            background: rgba(30, 30, 46, 0.95);
        }
        .cyber-card .card-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.9rem;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: #cba6f7;
            margin-bottom: 0.8rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(203, 166, 247, 0.3);
            text-shadow: 0 0 10px rgba(203, 166, 247, 0.5);
        }
        .cyber-card .cl { color: #bac2de; font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem;}
        .cyber-card .cv { color: #ffffff; font-weight: 500; font-family: 'Inter', sans-serif;}
        .cyber-card .cv-accent { 
            background: linear-gradient(to right, #89dceb, #a6e3a1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
            filter: drop-shadow(0 0 5px rgba(166,227,161,0.5));
        }

        /* ---- Hero FPS (Nebula Orb + Sharp Frame) ---- */
        .hero-fps {
            text-align: center;
            padding: 4rem 1rem;
            background: radial-gradient(circle, rgba(203,166,247,0.2) 0%, rgba(137,180,250,0.05) 50%, rgba(17,17,27,0.8) 100%);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(203, 166, 247, 0.4);
            border-radius: 0px;
            margin-bottom: 1.5rem;
            box-shadow: 0 0 50px rgba(203,166,247,0.3), inset 0 0 30px rgba(137,180,250,0.15);
            transition: all 0.5s ease;
            position: relative;
            overflow: hidden;
        }
        .hero-fps::before {
            content: '';
            position: absolute;
            top: -50%; left: -50%; width: 200%; height: 200%;
            background: conic-gradient(from 0deg, transparent, rgba(203,166,247,0.25), transparent);
            animation: rotate 10s linear infinite;
            z-index: -1;
        }
        @keyframes rotate {
            100% { transform: rotate(360deg); }
        }
        .hero-fps:hover {
            box-shadow: 0 0 80px rgba(137,180,250,0.4), inset 0 0 50px rgba(203,166,247,0.3);
            border-color: rgba(137, 180, 250, 0.7);
            transform: scale(1.02);
        }
        .hero-fps .fps-number {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 6.5rem;
            font-weight: 800;
            background: linear-gradient(to bottom right, #ffffff, #89dceb, #a6e3a1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1;
            filter: drop-shadow(0 0 15px rgba(137,180,250,0.6));
        }
        .hero-fps .fps-unit {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1rem;
            font-weight: 600;
            color: #ffffff;
            letter-spacing: 4px;
            text-transform: uppercase;
            margin-top: 1.5rem;
        }

        /* ---- Phase badges ---- */
        .phase-badge {
            display: inline-block;
            padding: 0.6rem 1.4rem;
            border-radius: 0px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 0.5rem;
            backdrop-filter: blur(12px);
            transition: all 0.3s ease;
            letter-spacing: 1px;
        }
        .phase-0 {
            background: rgba(249, 226, 175, 0.2);
            color: #f9e2af;
            border: 1px solid rgba(249, 226, 175, 0.5);
            box-shadow: 0 0 20px rgba(249, 226, 175, 0.3);
        }
        .phase-1 {
            background: rgba(137, 180, 250, 0.2);
            color: #89b4fa;
            border: 1px solid rgba(137, 180, 250, 0.5);
            box-shadow: 0 0 20px rgba(137, 180, 250, 0.3);
        }
        .phase-2 {
            background: rgba(166, 227, 161, 0.2);
            color: #a6e3a1;
            border: 1px solid rgba(166, 227, 161, 0.5);
            box-shadow: 0 0 20px rgba(166, 227, 161, 0.3);
        }

        /* ---- Verdict boxes ---- */
        .verdict-box {
            padding: 1.4rem 1.8rem;
            border-radius: 0px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(20px);
            display: flex;
            align-items: center;
            letter-spacing: 0.5px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        }
        .verdict-esports {
            background: linear-gradient(135deg, rgba(166,227,161,0.2), rgba(17,17,27,0.8));
            color: #a6e3a1;
            border: 1px solid rgba(166,227,161,0.4);
            border-left: 5px solid #a6e3a1;
            text-shadow: 0 0 10px rgba(166,227,161,0.4);
        }
        .verdict-smooth {
            background: linear-gradient(135deg, rgba(137,180,250,0.2), rgba(17,17,27,0.8));
            color: #89b4fa;
            border: 1px solid rgba(137,180,250,0.4);
            border-left: 5px solid #89b4fa;
            text-shadow: 0 0 10px rgba(137,180,250,0.5);
        }
        .verdict-playable {
            background: linear-gradient(135deg, rgba(249,226,175,0.2), rgba(17,17,27,0.8));
            color: #f9e2af;
            border: 1px solid rgba(249,226,175,0.4);
            border-left: 5px solid #f9e2af;
        }
        .verdict-unplayable {
            background: linear-gradient(135deg, rgba(243,139,168,0.2), rgba(17,17,27,0.8));
            color: #f38ba8;
            border: 1px solid rgba(243,139,168,0.4);
            border-left: 5px solid #f38ba8;
        }

        /* ---- Bottleneck cards ---- */
        .consul-optimal {
            background: rgba(166, 227, 161, 0.15);
            color: #a6e3a1;
            border: 1px solid rgba(166, 227, 161, 0.4);
            padding: 1.2rem 1.6rem;
            border-radius: 0px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.95rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 0 20px rgba(166, 227, 161, 0.2);
            backdrop-filter: blur(10px);
        }
        .consul-warning {
            background: rgba(243, 139, 168, 0.15);
            color: #f38ba8;
            border: 1px solid rgba(243, 139, 168, 0.4);
            padding: 1.2rem 1.6rem;
            border-radius: 0px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.95rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 0 20px rgba(243, 139, 168, 0.2);
            backdrop-filter: blur(10px);
        }

        /* ---- Buttons ---- */
        .stButton > button {
            background: linear-gradient(45deg, #9a75c4 0%, #5a85c9 100%);
            color: #ffffff;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: 1.5px;
            border: none;
            padding: 0.6rem 2.5rem;
            border-radius: 0px;
            transition: all 0.3s ease;
            box-shadow: 0 8px 20px rgba(154, 117, 196, 0.4);
            text-transform: uppercase;
        }
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 12px 30px rgba(90, 133, 201, 0.5);
            background: linear-gradient(45deg, #5a85c9 0%, #9a75c4 100%);
            color: #ffffff;
        }

        /* ---- Dividers ---- */
        hr {
            border-color: rgba(203, 166, 247, 0.3) !important;
            margin: 2.5rem 0 !important;
        }
        
        /* Dropdowns/Inputs */
        .stTextInput > div > div > input, .stSelectbox > div > div {
            border-radius: 0px !important;
            background: rgba(17, 17, 27, 0.8) !important;
            color: #ffffff !important;
            border: 1px solid rgba(137, 180, 250, 0.4) !important;
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
        <div style='text-align: center; padding: 1.5rem 0 1rem 0;'>
            <h2 style='font-family: Space Grotesk; color: #ffffff; font-size: 2.6rem; font-weight: 800; margin-bottom: 0; letter-spacing: 2px; text-shadow: 0 0 20px rgba(137, 180, 250, 0.8), 0 0 10px rgba(203, 166, 247, 0.6);'>SOFT-CUDA</h2>
            <p style='color: #bac2de; font-size: 1rem; font-family: Inter; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; margin-top: 0.2rem;'>Neural Engine</p>
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
        ph_samples = st.empty()

        st.markdown("---")
        st.markdown("##### Learning Status")
        ph_phase = st.empty()

    with col_graph:
        st.markdown("##### Loss Curve (Real-Time)")
        ph_chart = st.empty()

    st.markdown("---")
    st.markdown("##### System Diagnostics")
    
    sys_col1, sys_col2, sys_col3 = st.columns(3, gap="large")

    with sys_col1:
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

    with sys_col2:
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

    with sys_col3:
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
