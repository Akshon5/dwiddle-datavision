import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import numpy as np
import streamlit as st
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Please install streamlit-drawable-canvas: pip install streamlit-drawable-canvas")
    st.stop()

from src.challenges import get_challenges, get_challenge, get_riddles, RIDDLES_BY_DIFFICULTY
from src.game_engine import GameEngine, POINTS_PER_ROOM, FIRST_TRY_BONUS, BONUS_PER_LIFE
from src.inference import QuickDrawPredictor


# Page Setup
st.set_page_config(
    page_title="Dwiddle — AI Drawing Riddle Game",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Polished Minimal Dark Theme CSS
st.markdown("""
<style>
    /* Base Background and Typography */
    .stApp {
        background-color: #0b0f17;
        color: #f3f4f6;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Eliminate top white strip and normalize header */
    header[data-testid="stHeader"] {
        background-color: #0b0f17 !important;
        border-bottom: 1px solid #1f2937 !important;
    }
    
    [data-testid="stDecoration"] {
        display: none !important;
    }
    
    /* Layout Container & Spacing */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1140px !important;
        margin: 0 auto !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d121c !important;
        border-right: 1px solid #1f2937 !important;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #f9fafb !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    p, span, label {
        color: #d1d5db;
    }
    
    /* Riddle Presentation Card */
    .riddle-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    
    .riddle-header {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #60a5fa;
        margin-bottom: 8px;
    }
    
    .riddle-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: #f9fafb;
        margin-bottom: 14px;
    }
    
    .riddle-text {
        font-size: 1.15rem;
        line-height: 1.65;
        color: #f3f4f6;
        font-style: italic;
        background: #161f30;
        border: 1px solid #233147;
        border-radius: 6px;
        padding: 16px 20px;
        margin: 12px 0 16px 0;
    }
    
    .mission-prompt {
        font-size: 0.95rem;
        font-weight: 500;
        color: #93c5fd;
        background-color: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 6px;
        padding: 10px 14px;
    }
    
    /* General Surface Card */
    .surface-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 16px;
    }
    
    /* Result & Feedback Cards */
    .feedback-success {
        background: #062e20;
        border: 1px solid #059669;
        border-radius: 8px;
        padding: 18px 20px;
        margin: 14px 0;
        color: #ecfdf5;
    }
    
    .feedback-success h3 {
        color: #34d399 !important;
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 1.25rem;
    }
    
    .feedback-error {
        background: #2b0f12;
        border: 1px solid #dc2626;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #fee2e2;
    }
    
    .feedback-info {
        background: #162235;
        border: 1px solid #2563eb;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #dbeafe;
    }
    
    /* End Screens */
    .end-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        margin-bottom: 24px;
    }
    
    .end-card-victory {
        border-top: 4px solid #3b82f6;
    }
    
    .end-card-gameover {
        border-top: 4px solid #ef4444;
    }
    
    /* Button Overrides */
    div.stButton > button {
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        padding: 0.45rem 1rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    
    div.stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        border: 1px solid #3b82f6 !important;
        color: #ffffff !important;
    }
    
    div.stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        border-color: #60a5fa !important;
    }
    
    div.stButton > button[kind="secondary"] {
        background-color: #161f30 !important;
        border: 1px solid #2a384f !important;
        color: #e2e8f0 !important;
    }
    
    div.stButton > button[kind="secondary"]:hover {
        background-color: #1e2b42 !important;
        border-color: #3b82f6 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# Session State Initialization
if 'model_version' not in st.session_state:
    st.session_state.model_version = 20

if 'difficulty' not in st.session_state:
    st.session_state.difficulty = 'easy'

if 'show_exit_confirm' not in st.session_state:
    st.session_state.show_exit_confirm = False

if 'canvas_nonce' not in st.session_state:
    st.session_state.canvas_nonce = 0

if 'last_meta' not in st.session_state:
    st.session_state.last_meta = None

if 'last_top3' not in st.session_state:
    st.session_state.last_top3 = []


# Initialize Predictor (cached on CPU)
@st.cache_resource
def get_predictor(version: int):
    return QuickDrawPredictor(model_version=version)


try:
    predictor = get_predictor(st.session_state.model_version)
except Exception as e:
    st.error(f"Error loading trained CNN model (v{st.session_state.model_version}): {e}")
    st.stop()


# Initialize Game Engine with matching challenges
active_engine_key = (
    st.session_state.model_version,
    st.session_state.difficulty if st.session_state.model_version == 20 else 'facility_8'
)
if 'engine' not in st.session_state or st.session_state.get('engine_key') != active_engine_key:
    if st.session_state.model_version == 20:
        active_challenges = get_riddles(st.session_state.difficulty)
        diff_label = st.session_state.difficulty
    else:
        active_challenges = get_challenges(8)
        diff_label = None
    st.session_state.engine = GameEngine(challenges=active_challenges, difficulty=diff_label)
    st.session_state.engine_key = active_engine_key

engine: GameEngine = st.session_state.engine

# Clean Display Names for Supported Categories
CLASS_DISPLAY = {
    'airplane': 'Airplane',
    'apple': 'Apple',
    'axe': 'Axe',
    'bicycle': 'Bicycle',
    'book': 'Book',
    'candle': 'Candle',
    'chair': 'Chair',
    'clock': 'Clock',
    'cup': 'Cup',
    'door': 'Door',
    'fish': 'Fish',
    'flower': 'Flower',
    'house': 'House',
    'key': 'Key',
    'parachute': 'Parachute',
    'scissors': 'Scissors',
    'shoe': 'Shoe',
    'star': 'Star',
    'tree': 'Tree',
    'umbrella': 'Umbrella'
}
CLASS_ICONS = CLASS_DISPLAY  # Backward-compatibility alias

# Tactical Drawing Advice
DRAWING_TIPS = {
    'airplane': 'Draw a central fuselage with perpendicular wings and a tail fin.',
    'apple': 'Draw a round body with a small vertical stem and leaf on top.',
    'axe': 'Draw a straight vertical handle with a sharp wedge/triangular blade at one side.',
    'bicycle': 'Draw two circular wheels connected by a triangular frame and handlebars.',
    'book': 'Draw an open book with curved pages or a clear rectangular closed volume.',
    'candle': 'Draw a vertical cylindrical pillar topped with a small teardrop flame.',
    'chair': 'Draw a flat seat with 4 legs and a tall vertical backrest.',
    'clock': 'Draw a circular outline with hour/minute hands pointing out from the center.',
    'cup': 'Draw a cylindrical or tapered mug with a curved C-shaped handle on the side.',
    'door': 'Draw a tall rectangular frame with a distinct round doorknob on one side.',
    'fish': 'Draw a streamlined oval body tapering into a triangular tail fin.',
    'flower': 'Draw a center circular disc surrounded by petals and a stem below.',
    'house': 'Draw a square base with a triangular pitched roof and a central door.',
    'key': 'Draw horizontally: a circular loop/head, straight shaft, and 2 notched teeth.',
    'parachute': 'Draw an arched dome canopy with suspension lines converging downward.',
    'scissors': 'Draw two crossed diagonal blades with two round finger loop handles.',
    'shoe': 'Draw the profile of a shoe: flat sole, curved heel, toe bump, and ankle collar.',
    'star': 'Draw a classic five-pointed star outline.',
    'tree': 'Draw a straight vertical trunk topped with a bushy cloud-shaped canopy.',
    'umbrella': 'Draw a curved dome umbrella top with a center pole and curved J-hook handle.'
}


# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
st.sidebar.markdown("### Dwiddle")
st.sidebar.caption("AI-Powered Drawing Riddle Trial • Edge AI Vision")

st.sidebar.markdown("---")
st.sidebar.markdown("##### Vision Model")
model_choice = st.sidebar.selectbox(
    "Active Mode",
    options=["20-Class Riddle Trial", "8-Class Original Facility"],
    index=0 if st.session_state.model_version == 20 else 1,
    help="Select between the 20-class riddle trial (15 riddles across 3 difficulties) and the original 8-class facility mode."
)
selected_version = 20 if "20" in model_choice else 8
if selected_version != st.session_state.model_version:
    st.session_state.model_version = selected_version
    st.session_state.pop('engine_key', None)
    st.session_state.canvas_nonce += 1
    st.session_state.last_meta = None
    st.session_state.last_top3 = []
    st.session_state.show_exit_confirm = False
    st.rerun()

if st.session_state.model_version == 20:
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### Difficulty Mode")
    diff_labels = {"easy": "Easy (5 Riddles)", "medium": "Medium (5 Riddles)", "hard": "Hard (5 Riddles)"}
    diff_keys = ["easy", "medium", "hard"]
    curr_diff_idx = diff_keys.index(st.session_state.difficulty) if st.session_state.difficulty in diff_keys else 0
    selected_diff_label = st.sidebar.selectbox(
        "Riddle Tier",
        options=[diff_labels[k] for k in diff_keys],
        index=curr_diff_idx,
        help="Select riddle tier: Easy, Medium, or Hard."
    )
    chosen_diff = diff_keys[[diff_labels[k] for k in diff_keys].index(selected_diff_label)]
    if chosen_diff != st.session_state.difficulty:
        st.session_state.difficulty = chosen_diff
        st.session_state.pop('engine_key', None)
        st.session_state.canvas_nonce += 1
        st.session_state.last_meta = None
        st.session_state.last_top3 = []
        st.session_state.show_exit_confirm = False
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("##### Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Cutoff (%)",
    min_value=20,
    max_value=80,
    value=40,
    step=5,
    help="Sketches with top confidence below this threshold are rejected. Softmax confidence reflects model certainty."
) / 100.0

stroke_width = st.sidebar.slider("Brush Width (px)", min_value=4, max_value=22, value=10, step=2)
show_debug = st.sidebar.checkbox("Show 28×28 Neural Input", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("##### Game Actions")
sb_col1, sb_col2 = st.sidebar.columns(2)
with sb_col1:
    if st.button("Restart Trial", use_container_width=True):
        engine.restart_game()
        st.session_state.canvas_nonce += 1
        st.session_state.last_meta = None
        st.session_state.last_top3 = []
        st.session_state.show_exit_confirm = False
        st.rerun()
with sb_col2:
    if engine.game_state != "WELCOME":
        if st.button("Exit to Home", use_container_width=True):
            st.session_state.show_exit_confirm = True
            st.rerun()


# -------------------------------------------------------------
# SCREEN 1: HOME SCREEN (WELCOME)
# -------------------------------------------------------------
if engine.game_state == "WELCOME":
    if st.session_state.model_version == 20:
        st.title("Dwiddle")
        st.caption("An AI-powered drawing riddle trial")
        
        st.markdown("""
        <div class="surface-card">
            <h4 style="margin-top:0; color:#f3f4f6;">The Dwiddle Challenge</h4>
            <p style="margin-bottom:0; line-height: 1.6; color:#9ca3af;">
                Choose your difficulty and solve five clever riddles by drawing your answers. Each sketch is put to the test by our AI vision model. Think carefully, draw clearly, and see if you can crack all five riddles!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Choose Difficulty Tier")
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            btn_kind = "primary" if st.session_state.difficulty == "easy" else "secondary"
            if st.button("Easy (5 Riddles)", use_container_width=True, type=btn_kind):
                st.session_state.difficulty = "easy"
                st.session_state.pop('engine_key', None)
                st.rerun()
            st.caption("Clear poetic imagery and everyday items.")
        with d_col2:
            btn_kind = "primary" if st.session_state.difficulty == "medium" else "secondary"
            if st.button("Medium (5 Riddles)", use_container_width=True, type=btn_kind):
                st.session_state.difficulty = "medium"
                st.session_state.pop('engine_key', None)
                st.rerun()
            st.caption("Metaphorical clues requiring lateral deduction.")
        with d_col3:
            btn_kind = "primary" if st.session_state.difficulty == "hard" else "secondary"
            if st.button("Hard (5 Riddles)", use_container_width=True, type=btn_kind):
                st.session_state.difficulty = "hard"
                st.session_state.pop('engine_key', None)
                st.rerun()
            st.caption("Cryptic paradoxes and abstract contradictions.")
            
        st.markdown("#### Trial Rules")
        st.write("• **3 Lives:** You begin with 3 lives. An incorrect or unrecognized drawing costs 1 life.")
        st.write("• **Blank Canvas Safe:** Submitting an empty canvas never consumes a life.")
        st.write(f"• **5 Riddles:** Complete all 5 riddles in {st.session_state.difficulty.upper()} tier.")
        st.write("• **Scoring:**")
        st.write("   - `+100 pts` base for every riddle solved")
        st.write("   - `+50 pts` first-attempt bonus if solved on your first try")
        st.write("   - `+25 pts` per remaining life awarded upon victory")
                
        st.markdown("---")
        start_label = f"Start {st.session_state.difficulty.capitalize()} Trial (5 Riddles)"
        if st.button(start_label, type="primary", use_container_width=True):
            engine.start_game()
            st.session_state.show_exit_confirm = False
            st.rerun()
            
    else:
        st.title("Dwiddle: Original Facility Escape")
        st.caption("8-Class Subterranean Containment Escape")
        
        st.markdown("""
        <div class="surface-card">
            <h4 style="margin-top:0; color:#f3f4f6;">Emergency Containment Protocol</h4>
            <p style="margin-bottom:0; line-height: 1.6; color:#9ca3af;">
                You awaken trapped in a reinforced subterranean research facility. Alarms are sounding,
                and emergency blast doors have sealed every corridor. Draw the exact object needed
                to overcome each sector's crisis to unlock your escape.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Facility Rules")
        st.write("• **3 Lives:** Each failed or unrecognized drawing costs 1 life.")
        st.write("• **Blank Canvas Safe:** A blank canvas does not consume a life.")
        st.write(f"• **{engine.total_rooms} Sectors:** Escape through {engine.total_rooms} consecutive crisis chambers.")
        st.write("• **Scoring:** `+100 pts` per sector, `+50 pts` first-attempt bonus, `+25 pts` per life retained.")
                
        st.markdown("---")
        if st.button("Start Facility Escape (8 Chambers)", type="primary", use_container_width=True):
            engine.start_game()
            st.session_state.show_exit_confirm = False
            st.rerun()


# -------------------------------------------------------------
# SCREEN 2: GAMEPLAY (PLAYING & ROOM_SUCCESS)
# -------------------------------------------------------------
elif engine.game_state in ("PLAYING", "ROOM_SUCCESS"):
    summary = engine.get_summary()
    current_ch = engine.get_current_challenge()
    
    # Scoped Gameplay CSS (only active during gameplay)
    st.markdown("""
    <style>
        /* Header clearance & compact gameplay viewport spacing */
        .gameplay-hud {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 6px;
            padding: 6px 14px;
            margin-top: 0.25rem;
            margin-bottom: 6px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
        }
        
        .hud-items-row {
            display: flex;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }
        
        .hud-cell {
            display: flex;
            align-items: baseline;
            gap: 6px;
        }
        
        .hud-cell-label {
            font-size: 0.70rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #9ca3af;
        }
        
        .hud-cell-value {
            font-size: 0.90rem;
            font-weight: 700;
            color: #f3f4f6;
        }
        
        .hud-tier-easy {
            color: #34d399;
            background: rgba(16, 185, 129, 0.12);
            padding: 1px 6px;
            border-radius: 4px;
            font-size: 0.78rem;
        }
        
        .hud-tier-medium {
            color: #fbbf24;
            background: rgba(245, 158, 11, 0.12);
            padding: 1px 6px;
            border-radius: 4px;
            font-size: 0.78rem;
        }
        
        .hud-tier-hard {
            color: #f87171;
            background: rgba(239, 68, 68, 0.12);
            padding: 1px 6px;
            border-radius: 4px;
            font-size: 0.78rem;
        }
        
        .hud-sep {
            width: 1px;
            height: 16px;
            background: #283548;
        }
        
        /* Thin Mini Progress Track */
        .hud-progress-bar {
            width: 100%;
            height: 3px;
            background: #1f2937;
            border-radius: 2px;
            margin-bottom: 8px;
            overflow: hidden;
        }
        
        .hud-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #2563eb, #60a5fa);
            border-radius: 2px;
            transition: width 0.3s ease;
        }
        
        /* Compact Riddle Card */
        .gameplay-riddle-card {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-left: 3px solid #3b82f6;
            border-radius: 6px;
            padding: 9px 14px;
            margin-bottom: 8px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15);
        }
        
        .gameplay-riddle-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 4px;
        }
        
        .gameplay-riddle-tag {
            font-size: 0.70rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #60a5fa;
        }
        
        .gameplay-riddle-title {
            font-size: 1.0rem;
            font-weight: 600;
            color: #f9fafb;
            margin: 0;
            line-height: 1.25;
        }
        
        .gameplay-riddle-poem {
            font-size: 0.90rem;
            line-height: 1.40;
            color: #e5e7eb;
            font-style: italic;
            background: #151d2a;
            border: 1px solid #1e293b;
            border-radius: 5px;
            padding: 6px 10px;
            margin: 5px 0 5px 0;
        }
        
        .gameplay-riddle-prompt {
            font-size: 0.82rem;
            font-weight: 500;
            color: #93c5fd;
            background-color: rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 4px;
            padding: 4px 8px;
            margin: 0;
        }
        
        /* Column Titles & Canvas */
        .gameplay-col-header {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #9ca3af;
            margin-bottom: 4px;
        }
        
        .canvas-wrapper {
            border: 1.5px solid #374151;
            border-radius: 6px;
            background: #ffffff;
            width: 460px;
            height: 260px;
            margin-bottom: 6px;
            overflow: hidden;
        }
        
        /* Compact Detection Probabilities */
        .top3-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #141b27;
            border: 1px solid #1f2937;
            border-radius: 5px;
            padding: 3px 8px;
            margin-bottom: 3px;
            font-size: 0.82rem;
        }
        
        .top3-label {
            font-weight: 600;
            color: #f3f4f6;
            min-width: 70px;
        }
        
        .top3-bar-container {
            flex-grow: 1;
            height: 5px;
            background: #1f2937;
            border-radius: 3px;
            overflow: hidden;
            margin: 0 8px;
        }
        
        .top3-bar-active {
            height: 100%;
            background: #3b82f6;
            border-radius: 3px;
        }
        
        .top3-percentage {
            font-size: 0.78rem;
            font-family: monospace;
            color: #93c5fd;
            min-width: 44px;
            text-align: right;
        }
        
        /* Compact In-Game Expanders */
        div[data-testid="stExpander"] {
            border-color: #1f2937 !important;
            background-color: #111827 !important;
            margin-bottom: 4px !important;
            border-radius: 6px !important;
        }
        
        div[data-testid="stExpander"] summary {
            padding: 0.25rem 0.5rem !important;
            font-size: 0.80rem !important;
            color: #cbd5e1 !important;
        }
        
        div[data-testid="stExpander"] div[role="region"] {
            padding: 0.4rem 0.6rem !important;
            font-size: 0.82rem !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Exit to Home Confirmation Dialogue
    if st.session_state.show_exit_confirm:
        st.markdown("""
        <div style="background: #2b1115; border: 1px solid #dc2626; border-radius: 6px; padding: 8px 14px; margin-bottom: 8px;">
            <span style="color: #fecaca; font-size: 0.85rem; font-weight: 500;">Exit current trial? All progress in this round will be lost.</span>
        </div>
        """, unsafe_allow_html=True)
        c_conf1, c_conf2 = st.columns([1, 1])
        with c_conf1:
            if st.button("Yes, Exit to Home", type="primary", use_container_width=True, key="btn_confirm_exit_yes"):
                engine.restart_game()
                st.session_state.show_exit_confirm = False
                st.session_state.canvas_nonce += 1
                st.session_state.last_meta = None
                st.session_state.last_top3 = []
                st.rerun()
        with c_conf2:
            if st.button("Cancel & Resume", use_container_width=True, key="btn_confirm_exit_cancel"):
                st.session_state.show_exit_confirm = False
                st.rerun()
    
    # Compact Top HUD Bar
    diff_val = current_ch.difficulty.capitalize() if current_ch.difficulty else "Normal"
    diff_css = "hud-tier-easy" if diff_val.lower() == "easy" else ("hud-tier-medium" if diff_val.lower() == "medium" else "hud-tier-hard")
    stage_type = "Riddle" if st.session_state.model_version == 20 else "Sector"
    lives_str = "❤️" * summary['lives'] + "🖤" * (summary['initial_lives'] - summary['lives']) if summary['lives'] <= 5 else f"{summary['lives']} / {summary['initial_lives']}"
    
    col_hud_left, col_hud_right = st.columns([4.2, 0.8])
    with col_hud_left:
        st.markdown(f"""
        <div class="gameplay-hud">
            <div class="hud-items-row">
                <div class="hud-cell">
                    <span class="hud-cell-label">{stage_type}</span>
                    <span class="hud-cell-value">{summary['current_room_number']} of {summary['total_rooms']}</span>
                </div>
                <div class="hud-sep"></div>
                <div class="hud-cell">
                    <span class="hud-cell-label">Difficulty</span>
                    <span class="{diff_css}">{diff_val}</span>
                </div>
                <div class="hud-sep"></div>
                <div class="hud-cell">
                    <span class="hud-cell-label">Lives</span>
                    <span class="hud-cell-value" style="color: #f87171; letter-spacing: 1px;">{lives_str}</span>
                </div>
                <div class="hud-sep"></div>
                <div class="hud-cell">
                    <span class="hud-cell-label">Score</span>
                    <span class="hud-cell-value" style="color: #60a5fa;">{summary['score']} pts</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_hud_right:
        if st.button("Exit to Home", use_container_width=True, key="btn_hud_exit"):
            st.session_state.show_exit_confirm = True
            st.rerun()
            
    # Mini Progress Track
    pct = int((summary['current_room_idx'] / summary['total_rooms']) * 100)
    st.markdown(f"""
    <div class="hud-progress-bar">
        <div class="hud-progress-fill" style="width: {pct}%;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Compact Riddle Presentation Card
    card_header = f"Riddle {summary['current_room_number']} of {summary['total_rooms']}" if st.session_state.model_version == 20 else f"Sector {summary['current_room_number']}"
    diff_tag = current_ch.difficulty.upper() if current_ch.difficulty else "NORMAL"
    st.markdown(f"""
    <div class="gameplay-riddle-card">
        <div class="gameplay-riddle-top">
            <span class="gameplay-riddle-tag">{card_header} • {diff_tag}</span>
            <span class="gameplay-riddle-title">{current_ch.title}</span>
        </div>
        <div class="gameplay-riddle-poem">{current_ch.scenario}</div>
        <div class="gameplay-riddle-prompt">{current_ch.prompt}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Drawing Canvas & Feedback Panel in Viewport-Efficient 2-Column Split
    col_draw, col_pred = st.columns([1.0, 1.0], gap="medium")
    
    with col_draw:
        st.markdown('<div class="gameplay-col-header">Drawing Canvas</div>', unsafe_allow_html=True)
        canvas_key = f"canvas_room_{summary['current_room_number']}_{st.session_state.canvas_nonce}"
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=stroke_width,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=260,
            width=460,
            drawing_mode="freedraw",
            return_image_data=True,
            key=canvas_key
        )
        
        c_btn1, c_btn2 = st.columns([1.2, 0.8])
        with c_btn1:
            submit_pressed = st.button("Submit Drawing", type="primary", use_container_width=True, key="btn_submit_draw")
        with c_btn2:
            if st.button("Clear Canvas", use_container_width=True, key="btn_clear_draw"):
                st.session_state.canvas_nonce += 1
                st.rerun()
                
        st.markdown('<div style="color: #6b7280; font-size: 0.74rem; margin-top: 2px;">Draw centered silhouette with black ink.</div>', unsafe_allow_html=True)
        
    with col_pred:
        st.markdown('<div class="gameplay-col-header">AI Vision Analysis</div>', unsafe_allow_html=True)
        
        # If Room is Solved -> Show Success Feedback and Next Button
        if engine.game_state == "ROOM_SUCCESS":
            pts = engine.last_attempt_feedback.get('points_awarded', 100)
            bonus_note = " (+50 First-Attempt Bonus!)" if engine.last_attempt_feedback.get('is_first_try', False) else ""
            solved_title = "Riddle Solved" if st.session_state.model_version == 20 else "Sector Solved"
            
            st.markdown(f"""
            <div class="feedback-success" style="padding: 10px 14px; margin: 0 0 8px 0;">
                <h4 style="margin: 0 0 4px 0; color: #34d399; font-size: 1.05rem;">{solved_title}</h4>
                <p style="font-size: 0.90rem; line-height: 1.4; margin-bottom: 6px;">{current_ch.success_message}</p>
                <p style="margin: 0; font-size: 0.82rem; font-weight: 600; color: #a7f3d0;">Points Earned: +{pts} pts{bonus_note}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if summary['current_room_idx'] < summary['total_rooms'] - 1:
                next_btn_label = "Next Riddle" if st.session_state.model_version == 20 else "Next Sector"
            else:
                next_btn_label = "Complete Riddle Trial" if st.session_state.model_version == 20 else "Complete Final Evacuation"
                
            if st.button(next_btn_label, type="primary", use_container_width=True, key="btn_next_room"):
                engine.advance_room()
                st.session_state.canvas_nonce += 1
                st.session_state.last_meta = None
                st.session_state.last_top3 = []
                st.rerun()
                
        else:
            # Active attempt handling
            if submit_pressed:
                if canvas_result is not None and canvas_result.image_data is not None:
                    # Run CPU inference
                    pred_class, conf, top3, meta = predictor.predict_canvas(canvas_result.image_data)
                    st.session_state.last_meta = meta
                    st.session_state.last_top3 = top3
                    
                    feedback = engine.submit_attempt(
                        predicted_class=pred_class,
                        confidence=conf,
                        threshold=confidence_threshold,
                        is_empty=meta.get('is_empty', False)
                    )
                    
                    if feedback.get("is_empty", False):
                        st.markdown(f"""
                        <div class="feedback-info" style="padding: 8px 12px; margin-bottom: 6px; font-size: 0.84rem;">
                            <strong>Blank Canvas:</strong> {feedback['message']}
                        </div>
                        """, unsafe_allow_html=True)
                    elif feedback["success"]:
                        st.rerun()  # Triggers ROOM_SUCCESS rendering
                    else:
                        st.markdown(f"""
                        <div class="feedback-error" style="padding: 8px 12px; margin-bottom: 6px; font-size: 0.84rem;">
                            <strong>Incorrect Attempt:</strong> {feedback['message']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="feedback-info" style="padding: 8px 12px; margin-bottom: 6px; font-size: 0.84rem;">
                        <strong>Blank Canvas:</strong> Please sketch your answer before submitting.
                    </div>
                    """, unsafe_allow_html=True)
                    
            elif engine.last_attempt_feedback and not engine.last_attempt_feedback.get("success", False):
                fb = engine.last_attempt_feedback
                if fb.get("life_lost", False):
                    st.markdown(f"""
                    <div class="feedback-error" style="padding: 8px 12px; margin-bottom: 6px; font-size: 0.84rem;">
                        <strong>Previous Attempt:</strong> {fb['message']}
                    </div>
                    """, unsafe_allow_html=True)
                elif fb.get("is_empty", False):
                    st.markdown(f"""
                    <div class="feedback-info" style="padding: 8px 12px; margin-bottom: 6px; font-size: 0.84rem;">
                        <strong>Blank Canvas:</strong> {fb['message']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="feedback-info" style="padding: 8px 12px; margin-bottom: 6px; font-size: 0.84rem;">
                    Sketch your solution on the canvas and click <strong>Submit Drawing</strong> to scan.
                </div>
                """, unsafe_allow_html=True)
                
            # Top-3 Probabilities
            if st.session_state.last_top3:
                st.markdown('<div style="font-size:0.75rem; font-weight:600; text-transform:uppercase; color:#9ca3af; margin: 4px 0 3px 0;">Detection Probabilities</div>', unsafe_allow_html=True)
                for rank, (c_name, p) in enumerate(st.session_state.last_top3, 1):
                    display_name = CLASS_DISPLAY.get(c_name, c_name.capitalize())
                    bar_w = int(p * 100)
                    st.markdown(f"""
                    <div class="top3-item">
                        <span class="top3-label">{rank}. {display_name}</span>
                        <div class="top3-bar-container">
                            <div class="top3-bar-active" style="width: {bar_w}%;"></div>
                        </div>
                        <span class="top3-percentage">{p*100:4.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
            # Hint Expander
            if current_ch.progressive_hints:
                with st.expander("Hint", expanded=False):
                    st.markdown(f"**Clue 1:** {current_ch.progressive_hints[0]}")
                        
            # Contextual Drawing Tips Expander
            with st.expander("Drawing Tips for This Riddle", expanded=False):
                for ac in current_ch.accepted_classes:
                    tip = DRAWING_TIPS.get(ac, "Draw a clean, recognizable outline in the center.")
                    name_lbl = CLASS_DISPLAY.get(ac, ac.capitalize())
                    st.write(f"• **{name_lbl}:** {tip}")
                st.caption("AI recognition works best when sketches are centered and drawn with decisive strokes.")
                
            # Debug Preprocessed View (Optional Sidebar Toggle)
            if show_debug and st.session_state.last_meta and 'processed_28x28' in st.session_state.last_meta:
                with st.expander("Neural Input Inspection (28×28 Grayscale)", expanded=False):
                    meta = st.session_state.last_meta
                    cd1, cd2, cd3 = st.columns([1, 1, 2])
                    with cd1:
                        st.image(255 - meta['processed_28x28'], width=70, caption="Player Canvas")
                    with cd2:
                        st.image(meta['processed_28x28'], width=70, caption="CNN Input Tensor")
                    with cd3:
                        st.write(f"• **Stroke Pixels:** {meta.get('stroke_pixels', 'N/A')}")
                        st.write(f"• **Bounding Box:** {meta.get('bbox', 'N/A')}")


# -------------------------------------------------------------
# SCREEN 3: GAME OVER
# -------------------------------------------------------------
elif engine.game_state == "GAME_OVER":
    summary = engine.get_summary()
    fail_title = "Trial Failed" if st.session_state.model_version == 20 else "Critical Containment Failure"
    fail_sub = "You have run out of lives. Review the riddle clues and try again." if st.session_state.model_version == 20 else "Your vital signs have flatlined. The facility lockdown seals forever."
    
    st.markdown(f"""
    <div class="end-card end-card-gameover">
        <h1 style="color:#ef4444 !important;">{fail_title}</h1>
        <p style="font-size:1.15rem; color:#fca5a5; margin-bottom: 20px;">{fail_sub}</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        solved_label = "Riddles Solved" if st.session_state.model_version == 20 else "Sectors Survived"
        st.metric(solved_label, f"{summary['rooms_solved']} of {summary['total_rooms']}")
    with c2:
        st.metric("First-Try Solves", f"{summary['first_try_bonuses']}")
    with c3:
        st.metric("Final Score", f"{summary['score']} pts")
        
    st.markdown("---")
    col_go1, col_go2 = st.columns(2)
    with col_go1:
        if st.button("Retry Trial", type="primary", use_container_width=True):
            engine.restart_game()
            st.session_state.canvas_nonce += 1
            st.session_state.last_meta = None
            st.session_state.last_top3 = []
            st.session_state.show_exit_confirm = False
            st.rerun()
    with col_go2:
        if st.button("Exit to Home", type="secondary", use_container_width=True):
            engine.restart_game()
            st.session_state.canvas_nonce += 1
            st.session_state.last_meta = None
            st.session_state.last_top3 = []
            st.session_state.show_exit_confirm = False
            st.rerun()


# -------------------------------------------------------------
# SCREEN 4: VICTORY
# -------------------------------------------------------------
elif engine.game_state == "VICTORY":
    summary = engine.get_summary()
    diff_name = engine.difficulty.capitalize() if engine.difficulty else "Normal"
    vic_title = "Trial Completed" if st.session_state.model_version == 20 else "Facility Evacuated"
    vic_sub = f"You successfully solved all {summary['total_rooms']} riddles on {diff_name} difficulty." if st.session_state.model_version == 20 else f"You navigated all {summary['total_rooms']} crisis chambers and escaped to freedom."
    
    st.markdown(f"""
    <div class="end-card end-card-victory">
        <h1 style="color:#60a5fa !important;">{vic_title}</h1>
        <p style="font-size:1.15rem; color:#93c5fd; margin-bottom: 20px;">{vic_sub}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.balloons()
    
    # Detailed score breakdown
    st.markdown("#### Final Scorecard")
    base_points = summary['rooms_solved'] * POINTS_PER_ROOM
    first_try_pts = summary['first_try_bonuses'] * FIRST_TRY_BONUS
    life_pts = summary['remaining_lives_bonus']
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        room_label = "Riddles Solved" if st.session_state.model_version == 20 else "Chambers Solved"
        st.metric(room_label, f"{summary['rooms_solved']} / {summary['total_rooms']}", f"+{base_points} pts")
    with c2:
        st.metric("First-Attempt Bonuses", f"{summary['first_try_bonuses']}", f"+{first_try_pts} pts")
    with c3:
        st.metric("Lives Retained", f"{summary['lives']} / 3", f"+{life_pts} pts")
    with c4:
        st.metric("Grand Total", f"{summary['score']} pts")
        
    narrative_p = (
        "The neural vision system acknowledges your wits. Every poetic riddle has been unraveled through your drawings."
        if st.session_state.model_version == 20 else
        "As your aircraft banks over the breaking sunrise, the self-destruct sequence collapses the abandoned complex into dust. Your rapid ingenuity and quick sketches outsmarted the facility's neural surveillance. You have survived."
    )
    st.markdown(f"""
    <div class="surface-card" style="margin-top: 15px;">
        <p style="margin: 0; line-height: 1.6; color: #9ca3af;">{narrative_p}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        if st.button("Play Again (Same Difficulty)", type="primary", use_container_width=True):
            engine.restart_game()
            st.session_state.canvas_nonce += 1
            st.session_state.last_meta = None
            st.session_state.last_top3 = []
            st.session_state.show_exit_confirm = False
            st.rerun()
    with col_v2:
        if st.button("Exit to Home", type="secondary", use_container_width=True):
            engine.restart_game()
            st.session_state.canvas_nonce += 1
            st.session_state.last_meta = None
            st.session_state.last_top3 = []
            st.session_state.show_exit_confirm = False
            st.rerun()


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85em;'>"
    "Dwiddle · Your mind. Your sketch. Your victory."
    "</div>",
    unsafe_allow_html=True
)
