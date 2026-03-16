import streamlit as st
import base64
import os
import requests
import numpy as np

# =========================================
# TENSORFLOW SILENCER (Wajib Paling Atas)
# =========================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

try:
    import prediction
    import eda
except ImportError as e:
    st.error(f"Missing Internal Module: {e}")

# =========================================
# CONFIG & THEME
# =========================================
st.set_page_config(
    page_title="Matrix Drone Intelligence", 
    page_icon="🛸", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Proteksi library Lottie
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

# =========================================
# UTILITIES & ASSET LOADERS
# =========================================
def load_lottieurl(url: str):
    if not LOTTIE_AVAILABLE: return None
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def sidebar_bg(gif_path):
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        st.markdown(f"""
            <style>
            [data-testid="stSidebar"] {{
                background-image: url("data:image/gif;base64,{bin_str}") !important;
                background-size: cover !important;
                background-repeat: no-repeat !important;
                background-position: center !important;
            }}
            </style>
            """, unsafe_allow_html=True)

# =========================================
# INITIALIZE INTERFACE
# =========================================
local_css("src/style.css") 
sidebar_bg("src/background.gif")

# =========================================
# SIDEBAR NAVIGATION
# =========================================
with st.sidebar:
    if LOTTIE_AVAILABLE:
        lottie_url = "https://assets5.lottiefiles.com/packages/lf20_fp7vscdz.json"
        lottie_json = load_lottieurl(lottie_url)
        if lottie_json:
            st_lottie(lottie_json, height=150, key="drone_main")
        else:
            st.markdown('<h1 style="text-align:center;">🛸</h1>', unsafe_allow_html=True)
    else:
        st.markdown('<h1 style="text-align:center;">🛸</h1>', unsafe_allow_html=True)
    
    st.markdown('<p class="matrix-menu-text">🛰️ NEURAL UPLINK</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # NAVIGATION SELECTBOX (Dropdown Mode)
    st.markdown('<div class="matrix-select-container">', unsafe_allow_html=True)
    page = st.selectbox("NAVIGATION SYSTEM", ["Intel Prediction", "EDA Analysis"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---") 
    
    st.markdown("""
        <div class="status-container">
            <p class="status-online">● SYSTEM: ONLINE</p>
        </div>
    """, unsafe_allow_html=True)

# ROUTING
if page == "Intel Prediction":
    prediction.menu1()
elif page == "EDA Analysis":
    if hasattr(eda, 'run_analysis'):
        eda.run_analysis()
    else:
        st.warning("EDA module does not have run_analysis() function.")

# Footer Watermark
st.sidebar.markdown("""
    <p style="
        position: fixed; 
        bottom: 10px;
        font-size: 13px; 
        color: #00ff00; 
        font-weight: 900; 
        text-shadow: 1px 1px 2px #000, 0 0 10px #00ff00; 
        font-family: 'Courier New', monospace;
        letter-spacing: 1px;
    ">
    v1.0.0-Computer-VisionApp
    </p>
""", unsafe_allow_html=True)