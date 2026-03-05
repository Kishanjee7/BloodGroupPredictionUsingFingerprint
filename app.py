import os
import cv2
import time
import joblib
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from skimage.feature import hog
from skimage.filters import gabor
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="BloodAI Neural Interface",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

IMG_SIZE = (128, 128)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    rf_hog = joblib.load(os.path.join(MODEL_DIR, "rf_hog.joblib"))
    rf_gabor = joblib.load(os.path.join(MODEL_DIR, "rf_gabor.joblib"))
    ensemble = joblib.load(os.path.join(MODEL_DIR, "ensemble_logreg.joblib"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

    cnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, "cnn_model.h5"), compile=False)
    mnet = tf.keras.models.load_model(os.path.join(MODEL_DIR, "mobilenet_model.h5"), compile=False)

    return rf_hog, rf_gabor, cnn, mnet, ensemble, label_encoder

rf_hog, rf_gabor, cnn_model, mnet_model, ensemble_model, label_encoder = load_models()

# ==============================
# TESLA UI CSS
# ==============================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #000;
    color: #E0E0E0;
    font-family: 'Orbitron', sans-serif;
}
.navbar {
    position: sticky;
    top: 0;
    padding: 1rem 2rem;
    backdrop-filter: blur(10px);
    background: rgba(0,0,0,0.6);
    display:flex;
    justify-content:space-between;
    border-bottom:1px solid rgba(255,255,255,0.1);
}
.nav-title { font-weight:700; font-size:1.3rem; }
.neon {
    border:2px solid #00F5FF;
    box-shadow:0 0 10px #00F5FF,0 0 30px #00F5FF;
    border-radius:16px;
    padding:2rem;
}
@keyframes rotate3d {
  0% { transform: rotateY(0deg); }
  100% { transform: rotateY(360deg); }
}
.rotate3d { animation: rotate3d 8s linear infinite; }
@keyframes scan {
  0% { top:0%; }
  100% { top:100%; }
}
.scan-line {
    position:absolute;
    width:100%;
    height:4px;
    background:rgba(0,255,255,0.6);
    animation: scan 2s linear infinite;
}
* { transition: all 0.3s ease; }
</style>
""", unsafe_allow_html=True)

# ==============================
# NAVBAR
# ==============================
st.markdown("""
<div class="navbar">
  <div class="nav-title">🧬 BLOODAI NEURAL SYSTEM</div>
  <div>Fingerprint Intelligence Engine</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;margin-top:30px;'>Neural Blood Group Detection</h1>", unsafe_allow_html=True)

# ==============================
# PREPROCESS FUNCTIONS
# ==============================
def enhance(img):
    img = cv2.GaussianBlur(img, (5,5), 0)
    return img

def get_hog_feature(img):
    f = hog(img, orientations=9, pixels_per_cell=(16,16),
            cells_per_block=(2,2), block_norm="L2-Hys")
    return f.reshape(1,-1)

def get_gabor_feature(img):
    real, imag = gabor(img, frequency=0.3)
    return np.array([real.mean(), real.var(), imag.mean(), imag.var()]).reshape(1,-1)

def prep_cnn(img):
    return np.expand_dims(img, axis=(0,-1))

def prep_mnet(img):
    x = np.repeat(np.expand_dims(img,axis=-1),3,axis=-1)
    x = preprocess_input(x*255.0)
    return np.expand_dims(x,axis=0)

# ==============================
# FILE UPLOAD
# ==============================
uploaded = st.file_uploader("Upload Fingerprint Image", type=["png","jpg","jpeg"])

if uploaded:

    image = Image.open(uploaded).convert("L")
    image = np.array(image)/255.0
    image = cv2.resize(image, IMG_SIZE)
    enhanced = enhance(image)

    # 3D rotating fingerprint animation
    st.markdown("""
    <div style="position:relative;text-align:center;">
        <img src="https://cdn-icons-png.flaticon.com/512/565/565547.png"
             width="220"
             class="rotate3d">
        <div class="scan-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # Sound effect
    st.markdown("""
    <audio autoplay>
      <source src="https://assets.mixkit.co/sfx/preview/mixkit-robotic-interface-beep-2579.mp3" type="audio/mpeg">
    </audio>
    """, unsafe_allow_html=True)

    # Progress animation
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progress.progress(i+1)
    progress.empty()

    # ==============================
    # PREDICTION
    # ==============================
    p_rf_hog = rf_hog.predict_proba(get_hog_feature(enhanced))[0]
    p_rf_gabor = rf_gabor.predict_proba(get_gabor_feature(enhanced))[0]
    p_cnn = cnn_model.predict(prep_cnn(enhanced), verbose=0)[0]
    p_mnet = mnet_model.predict(prep_mnet(enhanced), verbose=0)[0]

    stack = np.concatenate([p_rf_hog,p_rf_gabor,p_cnn,p_mnet]).reshape(1,-1)
    p_ens = ensemble_model.predict_proba(stack)[0]

    classes = label_encoder.classes_
    idx = np.argmax(p_ens)
    final_label = classes[idx]
    final_conf = round(p_ens[idx]*100,1)

    # ==============================
    # RESULT DISPLAY
    # ==============================
    st.markdown(f"""
    <div class="neon" style="text-align:center;margin-top:40px;">
        <h2 style="font-size:4rem;color:#00F5FF;">{final_label}</h2>
        <p style="font-size:1.5rem;">Confidence: {final_conf}%</p>
    </div>
    """, unsafe_allow_html=True)

    # ==============================
    # MODEL COMPARISON CHART
    # ==============================
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["RF-HOG","RF-Gabor","CNN","MobileNet","Ensemble"],
        y=[
            max(p_rf_hog)*100,
            max(p_rf_gabor)*100,
            max(p_cnn)*100,
            max(p_mnet)*100,
            max(p_ens)*100
        ]
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Model Confidence Comparison",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("<h3 style='text-align:center;margin-top:100px;color:#444;'>Awaiting Neural Input...</h3>", unsafe_allow_html=True)