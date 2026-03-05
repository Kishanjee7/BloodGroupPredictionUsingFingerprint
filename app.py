import streamlit as st
import numpy as np
import cv2
from PIL import Image

# -----------------------
# Config
# -----------------------
IMG_SIZE = (128, 128)

st.set_page_config(
    page_title="Blood Group Predictor",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------
# Sidebar (UI only)
# -----------------------
with st.sidebar:
    st.markdown("## 🩸 Blood Group Predictor")
    st.markdown("---")

    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. Upload a fingerprint image  
    2. Preview image  
    3. Click Analyze  
    4. View prediction  
    """)

    st.markdown("---")
    st.markdown("### 🧠 Models (Demo)")
    st.markdown("""
    - RF-HOG  
    - RF-Gabor  
    - CNN  
    - MobileNetV2  
    - Ensemble  
    """)

# -----------------------
# Hero Section
# -----------------------
st.markdown("""
<div style="text-align:center;padding:2rem;background:linear-gradient(90deg,#1D3557,#E63946);border-radius:20px;color:white;">
<h1>🩸 Blood Group Prediction Using Fingerprint</h1>
<p>Frontend UI Demo Version (No Backend)</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# Model Status Cards (Static)
# -----------------------
cols = st.columns(5)
models = ["RF-HOG", "RF-Gabor", "CNN", "MobileNetV2", "Ensemble"]

for col, model in zip(cols, models):
    col.markdown(f"""
    <div style="padding:1rem;background:#f4f4f4;border-radius:12px;text-align:center;">
    <h4>{model}</h4>
    <span style="color:green;font-weight:bold;">Ready (Demo)</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# Upload Section
# -----------------------
uploaded = st.file_uploader(
    "📤 Upload a fingerprint image",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded is not None:

    img = Image.open(uploaded).convert("L")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = cv2.resize(img_np, IMG_SIZE)

    st.markdown("### 🔍 Image Preview")
    c1, c2 = st.columns(2)

    with c1:
        st.image(img_np, caption="Original", use_container_width=True)

    with c2:
        st.image(img_np, caption="Enhanced (Demo)", use_container_width=True)

    st.markdown("---")

    if st.button("🔬 Analyze Fingerprint", use_container_width=True):

        # -----------------------
        # Dummy Prediction (Static)
        # -----------------------
        blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
        probs = np.array([10, 5, 15, 8, 12, 6, 35, 9])
        probs = probs / probs.sum()

        final_idx = np.argmax(probs)
        final_label = blood_groups[final_idx]
        final_conf = probs[final_idx] * 100

        st.markdown("### 🎯 Final Prediction (Demo)")

        st.markdown(f"""
        <div style="text-align:center;">
        <h1 style="color:#E63946;font-size:60px;">{final_label}</h1>
        <h3>Confidence: {final_conf:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📈 Probability Distribution (Demo)")

        for bg, p in zip(blood_groups, probs):
            st.progress(float(p))

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem;">
    <h3>Upload a fingerprint image to see the UI demo.</h3>
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# Footer
# -----------------------
st.markdown("""
<hr>
<p style="text-align:center;color:gray;">
Frontend UI Demo Only • No ML Backend Connected
</p>
""", unsafe_allow_html=True)
