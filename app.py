import os
import cv2
import json
import zipfile
import joblib
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import tensorflow as tf
from skimage.feature import hog
from skimage.filters import gabor
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

try:
    import keras as standalone_keras
except Exception:
    standalone_keras = None

# -----------------------
# Config
# -----------------------
IMG_SIZE = (128, 128)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

PATHS = {
    "rf_hog": os.path.join(MODEL_DIR, "rf_hog.joblib"),
    "rf_gabor": os.path.join(MODEL_DIR, "rf_gabor.joblib"),
    "cnn_h5": os.path.join(MODEL_DIR, "cnn_model.h5"),
    "cnn_keras": os.path.join(MODEL_DIR, "cnn_model.keras"),
    "mnet_h5": os.path.join(MODEL_DIR, "mobilenet_model.h5"),
    "mnet_keras": os.path.join(MODEL_DIR, "mobilenet_model.keras"),
    "ensemble": os.path.join(MODEL_DIR, "ensemble_logreg.joblib"),
    "label_encoder": os.path.join(MODEL_DIR, "label_encoder.joblib"),
}

# -----------------------
# Keras compatibility (Keras 3 config -> Keras 2)
# -----------------------
import h5py

def _extract_keras_history(tensor_obj):
    if isinstance(tensor_obj, dict):
        cls = tensor_obj.get("class_name", "")
        if cls in ("__keras_tensor__", "keras_tensor"):
            cfg = tensor_obj.get("config", {})
            history = cfg.get("keras_history", [])
            if len(history) == 3:
                return history
    return tensor_obj

def _convert_inbound_nodes_k3_to_k2(nodes):
    if not nodes:
        return nodes
    if isinstance(nodes, list) and len(nodes) > 0:
        first = nodes[0]
        if isinstance(first, list):
            return nodes
        if isinstance(first, dict) and "args" in first:
            converted = []
            for node in nodes:
                args = node.get("args", [])
                call_inputs = []
                for arg in args:
                    if isinstance(arg, dict) and arg.get("class_name") in ("__keras_tensor__", "keras_tensor"):
                        h = _extract_keras_history(arg)
                        call_inputs.append(h)
                    elif isinstance(arg, list):
                        for item in arg:
                            h = _extract_keras_history(item)
                            call_inputs.append(h)
                    else:
                        call_inputs.append(arg)
                converted.append(call_inputs)
            return converted
    return nodes

def _fix_keras3_config(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("build_config", "shared_object_id", "registered_name"):
                continue
            nk = "batch_input_shape" if k == "batch_shape" else k
            if nk == "module" and isinstance(v, str):
                if v.startswith("keras.src."):
                    v = "keras." + v[len("keras.src."):]
            if nk == "dtype" and isinstance(v, dict):
                cls = v.get("class_name", "")
                if cls == "DTypePolicy":
                    v = v.get("config", {}).get("name", "float32")
                else:
                    v = _fix_keras3_config(v)
            if nk == "inbound_nodes":
                out[nk] = _convert_inbound_nodes_k3_to_k2(v)
                continue
            if nk != "dtype" or isinstance(v, dict):
                out[nk] = _fix_keras3_config(v)
            else:
                out[nk] = v
        return out
    if isinstance(obj, list):
        return [_fix_keras3_config(x) for x in obj]
    return obj

def _strip_module_keys(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "module":
                continue
            out[k] = _strip_module_keys(v)
        return out
    if isinstance(obj, list):
        return [_strip_module_keys(x) for x in obj]
    return obj

def patch_keras_file(src_path: str) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="keras_patch_")
    patched = os.path.join(tmp_dir, os.path.basename(src_path))
    with zipfile.ZipFile(src_path, "r") as zin:
        with zipfile.ZipFile(patched, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for name in zin.namelist():
                data = zin.read(name)
                if name == "config.json":
                    cfg = json.loads(data.decode("utf-8"))
                    cfg = _fix_keras3_config(cfg)
                    cfg = _strip_module_keys(cfg)
                    data = json.dumps(cfg).encode("utf-8")
                zout.writestr(name, data)
    return patched

def patch_h5_file(src_path: str) -> str:
    import shutil
    tmp_dir = tempfile.mkdtemp(prefix="h5_patch_")
    patched = os.path.join(tmp_dir, os.path.basename(src_path))
    shutil.copy2(src_path, patched)
    try:
        with h5py.File(patched, "r+") as f:
            if "model_config" in f.attrs:
                raw = f.attrs["model_config"]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                cfg = json.loads(raw)
                cfg = _fix_keras3_config(cfg)
                cfg = _strip_module_keys(cfg)
                f.attrs["model_config"] = json.dumps(cfg)
    except Exception:
        pass
    return patched

def _try_load_model(path: str, loader_name: str, errs: list):
    if not os.path.exists(path):
        errs.append(f"{loader_name}: file not found ({path})")
        return None
    try:
        if loader_name.startswith("tf.keras"):
            return tf.keras.models.load_model(path, compile=False)
        if loader_name.startswith("keras") and standalone_keras is not None:
            return standalone_keras.models.load_model(path, compile=False)
    except Exception as e:
        errs.append(f"{loader_name} failed: {e}")
    return None

def load_deep_model(primary_h5: str, fallback_keras: str, model_name: str):
    errs = []
    if os.path.exists(primary_h5):
        m = _try_load_model(primary_h5, "tf.keras", errs)
        if m is not None: return m, None
        m = _try_load_model(primary_h5, "keras", errs)
        if m is not None: return m, None
        try:
            patched_h5 = patch_h5_file(primary_h5)
            m = _try_load_model(patched_h5, "tf.keras (patched h5)", errs)
            if m is not None: return m, None
            m = _try_load_model(patched_h5, "keras (patched h5)", errs)
            if m is not None: return m, None
        except Exception as e:
            errs.append(f"patch_h5_file failed: {e}")
    else:
        errs.append(f".h5 not found ({primary_h5})")

    if os.path.exists(fallback_keras):
        m = _try_load_model(fallback_keras, "tf.keras", errs)
        if m is not None: return m, None
        m = _try_load_model(fallback_keras, "keras", errs)
        if m is not None: return m, None
        try:
            patched = patch_keras_file(fallback_keras)
            m = _try_load_model(patched, "tf.keras (patched keras)", errs)
            if m is not None: return m, None
            m = _try_load_model(patched, "keras (patched keras)", errs)
            if m is not None: return m, None
        except Exception as e:
            errs.append(f"patch_keras_file failed: {e}")
    else:
        errs.append(f".keras not found ({fallback_keras})")

    return None, f"{model_name} unavailable | " + " | ".join(errs)

# -----------------------
# Preprocessing
# -----------------------
def enhance_fingerprint(img_gray_01: np.ndarray) -> np.ndarray:
    img_u8 = (img_gray_01 * 255).astype(np.uint8)
    img_denoised = cv2.GaussianBlur(img_u8, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_denoised)
    blur = cv2.GaussianBlur(img_clahe, (9, 9), 10)
    img_sharp = cv2.addWeighted(img_clahe, 1.5, blur, -0.5, 0)
    return img_sharp.astype(np.float32) / 255.0

def get_hog_feature(img_gray_01: np.ndarray) -> np.ndarray:
    f = hog(img_gray_01, orientations=9, pixels_per_cell=(16, 16),
            cells_per_block=(2, 2), block_norm="L2-Hys")
    return f.reshape(1, -1).astype(np.float32)

def get_gabor_feature(img_gray_01: np.ndarray, freqs=(0.1, 0.3, 0.5)) -> np.ndarray:
    vals = []
    for freq in freqs:
        real, imag = gabor(img_gray_01, frequency=freq)
        vals += [real.mean(), real.var(), imag.mean(), imag.var()]
    return np.array(vals, dtype=np.float32).reshape(1, -1)

def prep_cnn_input(img_gray_01: np.ndarray) -> np.ndarray:
    return np.expand_dims(img_gray_01, axis=(0, -1)).astype(np.float32)

def prep_mnet_input(img_gray_01: np.ndarray) -> np.ndarray:
    x = np.repeat(np.expand_dims(img_gray_01, axis=-1), 3, axis=-1)
    x = preprocess_input(x * 255.0)
    return np.expand_dims(x, axis=0).astype(np.float32)

def top_prediction(prob: np.ndarray, classes: np.ndarray):
    idx = int(np.argmax(prob))
    return classes[idx], float(prob[idx] * 100.0)

# -----------------------
# Load artifacts
# -----------------------
@st.cache_resource
def load_artifacts():
    required = ["rf_hog", "rf_gabor", "ensemble", "label_encoder"]
    missing = [k for k in required if not os.path.exists(PATHS[k])]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    rf_hog = joblib.load(PATHS["rf_hog"])
    rf_gabor = joblib.load(PATHS["rf_gabor"])
    ensemble = joblib.load(PATHS["ensemble"])
    label_encoder = joblib.load(PATHS["label_encoder"])

    cnn_model, cnn_err = load_deep_model(PATHS["cnn_h5"], PATHS["cnn_keras"], "CNN")
    mnet_model, mnet_err = load_deep_model(PATHS["mnet_h5"], PATHS["mnet_keras"], "MobileNetV2")

    return rf_hog, rf_gabor, cnn_model, mnet_model, ensemble, label_encoder, cnn_err, mnet_err

# -----------------------
# UI
# -----------------------
st.set_page_config(
    page_title="Blood Group Predictor",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&display=swap');

/* ═══ Keyframe Animations ═══ */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes barGrow {
    from { width: 0%; }
}

/* ═══ Root Variables (High Contrast) ═══ */
:root {
    --primary: #E63946;
    --primary-dark: #B92B37;
    --text-main: #1E293B;      /* Highly visible dark blue/grey */
    --text-muted: #475569;
    --bg-main: #F4F7F9;
    --bg-card: rgba(255, 255, 255, 0.95);
    --card-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-main);
}

/* Override default Streamlit text colors for perfect visibility */
p, li, span, div {
    color: var(--text-main);
}

/* ═══ Clean Background ═══ */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
}

/* ═══ Sidebar Styling ═══ */
[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #FF6B6B !important;
    font-family: 'Poppins', sans-serif;
}

/* ═══ Hero Banner (Deep Contrast) ═══ */
.hero-banner {
    background: linear-gradient(-45deg, #1D2671, #C33764);
    background-size: 200% 200%;
    animation: gradientShift 10s ease infinite;
    border-radius: 16px;
    padding: 3rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 15px 35px rgba(195, 55, 100, 0.2);
}
.hero-banner h1 {
    font-family: 'Poppins', sans-serif;
    color: #FFFFFF !important;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.hero-banner p {
    color: #F8FAFC !important;
    font-size: 1.2rem;
    margin-top: 0.8rem;
    font-weight: 500;
    opacity: 0.9;
}

/* ═══ Stat Cards ═══ */
.stat-card {
    background: var(--bg-card);
    border: 1px solid rgba(0,0,0,0.05);
    border-radius: 12px;
    padding: 1.5rem 1rem;
    text-align: center;
    box-shadow: var(--card-shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: slideUp 0.6s ease forwards;
}
.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(15, 23, 42, 0.12);
}
.stat-card-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--text-main) !important;
    margin: 8px 0;
}

/* ═══ Blood Group Result Badge ═══ */
.blood-badge {
    display: inline-block;
    background: linear-gradient(135deg, #E63946, #900C3F);
    color: #FFFFFF !important;
    font-family: 'Poppins', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    padding: 1rem 3rem;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(230,57,70,0.4);
}

/* ═══ Model Result Cards ═══ */
.model-card {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border-left: 5px solid #3B82F6;
    box-shadow: var(--card-shadow);
    margin-bottom: 1rem;
    animation: slideUp 0.5s ease forwards;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.model-card.ensemble {
    border-left-color: var(--primary);
    background: #FFF5F5;
    border: 1px solid rgba(230,57,70,0.1);
    border-left-width: 6px;
}
.model-card .model-name {
    font-weight: 700;
    color: var(--text-main) !important;
    font-size: 1.05rem;
}
.model-card .model-prediction {
    font-family: 'Poppins', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--primary) !important;
}

/* ═══ Status Pill ═══ */
.status-pill {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    text-transform: uppercase;
}
.status-pill.loaded {
    background: #DCFCE7;
    color: #166534 !important;
}
.status-pill.unavailable {
    background: #FEE2E2;
    color: #991B1B !important;
}

/* ═══ Probability Bars ═══ */
.prob-bar-container {
    margin: 12px 0;
    animation: slideUp 0.5s ease forwards;
}
.prob-bar-bg {
    background: #E2E8F0;
    border-radius: 8px;
    height: 18px;
    overflow: hidden;
    position: relative;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    animation: barGrow 1s cubic-bezier(0.1, 0.8, 0.2, 1) forwards;
}
.prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.95rem;
    color: var(--text-main) !important;
    font-weight: 600;
    margin-bottom: 6px;
}

/* ═══ Upload Area ═══ */
[data-testid="stFileUploader"] {
    border: 2px dashed #94A3B8 !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    background: #FFFFFF !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--primary) !important;
    background: #F8FAFC !important;
}

/* ═══ Buttons ═══ */
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #E63946, #B92B37) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 1.5rem !important;
    font-weight: 800 !important;
    font-size: 1.2rem !important;
    box-shadow: 0 8px 20px rgba(230,57,70,0.25) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 25px rgba(230,57,70,0.35) !important;
}

/* ═══ Headers ═══ */
h1, h2, h3 {
    color: #0F172A !important;
    font-family: 'Poppins', sans-serif !important;
}

/* ═══ Footer ═══ */
.footer-text {
    text-align: center;
    color: var(--text-muted) !important;
    font-size: 0.9rem;
    margin-top: 3rem;
    font-weight: 500;
}
.footer-text a {
    color: #2563EB !important;
    text-decoration: none;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🩸 Blood Group Predictor")
    st.markdown("---")

    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Upload** a fingerprint image
    2. **Preview** original & enhanced images
    3. Click **🔬 Analyze Fingerprint**
    4. View predictions from **5 models**
    """)

    st.markdown("---")
    st.markdown("### 🧠 Models")
    model_info = {
        "RF-HOG": "Random Forest on HOG features",
        "RF-Gabor": "Random Forest on Gabor features",
        "CNN": "Custom convolutional neural network",
        "MobileNetV2": "Transfer learning (pretrained)",
        "Ensemble": "Stacking meta-learner (final)",
    }
    for name, desc in model_info.items():
        st.markdown(f"**{name}** — {desc}")

    st.markdown("---")
    st.markdown("### 🩸 Supported Blood Groups")
    st.markdown("`A+` `A-` `B+` `B-` `AB+` `AB-` `O+` `O-`")

    st.markdown("---")
    with st.expander("⚙️ Runtime diagnostics"):
        st.caption(f"TensorFlow: {tf.__version__}")
        st.caption(f"Keras available: {standalone_keras is not None}")
        diag_rows = []
        for k, p in PATHS.items():
            diag_rows.append({"artifact": k, "exists": "✅" if os.path.exists(p) else "❌"})
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)

# ── Load models ──
try:
    rf_hog, rf_gabor, cnn_model, mnet_model, ensemble_model, label_encoder, cnn_err, mnet_err = load_artifacts()
except Exception as e:
    st.error(f"⚠️ Model loading failed: {e}")
    st.stop()

# ── Hero Banner ──
st.markdown("""
<div class="hero-banner">
    <h1>Blood Group Prediction Using Fingerprint</h1>
    <p>Multi-model system that analyzes fingerprint ridge patterns to predict your blood group</p>
</div>
""", unsafe_allow_html=True)

# ── Model status indicators ──
status_cols = st.columns(5)
model_statuses = [
    ("RF-HOG", True, "🌲"),
    ("RF-Gabor", True, "🌀"),
    ("CNN", cnn_model is not None, "🧠"),
    ("MobileNetV2", mnet_model is not None, "📱"),
    ("Ensemble", True, "🎯"),
]
for col, (name, loaded, icon) in zip(status_cols, model_statuses):
    status_cls = "loaded" if loaded else "unavailable"
    status_text = "Ready" if loaded else "N/A"
    col.markdown(f"""
    <div class="stat-card">
        <div style="font-size:2rem; margin-bottom:8px;">{icon}</div>
        <div class="stat-card-title">{name}</div>
        <span class="status-pill {status_cls}">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer

# ── Upload ──
uploaded = st.file_uploader(
    "📤 Upload a fingerprint image",
    type=["png", "jpg", "jpeg", "bmp"],
    help="Supported formats: PNG, JPG, JPEG, BMP — max 200MB"
)

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("L")
    orig = np.array(pil_img).astype(np.float32) / 255.0
    orig = cv2.resize(orig, IMG_SIZE)
    enhanced = enhance_fingerprint(orig)

    st.markdown("### 🔍 Image Preview")
    c1, c2 = st.columns(2)
    with c1:
        st.image(orig, caption="📷 Original Fingerprint", use_container_width=True, clamp=True)
    with c2:
        st.image(enhanced, caption="✨ Enhanced Fingerprint", use_container_width=True, clamp=True)

    st.write("") 

    col_l, col_btn, col_r = st.columns([1, 2, 1])
    with col_btn:
        predict_clicked = st.button("🔬 Analyze Fingerprint", use_container_width=True, type="primary")

    if predict_clicked:
        classes = label_encoder.classes_
        n_cls = len(classes)

        with st.spinner("🧬 Running analysis across all models..."):
            p_rf_hog = rf_hog.predict_proba(get_hog_feature(enhanced))[0]
            p_rf_gabor = rf_gabor.predict_proba(get_gabor_feature(enhanced))[0]

            if cnn_model is not None:
                p_cnn = cnn_model.predict(prep_cnn_input(enhanced), verbose=0)[0]
            else:
                p_cnn = np.zeros(n_cls, dtype=np.float32)

            if mnet_model is not None:
                p_mnet = mnet_model.predict(prep_mnet_input(enhanced), verbose=0)[0]
            else:
                p_mnet = np.zeros(n_cls, dtype=np.float32)

            base_stack = np.concatenate([p_rf_hog, p_rf_gabor, p_cnn, p_mnet], axis=0).reshape(1, -1)
            expected = getattr(ensemble_model, "n_features_in_", base_stack.shape[1])
            if base_stack.shape[1] < expected:
                pad = np.zeros((1, expected - base_stack.shape[1]), dtype=np.float32)
                stack = np.concatenate([base_stack, pad], axis=1)
            else:
                stack = base_stack[:, :expected]

            p_ens = ensemble_model.predict_proba(stack)[0]

        # ── Final Prediction Hero ──
        final_label, final_conf = top_prediction(p_ens, classes)
        
        st.write("---")
        st.markdown("<h2 style='text-align: center;'>🎯 Final Prediction</h2>", unsafe_allow_html=True)

        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
        with res_col2:
            st.markdown(f"""
            <div style="text-align:center; padding:1.5rem 0;">
                <div class="blood-badge">{final_label}</div>
                <div style="margin-top:16px; font-size:1.3rem; color:var(--text-main); font-weight: 600;">
                    Confidence: <strong style="color:#E63946;">{final_conf:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("---")

        col_models, col_probs = st.columns([1.2, 1])
        
        # ── Individual Model Results ──
        with col_models:
            st.markdown("### 📊 Model Predictions")
            model_results = [
                ("🌲 RF-HOG", p_rf_hog, True, False),
                ("🌀 RF-Gabor", p_rf_gabor, True, False),
                ("🧠 CNN", p_cnn, cnn_model is not None, False),
                ("📱 MobileNetV2", p_mnet, mnet_model is not None, False),
                ("🎯 Ensemble (Final)", p_ens, True, True),
            ]

            for name, prob, enabled, is_ensemble in model_results:
                label, conf = top_prediction(prob, classes)
                card_class = "model-card ensemble" if is_ensemble else "model-card"
                status_cls = "loaded" if enabled else "unavailable"
                status_text = "Loaded" if enabled else "Unavailable"

                st.markdown(f"""
                <div class="{card_class}">
                    <div>
                        <span class="model-name">{name}</span>
                        <span class="status-pill {status_cls}" style="margin-left:10px;">{status_text}</span>
                    </div>
                    <div style="text-align:right;">
                        <span class="model-prediction">{label}</span>
                        <div style="font-size:0.85rem; color:var(--text-muted); font-weight: 600;">{conf:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Probability Distribution ──
        with col_probs:
            st.markdown("### 📈 Probability Breakdown")

            sorted_indices = np.argsort(p_ens)[::-1]
            colors = ["#E63946", "#F59E0B", "#3B82F6", "#10B981", "#8B5CF6", "#EC4899", "#14B8A6", "#64748B"]

            prob_html = ""
            for rank, idx in enumerate(sorted_indices):
                bg = classes[idx]
                pct = p_ens[idx] * 100
                bar_color = colors[rank % len(colors)]
                prob_html += f"""
                <div class="prob-bar-container">
                    <div class="prob-bar-label">
                        <span>{bg}</span>
                        <span>{pct:.1f}%</span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{max(pct, 1)}%; background:{bar_color};"></div>
                    </div>
                </div>
                """

            st.markdown(prob_html, unsafe_allow_html=True)

        # ── Detailed Data Table ──
        st.write("")
        with st.expander("📋 View Detailed Data Table"):
            rows = []
            for name, prob, enabled in [
                ("RF-HOG", p_rf_hog, True),
                ("RF-Gabor", p_rf_gabor, True),
                ("CNN", p_cnn, cnn_model is not None),
                ("MobileNetV2", p_mnet, mnet_model is not None),
                ("Ensemble", p_ens, True),
            ]:
                label, conf = top_prediction(prob, classes)
                rows.append({
                    "Model": name,
                    "Status": "✅ Loaded" if enabled else "❌ Unavailable",
                    "Prediction": label,
                    "Confidence": f"{conf:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

else:
    # ── Empty State ──
    st.markdown("""
    <div style="text-align:center; padding:4rem 1rem;">
        <div style="font-size:4.5rem; margin-bottom:1rem;">👆</div>
        <h3 style="color:#0F172A; margin-bottom:0.5rem;">Upload a Fingerprint Image</h3>
        <p style="color:#475569; font-size:1.1rem; font-weight: 500;">
            Drag and drop or click above to upload a fingerprint image.<br>
            The AI will analyze the ridge patterns and predict the blood group.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div class="footer-text">
    Built with ❤️ using Streamlit & TensorFlow • 
    <a href="https://github.com/Kishanjee7/BloodGroupPredictionUsingFingerprint" target="_blank">GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
