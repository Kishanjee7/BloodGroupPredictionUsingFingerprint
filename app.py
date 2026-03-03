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
    """Extract [layer_name, node_index, tensor_index] from a Keras 3 tensor ref."""
    if isinstance(tensor_obj, dict):
        cls = tensor_obj.get("class_name", "")
        if cls in ("__keras_tensor__", "keras_tensor"):
            cfg = tensor_obj.get("config", {})
            history = cfg.get("keras_history", [])
            if len(history) == 3:
                return history  # [layer_name, node_index, tensor_index]
    return tensor_obj


def _convert_inbound_nodes_k3_to_k2(nodes):
    """
    Convert Keras 3 inbound_nodes format to Keras 2 format.

    Keras 3: [{"args": [tensor_or_list], "kwargs": {...}}, ...]
    Keras 2: [[[layer_name, node_idx, tensor_idx], ...], ...]

    For single-input layers:  args = [tensor]         -> [[history]]
    For multi-input layers:   args = [[t1, t2, ...]]  -> [[h1, h2, ...]]
    """
    if not nodes:
        return nodes

    # Check if nodes are already in Keras 2 format (list of lists)
    if isinstance(nodes, list) and len(nodes) > 0:
        first = nodes[0]
        if isinstance(first, list):
            return nodes  # Already Keras 2 format
        if isinstance(first, dict) and "args" in first:
            # Keras 3 format -> convert
            converted = []
            for node in nodes:
                args = node.get("args", [])
                call_inputs = []
                for arg in args:
                    if isinstance(arg, dict) and arg.get("class_name") in ("__keras_tensor__", "keras_tensor"):
                        # Single tensor input
                        h = _extract_keras_history(arg)
                        call_inputs.append(h)
                    elif isinstance(arg, list):
                        # List of tensors (e.g., Add, Concatenate)
                        for item in arg:
                            h = _extract_keras_history(item)
                            call_inputs.append(h)
                    else:
                        call_inputs.append(arg)
                converted.append(call_inputs)
            return converted
    return nodes


def _fix_keras3_config(obj):
    """Recursively convert a Keras 3 serialized config to Keras 2 format."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # 1) Skip keys that only exist in Keras 3
            if k in ("build_config", "shared_object_id", "registered_name"):
                continue

            # 2) Rename batch_shape -> batch_input_shape (InputLayer)
            nk = "batch_input_shape" if k == "batch_shape" else k

            # 3) Rewrite keras.src.* module paths -> keras.*
            if nk == "module" and isinstance(v, str):
                if v.startswith("keras.src."):
                    v = "keras." + v[len("keras.src."):]

            # 4) Flatten DTypePolicy dicts to plain strings
            if nk == "dtype" and isinstance(v, dict):
                cls = v.get("class_name", "")
                if cls == "DTypePolicy":
                    v = v.get("config", {}).get("name", "float32")
                else:
                    v = _fix_keras3_config(v)

            # 5) Convert inbound_nodes format
            if nk == "inbound_nodes":
                out[nk] = _convert_inbound_nodes_k3_to_k2(v)
                continue

            # 6) Recurse into the value
            if nk != "dtype" or isinstance(v, dict):
                out[nk] = _fix_keras3_config(v)
            else:
                out[nk] = v
        return out
    if isinstance(obj, list):
        return [_fix_keras3_config(x) for x in obj]
    return obj


def _strip_module_keys(obj):
    """Remove 'module' keys from layer dicts (Keras 2 doesn't use them)."""
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
    """Patch a .keras (zip) file by transforming its config.json."""
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
    """Patch an .h5 file by transforming its embedded model JSON config."""
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
    """Attempt to load a model, appending errors to errs on failure."""
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

    # 1) Try .h5 directly
    if os.path.exists(primary_h5):
        m = _try_load_model(primary_h5, "tf.keras", errs)
        if m is not None:
            return m, None
        m = _try_load_model(primary_h5, "keras", errs)
        if m is not None:
            return m, None
        # 1b) Try patched .h5
        try:
            patched_h5 = patch_h5_file(primary_h5)
            m = _try_load_model(patched_h5, "tf.keras (patched h5)", errs)
            if m is not None:
                return m, None
            m = _try_load_model(patched_h5, "keras (patched h5)", errs)
            if m is not None:
                return m, None
        except Exception as e:
            errs.append(f"patch_h5_file failed: {e}")
    else:
        errs.append(f".h5 not found ({primary_h5})")

    # 2) Try .keras directly, then patched
    if os.path.exists(fallback_keras):
        m = _try_load_model(fallback_keras, "tf.keras", errs)
        if m is not None:
            return m, None
        m = _try_load_model(fallback_keras, "keras", errs)
        if m is not None:
            return m, None
        try:
            patched = patch_keras_file(fallback_keras)
            m = _try_load_model(patched, "tf.keras (patched keras)", errs)
            if m is not None:
                return m, None
            m = _try_load_model(patched, "keras (patched keras)", errs)
            if m is not None:
                return m, None
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
    f = hog(
        img_gray_01, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm="L2-Hys"
    )
    return f.reshape(1, -1).astype(np.float32)

def get_gabor_feature(img_gray_01: np.ndarray, freqs=(0.1, 0.3, 0.5)) -> np.ndarray:
    vals = []
    for freq in freqs:
        real, imag = gabor(img_gray_01, frequency=freq)
        vals += [real.mean(), real.var(), imag.mean(), imag.var()]
    return np.array(vals, dtype=np.float32).reshape(1, -1)

def prep_cnn_input(img_gray_01: np.ndarray) -> np.ndarray:
    return np.expand_dims(img_gray_01, axis=(0, -1)).astype(np.float32)  # (1,H,W,1)

def prep_mnet_input(img_gray_01: np.ndarray) -> np.ndarray:
    x = np.repeat(np.expand_dims(img_gray_01, axis=-1), 3, axis=-1)      # (H,W,3)
    x = preprocess_input(x * 255.0)
    return np.expand_dims(x, axis=0).astype(np.float32)                   # (1,H,W,3)

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800;900&display=swap');

/* ═══ Keyframe Animations ═══ */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.6; }
    50% { transform: translateY(-20px) rotate(5deg); opacity: 1; }
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(230,57,70,0.3); }
    50% { box-shadow: 0 0 40px rgba(230,57,70,0.6), 0 0 60px rgba(230,57,70,0.2); }
}
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes barGrow {
    from { width: 0%; }
}
@keyframes ripple {
    0% { transform: scale(1); opacity: 0.4; }
    100% { transform: scale(1.5); opacity: 0; }
}

/* ═══ Root Variables ═══ */
:root {
    --primary: #E63946;
    --primary-dark: #C1121F;
    --primary-glow: rgba(230,57,70,0.3);
    --accent: #457B9D;
    --accent-light: #A8DADC;
    --accent-vivid: #2EC4B6;
    --dark: #1D3557;
    --dark-rich: #0D1B2A;
    --light: #F1FAEE;
    --warm: #F4A261;
    --warm-dark: #E76F51;
    --purple: #6C63FF;
    --bg-card: rgba(255,255,255,0.92);
    --bg-glass: rgba(255,255,255,0.15);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ═══ Animated Background ═══ */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #F1FAEE, #E8F4F8, #FFF0E8, #F0E6FF, #E8F8F0);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}

/* Floating particles via pseudo-elements on main block */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 10%;
    left: 5%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(230,57,70,0.08) 0%, transparent 70%);
    border-radius: 50%;
    animation: float 8s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    bottom: 15%;
    right: 10%;
    width: 250px;
    height: 250px;
    background: radial-gradient(circle, rgba(69,123,157,0.08) 0%, transparent 70%);
    border-radius: 50%;
    animation: float 10s ease-in-out infinite reverse;
    pointer-events: none;
    z-index: 0;
}

/* ═══ Sidebar Styling ═══ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #1D3557 60%, #264573 100%) !important;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] hr {
    border-color: #d70f32 !important;
}
[data-testid="stSidebar"] strong {
    color: #A8DADC !important;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #d70f32 !important;
}

/* ═══ Hero Banner ═══ */
.hero-banner {
    background: linear-gradient(-45deg, #0D1B2A, #1D3557, #457B9D, #E63946, #F4A261);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
    border-radius: 20px;
    padding: 3rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 12px 40px rgba(29,53,87,0.3);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 1px, transparent 1px);
    background-size: 30px 30px;
    animation: float 20s linear infinite;
    pointer-events: none;
}
.hero-banner h1 {
    font-family: 'Poppins', sans-serif;
    color: white;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    position: relative;
    z-index: 1;
}
.hero-banner p {
    color: rgba(168,218,220,0.9);
    font-size: 1.1rem;
    margin: 0.6rem 0 0 0;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

/* ═══ Stat Cards ═══ */
.stat-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(168,218,220,0.25);
    border-radius: 16px;
    padding: 1.3rem 1rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: slideUp 0.6s ease forwards;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--primary), var(--warm));
    background-size: 200% 100%;
    animation: shimmer 3s ease infinite;
}
.stat-card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 8px 30px rgba(69,123,157,0.15);
    border-color: var(--accent-light);
}

/* ═══ Blood Group Result Badge ═══ */
.blood-badge {
    display: inline-block;
    background: linear-gradient(135deg, #E63946, #C1121F, #E63946);
    background-size: 200% 200%;
    animation: gradientShift 3s ease infinite, pulse 2s ease-in-out infinite;
    color: white;
    font-family: 'Poppins', sans-serif;
    font-size: 3.5rem;
    font-weight: 900;
    padding: 0.7rem 2.5rem;
    border-radius: 20px;
    letter-spacing: 2px;
    margin: 0.5rem 0;
    text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    position: relative;
}
.blood-badge::before {
    content: '';
    position: absolute;
    inset: -3px;
    border-radius: 23px;
    background: linear-gradient(135deg, #E63946, #F4A261, #E63946);
    z-index: -1;
    animation: gradientShift 4s ease infinite;
    background-size: 200% 200%;
}

/* ═══ Model Result Cards ═══ */
.model-card {
    background: var(--bg-card);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    border-left: 4px solid var(--accent);
    box-shadow: 0 3px 15px rgba(0,0,0,0.05);
    margin-bottom: 0.7rem;
    transition: all 0.3s ease;
    animation: slideUp 0.5s ease forwards;
}
.model-card:hover {
    transform: translateX(6px);
    box-shadow: 0 5px 20px rgba(69,123,157,0.12);
    border-left-width: 6px;
}
.model-card.ensemble {
    border-left-color: var(--primary);
    background: linear-gradient(135deg, rgba(230,57,70,0.04), rgba(244,162,97,0.04), rgba(255,255,255,0.95));
    border-left-width: 5px;
}
.model-card.ensemble:hover {
    box-shadow: 0 5px 25px rgba(230,57,70,0.12);
}
.model-card .model-name {
    font-weight: 600;
    color: var(--dark);
    font-size: 0.95rem;
}
.model-card .model-prediction {
    font-family: 'Poppins', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--primary);
}
.model-card .model-conf {
    font-size: 0.85rem;
    color: #888;
}

/* ═══ Status Pill ═══ */
.status-pill {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 3px 12px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.status-pill.loaded {
    background: linear-gradient(135deg, #D4EDDA, #C3E6CB);
    color: #155724;
    box-shadow: 0 2px 6px rgba(40,167,69,0.15);
}
.status-pill.unavailable {
    background: linear-gradient(135deg, #F8D7DA, #F1AEB5);
    color: #721C24;
}

/* ═══ Probability Bars ═══ */
.prob-bar-container {
    margin: 6px 0;
    animation: slideUp 0.5s ease forwards;
}
.prob-bar-bg {
    background: linear-gradient(90deg, #E9ECEF, #F8F9FA);
    border-radius: 10px;
    height: 28px;
    overflow: hidden;
    position: relative;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.08);
}
.prob-bar-fill {
    height: 100%;
    border-radius: 10px;
    animation: barGrow 1s ease forwards;
    display: flex;
    align-items: center;
    padding-left: 10px;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    position: relative;
    overflow: hidden;
}
.prob-bar-fill::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255,255,255,0.2),
        transparent
    );
    animation: shimmer 2s ease infinite;
    background-size: 200% 100%;
}
.prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #444;
    margin-bottom: 3px;
    font-weight: 500;
}

/* ═══ Upload Area ═══ */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--accent) !important;
    border-radius: 20px !important;
    padding: 1.2rem !important;
    background: rgba(69,123,157,0.03) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--primary) !important;
    background: rgba(230,57,70,0.03) !important;
    box-shadow: 0 4px 20px rgba(230,57,70,0.08) !important;
}

/* ═══ Section Divider ═══ */
.section-divider {
    border: none;
    height: 2px;
    background: linear-gradient(to right, transparent, var(--accent-light), var(--primary), var(--accent-light), transparent);
    margin: 1.8rem 0;
    opacity: 0.6;
}

/* ═══ Image Preview ═══ */
[data-testid="stImage"] {
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 6px 25px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}
[data-testid="stImage"]:hover {
    transform: scale(1.02);
}

/* ═══ Buttons ═══ */
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #E63946, #C1121F) !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 15px rgba(230,57,70,0.3) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(230,57,70,0.4) !important;
}

/* ═══ Empty State ═══ */
.empty-state {
    text-align: center;
    padding: 3.5rem 1rem;
    animation: fadeIn 1s ease;
}
.empty-state .empty-icon {
    font-size: 4.5rem;
    margin-bottom: 1rem;
    animation: float 3s ease-in-out infinite;
}

/* ═══ Footer ═══ */
.footer-text {
    text-align: center;
    color: #aaa;
    font-size: 0.8rem;
    margin-top: 2.5rem;
    padding: 1.2rem 0;
    border-top: 1px solid rgba(168,218,220,0.2);
}
.footer-text a {
    color: var(--accent) !important;
    text-decoration: none;
    font-weight: 600;
    transition: color 0.2s;
}
.footer-text a:hover {
    color: var(--primary) !important;
}

/* ═══ Section Headers ═══ */
[data-testid="stMarkdown"] h3 {
    color: var(--dark) !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 700 !important;
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
        <div style="font-size:1.5rem;">{icon}</div>
        <div style="font-size:0.85rem;font-weight:600;color:#1D3557;margin:4px 0;">{name}</div>
        <span class="status-pill {status_cls}">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

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

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Centered predict button
    col_l, col_btn, col_r = st.columns([1, 2, 1])
    with col_btn:
        predict_clicked = st.button("🔬 Analyze Fingerprint", use_container_width=True, type="primary")

    if predict_clicked:
        classes = label_encoder.classes_
        n_cls = len(classes)

        with st.spinner("🧬 Running analysis across all models..."):
            # Classical model probs
            p_rf_hog = rf_hog.predict_proba(get_hog_feature(enhanced))[0]
            p_rf_gabor = rf_gabor.predict_proba(get_gabor_feature(enhanced))[0]

            # Deep model probs
            if cnn_model is not None:
                p_cnn = cnn_model.predict(prep_cnn_input(enhanced), verbose=0)[0]
            else:
                p_cnn = np.zeros(n_cls, dtype=np.float32)

            if mnet_model is not None:
                p_mnet = mnet_model.predict(prep_mnet_input(enhanced), verbose=0)[0]
            else:
                p_mnet = np.zeros(n_cls, dtype=np.float32)

            # Ensemble
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
        st.markdown("### 🎯 Final Prediction")

        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
        with res_col2:
            st.markdown(f"""
            <div style="text-align:center;padding:1.5rem 0;">
                <div class="blood-badge">{final_label}</div>
                <div style="margin-top:12px;font-size:1.1rem;color:#555;">
                    Confidence: <strong style="color:#E63946;">{final_conf:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Individual Model Results ──
        st.markdown("### 📊 Model-by-Model Predictions")

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
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span class="model-name">{name}</span>
                        <span class="status-pill {status_cls}" style="margin-left:8px;">{status_text}</span>
                    </div>
                    <div style="text-align:right;">
                        <span class="model-prediction">{label}</span>
                        <span class="model-conf" style="margin-left:8px;">{conf:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Probability Distribution ──
        st.markdown("### 📈 Blood Group Probability Distribution")

        # Sort by probability
        sorted_indices = np.argsort(p_ens)[::-1]
        colors = ["#E63946", "#C1121F", "#457B9D", "#1D3557", "#6C757D", "#A8DADC", "#90BE6D", "#F9C74F"]

        prob_html = ""
        for rank, idx in enumerate(sorted_indices):
            bg = classes[idx]
            pct = p_ens[idx] * 100
            bar_color = colors[rank % len(colors)]
            prob_html += f"""
            <div class="prob-bar-container">
                <div class="prob-bar-label">
                    <span><strong>{bg}</strong></span>
                    <span>{pct:.1f}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{max(pct, 2)}%;background:{bar_color};">
                    </div>
                </div>
            </div>
            """

        st.markdown(prob_html, unsafe_allow_html=True)

        # ── Detailed Data Table ──
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

            st.markdown("**Full Ensemble Probabilities:**")
            prob_df = pd.DataFrame({
                "Blood Group": classes,
                "Probability (%)": np.round(p_ens * 100, 2)
            }).sort_values("Probability (%)", ascending=False)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
else:
    # ── Empty State ──
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">👆</div>
        <h3 style="color:#1D3557;margin-bottom:0.5rem;">Upload a Fingerprint Image</h3>
        <p style="color:#888;font-size:1rem;">
            Drag and drop or click above to upload a fingerprint image.<br>
            The AI will analyze it and predict the blood group.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div class="footer-text">
    Built with ❤️ using Streamlit & TensorFlow  •
    <a href="https://github.com/Kishanjee7/BloodGroupPredictionUsingFingerprint" target="_blank" style="color:#457B9D;">GitHub</a>
</div>
""", unsafe_allow_html=True)
