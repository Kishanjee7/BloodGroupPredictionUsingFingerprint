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
st.set_page_config(page_title="Blood Group Predictor (All Models)", layout="centered")
st.title("Fingerprint Blood Group Prediction (All Models)")

with st.expander("Runtime diagnostics"):
    st.write(f"TensorFlow version: {tf.__version__}")
    st.write(f"Standalone keras available: {standalone_keras is not None}")
    st.write(f"Model directory: {MODEL_DIR}")
    diag_rows = []
    for k, p in PATHS.items():
        diag_rows.append({"artifact": k, "path": p, "exists": os.path.exists(p)})
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

try:
    rf_hog, rf_gabor, cnn_model, mnet_model, ensemble_model, label_encoder, cnn_err, mnet_err = load_artifacts()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

if cnn_err:
    st.warning(f"CNN unavailable: {cnn_err}")
if mnet_err:
    st.warning(f"MobileNetV2 unavailable: {mnet_err}")

uploaded = st.file_uploader("Upload fingerprint image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("L")
    orig = np.array(pil_img).astype(np.float32) / 255.0
    orig = cv2.resize(orig, IMG_SIZE)
    enhanced = enhance_fingerprint(orig)

    c1, c2 = st.columns(2)
    with c1:
        st.image(orig, caption="Original", use_container_width=True, clamp=True)
    with c2:
        st.image(enhanced, caption="Enhanced", use_container_width=True, clamp=True)

    if st.button("Predict"):
        classes = label_encoder.classes_
        n_cls = len(classes)

        # Classical model probs
        p_rf_hog = rf_hog.predict_proba(get_hog_feature(enhanced))[0]
        p_rf_gabor = rf_gabor.predict_proba(get_gabor_feature(enhanced))[0]

        # Deep model probs (or zeros if unavailable)
        if cnn_model is not None:
            p_cnn = cnn_model.predict(prep_cnn_input(enhanced), verbose=0)[0]
        else:
            p_cnn = np.zeros(n_cls, dtype=np.float32)

        if mnet_model is not None:
            p_mnet = mnet_model.predict(prep_mnet_input(enhanced), verbose=0)[0]
        else:
            p_mnet = np.zeros(n_cls, dtype=np.float32)

        # Ensemble input handling
        base_stack = np.concatenate([p_rf_hog, p_rf_gabor, p_cnn, p_mnet], axis=0).reshape(1, -1)
        expected = getattr(ensemble_model, "n_features_in_", base_stack.shape[1])

        if base_stack.shape[1] < expected:
            pad = np.zeros((1, expected - base_stack.shape[1]), dtype=np.float32)
            stack = np.concatenate([base_stack, pad], axis=1)
        else:
            stack = base_stack[:, :expected]

        p_ens = ensemble_model.predict_proba(stack)[0]

        # Table
        rows = []
        for name, prob, enabled in [
            ("RF-HOG", p_rf_hog, True),
            ("RF-Gabor", p_rf_gabor, True),
            ("CNN", p_cnn, cnn_model is not None),
            ("MobileNetV2", p_mnet, mnet_model is not None),
            ("Ensemble (Final)", p_ens, True),
        ]:
            label, conf = top_prediction(prob, classes)
            rows.append({
                "Model": name,
                "Status": "Loaded" if enabled else "Unavailable",
                "Predicted Blood Group": label,
                "Confidence (%)": round(conf, 2),
            })

        st.subheader("Model Predictions")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        final_label, final_conf = top_prediction(p_ens, classes)
        st.success(f"Final Predicted Blood Group: **{final_label}**")
        st.info(f"Final Confidence: **{final_conf:.2f}%**")

        st.subheader("Final Ensemble Class Probabilities")
        prob_df = pd.DataFrame({
            "Blood Group": classes,
            "Probability (%)": np.round(p_ens * 100, 2)
        }).sort_values("Probability (%)", ascending=False)
        st.dataframe(prob_df, use_container_width=True)
else:
    st.write("Upload an image to begin.")
