# 🩸 Blood Group Prediction Using Fingerprint

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bloodgrouppredictionusingfingerprint.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)

A machine learning system that predicts blood groups from fingerprint images using an ensemble of classical ML and deep learning models.

🔗 **Live Demo**: [bloodgrouppredictionusingfingerprint.streamlit.app](https://bloodgrouppredictionusingfingerprint.streamlit.app/)

---

## 📌 Overview

This project explores the correlation between fingerprint patterns and blood groups using multiple machine learning approaches. It combines hand-crafted feature extraction with deep learning to achieve robust predictions across **8 blood group classes** (A+, A−, B+, B−, AB+, AB−, O+, O−).

## 🏗️ Architecture


![tuvyu (1)](https://github.com/user-attachments/assets/6b7b3861-fc95-4984-a8a1-db7a94d053a0)

## 🚀 Models Used

| Model | Type | Input | Description |
|-------|------|-------|-------------|
| **RF-HOG** | Classical ML | HOG features | Random Forest on Histogram of Oriented Gradients |
| **RF-Gabor** | Classical ML | Gabor features | Random Forest on Gabor filter responses |
| **Custom CNN** | Deep Learning | 128×128 grayscale | Lightweight convolutional neural network |
| **MobileNetV2** | Deep Learning | 128×128 RGB | Transfer learning with pretrained MobileNetV2 |
| **Ensemble** | Meta-Learner | All model outputs | Logistic Regression stacking all 4 models |

## 🛠️ Tech Stack

- **ML/DL**: TensorFlow 2.15, Keras, scikit-learn
- **Feature Extraction**: scikit-image (HOG, Gabor filters)
- **Image Processing**: OpenCV, Pillow
- **UI**: Streamlit
- **Deployment**: Streamlit Cloud

## 📁 Project Structure

```
├── app.py                  # Streamlit web application
├── new_code.ipynb          # Training notebook
├── requirements.txt        # Python dependencies
├── .python-version         # Python version pin (3.11)
└── saved_models/
    ├── rf_hog.joblib        # Random Forest (HOG)
    ├── rf_gabor.joblib      # Random Forest (Gabor)
    ├── cnn_model.h5         # Custom CNN
    ├── cnn_model.keras      # Custom CNN (Keras format)
    ├── mobilenet_model.h5   # MobileNetV2
    ├── mobilenet_model.keras # MobileNetV2 (Keras format)
    ├── ensemble_logreg.joblib # Ensemble meta-learner
    └── label_encoder.joblib   # Label encoder
```

## ⚡ Quick Start

### Run Locally

```bash
# Clone the repository
git clone https://github.com/Kishanjee7/BloodGroupPredictionUsingFingerprint.git
cd BloodGroupPredictionUsingFingerprint

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### How to Use

1. Open the app (locally or via the [live demo](https://bloodgrouppredictionusingfingerprint.streamlit.app/))
2. Upload a fingerprint image (PNG, JPG, JPEG, or BMP)
3. View the enhanced fingerprint preview
4. Click **Predict** to see results from all models
5. Review the comparison table and probability distribution

## 📊 Features

- **Multi-Model Predictions** — Compare outputs from 5 different models side by side
- **Image Enhancement** — Automatic CLAHE-based fingerprint enhancement
- **Ensemble Voting** — Final prediction combines all model outputs via stacking
- **Confidence Scores** — Probability distribution across all blood groups
- **Runtime Diagnostics** — Built-in health checks for model availability

## 📄 License

This project is for academic purposes.

## 👤 Author

**Kishanjee7** — [GitHub Profile](https://github.com/Kishanjee7)
