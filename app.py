"""Streamlit app: upload a lung-extracted chest CT slice -> prediction, confidence, Grad-CAM.

Run with:  streamlit run app.py
Research prototype only. Not a medical device.
"""
from pathlib import Path

import numpy as np
import streamlit as st
import keras
from PIL import Image

from gradcam_utils import preprocess, check_input, make_gradcam, overlay, predict_prob

st.set_page_config(page_title="COVID-19 CT Explainer", page_icon="🫁", layout="wide")

MODEL_DIR = Path(__file__).parent / "models"
MODEL_FILES = {
    "ResNet50 (primary)": "resnet50_covid.keras",
    "MobileNetV2 (lightweight)": "mobilenetv2_covid.keras",
}
available = {name: f for name, f in MODEL_FILES.items() if (MODEL_DIR / f).exists()}


@st.cache_resource(show_spinner="Loading model...")
def load(path):
    model = keras.models.load_model(path, compile=False)
    return model, make_gradcam(model)


# ---------------------------------------------------------------- header
st.title("🫁 Explainable COVID-19 detection from chest CT")
st.warning(
    "**Research prototype - not a medical device.** It must not be used for diagnosis or "
    "treatment decisions. It was trained on lung-extracted CT slices from one public dataset "
    "and has not been validated on outside data."
)

if not available:
    st.error(f"No model file found. Put `resnet50_covid.keras` (and optionally "
             f"`mobilenetv2_covid.keras`) in `{MODEL_DIR}`.")
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", list(available))
    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, 0.50, 0.01,
        help="Probability above which a slice is flagged. For ResNet50, 0.30 reached 98% "
             "recall on validation data (fewer missed cases, more false alarms).")
    show_negative = st.checkbox(
        "Show heatmap even when no COVID-like pattern is flagged", value=False,
        help="Off by default: in our analysis, heatmaps of low-evidence (normal) predictions "
             "were often unspecific and drifted outside the lungs.")

# ---------------------------------------------------------------- input
uploaded = st.file_uploader(
    "Upload a lung-extracted chest CT slice (PNG or JPG)", type=["png", "jpg", "jpeg"])
st.caption("Expected input: a single 2D slice with the lungs isolated on a black background, "
           "like the 'Preprocessed CT scans' images of the training dataset.")

if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

try:
    pil_img = Image.open(uploaded)
    pil_img.load()
except Exception:
    st.error("This file could not be read as an image.")
    st.stop()

x = preprocess(pil_img)
gray = x.numpy()[..., 0]

ok, problems = check_input(gray)
if not ok:
    st.warning("This image does not look like the lung-extracted slices the model was trained "
               "on: " + "; ".join(problems) + ". The prediction may be meaningless.")

# ---------------------------------------------------------------- prediction
model, gradcam = load(str(MODEL_DIR / available[model_name]))
prob = predict_prob(model, x)
flagged = prob >= threshold
confidence = prob if flagged else 1.0 - prob

c1, c2, c3 = st.columns(3)
c1.metric("Result", "COVID-like" if flagged else "Not flagged")
c2.metric("P(COVID-like pattern)", f"{prob:.1%}")
c3.metric("Confidence in this result", f"{confidence:.1%}")
st.progress(min(max(prob, 0.0), 1.0))
st.caption(f"Threshold {threshold:.2f}. The percentages are the model's raw output probabilities; "
           "they have not been calibrated and should not be read as the chance of disease.")

# ---------------------------------------------------------------- Grad-CAM
st.subheader("Where the model looked (Grad-CAM)")
left, right = st.columns(2)
shown = (gray / max(float(gray.max()), 1.0) * 255).astype(np.uint8)
left.image(shown, caption="Input (resized to 224 x 224)", use_container_width=True)

if flagged or show_negative:
    cam, _ = gradcam(x, sign=1)
    right.image(overlay(gray, cam), caption="Red = regions that pushed the model toward "
                "'COVID-like pattern'", use_container_width=True)
    st.caption("The heatmap shows model attention, not a verified lesion location. It is coarse "
               "(7 x 7 grid) and can fall partly outside the lungs.")
else:
    right.info("No heatmap shown because no COVID-like pattern was flagged. "
               "Enable it in the sidebar if you want to see it anyway.")

with st.expander("About this model and its limits"):
    st.markdown(
        "- Trained on lung-extracted CT slices from two hospitals (public Kaggle dataset); "
        "task: *COVID-like pattern visible* vs *normal* on a single slice.\n"
        "- ResNet50 on the held-out test set (similarity-grouped split): AUC 0.994, "
        "sensitivity 96.6%, specificity 95.6% at threshold 0.5.\n"
        "- No external validation, no patient-level evaluation, no lesion annotations.\n"
        "- Slices of mid-chest level were the hardest; errors were concentrated in a few "
        "image clusters.\n"
        "- Not intended for clinical use."
    )
