from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bank Account Prediction", page_icon="🏦", layout="centered")

ART_DIR = Path(__file__).parent
MODEL_PATH = ART_DIR / "gb_model.pkl"
FEATS_PATH = ART_DIR / "features.json"
SCALER_PATH = ART_DIR / "scaler.pkl"     
MAPS_PATH = ART_DIR / "maps.json"       

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(FEATS_PATH, "r") as f:
        features = json.load(f)

    scaler = None
    if SCALER_PATH.exists():
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception:
            scaler = None

    maps = None
    if MAPS_PATH.exists():
        try:
            with open(MAPS_PATH, "r") as f:
                maps = json.load(f)
        except Exception:
            maps = None

    return model, features, scaler, maps

st.title("🏦 African Financial Inclusion — Gradient Boosting")

try:
    model, FEATURES, scaler, maps = load_artifacts()
    st.success("Artifacts loaded.")
except Exception as e:
    st.error(f"Could not load artifacts: {e}")
    st.stop()

st.write("Enter the features, then click **Predict**. "
         "If `maps.json` is present, categorical fields will show as dropdowns. "
         "If `scaler.pkl` is present, raw values will be scaled exactly as in training.")

# --- Build inputs ---

# --- UI config for categorical dropdowns ---
# Edit the VALUE on the right to match how you encoded them during training.
CATEGORICAL_OPTIONS = {
    "cellphone_access": {"Yes": 1, "No": 0},
    "location_type": {"Urban": 1, "Rural": 0},
    "gender_of_respondent": {"Male": 1, "Female": 0},
    "marital_status": {
        "married": 1, "single": 0, "divorced/separated": 2, "widowed": 3
    },
    "education_level": {
        "No formal education": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3
    },
    "job_type": {
        "Government": 0, "Self employed": 1, "Formally employed": 2,
        "Informally employed": 3, "Farming": 4, "Other": 5, "No job": 6
    },
}


HIDDEN_FEATURE_DEFAULTS = {
    "relationship_with_head": 0  
}

# Build the list of features we will expose to users (drop relationship_with_head)
FEATURES_EXPOSED = [f for f in FEATURES if f not in HIDDEN_FEATURE_DEFAULTS]

st.header("📋 Inputs")
inputs = {}


for feat in FEATURES_EXPOSED:
    if feat in CATEGORICAL_OPTIONS:
        choice = st.selectbox(feat, options=list(CATEGORICAL_OPTIONS[feat].keys()))
        inputs[feat] = CATEGORICAL_OPTIONS[feat][choice]
    else:
        # numeric fallback
        inputs[feat] = st.number_input(feat, value=0.0, step=0.1)

for hidden_feat, default_val in HIDDEN_FEATURE_DEFAULTS.items():
    if hidden_feat in FEATURES:   # only add if the model expects it
        inputs[hidden_feat] = default_val


X_row = pd.DataFrame([inputs]).reindex(columns=FEATURES, fill_value=0.0)


threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)

if st.button("Predict"):
    try:
        X = pd.DataFrame([inputs]).reindex(columns=FEATURES, fill_value=0.0)
        proba = float(model.predict_proba(X)[:, 1])
        pred = int(proba >= threshold)

        st.subheader("Result")
        st.metric("Will open bank account?", "YES" if pred == 1 else "NO")
        st.write(f"Probability (class=1): **{proba:.3f}**  •  Threshold: **{threshold:.2f}**")

        with st.expander("Show model input row"):
            st.dataframe(X)

        if hasattr(model, "feature_importances_"):
            st.write("Top features (by importance):")
            imp = (pd.Series(model.feature_importances_, index=FEATURES)
                   .sort_values(ascending=False).head(10))
            st.write(imp)

    except Exception as e:
        st.error(f"Prediction failed: {e}")