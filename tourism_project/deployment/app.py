# ---- Streamlit-friendly startup: rely on Dockerfile to set HOME & dirs ----
import os
import io
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# ---------------------------
# Config & model loading
# ---------------------------
st.set_page_config(page_title="Visit With Us — Wellness Package Predictor", page_icon="🧭", layout="centered")

# Env vars (set at deploy time)
MODEL_REPO = os.getenv("MODEL_REPO", "Yashwanthsairam/package-tourism-predict")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "best_tourism_pipeline.joblib")
CLASSIFICATION_THRESHOLD = float(os.getenv("THRESHOLD", "0.45"))  # can be adjusted by user too

@st.cache_resource(show_spinner=True)
def load_model():
    """
    Downloads and loads the trained pipeline.
    Expectation: the artifact is a scikit-learn Pipeline that includes preprocessing + XGBClassifier.
    """
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, repo_type="model")
    model = joblib.load(path)
    return model

def ensure_columns(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """Add any missing columns with sensible defaults (zeros) to avoid KeyError."""
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0
    return df

# ---------------------------
# UI
# ---------------------------
st.title("🧭 Visit With Us — Wellness Tourism Purchase Prediction")

st.markdown(
    """
Predict whether a customer is likely to purchase the **Wellness Tourism Package**.
Provide customer & interaction details below, and the model will return a binary prediction
along with the predicted probability.

*This app uses a trained XGBoost model wrapped in a scikit-learn Pipeline (preprocessing included).*
"""
)

with st.expander("Set prediction options", expanded=False):
    threshold = st.slider("Classification threshold", 0.05, 0.95, CLASSIFICATION_THRESHOLD, 0.01)
    st.caption("Scores ≥ threshold → **Positive (Will Purchase)**; otherwise **Negative**.")

st.subheader("Customer Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)
    typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Other"])
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_persons = st.number_input("Number Of Person Visiting", min_value=1, max_value=20, value=2, step=1)
    pref_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5], index=2)
    num_trips = st.number_input("Number Of Trips (per year)", min_value=0, max_value=100, value=2, step=1)
    num_children = st.number_input("Number Of Children Visiting (<5y)", min_value=0, max_value=10, value=0, step=1)

st.subheader("Documents & Assets")
col3, col4 = st.columns(2)
with col3:
    passport = st.selectbox("Has Passport", [0, 1], index=1)
    owncar = st.selectbox("Owns Car", [0, 1], index=0)
    designation = st.selectbox(
        "Designation",
        ["Junior", "Executive", "Manager", "Senior Manager", "AVP", "VP", "Other"],
        index=2
    )
with col4:
    monthly_income = st.number_input("Monthly Income", min_value=0, max_value=10_000_000, value=60_000, step=1_000)
    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "Super Deluxe", "King", "Queen", "Other"]
    )
    pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=4, step=1)

st.subheader("Sales Interaction")
col5, col6 = st.columns(2)
with col5:
    num_followups = st.number_input("Number Of Followups", min_value=0, max_value=20, value=2, step=1)
with col6:
    duration_pitch = st.number_input("Duration Of Pitch (minutes)", min_value=0, max_value=240, value=20, step=1)

# Assemble a single-row dataframe in the schema expected by your training pipeline
input_data = pd.DataFrame([{
    # Numeric / binary
    "Age": age,
    "NumberOfPersonVisiting": num_persons,
    "PreferredPropertyStar": pref_star,
    "NumberOfTrips": num_trips,
    "NumberOfChildrenVisiting": num_children,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_score,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_pitch,
    "Passport": int(passport),
    "OwnCar": int(owncar),
    # Categoricals
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "Occupation": occupation,
    "Gender": gender,
    "MaritalStatus": marital,
    "Designation": designation,
    "ProductPitched": product_pitched,
}])

st.markdown("#### Preview")
st.dataframe(input_data, use_container_width=True)

# Load model lazily
with st.spinner("Loading model…"):
    try:
        model = load_model()
        st.success(f"Model loaded from `{MODEL_REPO}/{MODEL_FILENAME}`")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# Allow CSV batch scoring
st.markdown("---")
st.subheader("Batch Scoring (optional)")
csv_file = st.file_uploader("Upload a CSV with the same schema (no target column)", type=["csv"])

def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    """Run predictions with probability and label (using chosen threshold)."""
    # If your training pipeline included preprocessing, you can pass the raw columns directly.
    # Otherwise, you must replicate the exact preprocessing here.
    proba = model.predict_proba(df)[:, 1]
    label = (proba >= threshold).astype(int)
    out = df.copy()
    out["purchase_proba"] = proba
    out["purchase_pred"] = label
    return out

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("🔮 Predict (single row above)"):
        try:
            # Align columns if model expects extras (rare if pipeline is correct)
            expected = getattr(getattr(model, "feature_names_in_", None), "tolist", lambda: None)()
            if expected:
                input_aligned = ensure_columns(input_data.copy(), expected)
                input_aligned = input_aligned[expected]
            else:
                input_aligned = input_data
            result = predict_df(input_aligned)
            pred = "Will Purchase ✅" if result.loc[0, "purchase_pred"] == 1 else "Unlikely to Purchase ❌"
            st.metric("Prediction", pred, help=f"Probability: {result.loc[0, 'purchase_proba']:.3f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with col_btn2:
    if csv_file is not None and st.button("📦 Predict (uploaded CSV)"):
        try:
            df_in = pd.read_csv(io.BytesIO(csv_file.read()))
            expected = getattr(getattr(model, "feature_names_in_", None), "tolist", lambda: None)()
            if expected:
                df_in = ensure_columns(df_in, expected)
                df_in = df_in[expected]
            results = predict_df(df_in)
            st.success("Batch predictions complete.")
            st.dataframe(results.head(50), use_container_width=True)
            st.download_button("⬇️ Download predictions", data=results.to_csv(index=False), file_name="predictions.csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

st.caption(
    "Model & UI tailored to the *Visit With Us* Wellness Package project. "
    "Feature names mirror the dataset fields used in training."
)
