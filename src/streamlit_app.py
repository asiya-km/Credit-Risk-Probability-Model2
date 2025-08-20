import streamlit as st
import pandas as pd
import mlflow.sklearn
from explain import plot_global_shap, plot_local_shap
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

st.set_page_config(page_title="Credit Risk Probability Dashboard", layout="centered")
st.title("Credit Risk Probability Model Demo")
st.write("Upload borrower application data or enter features manually to see risk probability and explanations.")

# Load model from MLflow registry
def load_model():
    model_uri = "models:/credit-risk-best-model/Production"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        model_uri = "models:/credit-risk-best-model/1"
        model = mlflow.sklearn.load_model(model_uri)
    return model

model = load_model()

# Load reference data and SHAP global plot for demo
ref_data_path = os.path.join(os.path.dirname(__file__), '../data/processed/processed_with_target.csv')
global_shap_img = os.path.join(os.path.dirname(__file__), '../data/processed/shap_global.png')
if os.path.exists(ref_data_path):
    df = pd.read_csv(ref_data_path)
    feature_cols = [c for c in df.columns if c not in ['is_high_risk','CustomerId']]
else:
    df = pd.DataFrame()
    feature_cols = []

# User data input
st.header("Input Borrower Data")
if df.shape[0] > 0:
    sample_dict = {c: df[c].iloc[0] for c in feature_cols}
else:
    sample_dict = {}

input_mode = st.radio("Choose input mode:", ["Manual Entry", "Upload CSV"])

if input_mode == "Manual Entry" and feature_cols:
    user_data = {}
    for f in feature_cols:
        val = st.number_input(f, value=float(sample_dict.get(f, 0)))
        user_data[f] = val
    input_df = pd.DataFrame([user_data])
elif input_mode == "Upload CSV":
    upload_file = st.file_uploader("Upload feature CSV", type=["csv"])
    if upload_file is not None:
        input_df = pd.read_csv(upload_file)
    else:
        input_df = None
else:
    input_df = None

# Predict and explain
if input_df is not None and not input_df.empty:
    st.subheader("Prediction & Explanation")
    pred_proba = model.predict_proba(input_df)[:,1]
    st.write(f"**Predicted Risk Probability:** {pred_proba[0]:.3f}")

    # SHAP local explanation
    with st.spinner("Generating SHAP explanation..."):
        try:
            explainer = shap.Explainer(model, df[feature_cols].head(100) if df.shape[0]>100 else df[feature_cols])
            shap_values = explainer(input_df)
            # Plot SHAP force plot
            st.write("#### SHAP Local Feature Importance:")
            fig, ax = plt.subplots(figsize=(10,3))
            shap.plots.waterfall(shap_values[0], show=False, max_display=10)
            st.pyplot(fig)
        except Exception as ex:
            st.error(f"Could not generate SHAP local plot: {ex}")

if os.path.exists(global_shap_img):
    st.markdown("---")
    st.write("#### Model-Wide Feature Importance (SHAP):")
    st.image(global_shap_img)
    st.caption("This shows which features most influence risk, averaged across all borrowers.")

st.markdown("---")
st.write(
    "Model: `credit-risk-best-model` | Powered by MLflow, SHAP, Streamlit"
)
