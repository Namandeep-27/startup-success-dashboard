import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model and scaler
model = joblib.load("stacked_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define columns to drop
drop_cols = [
    'id', 'name', 'city', 'zip_code', 'founded_at', 'closed_at',
    'first_funding_at', 'last_funding_at', 'object_id',
    'category_code', 'status', 'labels', 'state', 'state_code'
]

# Streamlit UI
st.set_page_config(page_title="Startup Success Predictor", layout="centered")
st.title("🚀 Startup Success Probability Predictor")
st.markdown("Upload your startup dataset or use the default prediction file from the project.")

# Upload CSV (optional)
uploaded_file = st.file_uploader("📤 Upload cleaned test CSV", type=["csv"])

# Load Data
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.success("Custom file uploaded and loaded.")
elif os.path.exists("predicted_success_probabilities.csv"):
    test_df = pd.read_csv("predicted_success_probabilities.csv")
    st.info("No file uploaded. Using default file from repository.")
else:
    st.error("❌ No file uploaded and default file not found.")
    st.stop()

# Validate presence of 'name' column
if 'name' not in test_df.columns:
    st.error("❌ Dataset must contain a 'name' column.")
    st.stop()

# Step 1: Save startup names
startup_names = test_df['name']

# Step 2: Clean and scale data
X_test = test_df.drop(columns=drop_cols, errors='ignore')
X_test = X_test.fillna(X_test.median(numeric_only=True))
X_scaled = scaler.transform(X_test)

# Step 3: Predict
probs = model.predict_proba(X_scaled)[:, 1]
probs_rounded = np.round(probs, 2)

# Combine results
result_df = pd.DataFrame({
    "Startup Name": startup_names,
    "Success Probability": probs_rounded
})

# Step 4: Dropdown to search
st.markdown("### 🔍 Search or Select a Company")
selected_name = st.selectbox("Choose a startup", result_df["Startup Name"].sort_values(), key="search")

selected_prob = result_df[result_df["Startup Name"] == selected_name]["Success Probability"].values[0]

# Step 5: Show probability
st.markdown(f"### 🧠 Predicted Success Probability: **{selected_prob * 100:.2f}%**")
st.progress(min(selected_prob, 1.0))

# Optional: Show full table + download
with st.expander("📋 See All Predictions"):
    st.dataframe(result_df)

csv = result_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Full Predictions",
    data=csv,
    file_name="startup_success_predictions.csv",
    mime="text/csv"
)
