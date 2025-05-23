import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("rf_test_predictions.csv")

# Page settings
st.set_page_config(page_title="Startup Success Prediction", layout="wide")

# Sidebar
st.sidebar.title("🔍 Company Explorer")

selected_name = st.sidebar.selectbox("Select a startup", sorted(df["name"].dropna().unique()))
min_prob = st.sidebar.slider("Minimum Success Probability (Top list)", 0.0, 1.0, 0.7, 0.01)

# Selected startup info
selected_row = df[df["name"] == selected_name].iloc[0]
prob = selected_row["success_probability"]
pred = selected_row["predicted_success"]

st.sidebar.markdown(f"**Success Probability:** `{prob:.2f}`")

if pred == 1:
    st.sidebar.success("Prediction: Successful ✅")
else:
    st.sidebar.error("Prediction: Unsuccessful ❌")

# Main Title
st.title("🚀 Startup Success Prediction Dashboard")

# Filter top startups by probability
filtered_df = df[df["success_probability"] >= min_prob].sort_values("success_probability", ascending=False)

# Columns for layout
col1, col2 = st.columns([1.5, 1.5])

with col1:
    st.subheader(f"📊 Top Predicted Startups (Filtered by Min Probability ≥ {min_prob:.2f})")
    st.dataframe(filtered_df[["name", "success_probability", "predicted_success"]].reset_index(drop=True), use_container_width=True)

with col2:
    st.subheader("📉 Distribution of Success Probabilities")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["success_probability"], bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_title("Distribution of Success Probabilities (Random Forest)")
    ax.set_xlabel("Probability of Success")
    ax.set_ylabel("Number of Startups")
    st.pyplot(fig)
