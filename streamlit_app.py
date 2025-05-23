import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_csv("rf_test_predictions.csv")

st.set_page_config(page_title="Startup Success Dashboard", layout="wide")

# Sidebar - User Input
st.sidebar.title("🔍 Company Explorer")
selected_name = st.sidebar.selectbox("Select a startup", df["name"].dropna().unique())
min_prob = st.sidebar.slider("Minimum Success Probability (Top list)", 0.0, 1.0, 0.7)

# Display selected startup details
selected_row = df[df["name"] == selected_name].iloc[0]
st.sidebar.markdown(f"**Success Probability:** `{selected_row['success_probability']:.2f}`")

if selected_row["predicted_success"] == 1:
    st.sidebar.success("Prediction: Successful ✅")
else:
    st.sidebar.error("Prediction: Unsuccessful ❌")

# Main Layout
st.title("🚀 Startup Success Prediction Dashboard")

col1, col2 = st.columns([1.5, 1.5])

with col1:
    st.subheader(f"📊 Top Predicted Startups (Filtered by Min Probability ≥ {min_prob:.2f})")
    filtered_df = df[df["success_probability"] >= min_prob].copy()
    top_startups = filtered_df.sort_values("success_probability", ascending=False)[["name", "success_probability", "predicted_success"]]
    st.dataframe(top_startups.reset_index(drop=True))

with col2:
    st.subheader("📈 Distribution of Success Probabilities")
    plt.figure(figsize=(8, 5))
    sns.histplot(df["success_probability"], bins=20, kde=True, color="skyblue")
    plt.title("Distribution of Success Probabilities (Random Forest)")
    plt.xlabel("Probability of Success")
    plt.ylabel("Number of Startups")
    st.pyplot(plt.gcf())
    plt.clf()
