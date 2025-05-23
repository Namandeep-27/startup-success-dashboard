import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the predictions CSV
df = pd.read_csv("rf_test_predictions.csv")

# Set page config
st.set_page_config(page_title="Startup Success Dashboard", layout="wide")
st.title("🚀 Startup Success Prediction Dashboard")

# Sidebar: Company Explorer
st.sidebar.header("🔍 Company Explorer")

# Startup selector
startup_names = df["name"].dropna().unique()
selected_name = st.sidebar.selectbox("Select a startup", sorted(startup_names))

# Get info of the selected startup
company_info = df[df["name"] == selected_name].iloc[0]
success_prob = company_info["success_probability"]
prediction = "Successful ✅" if company_info["predicted_success"] == 1 else "Unsuccessful ❌"

st.sidebar.markdown(f"**Success Probability:** `{success_prob:.2f}`")
st.sidebar.markdown(f"**Prediction:** `{prediction}`")

# Sidebar: Minimum probability slider
min_prob = st.sidebar.slider("Minimum Success Probability (Top list)", 0.0, 1.0, 0.9, 0.01)

# Filter top startups by slider
filtered_df = df[df["success_probability"] >= min_prob]
top_startups = filtered_df.sort_values(by="success_probability", ascending=False).head(10)

# Main area: Top startups table
st.subheader(f"📊 Top Predicted Startups (≥ {min_prob})")
st.dataframe(top_startups[["name", "success_probability", "predicted_success"]].reset_index(drop=True))

# Plot: Success Probability Distribution
st.subheader("📈 Distribution of Success Probabilities")
plt.figure(figsize=(8, 5))
sns.histplot(df["success_probability"], bins=20, kde=True, color="skyblue")
plt.title("Distribution of Success Probabilities (Random Forest)")
plt.xlabel("Probability of Success")
plt.ylabel("Number of Startups")
plt.grid(True)
st.pyplot(plt)


