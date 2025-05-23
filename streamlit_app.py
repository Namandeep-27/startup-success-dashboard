import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("rf_test_predictions.csv")

# Page settings
st.set_page_config(page_title="Startup Success Dashboard", layout="wide")

# Sidebar
st.sidebar.title("🔍 Company Explorer")

# Select company
startup_names = sorted(df["name"].dropna().unique())
selected_name = st.sidebar.selectbox("Select a startup", startup_names)

# Show company prediction
selected_row = df[df["name"] == selected_name].iloc[0]
success_prob = selected_row["success_probability"]
predicted = selected_row["predicted_success"]

st.sidebar.markdown(f"**Success Probability:** `{success_prob:.2f}`")

if predicted == 1:
    st.sidebar.success("Prediction: Successful ✅")
else:
    st.sidebar.error("Prediction: Unsuccessful ❌")

# Slider for table filtering
min_prob = st.sidebar.slider("Minimum Success Probability (Top list)", 0.0, 1.0, 0.70, 0.01)

# Main page
st.title("🚀 Startup Success Prediction Dashboard")

# Filtered table
filtered_df = df[df["success_probability"] >= min_prob].sort_values("success_probability", ascending=False)
st.subheader(f"📊 Top Predicted Startups (Filtered by Min Probability ≥ {min_prob:.2f})")
st.dataframe(filtered_df[["name", "success_probability", "predicted_success"]].reset_index(drop=True), use_container_width=True)

# Spacer
st.markdown("---")

# Histogram
st.subheader("📉 Distribution of Success Probabilities (Random Forest)")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["success_probability"], bins=20, kde=True, color="skyblue", ax=ax)
ax.set_xlabel("Probability of Success")
ax.set_ylabel("Number of Startups")
ax.set_title("Distribution of Startup Success Probabilities")
st.pyplot(fig)
