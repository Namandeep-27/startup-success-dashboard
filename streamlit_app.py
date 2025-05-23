import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load prediction results
df = pd.read_csv("rf_test_predictions.csv")

# Set page config
st.set_page_config(page_title="Startup Success Dashboard", layout="wide")

# Sidebar: Startup selector and probability threshold
st.sidebar.title("🔍 Company Explorer")
selected_name = st.sidebar.selectbox("Select a startup", df['name'].sort_values().unique())
min_prob = st.sidebar.slider("Minimum Success Probability (Top list)", 0.0, 1.0, 0.7, 0.01)

# Display info for selected startup
selected_row = df[df['name'] == selected_name].iloc[0]
st.sidebar.markdown(f"**Success Probability:** `{selected_row['success_probability']:.2f}`")
if selected_row['predicted_success'] == 1:
    st.sidebar.success("Prediction: Successful ✅")
else:
    st.sidebar.error("Prediction: Unsuccessful ❌")

# Dashboard title
st.title("🚀 Startup Success Prediction Dashboard")

# Top startups filtered by probability
col1, col2 = st.columns([1.2, 1])
with col1:
    st.subheader(f"📊 Top Predicted Startups (Filtered by Min Probability ≥ {min_prob:.2f})")
    filtered_df = df[df["success_probability"] >= min_prob].sort_values(by="success_probability", ascending=False)
    st.dataframe(filtered_df[['name', 'success_probability', 'predicted_success']].reset_index(drop=True), use_container_width=True)

# Success probability distribution
with col2:
    st.subheader("📈 Distribution of Success Probabilities")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["success_probability"], bins=20, kde=True, color="skyblue", ax=ax)
    ax.set_xlabel("Probability of Success")
    ax.set_ylabel("Number of Startups")
    ax.grid(True)
    st.pyplot(fig)
