import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="Startup Success Dashboard", layout="wide")
st.title("🚀 Startup Success Prediction Dashboard")

# Load data
df = pd.read_csv("rf_test_predictions.csv")

# Sidebar filters
st.sidebar.header("🔍 Company Explorer")
startup_names = df["name"].sort_values().unique()
selected_name = st.sidebar.selectbox("Select a startup", startup_names)

# Slider for top predictions
min_prob = st.sidebar.slider(
    "Minimum Success Probability (Top list)", 0.0, 1.0, 0.7, step=0.01
)

# Show selected startup's info
selected_row = df[df["name"] == selected_name].iloc[0]
st.sidebar.markdown(f"**Success Probability:** `{selected_row['success_probability']:.2f}`")

if selected_row["predicted_success"] == 1:
    st.sidebar.success("Prediction: Successful ✅")
else:
    st.sidebar.error("Prediction: Unsuccessful ❌")

# Main content layout
col1, col2 = st.columns([1.5, 2])

with col1:
    st.subheader(f"📊 Top Predicted Startups (Filtered by Min Probability ≥ {min_prob})")
    filtered_df = df[df["success_probability"] >= min_prob]
    top_startups = filtered_df.sort_values(by="success_probability", ascending=False)
    st.dataframe(top_startups[["name", "success_probability", "predicted_success"]].reset_index(drop=True).head(10))

with col2:
    st.subheader("📈 Distribution of Success Probabilities")
    plt.figure(figsize=(8, 5))
    sns.histplot(df["success_probability"], bins=20, kde=True, color="skyblue")
    plt.title("Distribution of Success Probabilities (Random Forest)")
    plt.xlabel("Probability of Success")
    plt.ylabel("Number of Startups")
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.clf()

