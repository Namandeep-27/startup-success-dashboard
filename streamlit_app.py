import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("rf_test_predictions.csv")

# Sidebar — Company Explorer
st.sidebar.markdown("## 🔍 Company Explorer")
selected_name = st.sidebar.selectbox("Select a startup", sorted(df['name'].unique()))

# Slider for top prediction filtering
min_prob = st.sidebar.slider(
    "Minimum Success Probability (Top list)", 0.0, 1.0, 0.70, 0.01
)

# Show selected startup info
selected_row = df[df["name"] == selected_name].iloc[0]
st.sidebar.markdown(f"**Success Probability:** `{selected_row['success_probability']:.2f}`")

if selected_row["predicted_success"] == 1:
    st.sidebar.success("Prediction: Successful ✅")
else:
    st.sidebar.error("Prediction: Unsuccessful ❌")

# App title
st.markdown("## 🚀 Startup Success Prediction Dashboard")

# Split layout
col1, col2 = st.columns([1.5, 1.5])

# Column 1: Filtered Table
with col1:
    st.subheader(f"📊 Top Predicted Startups (Filtered by Min Probability ≥ {min_prob:.2f})")
    filtered_df = df[df["success_probability"] >= min_prob].sort_values(by="success_probability", ascending=False)
    st.dataframe(filtered_df[["name", "success_probability", "predicted_success"]].reset_index(drop=True))

# Column 2: Distribution Plot
with col2:
    st.subheader("📈 Distribution of Success Probabilities")
    plt.figure(figsize=(8, 5))
    sns.histplot(df["success_probability"], bins=20, kde=True, color='skyblue')
    plt.xlabel("Probability of Success")
    plt.ylabel("Number of Startups")
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.clf()

