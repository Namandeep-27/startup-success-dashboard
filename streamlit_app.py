import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the prediction results
df = pd.read_csv("rf_test_predictions.csv")
startup_names = df["name"].dropna().unique()

# --- Sidebar layout ---
st.sidebar.markdown("## 🔍 Company Explorer", unsafe_allow_html=True)

# Startup selector
st.sidebar.markdown("### Select a startup")
selected_name = st.sidebar.selectbox("", startup_names)
selected_row = df[df["name"] == selected_name].iloc[0]

# Display selected startup prediction
success_prob = selected_row["success_probability"]
predicted = selected_row["predicted_success"]

st.sidebar.markdown("<h4 style='margin-top: 20px;'>Success Probability:</h4>"
                    f"<div style='font-size: 28px; font-weight: bold; color: #00FFAA;'> {success_prob:.2f} </div>",
                    unsafe_allow_html=True)

if predicted == 1:
    st.sidebar.markdown("""
        <div style='margin-top: 10px; background-color: #167D32; padding: 16px; border-radius: 8px; 
        color: white; font-size: 22px; font-weight: bold;'>
        ✅ Prediction: Successful</div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
        <div style='margin-top: 10px; background-color: #911F1F; padding: 16px; border-radius: 8px; 
        color: white; font-size: 22px; font-weight: bold;'>
        ❌ Prediction: Unsuccessful</div>
    """, unsafe_allow_html=True)

# Slider for dynamic probability filtering
min_prob = st.sidebar.slider("Minimum Success Probability (Top list)", 0.0, 1.0, 0.70)

# --- Main layout ---
st.title("🚀 Startup Success Prediction Dashboard")

# Top 500 Companies
top500 = df[df["is_top500"] == 1].sort_values(by="success_probability", ascending=False)[
    ["name", "success_probability", "predicted_success"]
].reset_index(drop=True)

st.subheader("📊 Top 500 Startups")
st.dataframe(top500, use_container_width=True)

# Dynamic list based on slider
st.subheader(f" 🎲 Startups with Success Probability ≥ {min_prob:.2f}")
dynamic_filtered = df[df["success_probability"] >= min_prob].sort_values(
    by="success_probability", ascending=False
)[["name", "success_probability", "predicted_success"]].reset_index(drop=True)
st.dataframe(dynamic_filtered, use_container_width=True)

# --- Plot ---
st.subheader(":chart: Distribution of Success Probabilities")
plt.figure(figsize=(8, 4))
sns.histplot(df["success_probability"], bins=20, kde=True, color="skyblue")
plt.xlabel("Probability of Success")
plt.ylabel("Number of Startups")
plt.grid(True)
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

