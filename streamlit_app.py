import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the predictions
df = pd.read_csv("rf_test_predictions.csv")
df = df.sort_values("success_probability", ascending=False)
startup_names = df["name"].tolist()

# Sidebar layout
st.sidebar.markdown("## 🔍 Company Explorer", unsafe_allow_html=True)

selected_name = st.sidebar.selectbox("Select a startup", startup_names)

selected_row = df[df["name"] == selected_name].iloc[0]
success_prob = selected_row["success_probability"]
predicted = selected_row["predicted_success"]

st.sidebar.markdown(
    f"<h4>Success Probability:</h4>"
    f"<div style='font-size: 30px; font-weight: bold; color: #00FFAA;'> {success_prob:.2f} </div>",
    unsafe_allow_html=True
)

if predicted == 1:
    st.sidebar.markdown(
        "<div style='margin-top: 10px; background-color: #167D32; padding: 18px; border-radius: 8px; "
        "color: white; font-size: 22px; font-weight: bold;'>"
        "✅ Prediction: Successful</div>",
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        "<div style='margin-top: 10px; background-color: #911F1F; padding: 18px; border-radius: 8px; "
        "color: white; font-size: 22px; font-weight: bold;'>"
        "❌ Prediction: Unsuccessful</div>",
        unsafe_allow_html=True
    )

min_prob = st.sidebar.slider("Minimum Success Probability (Top list)", 0.0, 1.0, 0.7, step=0.01)

# Main layout
st.markdown("<h1 style='text-align: center;'>🚀 Startup Success Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"### 📊 Top Predicted Startups (Filtered by Min Probability ≥ {min_prob:.2f})")

filtered_df = df[df["success_probability"] >= min_prob].head(10)
st.dataframe(filtered_df[["name", "success_probability", "predicted_success"]], use_container_width=True)

st.markdown("### 📉 Distribution of Success Probabilities")
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df["success_probability"], bins=20, kde=True, color="skyblue", ax=ax)
ax.set_title("Distribution of Success Probabilities (Random Forest)")
ax.set_xlabel("Probability of Success")
ax.set_ylabel("Number of Startups")
ax.grid(True)
st.pyplot(fig)
