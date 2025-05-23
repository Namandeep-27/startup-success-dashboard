import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load predictions
df = pd.read_csv("rf_test_predictions.csv")
df = df.sort_values(by="success_probability", ascending=False)

# Unique startup names
startup_names = df["name"].unique()

# Sidebar – Company Explorer
with st.sidebar:
    st.markdown("## 🔍 Company Explorer")

    selected_name = st.selectbox("Select a startup", startup_names)

    selected_row = df[df["name"] == selected_name].iloc[0]
    success_prob = selected_row["success_probability"]
    predicted = selected_row["predicted_success"]

    st.markdown("### Success Probability:")
    st.markdown(f"<h1 style='color:#00ffaa'>{success_prob:.2f}</h1>", unsafe_allow_html=True)

    if predicted == 1:
        st.markdown(
            "<div style='background-color:#167D32;padding:16px;border-radius:10px;color:white;font-size:20px;font-weight:bold;'>"
            "✅ Prediction: Successful</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color:#911F1F;padding:16px;border-radius:10px;color:white;font-size:20px;font-weight:bold;'>"
            "❌ Prediction: Unsuccessful</div>",
            unsafe_allow_html=True
        )

    st.markdown("### Minimum Success Probability (Top list)")
    min_prob = st.slider("", 0.0, 1.0, 0.7, step=0.01)

# Main Dashboard
st.markdown("<h1 style='text-align: center;'>📊 Startup Success Prediction Dashboard</h1>", unsafe_allow_html=True)

# Top Startups
top_startups = df[df["success_probability"] >= min_prob]
st.markdown(f"### 🏆 Top Startups (Success ≥ {min_prob:.2f})")
st.dataframe(top_startups[["name", "success_probability", "predicted_success"]].reset_index(drop=True))

# Success Probability Distribution
st.markdown("### 📈 Distribution of Success Probabilities")
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["success_probability"], bins=20, kde=True, color='skyblue', ax=ax)
ax.set_xlabel("Probability of Success")
ax.set_ylabel("Number of Startups")
ax.set_title("Distribution of Success Probabilities (Random Forest)")
st.pyplot(fig)
