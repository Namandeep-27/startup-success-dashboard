import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("rf_test_predictions.csv")
df["name"] = df["name"].astype(str)

# Sidebar: Company Explorer
with st.sidebar:
    st.markdown("## 🔍 Company Explorer", unsafe_allow_html=True)
    
    startup_names = sorted(df["name"].unique())
    selected_name = st.selectbox("Select a startup", startup_names)

    selected_row = df[df["name"] == selected_name].iloc[0]
    success_prob = selected_row["success_probability"]
    predicted = selected_row["predicted_success"]

    st.markdown("### Success Probability:")
    st.markdown(f"<div style='font-size:36px; color:#00FFAA; font-weight:bold'>{success_prob:.2f}</div>", unsafe_allow_html=True)

    if predicted == 1:
        st.markdown(
            "<div style='background-color:#198754;padding:20px;border-radius:10px;font-size:24px;color:white;font-weight:bold'>✅ Prediction: Successful</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color:#C82333;padding:20px;border-radius:10px;font-size:24px;color:white;font-weight:bold'>❌ Prediction: Unsuccessful</div>",
            unsafe_allow_html=True
        )
    
    st.markdown("---", unsafe_allow_html=True)
    min_prob = st.slider("Minimum Success Probability (Top list)", 0.0, 1.0, 0.7)

# Main: Dashboard Layout
st.markdown("# 🚀 Startup Success Prediction Dashboard")

col1, col2 = st.columns([1.5, 1.5])

# Top 500 startups
with col1:
    st.subheader("🏅 Top 500 Startups")
    top_500_df = df[df["is_top500"] == 1][["name", "success_probability", "predicted_success"]].sort_values(by="success_probability", ascending=False)
    st.dataframe(top_500_df.head(10), use_container_width=True)

# Filtered by slider
with col2:
    st.subheader(f"📊 Startups with Probability ≥ {min_prob:.2f}")
    filtered_df = df[df["success_probability"] >= min_prob][["name", "success_probability", "predicted_success"]].sort_values(by="success_probability", ascending=False)
    st.dataframe(filtered_df.head(10), use_container_width=True)

# Full-width section: Distribution Plot
st.markdown("## 📈 Distribution of Success Probabilities")
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df["success_probability"], bins=20, kde=True, color="skyblue", ax=ax)
ax.set_xlabel("Probability of Success")
ax.set_ylabel("Number of Startups")
st.pyplot(fig)
