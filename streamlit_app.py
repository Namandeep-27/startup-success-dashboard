import streamlit as st
import pandas as pd

# ✅ Set up page
st.set_page_config(page_title="Startup Success Probability Predictor", layout="centered")
st.title("🚀 Startup Success Probability Predictor")
st.markdown("Select a startup to see its predicted success probability.")

# ✅ Load predictions from default CSV
@st.cache_data
def load_predictions():
    return pd.read_csv("predicted_success_probabilities.csv")

df = load_predictions()

# ✅ Startup selector
startup_names = df["name"].sort_values().tolist()
selected_name = st.selectbox("🔍 Search or Select a Startup", startup_names)

# ✅ Get and show probability
prob = df[df["name"] == selected_name]["success_probability"].values[0]
st.markdown(f"### 🎯 Success Probability: **{prob * 100:.2f}%**")
st.progress(min(prob, 1.0))

# ✅ Show full table
with st.expander("📊 Show Full Prediction Table"):
    st.dataframe(df)

# ✅ Download button
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Predictions CSV",
    data=csv,
    file_name="predicted_success_probabilities.csv",
    mime="text/csv"
)
