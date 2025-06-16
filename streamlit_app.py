import streamlit as st
import pandas as pd

# Load the predictions CSV from the repo
@st.cache_data
def load_data():
    return pd.read_csv("predicted_success_probabilities.csv")

df = load_data()

# Streamlit UI
st.set_page_config(page_title="Startup Success Predictor", layout="centered")
st.title("🚀 Startup Success Probability Predictor")
st.markdown("Select a startup to see its predicted success probability.")

# Dropdown to search/select startup
startup_name = st.selectbox("🔍 Search or Select a Startup", df["name"].sort_values())

# Get probability
prob = df[df["name"] == startup_name]["success_probability"].values[0]

# Display probability
st.metric(label="Success Probability", value=f"{prob*100:.2f}%")
st.progress(min(prob, 1.0))

# Optional: Show full table
with st.expander("📋 Show Full Prediction Table"):
    st.dataframe(df)

# Download option
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Full Predictions",
    data=csv,
    file_name="startup_success_predictions.csv",
    mime="text/csv"
)
