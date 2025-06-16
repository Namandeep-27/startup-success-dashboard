import streamlit as st
import pandas as pd
import numpy as np

Page configuration

st.set_page_config(page_title=“Startup Success Probability Predictor”, layout=“centered”)
st.title(“🚀 Startup Success Probability Predictor”)
st.markdown(”””
Select a startup to see its predicted success probability.
“””)

Load data from local CSV (already in repo)

@st.cache_data
def load_data():
df = pd.read_csv(“predicted_success_probabilities.csv”)
df[“success_probability”] = np.round(df[“success_probability”], 2)
return df

Load the prediction data

data = load_data()

Searchable input box

search_query = st.text_input(“🔍 Search or Select a Startup”)

Filter options based on search query

filtered_startups = data[data[“name”].str.contains(search_query, case=False, na=False)]

Dropdown for selecting startup

startup_list = filtered_startups[“name”].tolist()
if not startup_list:
st.warning(“No matching startup found.”)
st.stop()

selected_name = st.selectbox(“Select from matches below”, startup_list)
selected_prob = filtered_startups[filtered_startups[“name”] == selected_name][“success_probability”].values[0]

Display prediction

st.subheader(“Success Probability”)
st.markdown(f”### {selected_prob * 100:.2f}%”)
st.progress(min(selected_prob, 1.0))

Show full table (expandable)

with st.expander(“📋 Show Full Prediction Table”):
st.dataframe(data.sort_values(“success_probability”, ascending=False), use_container_width=True)

Footer

st.markdown(”””

Made with ❤️ for academic demonstration purposes.
“””)
