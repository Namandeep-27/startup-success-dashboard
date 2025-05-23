import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("rf_test_predictions.csv")
df.sort_values(by="success_probability", ascending=False, inplace=True)

# Streamlit page config
st.set_page_config(page_title="Startup Success Dashboard", layout="wide")

# Sidebar - Company Explorer
with st.sidebar:
    st.markdown("## 🔍 Company Explorer")
    selected_name = st.selectbox("Select a startup", df["name"].unique())
    
    selected_row = df[df["name"] == selected_name].iloc[0]
    success_prob = selected_row["success_probability"]
    predicted = selected_row["predicted_success"]

    st.markdown("**Success Probability:**")
    st.markdown(f"<h1 style='color:#00f5a0;'>{success_prob:.2f}</h1>", unsafe_allow_html=True)

    if predicted == 1:
        st.markdown(
            "<div style='background-color: #22aa44; padding: 18px; border-radius: 10px;'>"
            "<span style='font-size: 22px; color: white;'>✅ <strong>Prediction: Successful</strong></span></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color: #cc2233; padding: 18px; border-radius: 10px;'>"
            "<span style='font-size: 22px; color: white;'>❌ <strong>Prediction: Unsuccessful</strong></span></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    min_prob = st.slider("Minimum Success Probability (Top list)", 0.0, 1.0, 0.7)

# Header
st.markdown("<h1 style='text-align: center;'>🚀 Startup Success Prediction Dashboard</h1>", unsafe_allow_html=True)

# Top 500 Table
st.markdown("### 🥇 Top 500 Startups")
top_500 = df.head(500)[["name", "success_probability", "predicted_success"]]
st.dataframe(top_500, use_container_width=True)

# Filtered Table
filtered_names = df[df["success_probability"] == min_prob][["name"]]
st.markdown(f"### 📊 Startups with Probability = {min_prob:.2f}")
st.dataframe(filtered_names, use_container_width=True)

# Distribution Plot
st.markdown("### 📉 Distribution of Success Probabilities")
plt.figure(figsize=(10, 4))
sns.histplot(df["success_probability"], bins=20, kde=True, color='skyblue')
plt.xlabel("Probability of Success")
plt.ylabel("Number of Startups")
plt.grid(True)
st.pyplot(plt.gcf())
plt.clf()
