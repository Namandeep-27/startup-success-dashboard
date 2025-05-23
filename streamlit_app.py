import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("rf_test_predictions.csv")
df['name'] = df['name'].astype(str)
startup_names = df['name'].unique()

# Sidebar
with st.sidebar:
    st.markdown("## 🔍 Company Explorer")
    selected_name = st.selectbox("Select a startup", startup_names)
    selected_row = df[df["name"] == selected_name].iloc[0]
    
    success_prob = selected_row["success_probability"]
    predicted = selected_row["predicted_success"]

    st.markdown("### Success Probability:")
    st.markdown(f"<div style='font-size: 36px; font-weight: bold; color: #00FFAA;'>{success_prob:.2f}</div>", unsafe_allow_html=True)

    if predicted == 1:
        st.markdown("<div style='background-color: #0e8c43; padding: 18px; border-radius: 10px; color: white; font-size: 24px;'>✅ Prediction: Successful</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background-color: #c0392b; padding: 18px; border-radius: 10px; color: white; font-size: 24px;'>❌ Prediction: Unsuccessful</div>", unsafe_allow_html=True)

    st.markdown("---")
    min_prob = st.slider("Minimum Success Probability (Top list)", 0.0, 1.0, 0.70)

# Main dashboard
st.markdown("# 🚀 Startup Success Prediction Dashboard")

# Top 500 Startups Table
st.markdown("### 🥇 Top 500 Startups")
top_500 = df[df['is_top500'] == 1][['name', 'success_probability', 'predicted_success']]
st.dataframe(top_500.reset_index(drop=True), use_container_width=True)

# Filtered Startups Table with only names
st.markdown(f"### 📊 Startups with Probability ≥ {min_prob:.2f}")
filtered_names = df[df['success_probability'] >= min_prob][['name']]
st.dataframe(filtered_names.reset_index(drop=True), use_container_width=True)

# Distribution Chart
st.markdown("### 📉 Distribution of Success Probabilities")
plt.figure(figsize=(8, 5))
sns.histplot(df['success_probability'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Success Probabilities (Random Forest)')
plt.xlabel('Probability of Success')
plt.ylabel('Number of Startups')
plt.grid(True)
st.pyplot(plt.gcf())
plt.clf()
