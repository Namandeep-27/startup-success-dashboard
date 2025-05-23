import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load predictions
df = pd.read_csv("rf_test_predictions.csv")

# Streamlit configuration
st.set_page_config(page_title="Startup Success Dashboard", layout="wide")

st.title("🚀 Startup Success Prediction Dashboard")
st.markdown("""
This dashboard displays startup success predictions using a Random Forest model.
""")

# Sidebar filter
st.sidebar.header("Filter")
min_prob = st.sidebar.slider("Minimum Success Probability", 0.0, 1.0, 0.9, 0.01)

# Filtered view
filtered_df = df[df["success_probability"] >= min_prob]

st.subheader(f"Top Predicted Startups (Probability ≥ {min_prob})")
st.dataframe(
    filtered_df[["name", "success_probability", "predicted_success"]]
    .sort_values(by="success_probability", ascending=False)
    .head(10)
)

# Histogram plot
st.subheader("Distribution of Success Probabilities")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["success_probability"], bins=20, kde=True, color="skyblue", ax=ax)
ax.set_title("Distribution of Startup Success Probabilities")
ax.set_xlabel("Probability of Success")
ax.set_ylabel("Number of Startups")
st.pyplot(fig)

st.markdown("---")
st.markdown("Built by Namandeep Singh Nayyar | Model used: Random Forest")
