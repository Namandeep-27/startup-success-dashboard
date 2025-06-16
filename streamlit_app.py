import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("stacked_model.pkl")        # Save your model with joblib.dump()
scaler = joblib.load("scaler.pkl")              # Save scaler as well

# Columns to drop
drop_cols = [
    'id', 'name', 'city', 'zip_code', 'founded_at', 'closed_at',
    'first_funding_at', 'last_funding_at', 'object_id',
    'category_code', 'status', 'labels', 'state', 'state_code'
]

# Streamlit UI
st.title("Startup Success Probability Predictor")

uploaded_file = st.file_uploader("Upload your test CSV", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    if 'name' not in test_df.columns:
        st.error("The CSV must include a 'name' column.")
    else:
        names = test_df['name']
        X_test = test_df.drop(columns=drop_cols, errors='ignore')
        X_test = X_test.fillna(X_test.median(numeric_only=True))

        try:
            X_scaled = scaler.transform(X_test)
            probs = model.predict_proba(X_scaled)[:, 1]

            result_df = pd.DataFrame({
                'Startup Name': names,
                'Success Probability': np.round(probs, 2)
            })

            st.success("✅ Predictions generated successfully!")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
