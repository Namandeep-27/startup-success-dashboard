# Sidebar layout
with st.sidebar:
    st.markdown("## 🔍 Company Explorer", unsafe_allow_html=True)
    
    st.markdown("### Select a startup", unsafe_allow_html=True)
    selected_name = st.selectbox("", startup_names, key="startup_select", label_visibility="collapsed")

    selected_row = df[df["name"] == selected_name].iloc[0]
    success_prob = selected_row["success_probability"]
    predicted = selected_row["predicted_success"]

    st.markdown(
        f"<h4 style='margin-top: 20px;'>Success Probability:</h4>"
        f"<div style='font-size: 28px; font-weight: bold; color: #00FFAA;'> {success_prob:.2f} </div>",
        unsafe_allow_html=True
    )

    if predicted == 1:
        st.markdown(
            "<div style='margin-top: 10px; background-color: #167D32; padding: 16px; border-radius: 8px; "
            "color: white; font-size: 22px; font-weight: bold;'>"
            "✅ Prediction: Successful</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='margin-top: 10px; background-color: #911F1F; padding: 16px; border-radius: 8px; "
            "color: white; font-size: 22px; font-weight: bold;'>"
            "❌ Prediction: Unsuccessful</div>",
            unsafe_allow_html=True
        )
