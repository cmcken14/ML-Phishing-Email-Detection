import joblib
import streamlit as st

from email_parser import parse_email

st.set_page_config(
    page_title="Phishing Email Detection",
    page_icon="📧",
    layout="wide"
)

st.title("Phishing Email Detection")

st.write(
    "Upload an .eml file to view its contents "
    "and check if it's a phishing email."
)

uploaded_file = st.file_uploader(
    "Upload an .eml file",
    type=["eml"]
)

if uploaded_file is not None:
    email_data = parse_email(uploaded_file)

    st.subheader("Email Information")

    st.write("**From:**", email_data["sender"])
    st.write("**To:**", email_data["receiver"])
    st.write("**Subject:**", email_data["subject"])

    st.subheader("Email Body")

    st.text_area(
        "Message",
        email_data["body"],
        height=250,
        disabled=True
    )

    st.subheader("Model Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Logistic Regression")
        st.info("Model result will appear here.")

    with col2:
        st.write("### Random Forest")
        st.info("Model result will appear here.")

    with col3:
        st.write("### SVM")
        st.info("Model result will appear here.")

    st.subheader("Overall Result")

    st.warning(
        "Overall results will appear here "
        "after the models are connected."
    )