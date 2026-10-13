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

    # Load the saved models and vectorizer
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    logistic_model = joblib.load("models/logistic_regression.joblib")
    forest_model = joblib.load("models/random_forest.joblib")
    svm_model = joblib.load("models/svm.joblib")

    # Convert the email into TF-IDF features
    email_features = vectorizer.transform([email_data["text"]])

    # Get predictions from each model
    logistic_prediction = logistic_model.predict(email_features)[0]
    forest_prediction = forest_model.predict(email_features)[0]
    svm_prediction = svm_model.predict(email_features)[0]

    st.subheader("Model Analysis")


    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Logistic Regression")
        if logistic_prediction == 1:
            st.error("Likely Phishing")
        else:
            st.success("Likely Legitimate")

    with col2:
        st.write("### Random Forest")
        if forest_prediction == 1:
            st.error("Likely Phishing")
        else:
            st.success("Likely Legitimate")

    with col3:
        st.write("### SVM")
        if svm_prediction == 1:
            st.error("Likely Phishing")
        else:
            st.success("Likely Legitimate")

    st.subheader("Overall Result")

    phishing_votes = sum([
        logistic_prediction,
        forest_prediction,
        svm_prediction
    ])

    if phishing_votes >= 2:
        st.error(
            f"Likely Phishing — {phishing_votes} out of 3 models "
            "flagged this email."
        )
    else:
        st.success(
            f"Likely Legitimate — {3 - phishing_votes} out of 3 models "
            "classified this email as legitimate."
        )
