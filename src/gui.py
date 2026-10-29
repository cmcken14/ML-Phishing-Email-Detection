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


def explain_prediction(model, email_features, vectorizer, prediction):
    feature_names = vectorizer.get_feature_names_out()
    feature_values = email_features.toarray()[0]

    # Explain predictions using model coefficients when available.
    if hasattr(model, "coef_"):
        weights = model.coef_

        # Convert sparse coefficient matrices to a regular array.
        if hasattr(weights, "toarray"):
            weights = weights.toarray()

        weights = weights.ravel()
        contributions = feature_values * weights

        # Find words that support the model's prediction.
        if prediction == 1:
            indices = contributions.argsort()[::-1]
            indices = [
                i for i in indices
                if contributions[i] > 0
            ]
            explanation_title = (
                "Words supporting the phishing prediction:"
            )
        else:
            indices = contributions.argsort()
            indices = [
                i for i in indices
                if contributions[i] < 0
            ]
            explanation_title = (
                "Words supporting the legitimate prediction:"
            )

        top_indices = indices[:5]

        if not top_indices:
            st.write(
                "No strong word-level contributions were found "
                "for this prediction."
            )
            return

        st.write(explanation_title)

        for i in top_indices:
            st.write(
                f"- **{feature_names[i]}** "
                f"(contribution: {contributions[i]:.4f})"
            )

    # Random Forest does not provide linear coefficients.
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        present_indices = [
            i for i, value in enumerate(feature_values)
            if value > 0
        ]

        top_indices = sorted(
            present_indices,
            key=lambda i: importances[i],
            reverse=True
        )[:5]

        if not top_indices:
            st.write(
                "No recognized TF-IDF words were found to explain "
                "this prediction."
            )
            return

        st.write(
            "Important words present in this email "
            "(these are overall model importances, not exact "
            "reasons for this individual prediction):"
        )

        for i in top_indices:
            st.write(
                f"- **{feature_names[i]}** "
                f"(overall importance: {importances[i]:.4f})"
            )

    else:
        st.write(
            "This model does not expose feature information "
            "that this explanation function can display."
        )

    st.caption(
        "These features help explain the model's output, but they "
        "do not prove that an email is malicious or legitimate."
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

    # Load the saved models and vectorizer.
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    logistic_model = joblib.load("models/logistic_regression.joblib")
    forest_model = joblib.load("models/random_forest.joblib")
    svm_model = joblib.load("models/svm.joblib")

    # Convert the email into TF-IDF features.
    email_features = vectorizer.transform([email_data["text"]])

    # Get predictions from each model.
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

        with st.expander("See Why"):
            explain_prediction(
                logistic_model,
                email_features,
                vectorizer,
                logistic_prediction
            )

    with col2:
        st.write("### Random Forest")

        if forest_prediction == 1:
            st.error("Likely Phishing")
        else:
            st.success("Likely Legitimate")

        with st.expander("See Why"):
            explain_prediction(
                forest_model,
                email_features,
                vectorizer,
                forest_prediction
            )

    with col3:
        st.write("### SVM")

        if svm_prediction == 1:
            st.error("Likely Phishing")
        else:
            st.success("Likely Legitimate")

        with st.expander("See Why"):
            explain_prediction(
                svm_model,
                email_features,
                vectorizer,
                svm_prediction
            )

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