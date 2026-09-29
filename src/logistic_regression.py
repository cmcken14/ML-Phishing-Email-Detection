from sklearn.linear_model import LogisticRegression

from feature_extraction import prepare_features
from model_utils import evaluate_model, print_results


def main():
    print("Preparing data for Logistic Regression...")

    X_train, X_test, y_train, y_test, vectorizer = prepare_features()

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Training complete.")

    # Compare training and testing accuracy
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print("\nTraining vs Testing Accuracy")
    print("----------------------------------------")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy:  {test_accuracy:.4f}")

    # Make predictions using the test data
    y_pred = model.predict(X_test)

    # Evaluate the predictions
    results = evaluate_model(y_test, y_pred)

    print_results("Logistic Regression", results)

    # Show some of the words that affected the model the most
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    top_phishing = coefficients.argsort()[-10:][::-1]
    top_legitimate = coefficients.argsort()[:10]

    print("\nTop Phishing Indicators")
    print("----------------------------------------")

    for index in top_phishing:
        print(
            feature_names[index],
            round(coefficients[index], 4)
        )

    print("\nTop Legitimate Indicators")
    print("----------------------------------------")

    for index in top_legitimate:
        print(
            feature_names[index],
            round(coefficients[index], 4)
        )


if __name__ == "__main__":
    main()