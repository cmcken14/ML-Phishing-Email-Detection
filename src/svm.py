from sklearn.svm import SVC

from feature_extraction import prepare_features
from model_utils import evaluate_model, print_results


def main():
    print("Preparing data for SVM...")

    X_train, X_test, y_train, y_test, vectorizer = prepare_features()

    print("\nTraining SVM...")

    model = SVC(
        kernel="linear",
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

    print_results("SVM", results)


if __name__ == "__main__":
    main()