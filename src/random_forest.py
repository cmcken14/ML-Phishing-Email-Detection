import joblib
from sklearn.ensemble import RandomForestClassifier

from feature_extraction import prepare_features
from model_utils import evaluate_model, print_results


def main():
    print("Preparing data for Random Forest...")

    X_train, X_test, y_train, y_test, vectorizer = prepare_features()

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
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

    # Predict using the test data
    y_pred = model.predict(X_test)

    # Evaluate the results
    results = evaluate_model(y_test, y_pred)

    print_results("Random Forest", results)

    # Save the trained RF Model
    joblib.dump(
        model,
        "models/random_forest.joblib"
    )

    print("\nModel export complete.")
    print("Saved: models/random_forest.joblib")


if __name__ == "__main__":
    main()