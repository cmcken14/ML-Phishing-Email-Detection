import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from feature_extraction import prepare_features
from model_utils import evaluate_model


def main():
    print("Preparing dataset...")

    X_train, X_test, y_train, y_test, vectorizer = prepare_features()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    all_results = []

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        results = evaluate_model(
            y_test,
            y_pred
        )

        results["Model"] = model_name

        all_results.append(results)

    results_df = pd.DataFrame(all_results)

    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "False Positive Rate"
        ]
    ]

    print("\nCEAS_08 Model Comparison")
    print("=" * 90)

    print(
        results_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        "results/ceas_model_comparison.csv",
        index=False
    )

    print("\nResults saved to:")
    print("results/ceas_model_comparison.csv")


if __name__ == "__main__":
    main()