import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from model_utils import evaluate_model


CEAS_PATH = "data/processed/CEAS_08_preprocessed.csv"
ENRON_PATH = "data/processed/Enron_preprocessed.csv"


def main():
    # Load both datasets
    ceas = pd.read_csv(CEAS_PATH)
    enron = pd.read_csv(ENRON_PATH)

    print("Datasets loaded.")

    print("CEAS_08 training emails:", len(ceas))
    print("Enron testing emails:", len(enron))

    # CEAS_08 is only used for training
    X_train_text = ceas["text"]
    y_train = ceas["label"]

    # Enron is only used for external testing
    X_test_text = enron["text"]
    y_test = enron["label"]

    print("\nCreating TF-IDF features...")

    # TF-IDF is fit only on CEAS_08
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        lowercase=True
    )

    X_train = vectorizer.fit_transform(
        X_train_text
    )

    # Enron uses the TF-IDF vocabulary learned from CEAS_08
    X_test = vectorizer.transform(
        X_test_text
    )

    print("CEAS training matrix:", X_train.shape)
    print("Enron testing matrix:", X_test.shape)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

	"SVM": SVC(
	    kernel="linear",
            random_state=42
    	)
    }

    results_list = []

    for model_name, model in models.items():

        print(f"\nTraining {model_name} on CEAS_08...")

        model.fit(
            X_train,
            y_train
        )

        print(
            f"Testing {model_name} on Enron..."
        )

        predictions = model.predict(
            X_test
        )

        results = evaluate_model(
            y_test,
            predictions
        )

        print(f"\n{model_name} Results")
        print("----------------------------------------")

        for metric, score in results.items():
            print(
                f"{metric}: {score:.4f}"
            )

        results["Model"] = model_name

        results_list.append(results)

    # Put both model results into one table
    results_df = pd.DataFrame(
        results_list
    )

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

    print("\nCross-Dataset Results")
    print("=" * 90)

    print(
        results_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        "results/cross_dataset_enron.csv",
        index=False
    )

    print("\nResults saved to:")
    print("results/cross_dataset_enron.csv")


if __name__ == "__main__":
    main()