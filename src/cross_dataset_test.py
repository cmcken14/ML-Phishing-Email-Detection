import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from model_utils import evaluate_model


CEAS_PATH = "data/processed/CEAS_08_preprocessed.csv"
ENRON_PATH = "data/processed/Enron_preprocessed.csv"
DF_PATH = "data/processed/df_preprocessed.csv"
SPAMASSASIN_PATH = "data/processed/SpamAssasin_preprocessed.csv"
KD_PATH = "data/processed/kd_preprocessed.csv"


def main():
    # Load all datasets
    ceas = pd.read_csv(CEAS_PATH)
    enron = pd.read_csv(ENRON_PATH)
    df = pd.read_csv(DF_PATH)
    spamassasin = pd.read_csv(SPAMASSASIN_PATH)
    kd = pd.read_csv(KD_PATH)

    print("Datasets loaded.")

    print("CEAS_08 emails:", len(ceas))
    print("Enron emails:", len(enron))
    print("DF emails:", len(df))
    print("SpamAssasin emails:", len(spamassasin))
    print("KD emails:", len(kd))

    # Define training and testing dataset pairs
    experiments = [
        ("CEAS_08", ceas, "Enron", enron),
        ("Enron", enron, "CEAS_08", ceas),
        ("DF", df, "SpamAssasin", spamassasin),
        ("DF", df, "KD", kd)
    ]

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

    for train_name, train_df, test_name, test_df in experiments:

        print(f"\nTraining dataset: {train_name}")
        print(f"Testing dataset: {test_name}")

        # Training emails
        X_train_text = train_df["text"]
        y_train = train_df["label"]

        # External testing emails
        X_test_text = test_df["text"]
        y_test = test_df["label"]

        print("\nCreating TF-IDF features...")

        # TF-IDF is fit only on the training dataset
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            lowercase=True
        )

        X_train = vectorizer.fit_transform(
            X_train_text
        )

        # Testing dataset uses the TF-IDF vocabulary learned from training
        X_test = vectorizer.transform(
            X_test_text
        )

        print("Training matrix:", X_train.shape)
        print("Testing matrix:", X_test.shape)

        for model_name, model in models.items():

            print(f"\nTraining {model_name} on {train_name}...")

            model.fit(
                X_train,
                y_train
            )

            print(
                f"Testing {model_name} on {test_name}..."
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
                if metric in ["False Positives", "False Negatives"]:
                    print(f"{metric}: {score}")
                else:
                    print(f"{metric}: {score:.4f}")

            results["Training Dataset"] = train_name
            results["Testing Dataset"] = test_name
            results["Model"] = model_name

            results_list.append(results)

    # Put all model results into one table
    results_df = pd.DataFrame(
        results_list
    )

    results_df = results_df[
        [
            "Training Dataset",
            "Testing Dataset",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "False Positive Rate",
            "False Positives",
            "False Negative Rate",
            "False Negatives"
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
        "results/cross_dataset_comparison.csv",
        index=False
    )

    print("\nResults saved to:")
    print("results/cross_dataset_comparison.csv")


if __name__ == "__main__":
    main()
