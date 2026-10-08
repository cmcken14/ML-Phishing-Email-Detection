import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = "data/processed/CEAS_08_preprocessed.csv"


def prepare_features():
    # Load the cleaned dataset
    df = pd.read_csv(DATA_PATH)

    print("Processed dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    # Check for duplicate email text
    print("\nDuplicate Check")
    print("----------------------------------------")
    print("Total emails:", len(df))
    print("Exact duplicate text emails:", df["text"].duplicated().sum())
    print("Unique email texts:", df["text"].nunique())

    # Text will be used to predict the label
    X = df["text"]
    y = df["label"]

    # Split the dataset into 80% training and 20% testing
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain/Test Split")
    print("Training emails:", len(X_train_text))
    print("Testing emails:", len(X_test_text))

    # Turn the email text into numerical TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        lowercase=True
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Save the TF-IDF vectorizer
    joblib.dump(
        vectorizer,
        "models/tfidf_vectorizer.joblib"
    )

    print("\nTF-IDF Feature Extraction")
    print("Number of features:", X_train.shape[1])
    print("Training matrix shape:", X_train.shape)
    print("Testing matrix shape:", X_test.shape)

    print("\nVectorizer export complete.")
    print("Saved: models/tfidf_vectorizer.joblib")
    
    return X_train, X_test, y_train, y_test, vectorizer


if __name__ == "__main__":
    prepare_features()