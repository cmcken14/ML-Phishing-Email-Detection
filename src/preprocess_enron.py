import pandas as pd


RAW_PATH = "data/raw/Enron.csv"
OUTPUT_PATH = "data/processed/Enron_preprocessed.csv"


def main():
    df = pd.read_csv(RAW_PATH)

    print("Enron dataset loaded.")
    print("Original shape:", df.shape)

    # Make column names consistent
    df.columns = df.columns.str.lower().str.strip()

    print("Columns:", df.columns.tolist())

    # Fill any missing text
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    # Combine the subject and body like CEAS_08
    df["text"] = (
        df["subject"].astype(str)
        + " "
        + df["body"].astype(str)
    )

    # Remove emails with no usable text
    df = df[df["text"].str.strip() != ""]

    # Make sure labels are numbers
    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    # Only keep legitimate and phishing labels
    df = df[df["label"].isin([0, 1])]

    # Remove exact duplicate emails
    df = df.drop_duplicates(
        subset=["text"]
    )

    # Only save what is needed for testing
    df = df[
        [
            "text",
            "label"
        ]
    ]

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nEnron preprocessing complete.")
    print("Processed shape:", df.shape)

    print("\nLabel counts:")
    print(df["label"].value_counts())

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()