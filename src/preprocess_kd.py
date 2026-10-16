## Script for preprocessing the KD dataset for ML training

import pandas as pd

df = pd.read_csv("data/raw/phishing_legit_dataset_KD_10000.csv")

df = df.drop_duplicates()

df["text"] = df["text"].fillna("")

df = df[df["text"] != ""]

df = df[["text", "label"]]

df.to_csv("data/processed/kd_preprocessed.csv", index=False)

print("Rows after preprocessing:", len(df))
print("\nLabel distribution:")
print(df["label"].value_counts())