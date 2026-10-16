## Script for preprocessing the SpamAssasin dataset for ML training

import pandas as pd

df = pd.read_csv("data/raw/SpamAssasin.csv")

df = df.drop_duplicates()

df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")

df["text"] = (df["subject"] + " " + df["body"]).str.strip()

df = df[df["text"] != ""]

df = df[["text", "label"]]

df.to_csv("data/processed/SpamAssasin_preprocessed.csv", index=False)

print("Rows after preprocessing:", len(df))
print("\nLabel distribution:")
print(df["label"].value_counts())