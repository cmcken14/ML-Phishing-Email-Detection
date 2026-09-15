## Script for preprocessing the CEAS_08 dataset for ML training

import pandas as pd

df = pd.read_csv("data/raw/CEAS_08.csv")

df = df.drop_duplicates()

df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")

df["text"] = (df["subject"] + " " + df["body"]).str.strip()

df = df[df["text"] != ""]

df = df[["text", "urls", "label"]]

df.to_csv("data/processed/CEAS_08_preprocessed.csv", index=False)

print("Rows after preprocessing:", len(df))
print("\nLabel distribution:")
print(df["label"].value_counts())
print("\nURL distribution:")
print(df["urls"].value_counts())
