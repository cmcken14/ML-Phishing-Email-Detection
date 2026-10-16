## Script for preprocessing the df dataset for ML training

import pandas as pd

df = pd.read_csv("data/raw/df.csv")

df = df.drop_duplicates()

df["text"] = df["text"].fillna("")

df = df[df["text"] != ""]

df = df[df["label"].isin([0, 1])]

df = df[["text", "label"]]

df.to_csv("data/processed/df_preprocessed.csv", index=False)

print("Rows after preprocessing:", len(df))
print("\nLabel distribution:")
print(df["label"].value_counts())
