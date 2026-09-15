# Basic script to explore the data in the dataset CEAS_08 file, and print out some basic information about the dataset.

import pandas as pd

df = pd.read_csv("data/raw/CEAS_08.csv")

print("Shape:", df.shape)

print("\nColumns:", df.columns.tolist())

print("\nLabel counts:", df["label"].value_counts())

print("\nURL values:", df["urls"].value_counts().head(10))

print("\nMissing values:", df.isnull().sum())

print("\nFirst 5 rows:", df.head())