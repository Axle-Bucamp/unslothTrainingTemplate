import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import os
from transformers import AutoTokenizer

def load_parquet_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet file not found at: {path}")
    return pd.read_parquet(path)

def summarize_parquet(df):
    print("📊 Dataset Summary:")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("\n🧱 Column Names:")
    print(df.columns.tolist())

    print("\n📉 Null Values per Column:")
    print(df.isnull().sum())

    print("\n📈 Data Types:")
    print(df.dtypes)

def show_sample_rows(df, n=5):
    print(f"\n🔍 Showing {n} sample rows:")
    print(df.sample(n))

def tokenize_column_stats(df, column, tokenizer_name="unsloth/Llama-3.3-70B-Instruct-bnb-4bit"):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"\n🔢 Tokenizing column '{column}'...")

    df = df.dropna(subset=[column])
    token_lengths = df[column].astype(str).apply(lambda x: len(tokenizer.encode(x, truncation=False)))

    print("\n📏 Token Length Statistics:")
    print(token_lengths.describe())

    plt.figure(figsize=(10, 5))
    sns.histplot(token_lengths, bins=50, kde=True)
    plt.title(f"Token Length Distribution for '{column}'")
    plt.xlabel("Number of tokens")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def detect_issues(df, column):
    print(f"\n🛑 Checking data quality issues for column '{column}':")
    empty_rows = df[column].astype(str).str.strip() == ""
    if empty_rows.any():
        print(f"⚠️ Empty strings found in {empty_rows.sum()} rows.")
    else:
        print("✅ No empty strings found.")
    
    duplicates = df[column].duplicated().sum()
    print(f"📦 Duplicate entries: {duplicates}")

def visualize_parquet(path, column, tokenizer_name="unsloth/Llama-3.3-70B-Instruct-bnb-4bit"):
    df = load_parquet_file(path)
    summarize_parquet(df)
    show_sample_rows(df)
    tokenize_column_stats(df, column, tokenizer_name)
    detect_issues(df, column)

# Example CLI Usage:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize and validate a Parquet file for LLM training.")
    parser.add_argument("--file", required=True, help="Path to the Parquet file.")
    parser.add_argument("--column", required=True, help="Name of the column to tokenize.")
    parser.add_argument("--tokenizer", default="unsloth/Llama-3.3-70B-Instruct-bnb-4bit", help="Tokenizer name.")

    args = parser.parse_args()
    visualize_parquet(args.file, args.column, args.tokenizer)
