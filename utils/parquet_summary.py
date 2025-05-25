import pandas as pd
import pyarrow.parquet as pq
from tabulate import tabulate
import os
import matplotlib.pyplot as plt

REPORT_DIR = "report"
os.makedirs(REPORT_DIR, exist_ok=True)

def pretty_print_top_rows(df, n=5):
    top = df.head(n)
    text = tabulate(top, headers="keys", tablefmt="pretty", showindex=False)
    print(f"\n🔍 Top {n} Rows:")
    print(text)
    with open(os.path.join(REPORT_DIR, "top_rows.txt"), "w") as f:
        f.write(text + "\n")

def global_stats(df):
    stats_file = os.path.join(REPORT_DIR, "global_stats.txt")
    with open(stats_file, "w") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")

        log("\n📊 Global Dataset Stats:")
        log(f"- Total rows: {len(df)}")
        log(f"- Total columns: {len(df.columns)}")

        log("\n📉 Null Values:")
        nulls = df.isnull().sum()
        nulls_filtered = nulls[nulls > 0].to_frame(name='Nulls')
        nulls_text = tabulate(nulls_filtered, headers=['Column', 'Nulls'], tablefmt='pretty')
        log(nulls_text)

        # Plot nulls
        try:
            if nulls_filtered is not None and not nulls_filtered.empty:
                nulls_filtered.plot(kind='barh', legend=False, figsize=(10, 6), title='Null Values per Column')
                plt.tight_layout()
                plt.show()
            else:
                print("No null values to plot.")
        except Exception as e:
            print("Plotting error:", str(e))
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, "null_values_plot.png"))
        plt.close()

        log("\n📦 Duplicate Rows:")
        try:
            dupes = df.duplicated().sum()
            log(f"- Duplicate rows: {dupes}")
        except Exception as e:
            log(f"- Could not compute duplicates (unhashable types): {e}")

        log("\n🔠 Data Types:")
        dtypes_text = tabulate(df.dtypes.to_frame(name='DType'), headers=['Column', 'DType'], tablefmt='pretty')
        log(dtypes_text)

def summarize_parquet(path, top_n=5):
    print(f"\n📂 Loading Parquet file: {path}")
    df = pd.read_parquet(path)
    pretty_print_top_rows(df, n=top_n)
    global_stats(df)

# CLI Usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize a Parquet file with pretty output and reports.")
    parser.add_argument("--file", required=True, help="Path to the Parquet file.")
    parser.add_argument("--top", type=int, default=5, help="Number of top rows to show (default=5).")
    args = parser.parse_args()

    summarize_parquet(args.file, args.top)
