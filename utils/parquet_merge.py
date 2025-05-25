import pandas as pd
import glob

file_list = glob.glob("dataset/chunk_*.parquet")

# Load all files
dfs = [pd.read_parquet(f) for f in file_list]

# Get common columns across all dataframes
common_cols = set(dfs[0].columns)
for df in dfs[1:]:
    common_cols &= set(df.columns)

# Keep only common columns and concatenate
dfs_common = [df[list(common_cols)] for df in dfs]
merged_df = pd.concat(dfs_common, ignore_index=True)

# Save the result
merged_df.to_parquet("merged_rows_common_columns.parquet")
