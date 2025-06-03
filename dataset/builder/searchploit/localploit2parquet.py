import os
import pandas as pd
from pathlib import Path

SECTIONS = ["exploits", "shellcodes"]
ROOT_PATH = Path("/opt/exploitdb")
OUTPUT_FILE = "dataset/exploitdb_files.parquet"


def get_category_and_metadata(file_path: Path):
    # Extract category from path
    for section in SECTIONS:
        if section in file_path.parts:
            return section
    return "unknown"


def scan_files():
    records = []

    for section in SECTIONS:
        base_dir = ROOT_PATH / section
        if not base_dir.exists():
            print(f"[!] Missing section directory: {base_dir}")
            continue

        for root, dirs, files in os.walk(base_dir):
            for file in files:
                file_path = Path(root) / file

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        raw_text = f.read()
                except Exception as e:
                    print(f"[!] Failed to read {file_path}: {e}")
                    continue

                category = get_category_and_metadata(file_path)
                record = {
                    "path": str(file_path),
                    "category": category,
                    "filename": file_path.name,
                    "extension": file_path.suffix,
                    "raw_text": raw_text
                }
                records.append(record)

    return records


def save_parquet(records, output_path):
    df = pd.DataFrame(records)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"[✔] Saved {len(df)} entries to {output_path}")


def main():
    print(f"[+] Scanning {ROOT_PATH} for raw files...")
    records = scan_files()
    if records:
        save_parquet(records, OUTPUT_FILE)
    else:
        print("[!] No files found or processed.")


if __name__ == "__main__":
    main()
