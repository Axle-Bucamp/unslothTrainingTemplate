#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import tiktoken

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────

ENCODING_NAME = "cl100k_base"

def chunk_text(text: str, max_tokens: int, overlap: int, enc) -> list[tuple[int,int,int,str]]:
    """
    Break `text` into token-based chunks.
    Returns list of (chunk_id, char_start, char_end, chunk_str).
    """
    tokens = enc.encode(text)
    total = len(tokens)
    step = max_tokens - overlap
    chunks = []

    start_token = 0
    chunk_id = 0
    while start_token < total:
        end_token = min(total, start_token + max_tokens)
        token_slice = tokens[start_token:end_token]
        chunk_str = enc.decode(token_slice)

        # approximate char offsets by decoding prefix
        prefix = enc.decode(tokens[:start_token])
        full   = enc.decode(tokens[:end_token])
        cs = len(prefix)
        ce = len(full)

        chunks.append((chunk_id, cs, ce, chunk_str))

        chunk_id += 1
        start_token += step

    return chunks

# ───────────────────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Split a Parquet's text column into token-based chunks."
    )
    parser.add_argument("--input",  "-i", required=True, help="Input Parquet file")
    parser.add_argument("--output", "-o", required=True, help="Output Parquet file")
    parser.add_argument("--column", "-c", required=True, help="Text column to chunk")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens per chunk")
    parser.add_argument("--overlap",    type=int, default=64,  help="Token overlap between chunks")

    args = parser.parse_args()

    # load parquet
    df = pd.read_parquet(args.input)
    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not in input")

    # prepare tokenizer
    enc = tiktoken.get_encoding(ENCODING_NAME)

    records = []
    # For each row, chunk the text column
    for idx, row in df.iterrows():
        text = str(row[args.column] or "")
        chunks = chunk_text(text, args.max_tokens, args.overlap, enc)

        for chunk_id, cs, ce, chunk_str in chunks:
            # copy all other fields
            rec = {col: row[col] for col in df.columns if col != args.column}
            rec.update({
                "original_index": idx,
                "chunk_id": chunk_id,
                "char_start": cs,
                "char_end": ce,
                "chunk_content": chunk_str
            })
            records.append(rec)

    # create DataFrame and save
    out_df = pd.DataFrame(records)
    out_df.to_parquet(args.output, index=False)
    print(f"✅ Wrote {len(out_df)} chunk‑rows to {args.output}")

if __name__ == "__main__":
    main()
