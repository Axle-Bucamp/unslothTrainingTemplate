#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import tiktoken
from tqdm import tqdm

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
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per chunk")
    parser.add_argument("--overlap",    type=int, default=64,  help="Token overlap between chunks")
    parser.add_argument("--save_every", type=int, default=500, help="Save to output every N rows processed")

    args = parser.parse_args()

    # Load input
    df = pd.read_parquet(args.input)
    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not in input")

    # Prepare tokenizer
    enc = tiktoken.get_encoding(ENCODING_NAME)

    output_path = Path(args.output)
    if output_path.exists():
        print(f"🟡 Output file already exists: {args.output} — appending to it.")
        out_df = pd.read_parquet(output_path)
    else:
        out_df = pd.DataFrame()

    records = []
    # Iterate with progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔨 Chunking rows"):
        text = str(row[args.column] or "").replace('<|endoftext|>', '[END_OF_TEXT]')
        chunks = chunk_text(text, args.max_tokens, args.overlap, enc)

        for chunk_id, cs, ce, chunk_str in chunks:
            rec = {col: row[col] for col in df.columns if col != args.column}
            rec.update({
                "original_index": idx,
                "chunk_id": chunk_id,
                "char_start": cs,
                "char_end": ce,
                "chunk_content": chunk_str
            })
            records.append(rec)

        # Save every N rows
        if (idx + 1) % args.save_every == 0 or idx + 1 == len(df):
            if records:
                chunk_df = pd.DataFrame(records)
                out_df = pd.concat([out_df, chunk_df], ignore_index=True)
                out_df.to_parquet(output_path, index=False)
                print(f"💾 Intermediate save: {len(out_df)} total chunk-rows written to {args.output}")
                records.clear()

    print(f"\n✅ Finalized {len(out_df)} total chunk‑rows to {args.output}")

if __name__ == "__main__":
    main()
