#!/usr/bin/env python3
"""
Filter Raw EMG Data: Extract Only EMG + Timestamps
===================================================
Reads raw CSVs from raw_g1/ (each ~680MB with 112 columns) and
writes smaller CSVs to DATA/P{N}/Gesture_1.csv containing only the
16 columns needed (8 EMG values + 8 EMG timestamps).

Reduces file size from ~680MB to ~100MB per participant.

Usage:
    python3 scripts/filter_raw_emg.py             # process all
    python3 scripts/filter_raw_emg.py 1 3 5       # process specific files

Mapping:
    raw_g1/1.csv → DATA/P1/Gesture_1.csv
    raw_g1/2.csv → DATA/P2/Gesture_1.csv
    ...
"""

import pandas as pd
import os
import sys
import time

RAW_DIR = 'raw_g1'
OUT_DIR = 'DATA'
CHUNK_SIZE = 50000  # rows per chunk for streaming


def filter_one_file(raw_path: str, out_path: str) -> bool:
    """Stream-read EMG columns only and write to out_path."""
    if not os.path.exists(raw_path):
        print(f"  SKIP: {raw_path} not found")
        return False

    # Read header only
    header = pd.read_csv(raw_path, nrows=0)
    all_cols = header.columns.tolist()

    # Keep only EMG value + EMG time series columns
    emg_cols = [c for c in all_cols
                if 'EMG 1 (mV)' in c or 'EMG 1 Time Series' in c]

    if not emg_cols:
        print(f"  ERROR: No EMG columns found in {raw_path}")
        return False

    print(f"  Found {len(emg_cols)} EMG-related columns")
    print(f"  Streaming rows (chunk size={CHUNK_SIZE:,})...")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    start = time.time()
    total_rows = 0
    first_chunk = True

    try:
        for i, chunk in enumerate(pd.read_csv(
                raw_path, usecols=emg_cols,
                chunksize=CHUNK_SIZE, low_memory=False)):
            chunk.to_csv(out_path,
                         mode='w' if first_chunk else 'a',
                         header=first_chunk,
                         index=False)
            first_chunk = False
            total_rows += len(chunk)
            if i % 5 == 0:
                print(f"    chunk {i}: {total_rows:,} rows processed")
    except Exception as e:
        print(f"  ERROR during processing: {e}")
        return False

    elapsed = time.time() - start
    in_size = os.path.getsize(raw_path) / (1024 * 1024)
    out_size = os.path.getsize(out_path) / (1024 * 1024)

    print(f"  DONE: {total_rows:,} rows | {in_size:.0f}MB -> {out_size:.0f}MB "
          f"(-{(1 - out_size / in_size) * 100:.0f}%) | {elapsed:.1f}s")
    return True


def main():
    if len(sys.argv) > 1:
        indices = [int(x) for x in sys.argv[1:]]
    else:
        # Auto-discover all N.csv files
        indices = []
        for f in sorted(os.listdir(RAW_DIR)):
            name, ext = os.path.splitext(f)
            if ext == '.csv' and name.isdigit():
                indices.append(int(name))

    if not indices:
        print(f"No numbered .csv files found in {RAW_DIR}/")
        sys.exit(1)

    print(f"Processing: {indices}")
    print(f"  Source: {RAW_DIR}/N.csv")
    print(f"  Output: {OUT_DIR}/PN/Gesture_1.csv\n")

    success = 0
    total_start = time.time()

    for idx in indices:
        raw_path = os.path.join(RAW_DIR, f'{idx}.csv')
        out_path = os.path.join(OUT_DIR, f'P{idx}', 'Gesture_1.csv')

        print(f"{'=' * 60}")
        print(f"P{idx}: {raw_path} -> {out_path}")
        print(f"{'=' * 60}")

        if filter_one_file(raw_path, out_path):
            success += 1

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Done: {success}/{len(indices)} files processed in {total_elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
