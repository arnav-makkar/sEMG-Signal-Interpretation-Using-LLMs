#!/usr/bin/env python3
"""
Preprocess EMG data for all participants (P1-P5).
Filters to EMG columns only.
"""

import pandas as pd
import os

PARTICIPANTS = ['P1', 'P2', 'P3', 'P4', 'P5']
DATA_DIR = 'DATA'

def preprocess_emg(input_path, output_path):
    """Filter CSV to EMG columns only."""
    print(f"Processing {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"  ⚠️ File not found: {input_path}")
        return False

    # Read headers
    header_df = pd.read_csv(input_path, nrows=0)
    all_columns = header_df.columns.tolist()
    
    # Keep EMG columns
    cols_to_keep = [col for col in all_columns if "EMG" in col]
    
    if not cols_to_keep:
        print(f"  ⚠️ No EMG columns found")
        return False

    print(f"  Found {len(cols_to_keep)} EMG columns")
    
    # Process in chunks
    chunk_size = 50000
    df_iter = pd.read_csv(input_path, usecols=cols_to_keep, chunksize=chunk_size, low_memory=False)
    
    first_chunk = True
    for i, chunk in enumerate(df_iter):
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        chunk.to_csv(output_path, mode=mode, index=False, header=header)
        first_chunk = False
        if i % 10 == 0 and i > 0:
            print(f"    Chunk {i}...")
    
    print(f"  ✓ Saved: {output_path}")
    return True

def main():
    print("=" * 60)
    print("PREPROCESSING ALL PARTICIPANTS")
    print("=" * 60)
    
    success_count = 0
    for p in PARTICIPANTS:
        input_file = os.path.join(DATA_DIR, p, 'Gesture_1.csv')
        output_file = os.path.join(DATA_DIR, p, 'Gesture_1_filtered.csv')
        
        if preprocess_emg(input_file, output_file):
            success_count += 1
    
    print(f"\n✓ Preprocessed {success_count}/{len(PARTICIPANTS)} participants")

if __name__ == "__main__":
    main()
