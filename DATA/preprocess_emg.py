import pandas as pd
import os

dir = 'P5'

input_file = dir + '/Gesture_1.csv'
output_file = dir + '/Gesture_1_filtered.csv'

def preprocess_emg(input_path, output_path):
    print(f"Processing {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # Read only headers first to identify relevant columns
    try:
        header_df = pd.read_csv(input_path, nrows=0)
    except Exception as e:
        print(f"Error reading file headers: {e}")
        return

    all_columns = header_df.columns.tolist()
    
    # Filter columns: Keep if "EMG" is in the name
    cols_to_keep = [col for col in all_columns if "EMG" in col]
    
    if not cols_to_keep:
        print("No EMG columns found.")
        return

    print(f"Found {len(cols_to_keep)} EMG-related columns.")
    
    # Process in chunks to handle large files efficiently
    chunk_size = 50000
    try:
        df_iter = pd.read_csv(input_path, usecols=cols_to_keep, chunksize=chunk_size, low_memory=False)
        
        first_chunk = True
        for i, chunk in enumerate(df_iter):
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            chunk.to_csv(output_path, mode=mode, index=False, header=header)
            first_chunk = False
            if i % 10 == 0:
                print(f"Processed chunk {i}...")
                
        print(f"Successfully created {output_path}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    preprocess_emg(input_file, output_file)
