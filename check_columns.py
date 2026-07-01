import pandas as pd
import numpy as np
import os

def compare_data_shapes(raw_csv_path, clean_npy_path):
    """
    Compares the dimensions of your raw CSV file and your cleaned .npy file.
    """
    # 1. Load Raw CSV
    if not os.path.exists(raw_csv_path):
        print(f"Error: Raw file {raw_csv_path} not found.")
        return
    df_raw = pd.read_csv(raw_csv_path)
    
    # 2. Load Cleaned .npy
    if not os.path.exists(clean_npy_path):
        print(f"Error: Clean file {clean_npy_path} not found.")
        return
    data_clean = np.load(clean_npy_path)

    print(f"--- Dimension Comparison ---")
    print(f"Raw CSV:   {raw_csv_path}")
    print(f"   Shape: {df_raw.shape}")
    
    print(f"Clean .npy: {clean_npy_path}")
    print(f"   Shape: {data_clean.shape}")
    
    # Calculate difference
    # Note: CSV raw shape includes the label column, while .npy usually does not
    print(f"\nNote: The Raw CSV shape includes labels and metadata.")
    print(f"The Clean .npy shape represents only the features kept for training.")

if __name__ == "__main__":
    # Point these to your specific files
    raw_csv = "datasets/SWaT/swat_train2.csv"
    clean_npy = "datasets/SWaT/processed/train.npy"
    
    compare_data_shapes(raw_csv, clean_npy)
