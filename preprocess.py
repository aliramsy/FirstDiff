import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def run_preprocessing(train_file, test_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # Apply your specific logic
    df_train.columns = df_train.columns.astype(str).str.strip()
    df_test.columns = df_test.columns.astype(str).str.strip()
    
    labels_1d = pd.to_numeric(df_test['Normal/Attack'], errors='coerce').fillna(0).astype(int).values.reshape(-1, 1)
    
    df_train = df_train.drop("Normal/Attack", axis=1, errors='ignore').apply(pd.to_numeric, errors='coerce')
    #df_test = df_test.drop("Normal/Attack", axis=1, errors='ignore').apply(pd.to_numeric, errors='coerce')
    #labels_2d = pd.to_numeric(df_test['Normal/Attack'], errors='coerce').fillna(0).astype(int).values.reshape(-1, 1)
    labels_1d = pd.to_numeric(df_test['Normal/Attack'], errors='coerce').fillna(0).astype(int).values.reshape(-1)
    common_cols = [c for c in df_train.columns if c in df_test.columns]
    df_train, df_test = df_train[common_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0), df_test[common_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    std = df_train.std(axis=0)
    keep_cols = std[std > 1e-6].index
    df_train, df_test = df_train[keep_cols], df_test[keep_cols]

    scaler = StandardScaler()
    train = scaler.fit_transform(df_train.values.astype(np.float64))
    test  = scaler.transform(df_test.values.astype(np.float64))

    # Now save it:
    #np.save(os.path.join(output_folder, 'labels.npy'), labels_2d)
    np.save(os.path.join(output_folder, 'train.npy'), train)
    np.save(os.path.join(output_folder, 'labels.npy'), labels_1d)
    np.save(os.path.join(output_folder, 'test.npy'), test)
    #np.save(os.path.join(output_folder, 'labels.npy'), labels_1d)

if __name__ == '__main__':
    run_preprocessing("datasets/SWaT/swat_train2.csv", "datasets/SWaT/swat2.csv", "datasets/SWaT/processed")
