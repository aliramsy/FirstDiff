import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os

# Add the parent directory to sys.path so 'datasets' is findable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now you can import datasets normally
from datasets import * 

class MRDDFrequencyDecomposer(nn.Module):
    """
    Multi-Resolution Frequency Decomposition (MRDD-style)

    Splits a time series into:
        - low-frequency (trend)
        - mid-frequency (structure)
        - high-frequency (detail)

    Input:  x (B, T, C)
    Output: x_low, x_mid, x_high  (each B, T, C)
    """

    def __init__(self, low_cutoff=0.01, mid_cutoff=0.05):
        """
        low_cutoff  : fraction of Nyquist frequency (0-1)
        mid_cutoff  : fraction of Nyquist frequency (0-1)
        """
        super().__init__()
        assert 0 < low_cutoff < mid_cutoff < 1, \
            "Require 0 < low_cutoff < mid_cutoff < 1"

        self.low_cutoff = low_cutoff
        self.mid_cutoff = mid_cutoff

    def forward(self, x):
        """
        x: (B, T, C)
        """
        B, T, C = x.shape
        device = x.device

        # FFT along time axis
        X_f = torch.fft.rfft(x, dim=1)   # (B, F, C)

        #power = torch.mean(torch.abs(X_f)**2, dim=(0,2))  # average over batch and channels
#
        #import matplotlib.pyplot as plt
        #plt.plot(power.cpu().numpy())
        #plt.show()
        freqs = torch.fft.rfftfreq(T, device=device)  # (F,)

        # Normalize frequency to [0,1] (Nyquist = 0.5)
        # So divide by max(freqs)
        freqs_norm = freqs / freqs.max()

        # Create masks
        low_mask = freqs_norm < self.low_cutoff
        mid_mask = (freqs_norm >= self.low_cutoff) & (freqs_norm < self.mid_cutoff)
        high_mask = freqs_norm >= self.mid_cutoff

        # Expand masks for broadcasting
        low_mask = low_mask[None, :, None]
        mid_mask = mid_mask[None, :, None]
        high_mask = high_mask[None, :, None]

        # Apply masks
        X_low = X_f * low_mask
        X_mid = X_f * mid_mask
        X_high = X_f * high_mask

        # Inverse FFT (reconstruct each band)
        x_low = torch.fft.irfft(X_low, n=T, dim=1)
        x_mid = torch.fft.irfft(X_mid, n=T, dim=1)
        x_high = torch.fft.irfft(X_high, n=T, dim=1)
        #x_low = x_low / (x_low.std(dim=1, keepdim=True) + 1e-6)
        #x_mid = x_mid / (x_mid.std(dim=1, keepdim=True) + 1e-6)
        #x_high = x_high / (x_high.std(dim=1, keepdim=True) + 1e-6)

        return x_low, x_mid, x_high
    
if __name__ == "__main__":
    BATCH_SIZE = 64
    WINDOW_SIZE = 128
    STRIDE = 96
    GPU_ID = 0
    EPOCHS = 20
    NOISE_STEPS = 100
    HISTORY_SIZE = 1024
    DiT_num_layers = 6
    RUN = 1
    INFO = f"GAT_SA_EXPTMP_TMPTMP_DEL_RUN_{RUN}"
    D_MODEL = 256
    HISTORY = True
    TRAIN_PATH = {'MSL': "datasets/MSL/MSL_train.npy", 'SMAP': "datasets/SMAP/SMAP_train.npy", 'SWaT': "../datasets/SWaT/swat_train2.csv",
                'PSM': "datasets/PSM/train.csv"}
    TEST_PATH = {'MSL': "datasets/MSL/MSL_test.npy", 'SMAP': "datasets/SMAP/SMAP_test.npy",  'SWaT': "../datasets/SWaT/swat2.csv",
                 'PSM': "datasets/PSM/test.csv"}
    TEST_LABEL_PATH = {'MSL': "datasets/MSL/MSL_test_label.npy", 'SMAP': "datasets/SMAP/SMAP_test_label.npy",  'SWaT': None,
                       'PSM': "datasets/PSM/test_label.csv"}
    CHANNELS = {'MSL': 55, 'SMAP': 25,  'SWaT': 51, 'PSM': 25}


    CHANNELS = {'SWaT': 51}

    dataset = "SWaT"

    if dataset in ['SWaT', 'PSM']:
        df = pd.read_csv(TRAIN_PATH[dataset])
        if dataset == 'PSM':
            train_raw = df.values[:, 1:]
            train_raw = np.nan_to_num(train_raw)
        else:
            train_raw = df.values[:, :-1]
    else:
        train_raw = np.load(TRAIN_PATH[dataset])

    scaler = StandardScaler()
    scaler.fit(train_raw)

    if dataset in ['SWaT', 'PSM']:
        train = TrainDatasetCSV(TRAIN_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 0.85)
        val = ValDatasetCSV(TRAIN_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 0.85)
        test = TestDatasetCSV(TEST_PATH[dataset], TEST_LABEL_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 1.0)
    else:
        train = TrainDataset(TRAIN_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 0.85)
        val = ValDataset(TRAIN_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 0.85)
        test = TestDataset(TEST_PATH[dataset], TEST_LABEL_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 1.0)

    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    for i, data in enumerate(train_dataloader):
        inputs = data
        x = inputs
        break

    decomposer = MRDDFrequencyDecomposer()
    print(x.shape)
    x_low, x_mid, x_high = decomposer(x)
    x_list = [x_low, x_mid, x_high]
    for xs in x_list:
        recon = xs 
        print(torch.mean((recon - x)**2))