import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

import os
import pandas as pd


class TrainDataset(Dataset):
    """
    tmp_threshold - ratio of data to use for training (e.g., 0.8 or 1.0)
    scaler - Pass an un-fitted StandardScaler. It will be fitted here.
    """
    def __init__(self, data_path, transform, window_size, stride, tmp_threshold, history_size, scaler=None):
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.history_size = history_size
        self.scaler = scaler if scaler is not None else StandardScaler()

        data_np = np.load(data_path)
        
        # Split data before fitting scaler to prevent data leakage
        thre = int(data_np.shape[0] * tmp_threshold)
        train_data_np = data_np[:thre]
        
        # Fit and transform ONLY on train data
        train_data_scaled = self.scaler.fit_transform(train_data_np)
        self.data = torch.from_numpy(train_data_scaled).type(torch.float32)
        self.dimensions = self.data.shape[1]
        
        # Valid start indices account for the history size preceding the window
        max_start_idx = len(self.data) - self.window_size
        self.start_indices = list(range(self.history_size, max_start_idx + 1, self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_idx = self.start_indices[idx]
        
        # Target window ($X_{curr}$)
        sample = self.data[start_idx : start_idx + self.window_size]
        
        # History window ($X_{hist}$) strictly preceding the target window
        history = self.data[start_idx - self.history_size : start_idx]

        if self.transform:
            sample = self.transform(sample)
            history = self.transform(history)

        return sample, history


class ValDataset(Dataset):
    """
    tmp_threshold - ratio of data used for training (validation takes the rest)
    scaler - Pass the FITTED scaler from TrainDataset!
    """
    def __init__(self, data_path, transform, window_size, stride, tmp_threshold, history_size, scaler):
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.history_size = history_size

        data_np = np.load(data_path)
        
        # Extract validation split (the remaining data after Train)
        thre = int(data_np.shape[0] * tmp_threshold)
        val_data_np = data_np[thre:]
        
        # Transform using the fitted scaler (DO NOT FIT AGAIN)
        val_data_scaled = scaler.transform(val_data_np)
        self.data = torch.from_numpy(val_data_scaled).type(torch.float32)
        self.dimensions = self.data.shape[1]
        
        max_start_idx = len(self.data) - self.window_size
        self.start_indices = list(range(self.history_size, max_start_idx + 1, self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_idx = self.start_indices[idx]
        
        sample = self.data[start_idx : start_idx + self.window_size]
        history = self.data[start_idx - self.history_size : start_idx]

        if self.transform:
            sample = self.transform(sample)
            history = self.transform(history)

        return sample, history
    

class TestDataset(Dataset):
    """
    Returns sample, history, and labels.
    scaler - Pass the FITTED scaler from TrainDataset!
    """
    def __init__(self, data_path, labels_path, transform, window_size, stride, history_size, scaler):
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.history_size = history_size

        # Assuming test data is a separate file, load the whole thing.
        test_data_np = np.load(data_path)
        
        # Transform using the fitted scaler from training
        test_data_scaled = scaler.transform(test_data_np)
        self.data = torch.from_numpy(test_data_scaled).type(torch.float32)
        
        # Load labels 
        labels_np = np.load(labels_path)
        self.labels = torch.from_numpy(labels_np).type(torch.float32)
        self.dimensions = self.data.shape[1]
        
        # Ensure labels and data lengths match
        assert len(self.data) == len(self.labels), "Test data and labels must have the same length!"
        
        max_start_idx = len(self.data) - self.window_size
        self.start_indices = list(range(self.history_size, max_start_idx + 1, self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_idx = self.start_indices[idx]
        
        sample = self.data[start_idx : start_idx + self.window_size]
        history = self.data[start_idx - self.history_size : start_idx]
        labels = self.labels[start_idx : start_idx + self.window_size]

        if self.transform:
            sample = self.transform(sample)
            history = self.transform(history)

        return sample, history, labels
    
class TrainDatasetCSV(Dataset):
    """
    tmp_threshold - limit training data
    """
    def __init__(self, data_path, transform, window_size, stride, tmp_threshold, history_size):
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.history_size = history_size

        path = os.path.join(data_path)
        df = pd.read_csv(path)
        
        if 'PSM' in data_path:
            raw_data = df.values[:, 1:]
            raw_data = np.nan_to_num(raw_data)
        else:
            raw_data = df.values[:, :-1]
            
        # 1. Slice FIRST to prevent data leakage
        thre = int(raw_data.shape[0] * tmp_threshold)
        sliced_data = raw_data[:thre]
        
        # 2. Scale
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(sliced_data)
        
        self.data = torch.from_numpy(scaled_data).type(torch.float32)
        self.dimensions = self.data.shape[1]
        
        # 3. Calculate valid indices (History + Target Window)
        self.total_length = self.history_size + self.window_size
        max_start_idx = len(self.data) - self.total_length
        self.start_indices = list(range(0, max_start_idx + 1, self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_idx = self.start_indices[idx]
        hist_end = start_idx + self.history_size
        sample_end = hist_end + self.window_size
        
        history = self.data[start_idx:hist_end]
        sample = self.data[hist_end:sample_end]

        if self.transform:
            history = self.transform(history)

        return sample, history
    

class ValDatasetCSV(Dataset):
    """
    tmp_threshold - limit validation data
    """
    def __init__(self, data_path, transform, window_size, stride, tmp_threshold, history_size):
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.history_size = history_size

        path = os.path.join(data_path)
        df = pd.read_csv(path)
        
        if 'PSM' in data_path:
            raw_data = df.values[:, 1:]
            raw_data = np.nan_to_num(raw_data)
        else:
            raw_data = df.values[:, :-1]
            
        # Slice for validation (taking the remaining part after threshold)
        thre = int(raw_data.shape[0] * (1. - tmp_threshold))
        sliced_data = raw_data[thre:]
        
        # Note: Ideally you should use the scaler fitted on TrainDataset here. 
        # But keeping your standalone logic, we fit on the validation slice.
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(sliced_data)
        
        self.data = torch.from_numpy(scaled_data).type(torch.float32)
        self.dimensions = self.data.shape[1]
        
        self.total_length = self.history_size + self.window_size
        max_start_idx = len(self.data) - self.total_length
        self.start_indices = list(range(0, max_start_idx + 1, self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_idx = self.start_indices[idx]
        hist_end = start_idx + self.history_size
        sample_end = hist_end + self.window_size
        
        history = self.data[start_idx:hist_end]
        sample = self.data[hist_end:sample_end]

        if self.transform:
            history = self.transform(history)

        return sample, history
    

class TestDatasetCSV(Dataset):
    """
    returns sample, history, and labels for the sample window
    """
    def __init__(self, data_path, labels_path, transform, window_size, stride, tmp_threshold, history_size):
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.history_size = history_size

        path = os.path.join(data_path)
        df = pd.read_csv(path)
        
        if 'PSM' in data_path:
            raw_data = df.values[:, 1:]
        else:
            raw_data = df.values[:, :-1]
            
        thre = int(raw_data.shape[0] * tmp_threshold)
        sliced_data = raw_data[:thre]
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(sliced_data)
        self.data = torch.from_numpy(scaled_data).type(torch.float32)
        
        # Labels logic
        if labels_path is not None:
            df_labels = pd.read_csv(labels_path)
            labels_np = df_labels.values[:, 1:]
        else:
            labels_np = df.values[:, -1:]
            
        self.labels = torch.from_numpy(labels_np)[:thre].reshape(-1).type(torch.float32)
        self.dimensions = self.data.shape[1]
        
        self.total_length = self.history_size + self.window_size
        max_start_idx = len(self.data) - self.total_length
        self.start_indices = list(range(0, max_start_idx + 1, self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_idx = self.start_indices[idx]
        hist_end = start_idx + self.history_size
        sample_end = hist_end + self.window_size
        
        # for example idx is 1 and window size is 96 and history size is 1024 so
        # history is 0-1023 and winow 1024-1219
        # idx = 1, 96-1219 and window is 1220-1315
        history = self.data[start_idx:hist_end]
        sample = self.data[hist_end:sample_end]
        
        # Labels should correspond to the target sample being reconstructed
        labels = self.labels[hist_end:sample_end]
        hist_labels = self.labels[start_idx:hist_end]

        if self.transform:
            history = self.transform(history)

        return sample, history, labels, hist_labels
