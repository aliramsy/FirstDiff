import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

class TrainDataset(Dataset):
    """
    Updated to match CSV logic: Returns (sample, history)
    """
    def __init__(self, data_path, transform, window_size, stride, tmp_threshold, history_size, scaler=None):
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.history_size = history_size
        self.scaler = scaler if scaler is not None else StandardScaler()
        data_np = np.load(data_path)
        
        thre = int(data_np.shape[0] * tmp_threshold)
        train_data_np = data_np[:thre]
        
        train_data_scaled = self.scaler.fit_transform(train_data_np)
        self.data = torch.from_numpy(train_data_scaled).type(torch.float32)
        self.dimensions = self.data.shape[1]
        
        # Logic from TrainDatasetCSV
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
            sample = self.transform(sample)
            history = self.transform(history)
        return sample, history

class ValDataset(Dataset):
    """
    Updated to match CSV logic: Returns (sample, history)
    """
    def __init__(self, data_path, transform, window_size, stride, tmp_threshold, history_size, scaler):
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.history_size = history_size
        data_np = np.load(data_path)
        
        thre = int(data_np.shape[0] * tmp_threshold)
        val_data_np = data_np[thre:]
        
        val_data_scaled = scaler.transform(val_data_np)
        self.data = torch.from_numpy(val_data_scaled).type(torch.float32)
        self.dimensions = self.data.shape[1]
        
        # Logic from ValDatasetCSV
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
            sample = self.transform(sample)
            history = self.transform(history)
        return sample, history
    
class TestDataset(Dataset):
    """
    Updated to match CSV logic: Returns (sample, history, labels, hist_labels)
    """
    def __init__(self, data_path, labels_path, transform, window_size, stride, history_size, scaler):
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.history_size = history_size
        
        test_data_np = np.load(data_path)
        test_data_scaled = scaler.transform(test_data_np)
        self.data = torch.from_numpy(test_data_scaled).type(torch.float32)
        
        labels_np = np.load(labels_path)
        # Reshape to match CSV's .reshape(-1) behavior for consistency
        self.labels = torch.from_numpy(labels_np).reshape(-1).type(torch.float32)
        self.dimensions = self.data.shape[1]
        
        assert len(self.data) == len(self.labels), "Test data and labels must have the same length!"
        
        # Logic from TestDatasetCSV
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
        
        # Labels logic from TestDatasetCSV
        labels = self.labels[hist_end:sample_end]
        hist_labels = self.labels[start_idx:hist_end]
        
        if self.transform:
            sample = self.transform(sample)
            history = self.transform(history)
        return sample, history, labels, hist_labels
    
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
