import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd


class TrainDataset(Dataset):
    """
    tmp_threshold - limit training data
    """
    def __init__(self, data_path, scaler, transform, window_size, stride, tmp_threshold):

        self.transform = transform
        self.window_size = window_size
        self.stride = stride

        data_np = scaler.transform(np.load(data_path))
        thre = int(data_np.shape[0] * tmp_threshold)

        self.data = torch.from_numpy(data_np)[:thre].float()
        self.dimensions = self.data.shape[1]
        self.start_indices = list(range(0, len(self.data), self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        end = start + self.window_size

        if end <= len(self.data):
            sample = self.data[start:end]
        else:
            sample = self.data[start:]
            pad_len = self.window_size - (len(self.data) - start)
            padding = torch.zeros((pad_len, self.dimensions), dtype=self.data.dtype)
            sample = torch.cat([sample, padding], dim=0)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class ValDataset(Dataset):
    """
    tmp_threshold - limit validation data
    """
    def __init__(self, data_path, scaler, transform, window_size, stride, tmp_threshold):

        self.transform = transform
        self.window_size = window_size
        self.stride = stride

        data_np = scaler.transform(np.load(data_path))
        thre = int(data_np.shape[0] * (1. - tmp_threshold))

        self.data = torch.from_numpy(data_np)[thre:].float()
        self.dimensions = self.data.shape[1]
        self.start_indices = list(range(0, len(self.data), self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        end = start + self.window_size

        if end <= len(self.data):
            sample = self.data[start:end]
        else:
            sample = self.data[start:]
            pad_len = self.window_size - (len(self.data) - start)
            padding = torch.zeros((pad_len, self.dimensions), dtype=self.data.dtype)
            sample = torch.cat([sample, padding], dim=0)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class TestDataset(Dataset):
    """
    returns seq_len, sensors
    """
    def __init__(self, data_path, labels_path, scaler, transform, window_size, stride, tmp_threshold):

        self.transform = transform
        self.window_size = window_size
        self.stride = stride

        data_np = scaler.transform(np.load(data_path))
        thre = int(data_np.shape[0] * tmp_threshold)

        self.data = torch.from_numpy(data_np)[:thre].float()
        labels_np = np.load(labels_path)
        self.labels = torch.from_numpy(labels_np)[:thre].float()

        self.dimensions = self.data.shape[1]
        self.start_indices = list(range(0, len(self.data), self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        end = start + self.window_size

        if end <= len(self.data):
            sample = self.data[start:end]
            labels = self.labels[start:end]
        else:
            sample = self.data[start:]
            labels = self.labels[start:]

            pad_len = self.window_size - (len(self.data) - start)

            padding_data = torch.zeros((pad_len, self.dimensions), dtype=self.data.dtype)
            padding_labels = torch.zeros(pad_len, dtype=self.labels.dtype)

            sample = torch.cat([sample, padding_data], dim=0)
            labels = torch.cat([labels, padding_labels], dim=0)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, labels


class TrainDatasetCSV(Dataset):
    """
    tmp_threshold - limit training data
    """
    def __init__(self, data_path, scaler, transform, window_size, stride, tmp_threshold):

        self.transform = transform
        self.window_size = window_size
        self.stride = stride

        df = pd.read_csv(data_path)

        if 'PSM' in data_path:
            data_np = df.values[:, 1:]
            data_np = np.nan_to_num(data_np)
        else:
            data_np = df.values[:, :-1]

        data_np = scaler.transform(data_np)

        thre = int(data_np.shape[0] * tmp_threshold)

        self.data = torch.from_numpy(data_np)[:thre].float()
        self.dimensions = self.data.shape[1]
        self.start_indices = list(range(0, len(self.data), self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        end = start + self.window_size

        if end <= len(self.data):
            sample = self.data[start:end]
        else:
            sample = self.data[start:]
            pad_len = self.window_size - (len(self.data) - start)
            padding = torch.zeros((pad_len, self.dimensions), dtype=self.data.dtype)
            sample = torch.cat([sample, padding], dim=0)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class ValDatasetCSV(Dataset):
    """
    tmp_threshold - limit validation data
    """
    def __init__(self, data_path, scaler, transform, window_size, stride, tmp_threshold):

        self.transform = transform
        self.window_size = window_size
        self.stride = stride

        df = pd.read_csv(data_path)

        if 'PSM' in data_path:
            data_np = df.values[:, 1:]
            data_np = np.nan_to_num(data_np)
        else:
            data_np = df.values[:, :-1]

        data_np = scaler.transform(data_np)

        thre = int(data_np.shape[0] * (1. - tmp_threshold))

        self.data = torch.from_numpy(data_np)[thre:].float()
        self.dimensions = self.data.shape[1]
        self.start_indices = list(range(0, len(self.data), self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        end = start + self.window_size

        if end <= len(self.data):
            sample = self.data[start:end]
        else:
            sample = self.data[start:]
            pad_len = self.window_size - (len(self.data) - start)
            padding = torch.zeros((pad_len, self.dimensions), dtype=self.data.dtype)
            sample = torch.cat([sample, padding], dim=0)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class TestDatasetCSV(Dataset):
    """
    returns seq_len, sensors
    """
    def __init__(self, data_path, labels_path, scaler, transform, window_size, stride, tmp_threshold):

        self.transform = transform
        self.window_size = window_size
        self.stride = stride

        df = pd.read_csv(data_path)

        if 'PSM' in data_path:
            data_np = df.values[:, 1:]
        else:
            data_np = df.values[:, :-1]

        data_np = scaler.transform(data_np)

        thre = int(data_np.shape[0] * tmp_threshold)

        self.data = torch.from_numpy(data_np)[:thre].float()

        if labels_path is not None:
            df_labels = pd.read_csv(labels_path)
            labels_np = df_labels.values[:, 1:]
        else:
            labels_np = df.values[:, -1:]

        self.labels = torch.from_numpy(labels_np)[:thre].reshape(-1).float()

        self.dimensions = self.data.shape[1]
        self.start_indices = list(range(0, len(self.data), self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        end = start + self.window_size

        if end <= len(self.data):
            sample = self.data[start:end]
            labels = self.labels[start:end]
        else:
            sample = self.data[start:]
            labels = self.labels[start:]

            pad_len = self.window_size - (len(self.data) - start)

            padding_data = torch.zeros((pad_len, self.dimensions), dtype=self.data.dtype)
            padding_labels = torch.zeros(pad_len, dtype=self.labels.dtype)

            sample = torch.cat([sample, padding_data], dim=0)
            labels = torch.cat([labels, padding_labels], dim=0)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, labels