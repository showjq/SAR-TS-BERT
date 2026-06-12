import zarr
import numpy as np
import torch
from torch.utils.data import Dataset

class MonthlyParcelDataset(Dataset):
    def __init__(self, zarr_path):
        self.store = zarr.open(zarr_path, mode='r')
        self.series = self.store['time_series']
        self.monthly_labels = self.store['monthly_labels']
        self.parcel_ids = None
        if 'parcel_ids' in self.store:
            self.parcel_ids = self.store['parcel_ids']
        self.T = self.series.shape[1]
        self.class_names = self.store.attrs.get('class_names', None)
        self.month_names = self.store.attrs.get('month_names', None)
        self.index_base = int(self.store.attrs.get('index_base', 0))
        self.num_months = self.monthly_labels.shape[1]
        self.num_classes = len(self.class_names) if self.class_names else int(np.max(self.monthly_labels) + 1)
    
    def __len__(self):
        return self.series.shape[0]
    
    def __getitem__(self, idx):
        x = np.asarray(self.series[idx], dtype=np.float32)
        y = np.asarray(self.monthly_labels[idx], dtype=np.int64)
        ti = np.arange(self.T, dtype=np.int64)
        return {
            'series': torch.from_numpy(x),
            'time_idx': torch.from_numpy(ti),
            'monthly_labels': torch.from_numpy(y),
        }
