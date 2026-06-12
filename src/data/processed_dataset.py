import zarr
import numpy as np
import torch
from torch.utils.data import Dataset

class ProcessedParcelDataset(Dataset):
    def __init__(self, zarr_path, time_indices=None, pad_bands=None, band_indices=None):
        self.store=zarr.open(zarr_path,mode='r')
        self.series=self.store['time_series']
        self.labels=self.store['labels']
        self.parcel_ids = None
        if 'parcel_ids' in self.store:
            self.parcel_ids = self.store['parcel_ids']
        self.T_full=self.series.shape[1]
        if time_indices is not None:
            self.time_indices = np.array(time_indices, dtype=np.int64)
            self.T = len(self.time_indices)
        else:
            self.time_indices = None
            self.T = self.T_full
        self.class_names=self.store.attrs.get('class_names', None)
        self.index_base=int(self.store.attrs.get('index_base',0))
        self.pad_bands = pad_bands
        self.band_indices = band_indices
        self.original_bands = self.series.shape[2]
        if self.band_indices is not None:
            self.original_bands = len(self.band_indices)
        
        unique_labels = np.unique(np.asarray(self.labels))
        if len(unique_labels) > 1 or unique_labels[0] != 0:
            self.label_map = {}
            for i, label in enumerate(unique_labels):
                self.label_map[int(label)] = i
        else:
            self.label_map = {0: 0}
        
        if self.class_names is None:
            self.class_names = [f'Class_{i}' for i in range(len(self.label_map))]
        elif len(self.class_names) != len(self.label_map):
            print(f"警告: class_names长度 {len(self.class_names)} 与标签映射长度 {len(self.label_map)} 不一致")
            self.class_names = [f'Class_{i}' for i in range(len(self.label_map))]
    def __len__(self):
        return self.series.shape[0]
    def __getitem__(self, idx):
        x=np.asarray(self.series[idx],dtype=np.float32)
        if self.time_indices is not None:
            x = x[self.time_indices, :]
        if self.band_indices is not None:
            x = x[:, self.band_indices]
        nan_mask = np.isnan(x)
        if nan_mask.any():
            x = np.nan_to_num(x, nan=0.0)
        if self.pad_bands is not None and self.original_bands < self.pad_bands:
            pad_width = self.pad_bands - self.original_bands
            x = np.pad(x, ((0,0),(0,pad_width)), mode='constant', constant_values=0.0)
        y_original = int(self.labels[idx])
        y = self.label_map.get(y_original, 0)
        ti=np.arange(self.T,dtype=np.int64)
        return {
            'series':torch.from_numpy(x),
            'time_idx':torch.from_numpy(ti),
            'label':torch.tensor(y,dtype=torch.long),
        }
