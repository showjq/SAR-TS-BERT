import numpy as np
import zarr
import torch
from torch.utils.data import Dataset
import fiona
from rasterio import features
from rasterio.warp import transform_geom
import json
import os

class ParcelDataset(Dataset):
    def __init__(self, zarr_path, vector_path, class_field, split='train', train_ratio=0.8, val_ratio=0.1, seed=42, samples_per_epoch=10000, max_pixels_per_parcel=None, label_json_path=None):
        self.store=zarr.open(zarr_path,mode='r')
        self.data=self.store['observations']
        self.T,self.H,self.W,_=self.data.shape
        self.transform=tuple(self.store.attrs.get('transform'))
        self.crs=str(self.store.attrs.get('crs'))
        self.class_field=class_field
        self.rng=np.random.default_rng(seed)
        self.parcels=self._load_parcels(vector_path)
        self.classes,self.class_names=self._build_classes(self.parcels,label_json_path)
        self._encode_labels(self.parcels)
        self._split_parcels(split,train_ratio,val_ratio,seed)
        self.samples=samples_per_epoch
        self.max_pixels_per_parcel=max_pixels_per_parcel
    def _load_parcels(self, vector_path):
        parcels=[]
        with fiona.open(vector_path,'r') as src:
            vcrs=src.crs_wkt or src.crs
            for feat in src:
                geom=feat['geometry']
                if vcrs and self.crs and vcrs!=self.crs:
                    geom=transform_geom(vcrs, self.crs, geom)
                mask=features.geometry_mask([geom], out_shape=(self.H,self.W), transform=self.transform, invert=True)
                ys,xs=np.nonzero(mask)
                if len(ys)==0:
                    continue
                pixels=np.stack([ys,xs],axis=1)
                lab=feat['properties'].get(self.class_field)
                try:
                    lab=int(lab)
                except Exception:
                    # 若字段为字符串如 '1'，强制转为 int；如果为标签名，稍后通过映射转换
                    pass
                parcels.append({'pixels':pixels,'label':lab})
        return parcels
    def _build_classes(self, parcels, label_json_path):
        if label_json_path and os.path.exists(label_json_path):
            with open(label_json_path,'r',encoding='utf-8') as f:
                info=json.load(f)
            names=info.get('label_names', [])
            # 构造映射：假定 shapefile 中 label 为 0..C-1 的整数
            classes={i:i for i in range(len(names))}
            return classes, names
        labels=sorted({p['label'] for p in parcels})
        classes={lab:i for i,lab in enumerate(labels)}
        names=[str(lab) for lab in labels]
        return classes, names
    def _encode_labels(self, parcels):
        for p in parcels:
            p['y']=self.classes[p['label']]
    def _split_parcels(self, split, train_ratio, val_ratio, seed):
        idx=np.arange(len(self.parcels))
        self.rng.shuffle(idx)
        n_train=int(len(idx)*train_ratio)
        n_val=int(len(idx)*val_ratio)
        self.pool_idx=idx[:n_train] if split=='train' else idx[n_train:n_train+n_val] if split=='val' else idx[n_train+n_val:]
    def __len__(self):
        return self.samples
    def __getitem__(self, idx):
        pi=int(self.rng.choice(self.pool_idx))
        parcel=self.parcels[pi]
        pixels=parcel['pixels']
        if self.max_pixels_per_parcel:
            sel=self.rng.integers(0, len(pixels))
        else:
            sel=self.rng.integers(0, len(pixels))
        h,w=pixels[sel]
        x=np.asarray(self.data[:,h,w,:],dtype=np.float32)
        ti=np.arange(self.T,dtype=np.int64)
        return {
            'series':torch.from_numpy(x),
            'time_idx':torch.from_numpy(ti),
            'label':torch.tensor(parcel['y'],dtype=torch.long),
        }