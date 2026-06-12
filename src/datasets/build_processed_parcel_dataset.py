import argparse
import os
import json
import numpy as np
import zarr
import fiona
from affine import Affine
from shapely.geometry import shape
from rasterio.transform import rowcol
from rasterio import features
from rasterio.warp import transform_geom
from tqdm import tqdm

def _load_label_info(label_json_path):
    names=None
    counts=None
    if label_json_path and os.path.exists(label_json_path):
        with open(label_json_path,'r',encoding='utf-8') as f:
            info=json.load(f)
        names=info.get('label_names',None)
        counts_dict=info.get('num',None)
        if counts_dict:
            # numeric keys ("0","1",...) → list
            max_key=max(int(k) for k in counts_dict.keys() if k!='sample')
            counts=[counts_dict.get(str(i),0) for i in range(max_key+1)]
    return names, counts

def _window_mask(geom, aff, H, W):
    bx,by,bx2,by2=shape(geom).bounds
    r0,c0=rowcol(aff,bx,by2)
    r1,c1=rowcol(aff,bx2,by)
    r0=max(0,min(H-1,r0)); r1=max(0,min(H-1,r1))
    c0=max(0,min(W-1,c0)); c1=max(0,min(W-1,c1))
    if r1<r0 or c1<c0:
        return None
    h=r1-r0+1; w=c1-c0+1
    sub_aff=aff*Affine.translation(c0,r0)
    mask=features.geometry_mask([geom], out_shape=(h,w), transform=sub_aff, invert=True)
    ys,xs=np.nonzero(mask)
    if len(ys)==0:
        return None
    return r0,c0,ys,xs

def _select_parcels(src, zcrs, max_per_class, class_field='label', crop_name_map=None, seed=42):
    rng=np.random.default_rng(seed)
    by_class={}
    pool=[]
    for fi, feat in enumerate(src):
        if crop_name_map:
            crop_name=feat['properties'].get('crop_name','')
            lab=crop_name_map.get(crop_name, -1)
        else:
            lab=feat['properties'].get(class_field,0)
            try:
                lab=int(lab)
            except Exception:
                lab=0
        if lab < 0:
            continue
        pool.append({'lab':lab,'geom':feat['geometry'],'fid':fi})
        by_class.setdefault(lab,[]).append(pool[-1])
    selected=[]
    for lab, arr in by_class.items():
        rng.shuffle(arr)
        k=len(arr) if max_per_class is None else min(len(arr), int(max_per_class))
        selected.extend(arr[:k])
    return selected

def run(zarr_path, vector_path, label_json, out_path, max_pixels_per_parcel, max_per_class, seed, limit_parcels=None):
    store=zarr.open(zarr_path,mode='r')
    data=store['observations']
    T,H,W,C=data.shape
    aff=Affine(*tuple(store.attrs.get('transform')))
    zcrs=store.attrs.get('crs')
    class_names, counts=_load_label_info(label_json)
    crop_name_map=None
    if label_json and os.path.exists(label_json):
        with open(label_json,'r',encoding='utf-8') as f:
            info=json.load(f)
        crop_name_map=info.get('crop_name_to_label',None)
    with fiona.open(vector_path,'r') as src:
        vcrs=src.crs_wkt or src.crs
        selected=_select_parcels(src, zcrs, max_per_class, class_field='label', crop_name_map=crop_name_map, seed=seed)
    # count selected parcels per class
    sel_counts={}
    for item in selected:
        sel_counts[item['lab']]=sel_counts.get(item['lab'],0)+1
    if limit_parcels:
        selected=selected[:int(limit_parcels)]
    # first pass: compute total samples
    total=0
    masks_cache=[]
    with fiona.open(vector_path,'r') as src:
        vcrs=src.crs_wkt or src.crs
        for item in tqdm(selected, desc='counting', total=len(selected)):
            lab=item['lab']
            geom=item['geom']
            if vcrs and zcrs and vcrs!=zcrs:
                geom=transform_geom(vcrs,zcrs,geom)
            res=_window_mask(geom, aff, H, W)
            if res is None:
                masks_cache.append(None)
                continue
            r0,c0,ys,xs=res
            n=len(ys)
            k=min(n, max_pixels_per_parcel)
            total+=k
            masks_cache.append((r0,c0,ys,xs,k))
    if total==0:
        raise RuntimeError('No samples selected')
    os.makedirs(os.path.dirname(out_path),exist_ok=True)
    root=zarr.open_group(out_path,mode='w')
    arr_series=root.create('time_series', shape=(total,T,C), chunks=(max(1,min(256,total)),T,C), dtype='float32')
    arr_labels=root.create('labels', shape=(total,), chunks=(max(1,min(1024,total)),), dtype='int32')
    arr_parcel_ids=root.create('parcel_ids', shape=(total,), chunks=(max(1,min(1024,total)),), dtype='int32')
    root.attrs['crs']=zcrs
    root.attrs['transform']=tuple(store.attrs.get('transform'))
    if class_names:
        root.attrs['class_names']=class_names
    root.attrs['index_base']=0
    # second pass: write samples
    ptr=0
    selected_fids=[]
    with fiona.open(vector_path,'r') as src:
        vcrs=src.crs_wkt or src.crs
        for i,item in enumerate(tqdm(selected, desc='writing', total=len(selected))):
            lab=item['lab']
            geom=item['geom']
            selected_fids.append(int(item['fid']))
            if vcrs and zcrs and vcrs!=zcrs:
                geom=transform_geom(vcrs,zcrs,geom)
            cache=masks_cache[i]
            if cache is None:
                continue
            r0,c0,ys,xs,k=cache
            rng=np.random.default_rng(seed+i)
            idx=rng.permutation(len(ys))[:k]
            ys_sel=ys[idx]+r0
            xs_sel=xs[idx]+c0
            ys_idx=np.asarray(ys_sel,dtype=np.int64)
            xs_idx=np.asarray(xs_sel,dtype=np.int64)
            seq_list=[]
            for t in range(T):
                img=np.asarray(data[t],dtype=np.float32)  # H,W,3
                part=img[ys_idx, xs_idx, :]  # N,3
                seq_list.append(part)
            seq=np.stack(seq_list,axis=0)  # T,N,3
            seq=np.transpose(seq,(1,0,2))  # N,T,3
            n=seq.shape[0]
            arr_series[ptr:ptr+n]=seq
            arr_labels[ptr:ptr+n]=lab
            arr_parcel_ids[ptr:ptr+n]=i
            ptr+=n
    # counts_after
    uniq, cnt=np.unique(np.asarray(arr_labels), return_counts=True)
    counts_after={int(u):int(c) for u,c in zip(uniq,cnt)}
    root.attrs['counts_after']=counts_after
    root.attrs['selected_parcels_per_class']={str(k):int(v) for k,v in sel_counts.items()}
    if counts is not None:
        root.attrs['counts_before']={str(i):int(c) for i,c in enumerate(counts)}
    try:
        arr_sel=root.create('selected_feature_indices', shape=(len(selected_fids),), chunks=(max(1,min(2048,len(selected_fids))),), dtype='int32')
        arr_sel[:]=np.asarray(selected_fids,dtype=np.int32)
        root.attrs['source_vector_path']=vector_path
    except Exception:
        pass
    # write consolidated metadata (zarr.json)
    try:
        from zarr.convenience import consolidate_metadata
        consolidate_metadata(root.store)
    except Exception:
        pass
    return out_path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zarr_path',type=str,required=True)
    p.add_argument('--vector_path',type=str,required=True)
    p.add_argument('--label_json',type=str,default=None)
    p.add_argument('--out_path',type=str,required=True)
    p.add_argument('--max_pixels_per_parcel',type=int,default=100)
    p.add_argument('--max_per_class',type=int,default=None)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--limit_parcels',type=int,default=None)
    a=p.parse_args()
    run(a.zarr_path,a.vector_path,a.label_json,a.out_path,a.max_pixels_per_parcel,a.max_per_class,a.seed,a.limit_parcels)

if __name__=='__main__':
    main()
