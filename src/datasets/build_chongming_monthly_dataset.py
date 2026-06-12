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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timedelta

def get_month_list():
    months = []
    current = datetime(2023, 12, 1)
    end = datetime(2024, 10, 1)
    while current <= end:
        month_str = current.strftime('%Y-%m')
        if month_str != '2024-02':
            months.append(month_str)
        current += timedelta(days=32)
        current = current.replace(day=1)
    return months

def get_crop_mapping():
    crops = [
        '水稻', '小麦', '玉米', '大豆', '蔬菜', '油菜',
        '林地', '桔子树', '稻茬', '翻耕', '荒草'
    ]
    return {crop: i for i, crop in enumerate(crops)}

def _window_mask(geom, aff, H, W):
    bx, by, bx2, by2 = shape(geom).bounds
    r0, c0 = rowcol(aff, bx, by2)
    r1, c1 = rowcol(aff, bx2, by)
    r0 = max(0, min(H-1, r0))
    r1 = max(0, min(H-1, r1))
    c0 = max(0, min(W-1, c0))
    c1 = max(0, min(W-1, c1))
    if r1 < r0 or c1 < c0:
        return None
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    sub_aff = aff * Affine.translation(c0, r0)
    mask = features.geometry_mask([geom], out_shape=(h, w), transform=sub_aff, invert=True)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return r0, c0, ys, xs

def run(zarr_path, shp_path, json_path, out_path, max_pixels_per_parcel=100, seed=42, workers=4):
    store = zarr.open(zarr_path, mode='r')
    data = store['observations']
    T, H, W, C = data.shape
    ch = data.chunks[1]
    cw = data.chunks[2]
    aff = Affine(*tuple(store.attrs.get('transform')))
    zcrs = store.attrs.get('crs')
    
    months = get_month_list()
    num_months = len(months)
    crop_mapping = get_crop_mapping()
    class_names = list(crop_mapping.keys())
    
    print(f"月份列表: {months}")
    print(f"作物类别: {class_names}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        selected_parcels = json.load(f)
    
    with fiona.open(shp_path, 'r') as src:
        vcrs = src.crs_wkt or src.crs
        parcels = []
        id_field = None
        for col in src.schema['properties'].keys():
            if col.lower() in ['id', 'fid', 'plot_id', 'parcel_id']:
                id_field = col
                break
        if not id_field:
            id_field = list(src.schema['properties'].keys())[0]
        
        for fi, feat in enumerate(src):
            plot_id = str(feat['properties'][id_field])
            if plot_id in selected_parcels:
                crop_list = selected_parcels[plot_id]
                if len(crop_list) == num_months:
                    parcels.append({
                        'fid': fi,
                        'geom': feat['geometry'],
                        'plot_id': plot_id,
                        'crops': crop_list
                    })
    
    print(f"选中地块数量: {len(parcels)}")
    
    groups = {}
    rng = np.random.default_rng(seed)
    selected_fids = []
    
    with fiona.open(shp_path, 'r') as src:
        vcrs = src.crs_wkt or src.crs
        for i, parcel in enumerate(tqdm(parcels, desc='grouping', total=len(parcels))):
            geom = parcel['geom']
            fid = parcel['fid']
            selected_fids.append(int(fid))
            
            if vcrs and zcrs and vcrs != zcrs:
                geom = transform_geom(vcrs, zcrs, geom)
            
            res = _window_mask(geom, aff, H, W)
            if res is None:
                continue
            
            r0, c0, ys, xs = res
            n = len(ys)
            k = min(n, max_pixels_per_parcel)
            idx = rng.permutation(n)[:k]
            ys_sel = ys[idx] + r0
            xs_sel = xs[idx] + c0
            
            for y, x in zip(ys_sel, xs_sel):
                ry = y // ch
                rx = x // cw
                key = (ry, rx)
                groups.setdefault(key, {
                    'coords': [],
                    'monthly_crops': [],
                    'parcel_ids': []
                })
                groups[key]['coords'].append((int(y), int(x)))
                groups[key]['monthly_crops'].append(parcel['crops'])
                groups[key]['parcel_ids'].append(int(i))
    
    total = sum(len(g['coords']) for g in groups.values())
    if total == 0:
        raise RuntimeError('No samples selected')
    
    print(f"总像素数: {total}")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    root = zarr.open_group(out_path, mode='w')
    
    arr_series = root.create(
        'time_series', 
        shape=(total, T, C), 
        chunks=(max(1, min(512, total)), T, C), 
        dtype='float32'
    )
    arr_monthly_labels = root.create(
        'monthly_labels', 
        shape=(total, num_months), 
        chunks=(max(1, min(2048, total)), num_months), 
        dtype='int32'
    )
    arr_parcel_ids = root.create(
        'parcel_ids', 
        shape=(total,), 
        chunks=(max(1, min(2048, total)),), 
        dtype='int32'
    )
    arr_selected_fids = root.create(
        'selected_feature_indices', 
        shape=(len(selected_fids),), 
        chunks=(max(1, min(2048, len(selected_fids))),), 
        dtype='int32'
    )
    
    root.attrs['crs'] = zcrs
    root.attrs['transform'] = tuple(store.attrs.get('transform'))
    root.attrs['class_names'] = class_names
    root.attrs['month_names'] = months
    root.attrs['index_base'] = 0
    root.attrs['source_zarr_path'] = zarr_path
    root.attrs['source_vector_path'] = shp_path
    
    ptr = 0
    keys = list(groups.keys())
    
    def process_group(key):
        ry, rx = key
        y0 = ry * ch
        y1 = min(H, (ry + 1) * ch)
        x0 = rx * cw
        x1 = min(W, (rx + 1) * cw)
        window = np.asarray(data[:, y0:y1, x0:x1, :], dtype=np.float32)
        
        coords = groups[key]['coords']
        monthly_crops_list = groups[key]['monthly_crops']
        pids = groups[key]['parcel_ids']
        
        ys = np.array([c[0] - y0 for c in coords], dtype=np.int64)
        xs = np.array([c[1] - x0 for c in coords], dtype=np.int64)
        
        seq_list = []
        for t in range(T):
            frame = window[t]
            part = frame[ys, xs, :]
            seq_list.append(part)
        seq = np.stack(seq_list, axis=0).transpose(1, 0, 2)
        
        monthly_labels = np.zeros((len(coords), num_months), dtype=np.int32)
        for i, crops in enumerate(monthly_crops_list):
            for j, crop in enumerate(crops):
                monthly_labels[i, j] = crop_mapping.get(crop, 0)
        
        return seq, monthly_labels, np.array(pids, dtype=np.int32)
    
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_group, k): k for k in keys}
        for fut in tqdm(as_completed(futs), total=len(futs), desc='writing'):
            k = futs[fut]
            seq, monthly_labels, pids = fut.result()
            n = seq.shape[0]
            arr_series[ptr:ptr + n] = seq
            arr_monthly_labels[ptr:ptr + n] = monthly_labels
            arr_parcel_ids[ptr:ptr + n] = pids
            ptr += n
    
    arr_selected_fids[:] = np.asarray(selected_fids, dtype=np.int32)
    
    uniq, cnt = np.unique(np.asarray(arr_monthly_labels), return_counts=True)
    counts_after = {int(u): int(c) for u, c in zip(uniq, cnt)}
    root.attrs['counts_after'] = counts_after
    
    try:
        from zarr.convenience import consolidate_metadata
        consolidate_metadata(root.store)
    except Exception:
        pass
    
    return out_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zarr_path', type=str, default="F:/DATA/Chongming-6bands")
    p.add_argument('--shp_path', type=str, default="src/shpfiles/Chongming_plots/Chongming_plots.shp")
    p.add_argument('--json_path', type=str, default="src/shpfiles/Chongming_plots/selected_parcels.json")
    p.add_argument('--out_path', type=str, default="src/shpfiles/Chongming_monthly_crops/processed.zarr")
    p.add_argument('--max_pixels_per_parcel', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    run(a.zarr_path, a.shp_path, a.json_path, a.out_path, a.max_pixels_per_parcel, a.seed, a.workers)

if __name__ == '__main__':
    main()
