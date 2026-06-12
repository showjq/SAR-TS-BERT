import argparse
import os
import re
import numpy as np
import rasterio
import zarr
from dateutil import parser as dateparser

def _extract_date(path):
    name = os.path.basename(path)
    m = re.search(r"(\d{8})", name)
    if m:
        return dateparser.parse(m.group(1))
    return dateparser.parse("1970-01-01")

def _list_tifs(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.tif')]
    files.sort(key=lambda p: _extract_date(p))
    return files

def _read_shape_meta(path):
    with rasterio.open(path) as src:
        h = src.height
        w = src.width
        bands = src.count
        crs = str(src.crs)
        transform = tuple(src.transform)
    return h, w, bands, crs, transform

def _read_array(path):
    with rasterio.open(path) as src:
        arr = src.read(out_dtype='float32')
    return np.transpose(arr, (1, 2, 0))

def _check_consistent_shapes(files):
    """检查所有TIFF文件是否具有相同的尺寸和波段数"""
    if not files:
        return None
    
    first_h, first_w, first_bands, _, _ = _read_shape_meta(files[0])
    consistent = True
    
    for i, file in enumerate(files[1:], 1):
        h, w, bands, _, _ = _read_shape_meta(file)
        if h != first_h or w != first_w:
            print(f"警告: 文件 {os.path.basename(file)} 的尺寸 ({h}, {w}) 与第一个文件 ({first_h}, {first_w}) 不一致")
            consistent = False
        if bands != first_bands:
            print(f"警告: 文件 {os.path.basename(file)} 的波段数 ({bands}) 与第一个文件 ({first_bands}) 不一致")
            consistent = False
    
    if not consistent:
        # 找到最大尺寸
        max_h, max_w = first_h, first_w
        for file in files[1:]:
            h, w, _, _, _ = _read_shape_meta(file)
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        print(f"将使用最大尺寸: ({max_h}, {max_w})")
        return max_h, max_w, first_bands
    
    return first_h, first_w, first_bands

def _pad_array_to_size(arr, target_h, target_w):
    """将数组填充到目标尺寸"""
    h, w, c = arr.shape
    if h == target_h and w == target_w:
        return arr
    
    # 计算需要填充的边距
    pad_h = target_h - h
    pad_w = target_w - w
    
    # 在底部和右侧填充0
    if pad_h > 0 or pad_w > 0:
        padded_arr = np.zeros((target_h, target_w, c), dtype=arr.dtype)
        padded_arr[:h, :w, :] = arr
        return padded_arr
    
    # 如果目标尺寸更小，则裁剪（通常不会发生）
    return arr[:target_h, :target_w, :]

def run(input_dir, output_path, chunk, clevel):
    files = _list_tifs(input_dir)
    if not files:
        raise RuntimeError(f"No .tif files found in {input_dir}")
    
    # 检查并获取一致的尺寸和波段数
    h, w, bands = _check_consistent_shapes(files)
    _, _, _, crs, transform = _read_shape_meta(files[0])
    
    print(f"创建Zarr数据集，包含 {len(files)} 个文件")
    print(f"数据集尺寸: ({len(files)}, {h}, {w}, {bands})")
    print(f"分块大小: (1, {chunk}, {chunk}, {bands})")
    
    try:
        # 创建根组 - 使用无压缩模式
        root = zarr.open_group(output_path, mode='w')
        
        # 创建观测数据数组 - 无压缩
        data = root.create(
            name='observations',
            shape=(len(files), h, w, bands),
            chunks=(1, chunk, chunk, bands),
            dtype='float32'
        )
        
        # 创建时间戳数组
        ts = root.create(
            name='timestamps',
            shape=(len(files),),
            dtype='M8[s]'
        )
        
        # 保存元数据
        root.attrs['crs'] = crs
        root.attrs['transform'] = transform
        root.attrs['original_files'] = [os.path.basename(f) for f in files]
        root.attrs['bands'] = bands
        
        print("Zarr数据集结构创建成功")
        
    except Exception as e:
        raise RuntimeError(f"创建Zarr文件失败: {e}")
    
    # 写入数据
    print("开始写入数据...")
    for i, f in enumerate(files):
        try:
            arr = _read_array(f)
            # 确保数组尺寸一致
            arr_padded = _pad_array_to_size(arr, h, w)
            data[i] = arr_padded
            
            dt = _extract_date(f)
            ts[i] = np.datetime64(dt)
            
            print(f"处理进度: {i+1}/{len(files)} - {os.path.basename(f)}")
            
        except Exception as e:
            print(f"处理文件 {f} 时出错: {e}")
            continue
    
    print(f"Zarr数据集创建完成: {output_path}")
    print(f"数据集形状: {data.shape}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', type=str, default='E:/DATA/Chongming-6band')
    p.add_argument('--output_path', type=str, default="F:/DATA/Chongming-6bands")
    p.add_argument('--chunk', type=int, default=512)
    p.add_argument('--clevel', type=int, default=3)
    a = p.parse_args()
    run(a.input_dir, a.output_path, a.chunk, a.clevel)

if __name__ == '__main__':
    main()