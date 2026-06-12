import argparse
import os
import json
import numpy as np
import zarr
import rasterio
from affine import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def _load_label_info(dbf_path):
    """从tif的DBF文件中加载标签信息"""
    # 不依赖geopandas的简化实现
    # 直接返回None，后续会根据实际标签值生成class_names和label_map
    print("警告: 未安装geopandas，使用简化的标签映射")
    return None, None


def run(zarr_path, tif_path, out_path, max_per_class=None, min_samples=100000, seed=42, workers=4):
    """从tif标签创建像素级训练数据集"""
    """
    参数:
    zarr_path: 输入Zarr文件路径
    tif_path: 输入TIFF标签文件路径
    out_path: 输出Zarr文件路径
    max_per_class: 每类最多样本数
    min_samples: 最小类别样本阈值，少于此值的类别将被剔除
    seed: 随机种子
    workers: 并行处理的工作线程数
    """
    # 读取zarr数据
    print("读取Zarr数据...")
    store = zarr.open(zarr_path, mode='r')
    data = store['observations']
    T, H, W, C = data.shape
    ch = data.chunks[1]  # 高度方向的分块大小
    cw = data.chunks[2]  # 宽度方向的分块大小
    z_aff = Affine(*tuple(store.attrs.get('transform')))
    z_crs = store.attrs.get('crs')
    
    # 读取tif标签数据
    print("读取TIFF标签数据...")
    with rasterio.open(tif_path) as src:
        tif_data = src.read(1)
        tif_aff = src.transform
        tif_crs = src.crs
        tif_H, tif_W = src.height, src.width
        
        # 检查坐标系统是否一致
        if z_crs != tif_crs:
            print(f"警告: Zarr和TIFF的坐标系统不一致: {z_crs} vs {tif_crs}")
            return None
        
        # 检查地理范围是否重叠
        z_bounds = [z_aff.c, z_aff.f, z_aff.c + z_aff.a * W, z_aff.f + z_aff.e * H]
        tif_bounds = [tif_aff.c, tif_aff.f, tif_aff.c + tif_aff.a * tif_W, tif_aff.f + tif_aff.e * tif_H]
        
        print(f"Zarr地理范围: {z_bounds}")
        print(f"TIFF地理范围: {tif_bounds}")
        
        # 找到重叠区域
        overlap_left = max(z_bounds[0], tif_bounds[0])
        overlap_top = min(z_bounds[1], tif_bounds[1])
        overlap_right = min(z_bounds[2], tif_bounds[2])
        overlap_bottom = max(z_bounds[3], tif_bounds[3])
        
        if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
            print("错误: Zarr和TIFF的地理范围没有重叠")
            return None
        
        print(f"重叠区域: {[overlap_left, overlap_top, overlap_right, overlap_bottom]}")
        
        # 计算重叠区域在Zarr中的像素坐标
        z_x0 = int((overlap_left - z_aff.c) / z_aff.a)
        z_y0 = int((overlap_top - z_aff.f) / z_aff.e)
        z_x1 = int((overlap_right - z_aff.c) / z_aff.a)
        z_y1 = int((overlap_bottom - z_aff.f) / z_aff.e)
        
        # 确保坐标在有效范围内
        z_x0 = max(0, z_x0)
        z_y0 = max(0, z_y0)
        z_x1 = min(W, z_x1)
        z_y1 = min(H, z_y1)
        
        print(f"Zarr重叠区域像素坐标: ({z_x0}, {z_y0}) 到 ({z_x1}, {z_y1})")
        
        # 计算重叠区域在TIFF中的像素坐标
        tif_x0 = int((overlap_left - tif_aff.c) / tif_aff.a)
        tif_y0 = int((overlap_top - tif_aff.f) / tif_aff.e)
        tif_x1 = int((overlap_right - tif_aff.c) / tif_aff.a)
        tif_y1 = int((overlap_bottom - tif_aff.f) / tif_aff.e)
        
        # 确保坐标在有效范围内
        tif_x0 = max(0, tif_x0)
        tif_y0 = max(0, tif_y0)
        tif_x1 = min(tif_W, tif_x1)
        tif_y1 = min(tif_H, tif_y1)
        
        print(f"TIFF重叠区域像素坐标: ({tif_x0}, {tif_y0}) 到 ({tif_x1}, {tif_y1})")
        
        # 将TIFF数据重采样到Zarr的分辨率
        print("重采样TIFF数据到Zarr分辨率...")
        # 计算重采样后的尺寸
        resampled_H = z_y1 - z_y0
        resampled_W = z_x1 - z_x0
        
        # 使用rasterio的reproject函数进行高效重采样
        from rasterio.warp import reproject, Resampling
        
        # 创建重采样后的标签数据
        resampled_labels = np.zeros((resampled_H, resampled_W), dtype=np.uint8)
        
        # 定义Zarr的转换矩阵（重叠区域）
        z_overlap_aff = Affine(z_aff.a, z_aff.b, z_aff.c + z_x0 * z_aff.a, 
                              z_aff.d, z_aff.e, z_aff.f + z_y0 * z_aff.e)
        
        # 重采样TIFF数据到Zarr分辨率
        reproject(
            source=tif_data,
            destination=resampled_labels,
            src_transform=tif_aff,
            src_crs=tif_crs,
            dst_transform=z_overlap_aff,
            dst_crs=z_crs,
            resampling=Resampling.nearest
        )
    
    # 加载标签信息
    dbf_path = tif_path + '.vat.dbf'
    class_names, label_map = _load_label_info(dbf_path)
    
    # 统计各类别像素数量...
    unique_labels, counts = np.unique(resampled_labels, return_counts=True)
    # 将numpy uint8类型的键转换为Python整数
    label_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)}
    print(f"各类别像素数量: {label_counts}")
    
    # 过滤掉样本量少于min_samples的类别
    filtered_labels = {label: count for label, count in label_counts.items() if label > 0 and count >= min_samples}
    print(f"过滤后各类别像素数量 (阈值: {min_samples}): {filtered_labels}")
    
    if not filtered_labels:
        print(f"错误: 没有类别满足最小样本阈值 {min_samples}")
        return None
    
    # 生成class_names和label_map
    class_names = []
    label_map = {}
    for i, (label, count) in enumerate(sorted(filtered_labels.items())):
        class_names.append(f"Class_{i}")
        label_map[label] = i
    
    print(f"生成的类别名称: {class_names}")
    print(f"生成的标签映射: {label_map}")
    
    # 过滤出有效标签像素（只保留过滤后的类别）
    valid_coords = []
    for label in filtered_labels:
        coords = np.argwhere(resampled_labels == label)
        valid_coords.extend(coords)
    valid_coords = np.array(valid_coords)
    print(f"有效标签像素数量: {len(valid_coords)}")
    
    if len(valid_coords) == 0:
        print("错误: 没有找到有效标签像素")
        return None
    
    # 按类别进行采样（如果需要）
    if max_per_class:
        print(f"按类别采样，每类最多{max_per_class}个像素...")
        sampled_coords = []
        for label in filtered_labels:
            # 找到该类别的所有坐标
            class_coords = valid_coords[resampled_labels[valid_coords[:, 0], valid_coords[:, 1]] == label]
            
            # 采样不超过max_per_class个像素
            sample_size = min(len(class_coords), max_per_class)
            rng = np.random.default_rng(seed)
            sampled = rng.choice(len(class_coords), sample_size, replace=False)
            sampled_coords.extend(class_coords[sampled])
        
        valid_coords = np.array(sampled_coords)
        print(f"采样后有效标签像素数量: {len(valid_coords)}")
    
    # 将坐标转换为原始Zarr坐标
    valid_coords[:, 0] += z_y0  # 转换为原始Zarr的y坐标
    valid_coords[:, 1] += z_x0  # 转换为原始Zarr的x坐标
    
    # 按分块分组
    print("按分块分组坐标...")
    groups = {}
    for y, x in valid_coords:
        # 计算所属的分块
        ch_y = y // ch
        ch_x = x // cw
        key = (ch_y, ch_x)
        
        if key not in groups:
            groups[key] = []
        groups[key].append((y, x, resampled_labels[y - z_y0, x - z_x0]))
    
    # 创建输出zarr文件
    print("创建输出Zarr文件...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    root = zarr.open_group(out_path, mode='w')
    
    # 创建数据数组
    total_samples = len(valid_coords)
    arr_series = root.create('time_series', shape=(total_samples, T, C), 
                           chunks=(max(1, min(512, total_samples)), T, C), dtype='float32')
    arr_labels = root.create('labels', shape=(total_samples,), 
                          chunks=(max(1, min(2048, total_samples)),), dtype='int32')
    
    # 保存元数据
    root.attrs['crs'] = z_crs
    root.attrs['transform'] = tuple(store.attrs.get('transform'))
    if class_names:
        root.attrs['class_names'] = class_names
    root.attrs['index_base'] = 0
    root.attrs['source_zarr_path'] = zarr_path
    root.attrs['source_tif_path'] = tif_path
    root.attrs['label_counts'] = label_counts
    
    # 处理每个分块
    print("处理分块数据...")
    ptr = 0
    keys = list(groups.keys())
    
    def process_group(key):
        ch_y, ch_x = key
        coords = groups[key]
        
        # 计算分块的边界
        y0 = ch_y * ch
        y1 = min(H, (ch_y + 1) * ch)
        x0 = ch_x * cw
        x1 = min(W, (ch_x + 1) * cw)
        
        # 读取该分块的Zarr数据
        window_data = np.asarray(data[:, y0:y1, x0:x1, :], dtype=np.float32)
        
        # 提取每个坐标的时序数据
        seqs = []
        labels = []
        
        for y, x, label in coords:
            # 计算在分块内的相对坐标
            rel_y = y - y0
            rel_x = x - x0
            
            # 提取时序数据
            seq = window_data[:, rel_y, rel_x, :]  # T×3
            seqs.append(seq)
            labels.append(label)
        
        return np.array(seqs), np.array(labels, dtype=np.int32)
    
    # 并行处理分块
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_group, k): k for k in keys}
        
        for fut in tqdm(as_completed(futs), total=len(futs), desc='处理分块'):
            k = futs[fut]
            try:
                seqs, labels = fut.result()
                n = len(seqs)
                
                # 写入数据
                arr_series[ptr:ptr+n] = seqs
                arr_labels[ptr:ptr+n] = labels
                ptr += n
            except Exception as e:
                print(f"处理分块{k}时出错: {e}")
    
    # 保存统计信息
    print("保存统计信息...")
    final_counts = {}
    for label in filtered_labels:
        # 只统计过滤后的有效标签
        final_counts[label] = int(np.sum(arr_labels[:] == label))
    
    root.attrs['final_label_counts'] = final_counts
    
    # 合并元数据
    try:
        from zarr.convenience import consolidate_metadata
        consolidate_metadata(root.store)
    except Exception:
        pass
    
    print(f"处理完成! 生成的数据集位于: {out_path}")
    print(f"总样本数: {total_samples}")
    print(f"最终各类别样本数: {final_counts}")
    
    return out_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从TIFF标签创建像素级训练数据集')
    parser.add_argument('--zarr_path', type=str, default='F:/DATA/USA1/USA1-BS-norm/zarr.zarr', help='输入Zarr文件路径')
    parser.add_argument('--tif_path', type=str, default="F:/DATA/USA1/CDL_2024_filtered_color.tif", help='输入TIFF标签文件路径')
    parser.add_argument('--out_path', type=str, default="F:/DATA/USA1-BS-3bands-labeld-new", help='输出Zarr文件路径')
    parser.add_argument('--max_per_class', type=int, default=None, help='每类最多样本数')
    parser.add_argument('--min_samples', type=int, default=100000, help='最小类别样本阈值，少于此值的类别将被剔除')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--workers', type=int, default=4, help='并行处理的工作线程数')
    
    args = parser.parse_args()
    
    run(args.zarr_path, args.tif_path, args.out_path, args.max_per_class, args.min_samples, args.seed, args.workers)


if __name__ == '__main__':
    main()
