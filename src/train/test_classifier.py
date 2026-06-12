import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.processed_dataset import ProcessedParcelDataset
from src.models.parcel_classifier import ParcelClassifier

# 定义类别标签映射
#USA-1
""" CLASS_LABELS = {
    1: 'Corn',
    2: 'Cotton',
    24: 'Winter Wheat',
    36: 'Alfalfa',
    54: 'Tomatoes',
    63: 'Forest',
    69: 'Grapes',
    72: 'Citrus',
    75: 'Almonds',
    76: 'Walnuts',
    204: 'Pistachios'
} """
#Estonia-1
CLASS_LABELS = {
    6: 'clover',
    12: 'legumes_harvested_green',
    15: 'oats',
    17: 'pasture_meadow_grassland_grass',
    18: 'peas',
    22: 'spring_barley',
    23: 'spring_common_soft_wheat',
    27: 'winter_common_soft_wheat',
    28: 'winter_rapeseed_rape',
}

def test_model(zarr_path, model_weights, batch_size, d_model, nhead, layers, ff,
               train_ratio=0.04, val_ratio=0.01, test_ratio=None, seed=42, output_dir='outputs/test_results',
               use_dora=False, dora_rank=32, use_dual_branch=False, band_indices=None):
    # 加载完整数据集
    full_ds = ProcessedParcelDataset(zarr_path, band_indices=band_indices)
    
    # 计算划分大小
    train_size = int(train_ratio * len(full_ds))
    val_size = int(val_ratio * len(full_ds))
    
    # 如果指定了test_ratio，则使用该比例；否则使用剩余所有数据
    if test_ratio is not None:
        test_size = int(test_ratio * len(full_ds))
        unused_size = len(full_ds) - train_size - val_size - test_size
    else:
        test_size = len(full_ds) - train_size - val_size
        unused_size = 0
    
    print(f"数据集划分:")
    print(f"  训练集: {train_size:,} 样本 ({train_ratio*100:.1f}%)")
    print(f"  验证集: {val_size:,} 样本 ({val_ratio*100:.1f}%)")
    if test_ratio is not None:
        print(f"  测试集: {test_size:,} 样本 ({test_ratio*100:.1f}%)")
        if unused_size > 0:
            print(f"  未使用: {unused_size:,} 样本 ({unused_size/len(full_ds)*100:.1f}%)")
    else:
        print(f"  测试集: {test_size:,} 样本 ({(1-train_ratio-val_ratio)*100:.1f}%)")
    print(f"  总计: {len(full_ds):,} 样本")
    
    # 随机划分数据集
    generator = torch.Generator().manual_seed(seed)
    if unused_size > 0:
        train_ds, val_ds, test_ds, _ = random_split(
            full_ds, 
            [train_size, val_size, test_size, unused_size],
            generator=generator
        )
    else:
        train_ds, val_ds, test_ds = random_split(
            full_ds, 
            [train_size, val_size, test_size],
            generator=generator
        )
    
    # 获取类别信息
    num_classes = len(full_ds.label_map) if hasattr(full_ds, 'label_map') else len(full_ds.class_names)
    print(f"\n类别数量: {num_classes}")
    print(f"标签映射: {full_ds.label_map}")
    
    # 获取波段数量
    num_bands = full_ds.original_bands
    print(f"数据波段数量: {num_bands}" + (f" (选取波段索引: {band_indices})" if band_indices else ""))
    
    # 创建测试DataLoader
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"编码器模式: {'双分支跨模态融合' if use_dual_branch else '单分支'}")
    
    # 创建模型
    model = ParcelClassifier(
        num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=layers,
        dim_feedforward=ff,
        num_bands=num_bands,
        use_dora=use_dora,
        dora_rank=dora_rank,
        use_dual_branch=use_dual_branch
    ).to(device)
    
    # 加载模型权重
    if os.path.exists(model_weights):
        try:
            model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=False))
            print(f"\n成功加载模型权重: {model_weights}")
        except Exception as e:
            print(f"\n加载模型权重失败: {e}")
            return
    else:
        print(f"\n模型权重文件不存在: {model_weights}")
        return
    
    # 测试阶段
    model.eval()
    all_y = []
    all_p = []
    all_probs = []
    
    print("\n开始测试...")
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Testing"):
            x = batch['series'].to(device, non_blocking=True)
            t = batch['time_idx'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)
            
            logits = model(x.unsqueeze(0) if x.dim() == 2 else x, 
                         t.unsqueeze(0) if t.dim() == 1 else t)
            probs = F.softmax(logits, dim=1)
            
            all_y.append(y.cpu().numpy())
            all_p.append(logits.argmax(dim=1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # 合并所有预测结果
    all_y = np.concatenate(all_y)
    all_p = np.concatenate(all_p)
    all_probs = np.concatenate(all_probs)
    
    # 计算整体指标
    accuracy = accuracy_score(all_y, all_p)
    macro_f1 = f1_score(all_y, all_p, average='macro')
    weighted_f1 = f1_score(all_y, all_p, average='weighted')
    
    print("\n" + "="*80)
    print("整体分类结果")
    print("="*80)
    print(f"准确率: {accuracy:.4f}")
    print(f"宏平均F1分数: {macro_f1:.4f}")
    print(f"加权平均F1分数: {weighted_f1:.4f}")
    
    # 创建反向映射：从0基索引到原始标签值
    reverse_label_map = {v: k for k, v in full_ds.label_map.items()}
    
    # 将预测和真实标签映射回原始标签值
    all_y_original = np.array([reverse_label_map.get(y, 0) for y in all_y])
    all_p_original = np.array([reverse_label_map.get(p, 0) for p in all_p])
    
    # 过滤掉背景标签（0）
    mask = (all_y_original != 0) & (all_p_original != 0)
    all_y_filtered = all_y_original[mask]
    all_p_filtered = all_p_original[mask]
    
    # 获取所有存在的类别
    unique_labels = sorted(list(set(all_y_filtered) | set(all_p_filtered)))
    
    # 为每个类别创建名称映射
    target_names = [CLASS_LABELS.get(label, f'Class_{label}') for label in unique_labels]
    
    # 生成详细的分类报告
    print("\n" + "="*80)
    print("各类别详细分类报告")
    print("="*80)
    report = classification_report(
        all_y_filtered, 
        all_p_filtered, 
        target_names=target_names,
        labels=unique_labels,
        digits=4
    )
    print(report)
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_y_filtered, all_p_filtered, labels=unique_labels)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names,
                cbar_kws={'label': '样本数量'})
    plt.title('混淆矩阵', fontsize=16, fontweight='bold')
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\n混淆矩阵已保存到: {cm_path}")
    plt.close()
    
    # 绘制归一化混淆矩阵
    plt.figure(figsize=(12, 10))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names,
                cbar_kws={'label': '比例'})
    plt.title('归一化混淆矩阵 (按行)', fontsize=16, fontweight='bold')
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_norm_path = os.path.join(output_dir, 'confusion_matrix_normalized.png')
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    print(f"归一化混淆矩阵已保存到: {cm_norm_path}")
    plt.close()
    
    # 保存详细分类报告到文件
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("模型测试结果\n")
        f.write("="*80 + "\n\n")
        f.write(f"模型权重: {model_weights}\n")
        f.write(f"数据集: {zarr_path}\n")
        f.write(f"测试集大小: {test_size:,} 样本 ({(1-train_ratio-val_ratio)*100:.1f}%)\n\n")
        f.write("="*80 + "\n")
        f.write("整体分类结果\n")
        f.write("="*80 + "\n")
        f.write(f"准确率: {accuracy:.4f}\n")
        f.write(f"宏平均F1分数: {macro_f1:.4f}\n")
        f.write(f"加权平均F1分数: {weighted_f1:.4f}\n\n")
        f.write("="*80 + "\n")
        f.write("各类别详细分类报告\n")
        f.write("="*80 + "\n")
        f.write(report)
    print(f"分类报告已保存到: {report_path}")
    
    # 计算并保存每个类别的统计信息
    print("\n" + "="*80)
    print("各类别样本统计")
    print("="*80)
    class_stats = {}
    for label in unique_labels:
        true_count = np.sum(all_y_filtered == label)
        pred_count = np.sum(all_p_filtered == label)
        correct_count = np.sum((all_y_filtered == label) & (all_p_filtered == label))
        precision = correct_count / pred_count if pred_count > 0 else 0
        recall = correct_count / true_count if true_count > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_stats[label] = {
            'class_name': CLASS_LABELS.get(label, f'Class_{label}'),
            'true_count': true_count,
            'pred_count': pred_count,
            'correct_count': correct_count,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"{CLASS_LABELS.get(label, f'Class_{label}'):20s}: "
              f"真实={true_count:6d}, 预测={pred_count:6d}, "
              f"正确={correct_count:6d}, "
              f"精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}")
    
    # 保存类别统计到CSV
    import csv
    stats_path = os.path.join(output_dir, 'class_statistics.csv')
    with open(stats_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['类别ID', '类别名称', '真实样本数', '预测样本数', '正确预测数', 
                        '精确率', '召回率', 'F1分数'])
        for label in sorted(class_stats.keys()):
            stats = class_stats[label]
            writer.writerow([
                label, stats['class_name'], stats['true_count'], stats['pred_count'],
                stats['correct_count'], f"{stats['precision']:.4f}", 
                f"{stats['recall']:.4f}", f"{stats['f1']:.4f}"
            ])
    print(f"\n类别统计已保存到: {stats_path}")
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='测试作物分类模型并生成详细评估报告')
    parser.add_argument('--zarr_path', type=str, required=True, help='Zarr数据集路径')
    parser.add_argument('--model_weights', type=str, required=True, help='训练好的模型权重路径')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数')
    parser.add_argument('--layers', type=int, default=6, help='Transformer层数')
    parser.add_argument('--ff', type=int, default=512, help='前馈网络维度')
    parser.add_argument('--train_ratio', type=float, default=0.08, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.02, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=None, help='测试集比例（如果为None则使用剩余所有数据）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output_dir', type=str, default='outputs/test_results', help='输出目录')
    parser.add_argument('--use_dora', action='store_true', help='是否使用DoRA适配器（与训练时保持一致）')
    parser.add_argument('--dora_rank', type=int, default=32, help='DoRA适配器的低秩维度（与训练时保持一致）')
    parser.add_argument('--use_dual_branch', action='store_true', help='使用双分支跨模态融合编码器')
    parser.add_argument('--band_indices', type=str, default=None, help='选取的波段索引，逗号分隔，如"3,4,5"')

    args = parser.parse_args()
    
    band_indices = None
    if args.band_indices is not None:
        band_indices = [int(x.strip()) for x in args.band_indices.split(',')]
        print(f"选取波段索引: {band_indices} (共{len(band_indices)}个波段)")
    
    test_model(
        args.zarr_path, args.model_weights, args.batch_size,
        args.d_model, args.nhead, args.layers, args.ff,
        args.train_ratio, args.val_ratio, args.test_ratio, args.seed, args.output_dir,
        args.use_dora, args.dora_rank, args.use_dual_branch, band_indices
    )

if __name__ == '__main__':
    main()