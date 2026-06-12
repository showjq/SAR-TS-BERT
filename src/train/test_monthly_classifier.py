import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from src.data.monthly_dataset import MonthlyParcelDataset
from src.models.monthly_classifier import MonthlyParcelClassifier

def test_model(zarr_path, model_weights, batch_size, d_model, nhead, layers, ff,
               train_ratio=0.8, val_ratio=0.1, seed=42, output_dir='outputs/Chongming_monthly_test',
               use_dora=False, dora_rank=32):
    full_ds = MonthlyParcelDataset(zarr_path)
    total_size = len(full_ds)
    
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"数据集总大小: {total_size}")
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")
    print(f"测试集大小: {test_size}")
    
    rng = np.random.default_rng(seed)
    all_indices = np.arange(total_size)
    rng.shuffle(all_indices)
    
    test_indices = all_indices[train_size + val_size:]
    test_ds = Subset(full_ds, test_indices)
    
    print(f"测试集大小: {len(test_ds)}")
    
    num_classes = full_ds.num_classes
    num_months = full_ds.num_months
    print(f"类别数量: {num_classes}")
    print(f"月份数量: {num_months}")
    print(f"类别名称: {full_ds.class_names}")
    print(f"月份名称: {full_ds.month_names}")
    
    num_bands = full_ds.series.shape[2]
    print(f"数据波段数量: {num_bands}")
    
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    model = MonthlyParcelClassifier(
        num_classes,
        num_months,
        d_model=d_model,
        nhead=nhead,
        num_layers=layers,
        dim_feedforward=ff,
        num_bands=num_bands,
        use_dora=use_dora,
        dora_rank=dora_rank
    ).to(device)
    
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
    
    model.eval()
    all_y = [[] for _ in range(num_months)]
    all_p = [[] for _ in range(num_months)]
    all_parcel_ids = []
    
    print("\n开始测试...")
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Testing"):
            x = batch['series'].to(device, non_blocking=True)
            t = batch['time_idx'].to(device, non_blocking=True)
            y = batch['monthly_labels'].to(device, non_blocking=True)
            
            logits = model(x, t)
            
            for i in range(num_months):
                all_y[i].append(y[:, i].cpu().numpy())
                all_p[i].append(logits[:, i, :].argmax(dim=1).cpu().numpy())
            
            if hasattr(test_ds.dataset, 'parcel_ids') and test_ds.dataset.parcel_ids is not None:
                pass
    
    for i in range(num_months):
        all_y[i] = np.concatenate(all_y[i])
        all_p[i] = np.concatenate(all_p[i])
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*100)
    print("各月份分类结果")
    print("="*100)
    
    month_results = []
    for i in range(num_months):
        y_i = all_y[i]
        p_i = all_p[i]
        acc = accuracy_score(y_i, p_i)
        macro_f1 = f1_score(y_i, p_i, average='macro')
        weighted_f1 = f1_score(y_i, p_i, average='weighted')
        
        month_results.append({
            'month': full_ds.month_names[i],
            'accuracy': acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1
        })
        
        print(f"\n{full_ds.month_names[i]}:")
        print(f"  准确率: {acc:.4f}")
        print(f"  宏平均F1: {macro_f1:.4f}")
        print(f"  加权平均F1: {weighted_f1:.4f}")
    
    print("\n" + "="*100)
    print("各类别分类结果（合并所有月份）")
    print("="*100)
    
    all_y_combined = np.concatenate(all_y)
    all_p_combined = np.concatenate(all_p)
    
    report = classification_report(
        all_y_combined,
        all_p_combined,
        target_names=full_ds.class_names,
        labels=list(range(num_classes)),
        digits=4,
        zero_division=0
    )
    print(report)
    
    cm = confusion_matrix(all_y_combined, all_p_combined, labels=list(range(num_classes)))
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=full_ds.class_names,
                yticklabels=full_ds.class_names,
                cbar_kws={'label': '样本数量'})
    plt.title('混淆矩阵（所有月份）', fontsize=16, fontweight='bold')
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix_all_months.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\n混淆矩阵已保存到: {cm_path}")
    plt.close()
    
    print("\n" + "="*100)
    print("轮种模式分析")
    print("="*100)
    
    crop_patterns = []
    for idx in tqdm(range(len(test_ds)), desc="分析轮种模式"):
        y_true = []
        for i in range(num_months):
            y_true.append(all_y[i][idx])
        
        pattern = analyze_cropping_pattern(y_true, full_ds.class_names)
        crop_patterns.append(pattern)
    
    pattern_counter = Counter(crop_patterns)
    
    print("\n轮种模式统计:")
    print("-" * 100)
    for pattern, count in sorted(pattern_counter.items(), key=lambda x: -x[1]):
        print(f"{pattern:40s}: {count:6d} 个像素 ({count/len(crop_patterns)*100:.2f}%)")
    
    report_path = os.path.join(output_dir, 'test_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("崇明岛月度作物分类测试报告\n")
        f.write("="*100 + "\n\n")
        
        f.write("各月份分类结果\n")
        f.write("="*100 + "\n")
        for res in month_results:
            f.write(f"\n{res['month']}:\n")
            f.write(f"  准确率: {res['accuracy']:.4f}\n")
            f.write(f"  宏平均F1: {res['macro_f1']:.4f}\n")
            f.write(f"  加权平均F1: {res['weighted_f1']:.4f}\n")
        
        f.write("\n\n各类别分类结果（合并所有月份）\n")
        f.write("="*100 + "\n")
        f.write(report)
        
        f.write("\n\n轮种模式统计\n")
        f.write("="*100 + "\n")
        for pattern, count in sorted(pattern_counter.items(), key=lambda x: -x[1]):
            f.write(f"{pattern:40s}: {count:6d} 个像素 ({count/len(crop_patterns)*100:.2f}%)\n")
    
    print(f"\n测试报告已保存到: {report_path}")
    
    import csv
    month_stats_path = os.path.join(output_dir, 'month_statistics.csv')
    with open(month_stats_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['月份', '准确率', '宏平均F1', '加权平均F1'])
        for res in month_results:
            writer.writerow([res['month'], f"{res['accuracy']:.4f}", f"{res['macro_f1']:.4f}", f"{res['weighted_f1']:.4f}"])
    print(f"月份统计已保存到: {month_stats_path}")
    
    pattern_stats_path = os.path.join(output_dir, 'cropping_patterns.csv')
    with open(pattern_stats_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['轮种模式', '像素数量', '占比'])
        for pattern, count in sorted(pattern_counter.items(), key=lambda x: -x[1]):
            writer.writerow([pattern, count, f"{count/len(crop_patterns)*100:.2f}%"])
    print(f"轮种模式统计已保存到: {pattern_stats_path}")
    
    print("\n" + "="*100)
    print("测试完成！")
    print("="*100)

def analyze_cropping_pattern(monthly_labels, class_names):
    first_half = monthly_labels[:5]
    second_half = monthly_labels[5:]
    
    first_crops = [class_names[i] for i in first_half]
    second_crops = [class_names[i] for i in second_half]
    
    first_counter = Counter(first_crops)
    second_counter = Counter(second_crops)
    
    major_crops = ['水稻', '小麦', '玉米', '大豆', '蔬菜', '油菜', '林地', '桔子树']
    state_crops = ['稻茬', '翻耕', '荒草']
    
    def get_dominant(counter):
        for crop in major_crops:
            if counter.get(crop, 0) >= 3:
                return crop
        for crop in major_crops:
            if counter.get(crop, 0) >= 2:
                return crop
        return None
    
    first_dominant = get_dominant(first_counter)
    second_dominant = get_dominant(second_counter)
    
    all_labels = [class_names[i] for i in monthly_labels]
    all_counter = Counter(all_labels)
    
    if all_counter.get('林地', 0) >= 6:
        return '林地'
    if all_counter.get('桔子树', 0) >= 6:
        return '桔子树'
    
    pattern_parts = []
    if first_dominant:
        pattern_parts.append(first_dominant)
    if second_dominant and second_dominant != first_dominant:
        pattern_parts.append(second_dominant)
    
    if not pattern_parts:
        for crop in major_crops:
            if all_counter.get(crop, 0) >= 3:
                pattern_parts.append(crop)
                break
    
    if not pattern_parts:
        return '其他'
    
    return '-'.join(pattern_parts)

def main():
    parser = argparse.ArgumentParser(description='测试月度作物分类模型并生成详细评估报告')
    parser.add_argument('--zarr_path', type=str, default='src/shpfiles/Chongming_monthly_crops/processed.zarr', help='Zarr数据集路径')
    parser.add_argument('--model_weights', type=str, default='outputs/Chongming_monthly_nopre/parcel_cls_best.pt', help='训练好的模型权重路径')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数')
    parser.add_argument('--layers', type=int, default=6, help='Transformer层数')
    parser.add_argument('--ff', type=int, default=512, help='前馈网络维度')
    parser.add_argument('--train_ratio', type=float, default=0.4, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output_dir', type=str, default='outputs/Chongming_monthly_nopre_test1', help='输出目录')
    parser.add_argument('--use_dora', action='store_true', help='是否使用DoRA适配器（与训练时保持一致）')
    parser.add_argument('--dora_rank', type=int, default=32, help='DoRA适配器的低秩维度（与训练时保持一致）')
    
    args = parser.parse_args()
    
    test_model(
        args.zarr_path, args.model_weights, args.batch_size,
        args.d_model, args.nhead, args.layers, args.ff,
        args.train_ratio, args.val_ratio, args.seed, args.output_dir,
        args.use_dora, args.dora_rank
    )

if __name__ == '__main__':
    main()
