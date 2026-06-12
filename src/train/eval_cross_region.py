import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import zarr
from src.data.processed_dataset import ProcessedParcelDataset
from src.models.parcel_classifier import ParcelClassifier

ESTONIA_CLASS_NAMES = [
    'clover',
    'legumes_harvested_green',
    'oats',
    'pasture_meadow_grassland_grass',
    'peas',
    'spring_barley',
    'spring_common_soft_wheat',
    'winter_common_soft_wheat',
    'winter_rapeseed_rape',
]

LATVIA_CLASS_NAMES = [
    'clover',
    'oats',
    'pasture_meadow_grassland_grass',
    'peas',
    'spring_barley',
    'spring_common_soft_wheat',
    'winter_common_soft_wheat',
    'winter_rapeseed_rape',
]

LATVIA_ORIGINAL_LABELS = [0, 2, 3, 4, 5, 6, 7, 8]

ESTONIA_EXCLUDED_CLASS = 1


def test_same_region(zarr_path, model_weights, batch_size, d_model, nhead, layers, ff,
                     train_ratio, val_ratio, seed, output_dir, use_dora, dora_rank,
                     use_sar_norm, pad_bands):
    full_ds = ProcessedParcelDataset(zarr_path, pad_bands=pad_bands)
    total_size = len(full_ds)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    print(f"数据集总大小: {total_size}")
    print(f"训练集: {train_size}, 验证集: {val_size}, 测试集: {test_size}")

    rng = np.random.default_rng(seed)
    all_indices = np.arange(total_size)
    rng.shuffle(all_indices)
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]

    test_ds = Subset(full_ds, test_indices)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = len(full_ds.label_map)
    num_bands = pad_bands if pad_bands is not None else full_ds.series.shape[2]
    print(f"类别数量: {num_classes}, 波段数量: {num_bands}")
    print(f"标签映射: {full_ds.label_map}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParcelClassifier(
        num_classes, d_model=d_model, nhead=nhead, num_layers=layers,
        dim_feedforward=ff, num_bands=num_bands, use_dora=use_dora,
        dora_rank=dora_rank, use_sar_norm=use_sar_norm,
    ).to(device)

    sd = torch.load(model_weights, map_location=device, weights_only=False)
    model.load_state_dict(sd)
    print(f"Loaded model from {model_weights}")

    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Testing"):
            x = batch['series'].to(device, non_blocking=True)
            t = batch['time_idx'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                logits = model(x.unsqueeze(0) if x.dim() == 2 else x,
                             t.unsqueeze(0) if t.dim() == 1 else t)
            all_y.append(y.cpu().numpy())
            all_p.append(logits.argmax(dim=1).cpu().numpy())

    all_y = np.concatenate(all_y)
    all_p = np.concatenate(all_p)

    acc = accuracy_score(all_y, all_p)
    macro_f1 = f1_score(all_y, all_p, average='macro')
    weighted_f1 = f1_score(all_y, all_p, average='weighted')

    print(f"\nOA: {acc:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")

    class_names = full_ds.class_names if full_ds.class_names else [f'Class_{i}' for i in range(num_classes)]
    report = classification_report(all_y, all_p, target_names=class_names, digits=4)
    print(report)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"模型权重: {model_weights}\n")
        f.write(f"数据集: {zarr_path}\n")
        f.write(f"测试集大小: {test_size}\n\n")
        f.write(f"OA: {acc:.4f}\nMacro F1: {macro_f1:.4f}\nWeighted F1: {weighted_f1:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(all_y, all_p)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return acc, macro_f1


def test_cross_region(source_zarr, target_zarr, model_weights, batch_size,
                      d_model, nhead, layers, ff, seed, output_dir,
                      use_dora, dora_rank, use_sar_norm, pad_bands,
                      source_num_classes, source_class_names, target_class_names,
                      target_original_labels, estonia_excluded):
    source_ds = ProcessedParcelDataset(source_zarr, pad_bands=pad_bands)
    target_ds = ProcessedParcelDataset(target_zarr, pad_bands=pad_bands)

    num_bands = pad_bands if pad_bands is not None else source_ds.series.shape[2]
    print(f"源数据集标签映射: {source_ds.label_map}")
    print(f"目标数据集标签映射: {target_ds.label_map}")
    print(f"目标数据集类别: {target_ds.class_names}")
    print(f"目标数据集大小: {len(target_ds)}")
    print(f"目标数据集原始标签: {target_original_labels}")

    target_label_map = target_ds.label_map
    reverse_label_map = {v: k for k, v in target_label_map.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParcelClassifier(
        source_num_classes, d_model=d_model, nhead=nhead, num_layers=layers,
        dim_feedforward=ff, num_bands=num_bands, use_dora=use_dora,
        dora_rank=dora_rank, use_sar_norm=use_sar_norm,
    ).to(device)

    sd = torch.load(model_weights, map_location=device, weights_only=False)
    model.load_state_dict(sd)
    print(f"Loaded model from {model_weights}")

    target_dl = DataLoader(target_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model.eval()
    all_y_original = []
    all_p_original = []

    with torch.no_grad():
        for batch in tqdm(target_dl, desc="Cross-region Testing"):
            x = batch['series'].to(device, non_blocking=True)
            t = batch['time_idx'].to(device, non_blocking=True)
            y_remapped = batch['label'].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                logits = model(x.unsqueeze(0) if x.dim() == 2 else x,
                             t.unsqueeze(0) if t.dim() == 1 else t)

            pred_estonia = logits.float().argmax(dim=1).cpu().numpy()

            y_original = np.array([reverse_label_map.get(int(y), -1) for y in y_remapped.cpu().numpy()])

            all_y_original.extend(y_original.tolist())
            all_p_original.extend(pred_estonia.tolist())

    all_y_original = np.array(all_y_original)
    all_p_original = np.array(all_p_original)

    valid_mask = np.isin(all_y_original, target_original_labels)
    all_y_valid = all_y_original[valid_mask]
    all_p_valid = all_p_original[valid_mask]

    acc = accuracy_score(all_y_valid, all_p_valid)
    macro_f1 = f1_score(all_y_valid, all_p_valid, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_y_valid, all_p_valid, average='weighted', zero_division=0)

    print(f"\n跨区域零样本测试结果 (Estonia模型 → Latvia):")
    print(f"OA: {acc:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")

    report = classification_report(all_y_valid, all_p_valid,
                                   target_names=source_class_names, digits=4,
                                   zero_division=0)
    print(report)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"跨区域零样本测试: Estonia → Latvia\n")
        f.write(f"模型权重: {model_weights}\n")
        f.write(f"目标数据集: {target_zarr}\n")
        f.write(f"目标数据集大小: {len(target_ds)}\n")
        f.write(f"有效预测数: {len(all_y_valid)}\n\n")
        f.write(f"OA: {acc:.4f}\nMacro F1: {macro_f1:.4f}\nWeighted F1: {weighted_f1:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(all_y_valid, all_p_valid, labels=list(range(source_num_classes)))
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=source_class_names, yticklabels=source_class_names)
    plt.title('Cross-Region Confusion Matrix (Estonia→Latvia)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=source_class_names, yticklabels=source_class_names)
    plt.title('Cross-Region Normalized Confusion Matrix (Estonia→Latvia)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return acc, macro_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, choices=['same_region', 'cross_region'], required=True)
    p.add_argument('--zarr_path', type=str, required=True)
    p.add_argument('--source_zarr', type=str, default=None)
    p.add_argument('--target_zarr', type=str, default=None)
    p.add_argument('--model_weights', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--ff', type=int, default=512)
    p.add_argument('--train_ratio', type=float, default=0.8)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--use_dora', action='store_true')
    p.add_argument('--dora_rank', type=int, default=32)
    p.add_argument('--use_sar_norm', action='store_true')
    p.add_argument('--pad_bands', type=int, default=None)
    p.add_argument('--source_num_classes', type=int, default=9)
    a = p.parse_args()

    if a.mode == 'same_region':
        test_same_region(
            a.zarr_path, a.model_weights, a.batch_size,
            a.d_model, a.nhead, a.layers, a.ff,
            a.train_ratio, a.val_ratio, a.seed, a.output_dir,
            a.use_dora, a.dora_rank, a.use_sar_norm, a.pad_bands,
        )
    else:
        test_cross_region(
            a.source_zarr or a.zarr_path,
            a.target_zarr or a.zarr_path,
            a.model_weights, a.batch_size,
            a.d_model, a.nhead, a.layers, a.ff,
            a.seed, a.output_dir,
            a.use_dora, a.dora_rank, a.use_sar_norm, a.pad_bands,
            a.source_num_classes, ESTONIA_CLASS_NAMES, LATVIA_CLASS_NAMES,
            LATVIA_ORIGINAL_LABELS, ESTONIA_EXCLUDED_CLASS,
        )


if __name__ == '__main__':
    main()
