import argparse
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from src.data.monthly_dataset import MonthlyParcelDataset
from src.models.monthly_classifier import MonthlyParcelClassifier

def finetune(zarr_path, epochs, batch_size, d_model, nhead, layers, ff, lr_head, lr_enc, weight_decay, amp, clip_norm, patience, save_best, save_last, log_path, warmup_epochs, pretrain_weights=None, train_ratio=0.8, val_ratio=0.1, seed=42, use_dora=False, dora_rank=32, lr_dora=5e-4, freeze_encoder=False):
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
    
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]
    
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)
    test_ds = Subset(full_ds, test_indices)
    
    print(f"数据划分完成，训练集: {len(train_ds)}, 验证集: {len(val_ds)}, 测试集: {len(test_ds)}")
    
    num_classes = full_ds.num_classes
    num_months = full_ds.num_months
    print(f"使用的类别数量: {num_classes}")
    print(f"月份数量: {num_months}")
    print(f"类别名称: {full_ds.class_names}")
    print(f"月份名称: {full_ds.month_names}")
    
    num_bands = full_ds.series.shape[2]
    print(f"数据波段数量: {num_bands}")
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
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
    
    if pretrain_weights and os.path.exists(pretrain_weights):
        try:
            sd = torch.load(pretrain_weights, map_location=device, weights_only=False)
            model.encoder.load_state_dict(sd, strict=False)
            print(f"Loaded pretrained weights from {pretrain_weights}")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
    elif pretrain_weights:
        print(f"Pretrained weights file not found: {pretrain_weights}")
    
    if use_dora:
        encoder_core_params = model.get_encoder_core_params()
        obs_proj_params = model.get_obs_proj_params()
        dora_params = model.get_dora_params()
        head_params = model.get_head_params()
        
        if freeze_encoder:
            param_groups = [
                {'params': dora_params, 'lr': lr_dora, 'weight_decay': weight_decay},
                {'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay},
            ]
        else:
            param_groups = [
                {'params': encoder_core_params, 'lr': lr_enc, 'weight_decay': weight_decay},
                {'params': obs_proj_params, 'lr': lr_enc, 'weight_decay': weight_decay},
                {'params': dora_params, 'lr': lr_dora, 'weight_decay': weight_decay},
                {'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay},
            ]
        
        print(f"使用DoRA优化，参数组数量: {len(param_groups)}")
    else:
        enc_params = list(model.encoder.parameters())
        head_params = model.get_head_params()
        
        param_groups = [
            {'params': enc_params, 'lr': lr_enc, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay},
        ]
    
    opt = torch.optim.AdamW(param_groups)
    
    use_amp = amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    print(f"混合精度训练: {'启用' if use_amp else '禁用'}")
    
    os.makedirs(os.path.dirname(save_best), exist_ok=True)
    os.makedirs(os.path.dirname(save_last), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            w = csv.writer(f)
            header = ['epoch', 'train_loss', 'val_loss']
            for i in range(num_months):
                header.extend([f'acc_m{i}', f'f1_m{i}'])
            header.extend(['lr_head', 'lr_enc', 'best'])
            w.writerow(header)
        
        if full_ds.class_names is not None:
            with open(os.path.join(os.path.dirname(log_path), 'class_names.json'), 'w', encoding='utf-8') as f:
                import json
                json.dump({
                    'class_names': full_ds.class_names,
                    'month_names': full_ds.month_names
                }, f, ensure_ascii=False, indent=2)
    
    best = float('inf')
    wait = 0
    
    for epoch in range(epochs):
        model.train()
        
        freeze = (epoch < warmup_epochs)
        for p in model.encoder.parameters():
            p.requires_grad = not freeze
        
        train_se = 0.0
        train_cnt = 0
        pbar = tqdm(train_dl, desc=f"Train Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            x = batch['series'].to(device, non_blocking=True)
            t = batch['time_idx'].to(device, non_blocking=True)
            y = batch['monthly_labels'].to(device, non_blocking=True)
            
            opt.zero_grad(set_to_none=True)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(x, t)
                    loss = 0.0
                    for i in range(num_months):
                        loss += F.cross_entropy(logits[:, i, :], y[:, i])
                    loss /= num_months
                
                scaler.scale(loss).backward()
                if clip_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x, t)
                loss = 0.0
                for i in range(num_months):
                    loss += F.cross_entropy(logits[:, i, :], y[:, i])
                loss /= num_months
                loss.backward()
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                opt.step()
            
            train_se += loss.item() * y.shape[0]
            train_cnt += y.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        
        train_loss = train_se / max(1, train_cnt)
        
        model.eval()
        val_se = 0.0
        val_cnt = 0
        all_y = [[] for _ in range(num_months)]
        all_p = [[] for _ in range(num_months)]
        
        with torch.no_grad():
            for batch in val_dl:
                x = batch['series'].to(device, non_blocking=True)
                t = batch['time_idx'].to(device, non_blocking=True)
                y = batch['monthly_labels'].to(device, non_blocking=True)
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(x, t)
                else:
                    logits = model(x, t)
                
                loss = 0.0
                for i in range(num_months):
                    loss += F.cross_entropy(logits[:, i, :], y[:, i], reduction='sum').item()
                    all_y[i].append(y[:, i].cpu().numpy())
                    all_p[i].append(logits[:, i, :].argmax(dim=1).cpu().numpy())
                loss /= num_months
                val_se += loss * y.shape[0]
                val_cnt += y.shape[0]
        
        val_loss = val_se / max(1, val_cnt)
        
        accs = []
        f1s = []
        for i in range(num_months):
            y_i = np.concatenate(all_y[i]) if all_y[i] else np.array([])
            p_i = np.concatenate(all_p[i]) if all_p[i] else np.array([])
            acc = accuracy_score(y_i, p_i) if len(y_i) > 0 else 0.0
            f1 = f1_score(y_i, p_i, average='macro') if len(y_i) > 0 else 0.0
            accs.append(acc)
            f1s.append(f1)
        
        lr_head_val = opt.param_groups[1]['lr']
        lr_enc_val = opt.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs} - train_loss={train_loss:.6f} - val_loss={val_loss:.6f}")
        for i in range(num_months):
            print(f"  Month {i} ({full_ds.month_names[i]}): acc={accs[i]:.4f}, f1={f1s[i]:.4f}")
        
        with open(log_path, 'a', newline='') as f:
            w = csv.writer(f)
            row = [epoch+1, f"{train_loss:.6f}", f"{val_loss:.6f}"]
            for i in range(num_months):
                row.extend([f"{accs[i]:.4f}", f"{f1s[i]:.4f}"])
            row.extend([f"{lr_head_val:.6e}", f"{lr_enc_val:.6e}", int(val_loss < best)])
            w.writerow(row)
        
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), save_best)
            print(f"Saved best model with val_loss={val_loss:.6f}")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    torch.save(model.state_dict(), save_last)
    print(f"Saved final model to {save_last}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', type=str, default="src/shpfiles/Chongming_monthly_crops/processed.zarr")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--ff', type=int, default=512)
    parser.add_argument('--lr_head', type=float, default=1e-3)
    parser.add_argument('--lr_enc', type=float, default=5e-5)
    parser.add_argument('--lr_dora', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--log_path', type=str, default='outputs/Chongming_monthly_nopre/cls_train_log.csv')
    parser.add_argument('--save_best', type=str, default='outputs/Chongming_monthly_nopre/parcel_cls_best.pt')
    parser.add_argument('--save_last', type=str, default='outputs/Chongming_monthly_nopre/parcel_cls_last.pt')
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--pretrain_weights', type=str, default="")#F:/DATA/Chongming-6bands/encoder_best.pt
    parser.add_argument('--train_ratio', type=float, default=0.4)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_dora', action='store_true', help='启用DoRA适配器微调')
    parser.add_argument('--dora_rank', type=int, default=32, help='DoRA适配器的低秩维度')
    parser.add_argument('--freeze_encoder', action='store_true', help='冻结编码器，仅训练分类头和DoRA适配器')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    params = vars(args)
    params_file = os.path.join(os.path.dirname(args.log_path), 'hyperparameters.json')
    with open(params_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"Hyperparameters saved to {params_file}")
    
    use_amp = args.amp and torch.cuda.is_available()
    if use_amp:
        global scaler
        scaler = torch.amp.GradScaler('cuda')
        print("Using mixed precision training with GradScaler")
    else:
        scaler = None
    
    finetune(
        args.zarr_path, args.epochs, args.batch_size,
        args.d_model, args.nhead, args.layers, args.ff,
        args.lr_head, args.lr_enc, args.weight_decay,
        args.amp, args.clip_norm, args.patience,
        args.save_best, args.save_last, args.log_path,
        args.warmup_epochs, args.pretrain_weights,
        args.train_ratio, args.val_ratio, args.seed,
        use_dora=args.use_dora,
        dora_rank=args.dora_rank,
        lr_dora=args.lr_dora,
        freeze_encoder=args.freeze_encoder
    )

if __name__ == '__main__':
    main()
