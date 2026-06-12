import argparse
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from src.data.processed_dataset import ProcessedParcelDataset
from src.models.parcel_classifier import ParcelClassifier

def finetune(zarr_path, epochs, batch_size, d_model, nhead, layers, ff, lr_head, lr_enc, weight_decay, amp, clip_norm, patience, save_best, save_last, log_path, warmup_epochs, pretrain_weights=None, train_ratio=0.8, val_ratio=0.1, seed=42, use_dora=False, dora_rank=32, lr_dora=5e-4, freeze_encoder=False, use_dual_branch=False, use_ct_ae=False, ct_ae_fusion='cross_attn', use_htf_mst=False, htf_version=1, tcn_layers=2, tcn_channels=128, use_mamba=False, mamba_version=1, d_state=16, d_conv=4, expand=2, use_sar_norm=True, temporal_layers=4, spectral_layers=2, resume=None, start_epoch=0, use_mamba3=False, mimo_rank=2, time_indices=None, save_every_n_batch=0, save_epochs=None, norm_type=None, pad_bands=None, band_indices=None):
    full_ds = ProcessedParcelDataset(zarr_path, time_indices=time_indices, pad_bands=pad_bands, band_indices=band_indices)
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
    
    num_classes = len(full_ds.label_map) if hasattr(full_ds, 'label_map') else len(full_ds.class_names) if full_ds.class_names is not None else int(torch.max(torch.from_numpy(np.asarray(full_ds.labels)))+1)
    print(f"使用的类别数量: {num_classes}")
    print(f"类别名称: {full_ds.class_names}")
    print(f"标签映射: {full_ds.label_map}")
    
    num_bands = pad_bands if pad_bands is not None else full_ds.original_bands
    print(f"数据波段数量: {num_bands} (原始: {full_ds.series.shape[2]}, pad_bands: {pad_bands}, band_indices: {band_indices})")
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    encoder_mode = 'SAR-TS-Mamba' if use_mamba else ('HTF-MST层次化时序融合' if use_htf_mst else ('CNN-Transformer双分支交叉注意力' if use_ct_ae else ('双分支跨模态融合' if use_dual_branch else '单分支')))
    print(f"编码器模式: {encoder_mode}")
    
    model = ParcelClassifier(
        num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=layers,
        dim_feedforward=ff,
        num_bands=num_bands,
        use_dora=use_dora,
        dora_rank=dora_rank,
        use_dual_branch=use_dual_branch,
        use_ct_ae=use_ct_ae,
        ct_ae_fusion=ct_ae_fusion,
        use_htf_mst=use_htf_mst,
        htf_version=htf_version,
        tcn_layers=tcn_layers,
        tcn_channels=tcn_channels,
        use_mamba=use_mamba,
        mamba_version=mamba_version,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        use_sar_norm=use_sar_norm,
        temporal_layers=temporal_layers,
        spectral_layers=spectral_layers,
        use_mamba3=use_mamba3,
        mimo_rank=mimo_rank,
        norm_type=norm_type
    ).to(device)
    
    # 加载预训练权重
    if pretrain_weights and os.path.exists(pretrain_weights):
        try:
            # 使用weights_only=False避免版本兼容性问题
            sd = torch.load(pretrain_weights, map_location=device, weights_only=False)
            model.encoder.load_state_dict(sd, strict=False)
            print(f"Loaded pretrained weights from {pretrain_weights}")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
    elif pretrain_weights:
        print(f"Pretrained weights file not found: {pretrain_weights}")
    
    # 从checkpoint恢复训练
    if resume and os.path.exists(resume):
        try:
            sd = torch.load(resume, map_location=device, weights_only=False)
            model.load_state_dict(sd)
            print(f"Resumed from checkpoint: {resume}")
        except Exception as e:
            print(f"Failed to resume from checkpoint: {e}")
    
    # 参数分组
    enc_params = list(model.encoder.parameters())
    head_params = list(model.clf_head.parameters())
    for p in enc_params:
        p.requires_grad = True
    
    # 优化器设置
    if use_dora:
        # 获取不同部分的参数（确保没有重复）
        encoder_core_params = model.get_encoder_core_params()
        obs_proj_params = model.get_obs_proj_params()
        dora_params = model.get_dora_params()
        head_params = model.get_head_params()
        
        # 分层学习率配置
        param_groups = []
        
        # 仅训练DoRA适配器和分类头（如果冻结编码器）
        if freeze_encoder:
            param_groups = [
                {'params': dora_params, 'lr': lr_dora, 'weight_decay': weight_decay},
                {'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay},
            ]
        else:
            # 所有参数都参与训练，使用不同的学习率
            param_groups = [
                {'params': encoder_core_params, 'lr': lr_enc, 'weight_decay': weight_decay},
                {'params': obs_proj_params, 'lr': lr_enc, 'weight_decay': weight_decay},
                {'params': dora_params, 'lr': lr_dora, 'weight_decay': weight_decay},
                {'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay},
            ]
        
        print(f"使用DoRA优化，参数组数量: {len(param_groups)}")
        print(f"编码器核心参数数量: {sum(p.numel() for p in encoder_core_params)}")
        print(f"观测投影参数数量: {sum(p.numel() for p in obs_proj_params)}")
        print(f"DoRA适配器参数数量: {sum(p.numel() for p in dora_params)}")
        print(f"分类头参数数量: {sum(p.numel() for p in head_params)}")
        print(f"总参数数量: {sum(p.numel() for group in param_groups for p in group['params'])}")
    else:
        # 传统优化器设置
        enc_params = list(model.encoder.parameters())
        head_params = list(model.clf_head.parameters())
        
        param_groups = [
            {'params': enc_params, 'lr': lr_enc, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay},
        ]
    
    # 创建优化器
    opt = torch.optim.AdamW(param_groups)
    
    # 混合精度设置 - 根据PyTorch版本选择正确的API
    use_amp = amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    print(f"混合精度训练: {'启用' if use_amp else '禁用'}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(save_best), exist_ok=True)
    os.makedirs(os.path.dirname(save_last), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # 初始化日志文件
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'train_loss', 'val_loss', 'acc', 'macro_f1', 'lr_head', 'lr_enc', 'best'])
        
        # 保存类别名称映射
        if full_ds.class_names is not None:
            with open(os.path.join(os.path.dirname(log_path), 'class_names.json'), 'w', encoding='utf-8') as f:
                import json
                json.dump({'class_names': full_ds.class_names}, f, ensure_ascii=False, indent=2)
    
    # 训练循环
    best = float('inf')
    wait = 0
    
    for epoch in range(start_epoch, epochs):
        model.train()
        
        # warmup: 冻结编码器
        freeze = (epoch < warmup_epochs)
        for p in enc_params:
            p.requires_grad = not freeze
        
        # 训练阶段
        train_se = 0.0
        train_cnt = 0
        pbar = tqdm(train_dl, desc=f"Train Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            x = batch['series'].to(device, non_blocking=True)
            t = batch['time_idx'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)
            
            opt.zero_grad(set_to_none=True)
            
            # 使用混合精度训练
            if use_amp:
                # 使用autocast进行前向传播
                with torch.amp.autocast('cuda'):
                    logits = model(x.unsqueeze(0) if x.dim() == 2 else x, 
                                 t.unsqueeze(0) if t.dim() == 1 else t)
                    loss = F.cross_entropy(logits, y)
                
                # 使用GradScaler进行梯度缩放和优化步骤
                scaler.scale(loss).backward()
                if clip_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x.unsqueeze(0) if x.dim() == 2 else x, 
                             t.unsqueeze(0) if t.dim() == 1 else t)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                opt.step()
            
            train_se += loss.item() * y.shape[0]
            train_cnt += y.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        
        train_loss = train_se / max(1, train_cnt)
        
        # 验证阶段
        model.eval()
        val_se = 0.0
        val_cnt = 0
        all_y = []
        all_p = []
        
        with torch.no_grad():
            for batch in val_dl:
                x = batch['series'].to(device, non_blocking=True)
                t = batch['time_idx'].to(device, non_blocking=True)
                y = batch['label'].to(device, non_blocking=True)
                
                # 验证阶段使用混合精度
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(x.unsqueeze(0) if x.dim() == 2 else x, 
                                     t.unsqueeze(0) if t.dim() == 1 else t)
                else:
                    logits = model(x.unsqueeze(0) if x.dim() == 2 else x, 
                                 t.unsqueeze(0) if t.dim() == 1 else t)
                
                se = F.cross_entropy(logits, y, reduction='sum').item()
                val_se += se
                val_cnt += y.shape[0]
                all_y.append(y.cpu().numpy())
                all_p.append(logits.argmax(dim=1).cpu().numpy())
        
        val_loss = val_se / max(1, val_cnt)
        all_y = np.concatenate(all_y) if all_y else np.array([])
        all_p = np.concatenate(all_p) if all_p else np.array([])
        
        acc = accuracy_score(all_y, all_p) if len(all_y) > 0 else 0.0
        macro_f1 = f1_score(all_y, all_p, average='macro') if len(all_y) > 0 else 0.0
        
        lr_head = opt.param_groups[1]['lr']
        lr_enc = opt.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs} - train_loss={train_loss:.6f} - val_loss={val_loss:.6f} - acc={acc:.4f} - f1={macro_f1:.4f}")
        
        # 记录日志
        with open(log_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([epoch+1, f"{train_loss:.6f}", f"{val_loss:.6f}", 
                       f"{acc:.4f}", f"{macro_f1:.4f}", 
                       f"{lr_head:.6e}", f"{lr_enc:.6e}", int(val_loss < best)])
        
        # 早停和保存最佳模型
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
        
        # 每N个epoch保存checkpoint
        if save_every_n_batch > 0 and (epoch + 1) % save_every_n_batch == 0:
            ckpt_dir = os.path.dirname(save_best)
            ckpt_path = os.path.join(ckpt_dir, f"parcel_cls_ep{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved epoch checkpoint: {ckpt_path}")
        
        # 保存指定epoch的checkpoint
        if save_epochs and (epoch + 1) in save_epochs:
            ckpt_dir = os.path.dirname(save_best)
            ckpt_path = os.path.join(ckpt_dir, f"parcel_cls_ep{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved specified epoch checkpoint: {ckpt_path}")
    
    # 保存最终模型
    torch.save(model.state_dict(), save_last)
    print(f"Saved final model to {save_last}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=25)
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
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--log_path', type=str, default='outputs/cls_train_log.csv')
    parser.add_argument('--save_best', type=str, default='outputs/parcel_cls_best.pt')
    parser.add_argument('--save_last', type=str, default='outputs/parcel_cls_last.pt')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--pretrain_weights', type=str, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_dora', action='store_true', help='启用DoRA适配器微调')
    parser.add_argument('--dora_rank', type=int, default=32, help='DoRA适配器的低秩维度')
    parser.add_argument('--freeze_encoder', action='store_true', help='冻结编码器，仅训练分类头和DoRA适配器')
    parser.add_argument('--use_dual_branch', action='store_true', help='使用双分支跨模态融合编码器')
    parser.add_argument('--use_ct_ae', action='store_true', help='[DEPRECATED] 使用CNN-Transformer双分支交叉注意力编码器')
    parser.add_argument('--ct_ae_fusion', type=str, default='cross_attn', choices=['cross_attn','gate','add','concat'], help='[DEPRECATED] CT-AE融合方式')
    parser.add_argument('--use_htf_mst', action='store_true', help='[DEPRECATED] 使用HTF-MST层次化时序融合编码器')
    parser.add_argument('--htf_version', type=int, default=1, choices=[1,2], help='[DEPRECATED] HTF-MST版本')
    parser.add_argument('--tcn_layers', type=int, default=2, help='[DEPRECATED] TCN层数')
    parser.add_argument('--tcn_channels', type=int, default=128, help='[DEPRECATED] TCN通道数')
    parser.add_argument('--use_mamba', action='store_true', help='[DEPRECATED] 使用SAR-TS-Mamba编码器')
    parser.add_argument('--mamba_version', type=int, default=1, choices=[1, 2], help='[DEPRECATED] Mamba版本')
    parser.add_argument('--d_state', type=int, default=16, help='[DEPRECATED] Mamba状态维度')
    parser.add_argument('--d_conv', type=int, default=4, help='[DEPRECATED] Mamba卷积核大小')
    parser.add_argument('--expand', type=int, default=2, help='[DEPRECATED] Mamba扩展因子')
    parser.add_argument('--use_sar_norm', action='store_true', default=True, help='使用SAR自适应归一化')
    parser.add_argument('--temporal_layers', type=int, default=4, help='[DEPRECATED] Mamba V2时间层')
    parser.add_argument('--spectral_layers', type=int, default=2, help='[DEPRECATED] Mamba V2光谱层')
    parser.add_argument('--use_mamba3', action='store_true', help='[DEPRECATED] 使用Mamba-3架构')
    parser.add_argument('--mimo_rank', type=int, default=2, help='[DEPRECATED] Mamba-3 MIMO秩')
    parser.add_argument('--resume', type=str, default=None, help='从checkpoint恢复训练')
    parser.add_argument('--start_epoch', type=int, default=0, help='起始epoch')
    parser.add_argument('--time_indices', type=str, default=None, help='选取的时间步索引，逗号分隔，如"0,2,4,6"')
    parser.add_argument('--save_every_n_batch', type=int, default=0, help='每N个epoch保存一个checkpoint，0表示不保存')
    parser.add_argument('--save_epochs', type=str, default=None, help='指定保存的epoch编号，逗号分隔，如"6,7,8,9,10"')
    parser.add_argument('--norm_type', type=str, default=None, choices=['layer', 'batch', 'sar_adaptive'], help='输入归一化策略: layer/batch/sar_adaptive')
    parser.add_argument('--pad_bands', type=int, default=None, help='将输入波段补零到指定数量，用于跨区域迁移时适配不同波段数')
    parser.add_argument('--band_indices', type=str, default=None, help='选取的波段索引，逗号分隔，如"3,4,5"表示选取后三波段')

    args = parser.parse_args()
    
    time_indices = None
    if args.time_indices is not None:
        time_indices = [int(x.strip()) for x in args.time_indices.split(',')]
        print(f"选取时间步索引: {time_indices} (共{len(time_indices)}个)")
    
    band_indices = None
    if args.band_indices is not None:
        band_indices = [int(x.strip()) for x in args.band_indices.split(',')]
        print(f"选取波段索引: {band_indices} (共{len(band_indices)}个波段)")
    
    save_epochs = None
    if args.save_epochs is not None:
        save_epochs = set(int(x.strip()) for x in args.save_epochs.split(','))
        print(f"指定保存的epoch: {sorted(save_epochs)}")
    
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
        freeze_encoder=args.freeze_encoder,
        use_dual_branch=args.use_dual_branch,
        use_ct_ae=args.use_ct_ae,
        ct_ae_fusion=args.ct_ae_fusion,
        use_htf_mst=args.use_htf_mst,
        htf_version=args.htf_version,
        tcn_layers=args.tcn_layers,
        tcn_channels=args.tcn_channels,
        use_mamba=args.use_mamba,
        mamba_version=args.mamba_version,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        use_sar_norm=args.use_sar_norm,
        temporal_layers=args.temporal_layers,
        spectral_layers=args.spectral_layers,
        use_mamba3=args.use_mamba3,
        mimo_rank=args.mimo_rank,
        resume=args.resume,
        start_epoch=args.start_epoch,
        time_indices=time_indices,
        save_every_n_batch=args.save_every_n_batch,
        save_epochs=save_epochs,
        norm_type=args.norm_type,
        pad_bands=args.pad_bands,
        band_indices=band_indices
    )

if __name__ == '__main__':
    main()