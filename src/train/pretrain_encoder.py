import argparse
import os
import csv
import json
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from src.data.zarr_dataset import TemporalPixelDataset
from src.models.bert_sar_encoder import SarBertEncoder, DualBranchSarBertEncoder
from src.models.ct_ae_encoder import CT_AE_Encoder
from src.models.htf_mst_encoder import HTF_MST_Encoder, HTF_MST_Encoder_V2
from src.models.sar_ts_mamba import SARTSMambaEncoder, SARTSMambaEncoderV2


def train(zarr_path, epochs, batch_size, lr, d_model, nhead, layers, ff, mask_ratio, noise_scale, samples, seed, train_ratio, span_masking, span_len, weight_decay, patience, clip_norm, amp, log_path, save_best, save_last, use_dual_branch=False, physics_guided=False, span_physics=False, use_ct_ae=False, ct_ae_fusion='cross_attn', use_htf_mst=False, htf_version=1, tcn_layers=2, tcn_channels=128, use_mamba=False, mamba_version=1, d_state=16, d_conv=4, expand=2, use_sar_norm=True, temporal_layers=4, spectral_layers=2, resume=None, band_indices=None):
    import zarr
    store = zarr.open(zarr_path, mode='r')
    num_bands = store['observations'].shape[3]
    if band_indices is not None:
        num_bands = len(band_indices)
    print(f"波段数: {num_bands}" + (f" (选取波段索引: {band_indices})" if band_indices else ""))

    train_ds = TemporalPixelDataset(zarr_path, samples_per_epoch=samples, mask_ratio=mask_ratio, noise_scale=noise_scale, seed=seed, split='train', train_ratio=train_ratio, span_masking=span_masking, span_len=span_len, physics_guided=physics_guided, span_physics=span_physics, band_indices=band_indices)
    val_ds = TemporalPixelDataset(zarr_path, samples_per_epoch=max(1,int(samples*(1-train_ratio))), mask_ratio=mask_ratio, noise_scale=noise_scale, seed=seed, split='val', train_ratio=train_ratio, span_masking=span_masking, span_len=span_len, physics_guided=physics_guided, span_physics=span_physics, band_indices=band_indices)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_mamba:
        encoder_class = SARTSMambaEncoder if mamba_version == 1 else SARTSMambaEncoderV2
        encoder_kwargs = {
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': layers,
            'dim_feedforward': ff,
            'num_bands': num_bands,
            'd_state': d_state,
            'd_conv': d_conv,
            'expand': expand,
            'use_sar_norm': use_sar_norm
        }
        if mamba_version == 2:
            encoder_kwargs.update({
                'temporal_layers': temporal_layers,
                'spectral_layers': spectral_layers
            })
        model = encoder_class(**encoder_kwargs).to(device)
        print(f"使用SAR-TS-Mamba编码器 (version={mamba_version}, d_state={d_state}, d_conv={d_conv})")
    elif use_htf_mst:
        encoder_class = HTF_MST_Encoder if htf_version == 1 else HTF_MST_Encoder_V2
        model = encoder_class(d_model=d_model, nhead=nhead, num_layers=layers, dim_feedforward=ff, num_bands=num_bands, tcn_layers=tcn_layers, tcn_channels=tcn_channels).to(device)
        print(f"使用HTF-MST编码器 (version={htf_version}, tcn_layers={tcn_layers})")
    elif use_ct_ae:
        model=CT_AE_Encoder(d_model=d_model, nhead=nhead, num_layers=layers, dim_feedforward=ff, num_bands=num_bands, fusion_type=ct_ae_fusion).to(device)
        print(f"使用CNN-Transformer双分支交叉注意力编码器 (fusion={ct_ae_fusion})")
    elif use_dual_branch:
        model=DualBranchSarBertEncoder(d_model=d_model, nhead=nhead, num_layers=layers, dim_feedforward=ff, num_bands=num_bands).to(device)
        print(f"使用双分支跨模态融合编码器")
    else:
        model=SarBertEncoder(d_model=d_model, nhead=nhead, num_layers=layers, dim_feedforward=ff, num_bands=num_bands).to(device)
        print(f"使用单分支编码器")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
    scaler = GradScaler(enabled=amp and torch.cuda.is_available())
    best = float('inf')
    wait =0
    start_epoch = 0
    if resume and os.path.isfile(resume):
        print(f"从checkpoint恢复训练: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        if os.path.isfile(log_path):
            with open(log_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                rows = list(reader)
            if rows:
                start_epoch = int(rows[-1][0])
                best = min(float(r[2]) for r in rows)
                print(f"已完成epoch: {start_epoch}, 历史最佳val_mse: {best:.6f}")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            w =csv.writer(f)
            w.writerow(['epoch', 'train_loss', 'val_mse', 'lr', 'best'])
    for epoch in range(start_epoch, epochs):
        model.train()
        train_se=0.0
        train_cnt=0
        pbar=tqdm(train_dl, desc=f"epoch {epoch+1}/{epochs}")
        for batch in pbar:
            noisy=batch['noisy'].to(device)
            clean=batch['clean'].to(device)
            time_idx=batch['time_idx'].to(device)
            mask=batch['mask'].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp and torch.cuda.is_available()):
                pred=model(noisy, time_idx, mask)
                b=torch.where(mask)
                target=clean[b[0], b[1]]
                loss=F.mse_loss(pred, target)
            scaler.scale(loss).backward()
            if clip_norm>0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(opt)
            scaler.update()
            train_se+=loss.item()*pred.shape[0]
            train_cnt+=pred.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        train_loss=train_se/max(1, train_cnt)
        model.eval()
        val_se=0.0
        val_cnt=0
        with torch.no_grad():
            for batch in val_dl:
                noisy=batch['noisy'].to(device)
                clean=batch['clean'].to(device)
                time_idx=batch['time_idx'].to(device)
                mask=batch['mask'].to(device)
                pred=model(noisy, time_idx, mask)
                b=torch.where(mask)
                target=clean[b[0], b[1]]
                se=F.mse_loss(pred, target, reduction='sum').item()
                val_se+=se
                val_cnt+=pred.shape[0]
        val_mse=val_se/max(1, val_cnt)
        scheduler.step(val_mse)
        lr_now=opt.param_groups[0]['lr']
        print(f"epoch {epoch+1}/{epochs} - train_loss={train_loss:.6f} - val_loss={val_mse:.6f} - lr={lr_now:.6e}")
        with open(log_path, 'a', newline='') as f:
            w=csv.writer(f)
            w.writerow([epoch+1, f"{train_loss:.6f}", f"{val_mse:.6f}", f"{lr_now:.6e}", int(val_mse<best)])
        if val_mse<best:
            best=val_mse
            torch.save(model.state_dict(), save_best)
            wait=0
        else:
            wait+=1
            if wait>=patience:
                break
    torch.save(model.state_dict(), save_last)
    return model

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zarr_path', type=str, required=True)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--ff', type=int, default=512)
    p.add_argument('--mask_ratio', type=float, default=0.15)
    p.add_argument('--noise_scale', type=float, default=0.1)
    p.add_argument('--samples', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train_ratio', type=float, default=0.9)
    p.add_argument('--span_masking', action='store_true')
    p.add_argument('--span_len', type=int, default=3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--clip_norm', type=float, default=1.0)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--log_path', type=str, default='outputs/train_log.csv')
    p.add_argument('--save_best', type=str, default='outputs/encoder_best.pt')
    p.add_argument('--save_last', type=str, default='outputs/encoder_last.pt')
    p.add_argument('--resume', type=str, default=None, help='从checkpoint恢复训练')
    p.add_argument('--use_dual_branch', action='store_true', help='使用双分支跨模态融合编码器')
    p.add_argument('--physics_guided', action='store_true', help='使用物理引导掩蔽策略')
    p.add_argument('--span_physics', action='store_true', help='使用连续+物理引导掩蔽策略')
    p.add_argument('--use_ct_ae', action='store_true', help='使用CNN-Transformer双分支交叉注意力编码器')
    p.add_argument('--ct_ae_fusion', type=str, default='cross_attn', choices=['cross_attn', 'gate', 'add', 'concat'], help='CT-AE融合方式')
    p.add_argument('--use_htf_mst', action='store_true', help='使用HTF-MST层次化时序融合编码器')
    p.add_argument('--htf_version', type=int, default=1, choices=[1, 2], help='HTF-MST版本 (1: 单次融合, 2: 渐进融合)')
    p.add_argument('--tcn_layers', type=int, default=2, help='TCN层数')
    p.add_argument('--tcn_channels', type=int, default=128, help='TCN通道数')
    p.add_argument('--use_mamba', action='store_true', help='使用SAR-TS-Mamba编码器')
    p.add_argument('--mamba_version', type=int, default=1, choices=[1, 2], help='Mamba版本 (1: 标准, 2: 时空分离)')
    p.add_argument('--d_state', type=int, default=16, help='Mamba状态维度')
    p.add_argument('--d_conv', type=int, default=4, help='Mamba卷积核大小')
    p.add_argument('--expand', type=int, default=2, help='Mamba扩展因子')
    p.add_argument('--use_sar_norm', action='store_true', default=True, help='使用SAR自适应归一化')
    p.add_argument('--temporal_layers', type=int, default=4, help='Mamba V2时间层')
    p.add_argument('--spectral_layers', type=int, default=2, help='Mamba V2光谱层')
    p.add_argument('--band_indices', type=str, default=None, help='选取的波段索引，逗号分隔，如"3,4,5"表示选取后三波段')
    a=p.parse_args()
    os.makedirs(os.path.dirname(a.save_best),exist_ok=True)
    os.makedirs(os.path.dirname(a.save_last),exist_ok=True)
    os.makedirs(os.path.dirname(a.log_path),exist_ok=True)

    params = vars(a)
    
    # 解析band_indices
    band_indices = None
    if a.band_indices is not None:
        band_indices = [int(x.strip()) for x in a.band_indices.split(',')]
        print(f"选取波段索引: {band_indices} (共{len(band_indices)}个波段)")
    
    params_file = os.path.join(os.path.dirname(a.log_path), 'hyperparameters.json')
    with open(params_file, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"Hyperparameters saved to {params_file}")

    train(a.zarr_path,a.epochs,a.batch_size,a.lr,a.d_model,a.nhead,a.layers,a.ff,a.mask_ratio,a.noise_scale,a.samples,a.seed,a.train_ratio,a.span_masking,a.span_len,a.weight_decay,a.patience,a.clip_norm,a.amp,a.log_path,a.save_best,a.save_last,a.use_dual_branch,a.physics_guided,a.span_physics,a.use_ct_ae,a.ct_ae_fusion,a.use_htf_mst,a.htf_version,a.tcn_layers,a.tcn_channels,a.use_mamba,a.mamba_version,a.d_state,a.d_conv,a.expand,a.use_sar_norm,a.temporal_layers,a.spectral_layers,a.resume,band_indices)

if __name__=='__main__':
    main()
