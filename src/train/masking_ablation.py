import argparse
import os
import json
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from src.data.zarr_dataset import TemporalPixelDataset
from src.data.processed_dataset import ProcessedParcelDataset
from src.models.bert_sar_encoder import SarBertEncoder
from src.models.parcel_classifier import ParcelClassifier


def pretrain_masking(zarr_path, masking_type, epochs, batch_size, lr, d_model, nhead,
                     layers, ff, mask_ratio, noise_scale, samples, seed, train_ratio,
                     weight_decay, patience, clip_norm, amp, output_dir, span_len=3):
    os.makedirs(output_dir, exist_ok=True)
    save_best = os.path.join(output_dir, 'encoder_best.pt')
    save_last = os.path.join(output_dir, 'encoder_last.pt')
    log_path = os.path.join(output_dir, 'pretrain_log.csv')

    span_masking = (masking_type == 'span')
    physics_guided = (masking_type == 'physics')

    train_ds = TemporalPixelDataset(
        zarr_path, samples_per_epoch=samples, mask_ratio=mask_ratio,
        noise_scale=noise_scale, seed=seed, split='train', train_ratio=train_ratio,
        span_masking=span_masking, span_len=span_len, physics_guided=physics_guided
    )
    val_ds = TemporalPixelDataset(
        zarr_path, samples_per_epoch=max(1, int(samples * (1 - train_ratio))),
        mask_ratio=mask_ratio, noise_scale=noise_scale, seed=seed, split='val',
        train_ratio=train_ratio, span_masking=span_masking, span_len=span_len,
        physics_guided=physics_guided
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import zarr
    store = zarr.open(zarr_path, mode='r')
    num_bands = store['observations'].shape[3]

    model = SarBertEncoder(
        d_model=d_model, nhead=nhead, num_layers=layers,
        dim_feedforward=ff, num_bands=num_bands
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
    scaler = GradScaler(enabled=amp and torch.cuda.is_available())

    best_val = float('inf')
    wait = 0

    with open(log_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'val_mse', 'lr', 'best'])

    print(f"\n{'='*70}")
    print(f"Pre-training | Masking: {masking_type} | Epochs: {epochs}")
    print(f"{'='*70}")

    for epoch in range(epochs):
        model.train()
        train_se = 0.0
        train_cnt = 0
        pbar = tqdm(train_dl, desc=f"[{masking_type}] Pretrain Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            time_idx = batch['time_idx'].to(device)
            mask = batch['mask'].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp and torch.cuda.is_available()):
                pred = model(noisy, time_idx, mask)
                b = torch.where(mask)
                target = clean[b[0], b[1]]
                loss = F.mse_loss(pred, target)
            scaler.scale(loss).backward()
            if clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(opt)
            scaler.update()
            train_se += loss.item() * pred.shape[0]
            train_cnt += pred.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        train_loss = train_se / max(1, train_cnt)

        model.eval()
        val_se = 0.0
        val_cnt = 0
        with torch.no_grad():
            for batch in val_dl:
                noisy = batch['noisy'].to(device)
                clean = batch['clean'].to(device)
                time_idx = batch['time_idx'].to(device)
                mask = batch['mask'].to(device)
                pred = model(noisy, time_idx, mask)
                b = torch.where(mask)
                target = clean[b[0], b[1]]
                se = F.mse_loss(pred, target, reduction='sum').item()
                val_se += se
                val_cnt += pred.shape[0]
        val_mse = val_se / max(1, val_cnt)
        scheduler.step(val_mse)
        lr_now = opt.param_groups[0]['lr']

        is_best = " ***BEST***" if val_mse < best_val else ""
        print(f"[{masking_type}] Epoch {epoch+1:3d}/{epochs} | train_loss={train_loss:.6f} | val_mse={val_mse:.6f} | lr={lr_now:.6e}{is_best}")

        with open(log_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([epoch+1, f"{train_loss:.6f}", f"{val_mse:.6f}", f"{lr_now:.6e}", int(val_mse < best_val)])

        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), save_best)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[{masking_type}] Early stopping at epoch {epoch+1}")
                break

    torch.save(model.state_dict(), save_last)

    hp = {
        'zarr_path': zarr_path, 'masking_type': masking_type, 'epochs': epochs,
        'batch_size': batch_size, 'lr': lr, 'd_model': d_model, 'nhead': nhead,
        'layers': layers, 'ff': ff, 'mask_ratio': mask_ratio, 'noise_scale': noise_scale,
        'samples': samples, 'seed': seed, 'train_ratio': train_ratio,
        'span_masking': span_masking, 'span_len': span_len, 'physics_guided': physics_guided,
        'weight_decay': weight_decay, 'patience': patience, 'clip_norm': clip_norm, 'amp': amp,
        'best_val_mse': float(best_val)
    }
    with open(os.path.join(output_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hp, f, indent=2, ensure_ascii=False)

    del model
    torch.cuda.empty_cache()
    return save_best, best_val


def finetune_and_eval(zarr_path, pretrain_weights, output_dir, epochs, batch_size,
                      d_model, nhead, layers, ff, lr_head, lr_enc, lr_dora,
                      weight_decay, amp, clip_norm, patience, warmup_epochs,
                      use_dora, dora_rank, seed, train_ratio=0.4, val_ratio=0.1):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'cls_train_log.csv')
    save_best = os.path.join(output_dir, 'parcel_cls_best.pt')
    save_last = os.path.join(output_dir, 'parcel_cls_last.pt')

    full_ds = ProcessedParcelDataset(zarr_path)
    total_size = len(full_ds)
    num_classes = len(full_ds.label_map) if hasattr(full_ds, 'label_map') else int(torch.max(torch.from_numpy(np.asarray(full_ds.labels))) + 1)
    num_bands = full_ds.series.shape[2]

    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    rng = np.random.default_rng(seed)
    all_indices = np.arange(total_size)
    rng.shuffle(all_indices)
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]

    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)
    test_ds = Subset(full_ds, test_indices)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ParcelClassifier(
        num_classes, d_model=d_model, nhead=nhead, num_layers=layers,
        dim_feedforward=ff, num_bands=num_bands, use_dora=use_dora,
        dora_rank=dora_rank, use_sar_norm=True
    ).to(device)

    tag = os.path.basename(output_dir)
    if pretrain_weights and os.path.exists(pretrain_weights):
        sd = torch.load(pretrain_weights, map_location=device, weights_only=False)
        model.encoder.load_state_dict(sd, strict=False)
        print(f"[{tag}] Loaded pretrained weights from {pretrain_weights}")

    enc_params = list(model.encoder.parameters())
    if use_dora:
        encoder_core_params = model.get_encoder_core_params()
        obs_proj_params = model.get_obs_proj_params()
        dora_params = model.get_dora_params()
        head_params = model.get_head_params()
        param_groups = [
            {'params': encoder_core_params, 'lr': lr_enc, 'weight_decay': weight_decay},
            {'params': obs_proj_params, 'lr': lr_enc, 'weight_decay': weight_decay},
            {'params': dora_params, 'lr': lr_dora, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay},
        ]
    else:
        head_params = list(model.clf_head.parameters())
        param_groups = [
            {'params': enc_params, 'lr': lr_enc, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay},
        ]

    opt = torch.optim.AdamW(param_groups)
    scaler = GradScaler(enabled=amp and torch.cuda.is_available())
    best_val_loss = float('inf')
    best_state = None
    wait = 0

    with open(log_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'val_loss', 'acc', 'macro_f1', 'best'])

    print(f"\n{'='*70}")
    print(f"Fine-tuning | {tag} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"{'='*70}")

    for epoch in range(epochs):
        model.train()
        freeze = (epoch < warmup_epochs)
        for p in enc_params:
            p.requires_grad = not freeze

        train_se = 0.0
        train_cnt = 0
        pbar = tqdm(train_dl, desc=f"[{tag}] Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x = batch['series'].to(device, non_blocking=True)
            t = batch['time_idx'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp and torch.cuda.is_available()):
                logits = model(x.unsqueeze(0) if x.dim() == 2 else x,
                              t.unsqueeze(0) if t.dim() == 1 else t)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            if clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(opt)
            scaler.update()
            train_se += loss.item() * y.shape[0]
            train_cnt += y.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = train_se / max(1, train_cnt)

        model.eval()
        val_se = 0.0
        val_cnt = 0
        all_y, all_p = [], []
        with torch.no_grad():
            for batch in val_dl:
                x = batch['series'].to(device, non_blocking=True)
                t = batch['time_idx'].to(device, non_blocking=True)
                y = batch['label'].to(device, non_blocking=True)
                with autocast(enabled=amp and torch.cuda.is_available()):
                    logits = model(x.unsqueeze(0) if x.dim() == 2 else x,
                                  t.unsqueeze(0) if t.dim() == 1 else t)
                val_se += F.cross_entropy(logits, y, reduction='sum').item()
                val_cnt += y.shape[0]
                all_y.append(y.cpu().numpy())
                all_p.append(logits.argmax(dim=1).cpu().numpy())

        val_loss = val_se / max(1, val_cnt)
        all_y = np.concatenate(all_y)
        all_p = np.concatenate(all_p)
        val_acc = accuracy_score(all_y, all_p)
        val_f1 = f1_score(all_y, all_p, average='macro')

        is_best = " ***BEST***" if val_loss < best_val_loss else ""
        print(f"[{tag}] Epoch {epoch+1:3d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | acc={val_acc:.4f} | f1={val_f1:.4f}{is_best}")

        with open(log_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([epoch+1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_acc:.4f}", f"{val_f1:.4f}", int(val_loss < best_val_loss)])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            print(f"[{tag}] Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), save_best)
    torch.save(model.state_dict(), save_last)

    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for batch in test_dl:
            x = batch['series'].to(device, non_blocking=True)
            t = batch['time_idx'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)
            with autocast(enabled=amp and torch.cuda.is_available()):
                logits = model(x.unsqueeze(0) if x.dim() == 2 else x,
                              t.unsqueeze(0) if t.dim() == 1 else t)
            all_y.append(y.cpu().numpy())
            all_p.append(logits.argmax(dim=1).cpu().numpy())

    all_y = np.concatenate(all_y)
    all_p = np.concatenate(all_p)
    test_acc = accuracy_score(all_y, all_p)
    test_f1 = f1_score(all_y, all_p, average='macro')
    test_wf1 = f1_score(all_y, all_p, average='weighted')

    print(f"\n[{tag}] TEST RESULTS: OA={test_acc:.4f} | Macro F1={test_f1:.4f} | Weighted F1={test_wf1:.4f}")

    hp = {
        'zarr_path': zarr_path, 'pretrain_weights': pretrain_weights,
        'epochs': epochs, 'batch_size': batch_size, 'd_model': d_model,
        'nhead': nhead, 'layers': layers, 'ff': ff, 'lr_head': lr_head,
        'lr_enc': lr_enc, 'lr_dora': lr_dora, 'weight_decay': weight_decay,
        'amp': amp, 'clip_norm': clip_norm, 'patience': patience,
        'warmup_epochs': warmup_epochs, 'use_dora': use_dora, 'dora_rank': dora_rank,
        'seed': seed, 'train_ratio': train_ratio, 'val_ratio': val_ratio,
        'test_oa': float(test_acc), 'test_f1': float(test_f1), 'test_wf1': float(test_wf1),
        'best_val_loss': float(best_val_loss)
    }
    with open(os.path.join(output_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hp, f, indent=2, ensure_ascii=False)

    del model
    torch.cuda.empty_cache()
    return test_acc, test_f1, test_wf1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Masking Strategy Ablation for SAR-TS-BERT')
    p.add_argument('--zarr_path', type=str, default='F:\\DATA\\Estonia-6bands-labeld.zarr')
    p.add_argument('--zarr_unlabeled', type=str, default='F:\\DATA\\Estonia-6bands')
    p.add_argument('--output_base', type=str, default='outputs/masking_ablation')
    p.add_argument('--pretrain_epochs', type=int, default=50)
    p.add_argument('--pretrain_batch_size', type=int, default=512)
    p.add_argument('--pretrain_lr', type=float, default=1e-3)
    p.add_argument('--pretrain_samples', type=int, default=20000)
    p.add_argument('--pretrain_patience', type=int, default=10)
    p.add_argument('--finetune_epochs', type=int, default=300)
    p.add_argument('--finetune_batch_size', type=int, default=128)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--ff', type=int, default=512)
    p.add_argument('--mask_ratio', type=float, default=0.15)
    p.add_argument('--noise_scale', type=float, default=0.1)
    p.add_argument('--span_len', type=int, default=3)
    p.add_argument('--lr_head', type=float, default=1e-3)
    p.add_argument('--lr_enc', type=float, default=1e-4)
    p.add_argument('--lr_dora', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--clip_norm', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=35)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--use_dora', action='store_true', default=True)
    p.add_argument('--dora_rank', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--skip_pretrain', action='store_true', help='Skip pretrain, use existing weights')
    p.add_argument('--existing_pretrains', type=str, nargs='*', default=[],
                   help='Paths to existing pretrained weights [random, span, physics]')
    a = p.parse_args()

    masking_types = ['random', 'span', 'physics']
    results = {}

    pretrain_weights = {}
    if a.skip_pretrain and len(a.existing_pretrains) >= 3:
        for i, mt in enumerate(masking_types):
            pretrain_weights[mt] = a.existing_pretrains[i]
    else:
        for mt in masking_types:
            out_dir = os.path.join(a.output_base, f'pretrain_{mt}')
            pw, best_val = pretrain_masking(
                a.zarr_unlabeled, mt, a.pretrain_epochs, a.pretrain_batch_size,
                a.pretrain_lr, a.d_model, a.nhead, a.layers, a.ff, a.mask_ratio,
                a.noise_scale, a.pretrain_samples, a.seed, 0.9, a.weight_decay,
                a.pretrain_patience, a.clip_norm, a.amp, out_dir, a.span_len
            )
            pretrain_weights[mt] = pw
            print(f"\n[Pretrain] {mt}: best_val_mse={best_val:.6f}, weights={pw}")

    for mt in masking_types:
        ft_dir = os.path.join(a.output_base, f'finetune_{mt}')
        test_acc, test_f1, test_wf1 = finetune_and_eval(
            a.zarr_path, pretrain_weights[mt], ft_dir, a.finetune_epochs,
            a.finetune_batch_size, a.d_model, a.nhead, a.layers, a.ff,
            a.lr_head, a.lr_enc, a.lr_dora, a.weight_decay, a.amp, a.clip_norm,
            a.patience, a.warmup_epochs, a.use_dora, a.dora_rank, a.seed
        )
        results[mt] = {'oa': test_acc, 'f1': test_f1, 'wf1': test_wf1}

    print(f"\n{'='*80}")
    print(f"{'Masking Strategy':<25} {'OA':>12} {'Macro F1':>14} {'Weighted F1':>14}")
    print(f"{'-'*80}")
    for mt in masking_types:
        r = results[mt]
        print(f"{mt:<25} {r['oa']:>12.4f} {r['f1']:>14.4f} {r['wf1']:>14.4f}")
    print(f"{'-'*80}")

    summary_path = os.path.join(a.output_base, 'ablation_results.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {summary_path}")
