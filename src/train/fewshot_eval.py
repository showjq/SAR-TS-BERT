import argparse
import os
import json
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
from src.data.processed_dataset import ProcessedParcelDataset
from src.models.parcel_classifier import ParcelClassifier


def run_single_fewshot(zarr_path, pretrain_weights, train_indices, val_indices, test_indices,
                       full_ds, num_classes, num_bands, epochs, batch_size,
                       d_model, nhead, layers, ff, lr_head, lr_enc, weight_decay,
                       amp, clip_norm, patience, warmup_epochs, use_dora, dora_rank,
                       lr_dora, device, tag, use_dual_branch=False):

    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)
    test_ds = Subset(full_ds, test_indices)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ParcelClassifier(
        num_classes, d_model=d_model, nhead=nhead, num_layers=layers,
        dim_feedforward=ff, num_bands=num_bands, use_dora=use_dora,
        dora_rank=dora_rank, use_dual_branch=use_dual_branch, use_sar_norm=True
    ).to(device)

    if pretrain_weights and os.path.exists(pretrain_weights):
        sd = torch.load(pretrain_weights, map_location=device, weights_only=False)
        model.encoder.load_state_dict(sd, strict=False)

    enc_params = list(model.encoder.parameters())
    head_params = list(model.clf_head.parameters())

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
        param_groups = [
            {'params': enc_params, 'lr': lr_enc, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay},
        ]

    opt = torch.optim.AdamW(param_groups)
    use_amp = amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        freeze = (epoch < warmup_epochs)
        for p in enc_params:
            p.requires_grad = not freeze

        train_se = 0.0
        train_cnt = 0
        pbar = tqdm(train_dl, desc=f"[{tag}] Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            x = batch['series'].to(device, non_blocking=True)
            t = batch['time_idx'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(x.unsqueeze(0) if x.dim() == 2 else x,
                                 t.unsqueeze(0) if t.dim() == 1 else t)
                    loss = F.cross_entropy(logits, y)
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
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(x.unsqueeze(0) if x.dim() == 2 else x,
                                     t.unsqueeze(0) if t.dim() == 1 else t)
                else:
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

    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for batch in test_dl:
            x = batch['series'].to(device, non_blocking=True)
            t = batch['time_idx'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(x.unsqueeze(0) if x.dim() == 2 else x,
                                 t.unsqueeze(0) if t.dim() == 1 else t)
            else:
                logits = model(x.unsqueeze(0) if x.dim() == 2 else x,
                             t.unsqueeze(0) if t.dim() == 1 else t)
            all_y.append(y.cpu().numpy())
            all_p.append(logits.argmax(dim=1).cpu().numpy())

    all_y = np.concatenate(all_y)
    all_p = np.concatenate(all_p)
    test_acc = accuracy_score(all_y, all_p)
    test_f1 = f1_score(all_y, all_p, average='macro')
    test_wf1 = f1_score(all_y, all_p, average='weighted')

    del model
    torch.cuda.empty_cache()

    return test_acc, test_f1, test_wf1


def run_fewshot(zarr_path, pretrain_weights, label_ratios, epochs, batch_size,
                d_model, nhead, layers, ff, lr_head, lr_enc, weight_decay,
                amp, clip_norm, patience, warmup_epochs, use_dora, dora_rank,
                lr_dora, seed, output_dir, test_ratio=0.5, num_runs=3,
                use_dual_branch=False):

    full_ds = ProcessedParcelDataset(zarr_path)
    total_size = len(full_ds)
    num_classes = len(full_ds.label_map) if hasattr(full_ds, 'label_map') else int(torch.max(torch.from_numpy(np.asarray(full_ds.labels))) + 1)
    num_bands = full_ds.series.shape[2]

    print(f"=" * 70)
    print(f"SAR-TS-BERT Few-Shot Evaluation")
    print(f"Dataset: {total_size} samples, {num_classes} classes, {num_bands} bands")
    print(f"Label ratios: {[f'{r*100:.0f}%' for r in label_ratios]}")
    print(f"Pre-train weights: {pretrain_weights}")
    print(f"Num runs per config: {num_runs}")
    print(f"=" * 70)

    test_size = int(test_ratio * total_size)
    remaining = total_size - test_size

    rng = np.random.default_rng(seed)
    all_indices = np.arange(total_size)
    rng.shuffle(all_indices)

    test_indices = all_indices[remaining:]
    pool_indices = all_indices[:remaining]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = defaultdict(lambda: {'acc': [], 'f1': [], 'wf1': []})

    for ratio in label_ratios:
        n_labeled = max(1, int(ratio * remaining))
        n_val = max(1, int(0.2 * n_labeled))
        n_train = n_labeled - n_val

        print(f"\n{'='*70}")
        print(f"Label ratio: {ratio*100:.0f}% | Train: {n_train} | Val: {n_val} | Test: {len(test_indices)}")
        print(f"{'='*70}")

        for run in range(num_runs):
            run_seed = seed + run * 1000
            run_rng = np.random.default_rng(run_seed)
            perm = run_rng.permutation(len(pool_indices))
            train_idx = pool_indices[perm[:n_train]]
            val_idx = pool_indices[perm[n_train:n_train + n_val]]

            for pretrain_flag in [False, True]:
                tag = f"pretrain" if pretrain_flag else "scratch"
                pw = pretrain_weights if pretrain_flag else None

                print(f"\n--- {ratio*100:.0f}% | Run {run+1}/{num_runs} | {tag} ---")

                test_acc, test_f1, test_wf1 = run_single_fewshot(
                    zarr_path, pw, train_idx, val_idx, test_indices,
                    full_ds, num_classes, num_bands, epochs, batch_size,
                    d_model, nhead, layers, ff, lr_head, lr_enc, weight_decay,
                    amp, clip_norm, patience, warmup_epochs, use_dora, dora_rank,
                    lr_dora, device, tag, use_dual_branch=use_dual_branch
                )

                key = f"{ratio}_{tag}"
                results[key]['acc'].append(test_acc)
                results[key]['f1'].append(test_f1)
                results[key]['wf1'].append(test_wf1)

                print(f">>> [{tag}] Test OA={test_acc:.4f} F1={test_f1:.4f} WF1={test_wf1:.4f}")

    os.makedirs(output_dir, exist_ok=True)

    summary = {}
    print(f"\n{'='*80}")
    print(f"{'Label Ratio':<15} {'Method':<15} {'OA':>12} {'Macro F1':>14} {'Weighted F1':>14}")
    print(f"{'-'*80}")

    for ratio in label_ratios:
        for tag in ['scratch', 'pretrain']:
            key = f"{ratio}_{tag}"
            if key in results and results[key]['acc']:
                mean_acc = np.mean(results[key]['acc'])
                std_acc = np.std(results[key]['acc'])
                mean_f1 = np.mean(results[key]['f1'])
                std_f1 = np.std(results[key]['f1'])
                mean_wf1 = np.mean(results[key]['wf1'])
                std_wf1 = np.std(results[key]['wf1'])
                summary[key] = {
                    'ratio': ratio, 'method': tag,
                    'acc_mean': float(mean_acc), 'acc_std': float(std_acc),
                    'f1_mean': float(mean_f1), 'f1_std': float(std_f1),
                    'wf1_mean': float(mean_wf1), 'wf1_std': float(std_wf1),
                    'num_runs': len(results[key]['acc'])
                }
                label_str = f"{ratio*100:.0f}%"
                method_str = "w/ Pre-Train" if tag == 'pretrain' else "w/o Pre-Train"
                print(f"{label_str:<15} {method_str:<15} {mean_acc:.4f}+/-{std_acc:.4f} {mean_f1:.4f}+/-{std_f1:.4f} {mean_wf1:.4f}+/-{std_wf1:.4f}")

    print(f"{'-'*80}")
    print("Delta (Pre-Train advantage):")
    for ratio in label_ratios:
        s_key = f"{ratio}_scratch"
        p_key = f"{ratio}_pretrain"
        if s_key in summary and p_key in summary:
            delta_acc = summary[p_key]['acc_mean'] - summary[s_key]['acc_mean']
            delta_f1 = summary[p_key]['f1_mean'] - summary[s_key]['f1_mean']
            print(f"  {ratio*100:.0f}%: Delta_OA={delta_acc:+.4f} Delta_F1={delta_f1:+.4f}")

    with open(os.path.join(output_dir, 'fewshot_results.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'fewshot_results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label_ratio', 'method', 'acc_mean', 'acc_std', 'f1_mean', 'f1_std', 'wf1_mean', 'wf1_std'])
        for ratio in label_ratios:
            for tag in ['scratch', 'pretrain']:
                key = f"{ratio}_{tag}"
                if key in summary:
                    s = summary[key]
                    w.writerow([ratio, tag, f"{s['acc_mean']:.4f}", f"{s['acc_std']:.4f}",
                              f"{s['f1_mean']:.4f}", f"{s['f1_std']:.4f}",
                              f"{s['wf1_mean']:.4f}", f"{s['wf1_std']:.4f}"])

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ratios_pct = [r * 100 for r in label_ratios]
        scratch_acc = [summary[f"{r}_scratch"]['acc_mean'] for r in label_ratios if f"{r}_scratch" in summary]
        pretrain_acc = [summary[f"{r}_pretrain"]['acc_mean'] for r in label_ratios if f"{r}_pretrain" in summary]
        scratch_f1 = [summary[f"{r}_scratch"]['f1_mean'] for r in label_ratios if f"{r}_scratch" in summary]
        pretrain_f1 = [summary[f"{r}_pretrain"]['f1_mean'] for r in label_ratios if f"{r}_pretrain" in summary]

        scratch_acc_std = [summary[f"{r}_scratch"]['acc_std'] for r in label_ratios if f"{r}_scratch" in summary]
        pretrain_acc_std = [summary[f"{r}_pretrain"]['acc_std'] for r in label_ratios if f"{r}_pretrain" in summary]
        scratch_f1_std = [summary[f"{r}_scratch"]['f1_std'] for r in label_ratios if f"{r}_scratch" in summary]
        pretrain_f1_std = [summary[f"{r}_pretrain"]['f1_std'] for r in label_ratios if f"{r}_pretrain" in summary]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        x_vals = ratios_pct[:len(scratch_acc)]
        ax1.errorbar(x_vals, scratch_acc, yerr=scratch_acc_std, fmt='o--', color='#E74C3C', label='w/o Pre-Train', linewidth=2, markersize=8, capsize=4)
        ax1.errorbar(x_vals, pretrain_acc, yerr=pretrain_acc_std, fmt='s-', color='#2E86C1', label='w/ Pre-Train', linewidth=2, markersize=8, capsize=4)
        ax1.set_xlabel('Label Ratio (%)', fontsize=12)
        ax1.set_ylabel('Overall Accuracy', fontsize=12)
        ax1.set_title('Few-Shot Accuracy', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_xticks(ratios_pct)
        ax1.set_xticklabels([f'{r:.0f}%' for r in ratios_pct])

        x_vals2 = ratios_pct[:len(scratch_f1)]
        ax2.errorbar(x_vals2, scratch_f1, yerr=scratch_f1_std, fmt='o--', color='#E74C3C', label='w/o Pre-Train', linewidth=2, markersize=8, capsize=4)
        ax2.errorbar(x_vals2, pretrain_f1, yerr=pretrain_f1_std, fmt='s-', color='#2E86C1', label='w/ Pre-Train', linewidth=2, markersize=8, capsize=4)
        ax2.set_xlabel('Label Ratio (%)', fontsize=12)
        ax2.set_ylabel('Macro F1 Score', fontsize=12)
        ax2.set_title('Few-Shot Macro F1', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_xticks(ratios_pct)
        ax2.set_xticklabels([f'{r:.0f}%' for r in ratios_pct])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fewshot_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nFigure saved to {output_dir}/fewshot_curves.png")
    except ImportError:
        print("matplotlib not available, skipping figure generation")

    print(f"\nResults saved to {output_dir}/")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Few-Shot Evaluation for SAR-TS-BERT')
    parser.add_argument('--zarr_path', type=str, required=True)
    parser.add_argument('--pretrain_weights', type=str, default=None)
    parser.add_argument('--label_ratios', type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.5, 1.0])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--ff', type=int, default=512)
    parser.add_argument('--lr_head', type=float, default=1e-3)
    parser.add_argument('--lr_enc', type=float, default=1e-4)
    parser.add_argument('--lr_dora', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--use_dora', action='store_true', default=True)
    parser.add_argument('--dora_rank', type=int, default=32)
    parser.add_argument('--use_dual_branch', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='outputs/fewshot_eval')
    args = parser.parse_args()

    run_fewshot(
        zarr_path=args.zarr_path,
        pretrain_weights=args.pretrain_weights,
        label_ratios=args.label_ratios,
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        ff=args.ff,
        lr_head=args.lr_head,
        lr_enc=args.lr_enc,
        weight_decay=args.weight_decay,
        amp=args.amp,
        clip_norm=args.clip_norm,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        use_dora=args.use_dora,
        dora_rank=args.dora_rank,
        lr_dora=args.lr_dora,
        seed=args.seed,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
        use_dual_branch=args.use_dual_branch
    )
