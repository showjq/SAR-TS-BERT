import argparse
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from src.data.parcel_dataset import ParcelDataset
from src.models.parcel_classifier import ParcelClassifier

def finetune(zarr_path, vector_path, class_field, epochs, batch_size, d_model, nhead, layers, ff, lr_head, lr_enc, weight_decay, seed, train_ratio, val_ratio, amp, clip_norm, patience, save_best, save_last, log_path, warmup_epochs, pretrain_weights=None, label_json=None, train_samples=10000, val_samples=2000):
    train_ds=ParcelDataset(zarr_path, vector_path, class_field, split='train', train_ratio=train_ratio, val_ratio=val_ratio, seed=seed, samples_per_epoch=train_samples, label_json_path=label_json)
    val_ds=ParcelDataset(zarr_path, vector_path, class_field, split='val', train_ratio=train_ratio, val_ratio=val_ratio, seed=seed, samples_per_epoch=val_samples, label_json_path=label_json)
    num_classes=len(train_ds.class_names)
    train_dl=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0)
    val_dl=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=0)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ParcelClassifier(num_classes,d_model=d_model,nhead=nhead,num_layers=layers,dim_feedforward=ff).to(device)
    if pretrain_weights:
        sd=torch.load(pretrain_weights,map_location=device)
        model.encoder.load_state_dict(sd,strict=False)
    enc_params=list(model.encoder.parameters())
    head_params=list(model.clf_head.parameters())
    for p in enc_params:
        p.requires_grad=True
    opt=torch.optim.AdamW([
        {'params':enc_params,'lr':lr_enc,'weight_decay':weight_decay},
        {'params':head_params,'lr':lr_head,'weight_decay':weight_decay},
    ])
    scaler=GradScaler(enabled=amp and torch.cuda.is_available())
    best=float('inf')
    wait=0
    os.makedirs(os.path.dirname(save_best),exist_ok=True)
    os.makedirs(os.path.dirname(save_last),exist_ok=True)
    os.makedirs(os.path.dirname(log_path),exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path,'w',newline='') as f:
            w=csv.writer(f)
            w.writerow(['epoch','train_loss','val_loss','acc','macro_f1','lr_head','lr_enc','best'])
        # 保存类别名称映射
        with open(os.path.join(os.path.dirname(log_path),'class_names.json'),'w',encoding='utf-8') as f:
            import json
            json.dump({'class_names':train_ds.class_names},f,ensure_ascii=False,indent=2)
    for epoch in range(epochs):
        model.train()
        # warmup: 冻结编码器
        freeze = (epoch < warmup_epochs)
        for p in enc_params:
            p.requires_grad=not freeze
        train_se=0.0
        train_cnt=0
        pbar=tqdm(train_dl,desc=f"epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x=batch['series'].to(device)
            t=batch['time_idx'].to(device)
            y=batch['label'].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp and torch.cuda.is_available()):
                logits=model(x.unsqueeze(0) if x.dim()==2 else x, t.unsqueeze(0) if t.dim()==1 else t)
                loss=F.cross_entropy(logits,y)
            scaler.scale(loss).backward()
            if clip_norm>0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip_norm)
            scaler.step(opt)
            scaler.update()
            train_se+=loss.item()*y.shape[0]
            train_cnt+=y.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        train_loss=train_se/max(1,train_cnt)
        # validation
        model.eval()
        val_se=0.0
        val_cnt=0
        all_y=[]
        all_p=[]
        with torch.no_grad():
            for batch in val_dl:
                x=batch['series'].to(device)
                t=batch['time_idx'].to(device)
                y=batch['label'].to(device)
                logits=model(x.unsqueeze(0) if x.dim()==2 else x, t.unsqueeze(0) if t.dim()==1 else t)
                se=F.cross_entropy(logits,y,reduction='sum').item()
                val_se+=se
                val_cnt+=y.shape[0]
                all_y.append(y.cpu().numpy())
                all_p.append(logits.argmax(dim=1).cpu().numpy())
        val_loss=val_se/max(1,val_cnt)
        all_y=np.concatenate(all_y) if all_y else np.array([])
        all_p=np.concatenate(all_p) if all_p else np.array([])
        acc=accuracy_score(all_y,all_p) if len(all_y)>0 else 0.0
        macro_f1=f1_score(all_y,all_p,average='macro') if len(all_y)>0 else 0.0
        lr_head=opt.param_groups[1]['lr']
        lr_enc=opt.param_groups[0]['lr']
        print(f"epoch {epoch+1}/{epochs} - train_loss={train_loss:.6f} - val_loss={val_loss:.6f} - acc={acc:.4f} - f1={macro_f1:.4f}")
        with open(log_path,'a',newline='') as f:
            w=csv.writer(f)
            w.writerow([epoch+1,f"{train_loss:.6f}",f"{val_loss:.6f}",f"{acc:.4f}",f"{macro_f1:.4f}",f"{lr_head:.6e}",f"{lr_enc:.6e}",int(val_loss<best)])
        if val_loss<best:
            best=val_loss
            torch.save(model.state_dict(),save_best)
            wait=0
        else:
            wait+=1
            if wait>=patience:
                break
    torch.save(model.state_dict(),save_last)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zarr_path',type=str,required=True)
    p.add_argument('--vector_path',type=str,required=True)
    p.add_argument('--class_field',type=str,required=True)
    p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--batch_size',type=int,default=128)
    p.add_argument('--d_model',type=int,default=256)
    p.add_argument('--nhead',type=int,default=8)
    p.add_argument('--layers',type=int,default=6)
    p.add_argument('--ff',type=int,default=512)
    p.add_argument('--lr_head',type=float,default=1e-3)
    p.add_argument('--lr_enc',type=float,default=5e-5)
    p.add_argument('--weight_decay',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train_ratio',type=float,default=0.8)
    p.add_argument('--val_ratio',type=float,default=0.1)
    p.add_argument('--amp',action='store_true')
    p.add_argument('--clip_norm',type=float,default=1.0)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--log_path',type=str,default='outputs/cls_train_log.csv')
    p.add_argument('--save_best',type=str,default='outputs/parcel_cls_best.pt')
    p.add_argument('--save_last',type=str,default='outputs/parcel_cls_last.pt')
    p.add_argument('--warmup_epochs',type=int,default=5)
    p.add_argument('--pretrain_weights',type=str,default=None)
    p.add_argument('--label_json',type=str,default=None)
    p.add_argument('--train_samples',type=int,default=10000)
    p.add_argument('--val_samples',type=int,default=2000)
    a=p.parse_args()
    finetune(a.zarr_path,a.vector_path,a.class_field,a.epochs,a.batch_size,a.d_model,a.nhead,a.layers,a.ff,a.lr_head,a.lr_enc,a.weight_decay,a.seed,a.train_ratio,a.val_ratio,a.amp,a.clip_norm,a.patience,a.save_best,a.save_last,a.log_path,a.warmup_epochs,a.pretrain_weights,a.label_json,a.train_samples,a.val_samples)

if __name__=='__main__':
    main()