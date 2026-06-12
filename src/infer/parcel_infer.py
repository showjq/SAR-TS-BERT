import argparse
import os
import json
import numpy as np
import zarr
import torch
import fiona
from rasterio import features
from rasterio.warp import transform_geom
from rasterio.transform import rowcol
from affine import Affine
from shapely.geometry import shape
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from src.models.parcel_classifier import ParcelClassifier

def load_pixels(store, geom, max_pixels, rng):
    H=store['observations'].shape[1]
    W=store['observations'].shape[2]
    transform=tuple(store.attrs.get('transform'))
    aff=Affine(*transform)
    bx,by,bx2,by2=shape(geom).bounds
    r0,c0=rowcol(aff,bx,by2)
    r1,c1=rowcol(aff,bx2,by)
    r0=max(0,min(H-1,r0)); r1=max(0,min(H-1,r1))
    c0=max(0,min(W-1,c0)); c1=max(0,min(W-1,c1))
    if r1<r0 or c1<c0:
        return None
    h=r1-r0+1; w=c1-c0+1
    sub_aff=aff*Affine.translation(c0,r0)
    mask=features.geometry_mask([geom], out_shape=(h,w), transform=sub_aff, invert=True)
    ys,xs=np.nonzero(mask)
    if len(ys)==0:
        return None
    idx=np.arange(len(ys))
    rng.shuffle(idx)
    if max_pixels and len(idx)>max_pixels:
        idx=idx[:max_pixels]
    ys=ys[idx]+r0; xs=xs[idx]+c0
    seqs=[]
    for y,x in zip(ys,xs):
        seq=np.asarray(store['observations'][:,int(y),int(x),:],dtype=np.float32)  # T,3
        seqs.append(seq)
    series=np.stack(seqs,axis=0)  # N,T,3
    t=np.arange(store['observations'].shape[0],dtype=np.int64)
    return series,t

def infer(zarr_path, vector_path, label_json, weights, out_path, d_model, nhead, layers, ff, max_pixels, seed, limit_parcels=None, batch_size_infer=256, label_field='label', metrics_dir='outputs', use_dora=False, dora_rank=32, num_bands=3, use_sar_norm=True):
    rng=np.random.default_rng(seed)
    store=zarr.open(zarr_path,mode='r')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # classes
    names=None
    if label_json and os.path.exists(label_json):
        with open(label_json,'r',encoding='utf-8') as f:
            info=json.load(f)
            names=info.get('label_names',None)
    if names is None:
        # 尝试从 zarr attrs 读取
        z_names=store.attrs.get('class_names', None)
        if isinstance(z_names, (list, tuple)) and len(z_names)>0:
            names=list(z_names)
    # model
    num_classes=len(names) if names else None
    if num_classes is None:
        num_classes=2
    model=ParcelClassifier(num_classes,d_model=d_model,nhead=nhead,num_layers=layers,dim_feedforward=ff,use_dora=use_dora,dora_rank=dora_rank,num_bands=num_bands,use_sar_norm=use_sar_norm).to(device)
    try:
        sd=torch.load(weights,map_location=device,weights_only=True)
    except TypeError:
        sd=torch.load(weights,map_location=device)
    model.load_state_dict(sd,strict=False)
    model.eval()
    # io
    pixel_true=[]
    pixel_pred=[]
    parcel_true=[]
    parcel_pred=[]
    with fiona.open(vector_path,'r') as src:
        vcrs=src.crs_wkt or src.crs
        schema=dict(src.schema)
        schema['properties']=dict(schema['properties'])
        schema['properties']['pred']= 'int:10'
        schema['properties']['pred_name']= 'str:64'
        os.makedirs(os.path.dirname(out_path),exist_ok=True)
        with fiona.open(out_path,'w',driver=src.driver,crs=src.crs,schema=schema) as dst:
            it=src
            total=len(src) if hasattr(src,'__len__') else None
            if limit_parcels:
                total=min(total or limit_parcels, limit_parcels)
            pbar=tqdm(it,total=total,desc='parcels')
            count=0
            for feat in pbar:
                if limit_parcels and count>=limit_parcels:
                    break
                geom=feat['geometry']
                if vcrs and store.attrs.get('crs') and vcrs!=store.attrs.get('crs'):
                    geom=transform_geom(vcrs, store.attrs.get('crs'), geom)
                data=load_pixels(store, geom, max_pixels, rng)
                if data is None:
                    pred=0
                    pred_name= (names[pred] if names and pred < len(names) else str(pred))
                    parcel_true.append(int(feat['properties'].get(label_field,0)))
                    parcel_pred.append(pred)
                else:
                    series,t=data
                    preds=[]
                    with torch.no_grad():
                        for s in range(0,series.shape[0],batch_size_infer):
                            xb=torch.from_numpy(series[s:s+batch_size_infer]).to(device)
                            tib=torch.from_numpy(np.tile(t,(xb.shape[0],1))).to(device)
                            lb=model(xb,tib)
                            preds.append(lb)
                    logits=torch.cat(preds,dim=0)
                    pcls=logits.argmax(dim=1).cpu().numpy()
                    pixel_true.extend([int(feat['properties'].get(label_field,0))]*len(pcls))
                    pixel_pred.extend(list(pcls))
                    vote=np.bincount(pcls).argmax()
                    pred=int(vote)
                    parcel_true.append(int(feat['properties'].get(label_field,0)))
                    parcel_pred.append(pred)
                    pred_name= (names[pred] if names and pred < len(names) else str(pred))
                props=dict(feat['properties'])
                props['pred']=pred
                props['pred_name']=pred_name
                dst.write({'geometry':feat['geometry'],'properties':props})
                count+=1
    C=None
    if names:
        C=len(names)
    else:
        C=max(max(pixel_true or [0]),max(pixel_pred or [0]),max(parcel_true or [0]),max(parcel_pred or [0]))+1
    os.makedirs(metrics_dir,exist_ok=True)
    if pixel_true:
        pcm=confusion_matrix(pixel_true,pixel_pred,labels=list(range(C)))
        pr=classification_report(pixel_true,pixel_pred,output_dict=True,labels=list(range(C)))
        pacc=accuracy_score(pixel_true,pixel_pred)
        with open(os.path.join(metrics_dir,'pixel_class_report.csv'),'w',encoding='utf-8') as f:
            f.write('class,precision,recall,f1-score,support,name\n')
            for i in range(C):
                k=str(i)
                if k in pr:
                    r=pr[k]
                    name=names[i] if names else str(i)
                    f.write(f"{i},{r['precision']:.6f},{r['recall']:.6f},{r['f1-score']:.6f},{r['support']},{name}\n")
            f.write(f"overall_accuracy,{pacc:.6f}\n")
        with open(os.path.join(metrics_dir,'pixel_confusion.csv'),'w',encoding='utf-8') as f:
            for row in pcm:
                f.write(','.join(map(str,row))+'\n')
        plt.figure(figsize=(6,5))
        plt.imshow(pcm,cmap='Blues')
        plt.title('Pixel Confusion')
        plt.colorbar()
        plt.xticks(range(C), names if names else [str(i) for i in range(C)], rotation=45, ha='right')
        plt.yticks(range(C), names if names else [str(i) for i in range(C)])
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir,'pixel_confusion.png'),dpi=150)
        plt.close()
    if parcel_true:
        rcm=confusion_matrix(parcel_true,parcel_pred,labels=list(range(C)))
        rr=classification_report(parcel_true,parcel_pred,output_dict=True,labels=list(range(C)))
        racc=accuracy_score(parcel_true,parcel_pred)
        with open(os.path.join(metrics_dir,'parcel_class_report.csv'),'w',encoding='utf-8') as f:
            f.write('class,precision,recall,f1-score,support,name\n')
            for i in range(C):
                k=str(i)
                if k in rr:
                    r=rr[k]
                    name=names[i] if names else str(i)
                    f.write(f"{i},{r['precision']:.6f},{r['recall']:.6f},{r['f1-score']:.6f},{r['support']},{name}\n")
            f.write(f"overall_accuracy,{racc:.6f}\n")
        with open(os.path.join(metrics_dir,'parcel_confusion.csv'),'w',encoding='utf-8') as f:
            for row in rcm:
                f.write(','.join(map(str,row))+'\n')
        plt.figure(figsize=(6,5))
        plt.imshow(rcm,cmap='Greens')
        plt.title('Parcel Confusion')
        plt.colorbar()
        plt.xticks(range(C), names if names else [str(i) for i in range(C)], rotation=45, ha='right')
        plt.yticks(range(C), names if names else [str(i) for i in range(C)])
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir,'parcel_confusion.png'),dpi=150)
        plt.close()

def infer_processed(processed_zarr_path, weights, d_model, nhead, layers, ff, batch_size_infer=256, metrics_dir='outputs', vector_path=None, out_path=None, use_dora=False, dora_rank=32, num_bands=3, use_sar_norm=True, num_classes_override=None):
    store=zarr.open(processed_zarr_path,mode='r')
    series=store['time_series']
    labels=store['labels']
    pids=store['parcel_ids']
    names=store.attrs.get('class_names',None)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if num_classes_override is not None:
        num_classes=num_classes_override
    else:
        num_classes=len(names) if names else int(np.max(np.asarray(labels)))+1
    model=ParcelClassifier(num_classes,d_model=d_model,nhead=nhead,num_layers=layers,dim_feedforward=ff,use_dora=use_dora,dora_rank=dora_rank,num_bands=num_bands,use_sar_norm=use_sar_norm).to(device)
    sd=torch.load(weights,map_location=device)
    model.load_state_dict(sd,strict=False)
    model.eval()
    N=series.shape[0]
    T=series.shape[1]
    print(f"processed samples={N}, T={T}, classes={num_classes}")
    preds=[]
    with torch.no_grad():
        pbar=tqdm(range(0,N,batch_size_infer),desc='infer-batches')
        for s in pbar:
            xb=torch.from_numpy(np.asarray(series[s:s+batch_size_infer],dtype=np.float32)).to(device)
            t=np.arange(T,dtype=np.int64)
            tib=torch.from_numpy(np.tile(t,(xb.shape[0],1))).to(device)
            lb=model(xb,tib)
            preds.append(lb)
    logits=torch.cat(preds,dim=0)
    pixel_pred=list(logits.argmax(dim=1).cpu().numpy())
    pixel_true=list(np.asarray(labels,dtype=np.int32))
    C=num_classes
    os.makedirs(metrics_dir,exist_ok=True)
    pcm=confusion_matrix(pixel_true,pixel_pred,labels=list(range(C)))
    pr=classification_report(pixel_true,pixel_pred,output_dict=True,labels=list(range(C)))
    pacc=accuracy_score(pixel_true,pixel_pred)
    with open(os.path.join(metrics_dir,'pixel_class_report.csv'),'w',encoding='utf-8') as f:
        f.write('class,precision,recall,f1-score,support,name\n')
        for i in range(C):
            k=str(i)
            if k in pr:
                r=pr[k]
                name=names[i] if names else str(i)
                f.write(f"{i},{r['precision']:.6f},{r['recall']:.6f},{r['f1-score']:.6f},{r['support']},{name}\n")
        f.write(f"overall_accuracy,{pacc:.6f}\n")
    with open(os.path.join(metrics_dir,'pixel_confusion.csv'),'w',encoding='utf-8') as f:
        for row in pcm:
            f.write(','.join(map(str,row))+'\n')
    plt.figure(figsize=(6,5))
    plt.imshow(pcm,cmap='Blues')
    plt.title('Pixel Confusion')
    plt.colorbar()
    plt.xticks(range(C), names if names else [str(i) for i in range(C)], rotation=45, ha='right')
    plt.yticks(range(C), names if names else [str(i) for i in range(C)])
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir,'pixel_confusion.png'),dpi=150)
    plt.close()
    # prepare arrays for parcel-level aggregation
    pid_arr=np.asarray(pids[:],dtype=np.int32) if hasattr(pids,'__getitem__') else np.asarray(pids,dtype=np.int32)
    lbl_arr=np.asarray(labels[:],dtype=np.int32) if hasattr(labels,'__getitem__') else np.asarray(labels,dtype=np.int32)
    pred_arr=np.asarray(pixel_pred,dtype=np.int32)
    # optional shapefile export if vector_path provided
    if vector_path and out_path:
        sel_fids=None
        if 'selected_feature_indices' in store:
            sel_fids=np.asarray(store['selected_feature_indices'][:],dtype=np.int32)
        else:
            print('warning: selected_feature_indices not found in processed zarr; shapefile export will set pred=-1 for non-selected features. Rebuild processed.zarr with fast builder to enable alignment.')
        os.makedirs(os.path.dirname(out_path),exist_ok=True)
        with fiona.open(vector_path,'r') as src:
            schema=dict(src.schema)
            schema['properties']=dict(schema['properties'])
            schema['properties']['pred']='int:10'
            schema['properties']['pred_name']='str:64'
            with fiona.open(out_path,'w',driver=src.driver,crs=src.crs,schema=schema) as dst:
                pbar=tqdm(enumerate(src), total=len(src) if hasattr(src,'__len__') else None, desc='export-shp')
                for fi, feat in pbar:
                    props=dict(feat['properties'])
                    pred=-1
                    name=str(pred)
                    idx=np.where(pid_arr==fi)[0]
                    if idx.size>0:
                        votes=np.bincount(pred_arr[idx],minlength=C)
                        pred=int(votes.argmax())
                        name=names[pred] if names else str(pred)
                    props['pred']=pred
                    props['pred_name']=name
                    dst.write({'geometry':feat['geometry'],'properties':props})
    pid_arr=np.asarray(pids,dtype=np.int32)
    lbl_arr=np.asarray(labels,dtype=np.int32)
    pred_arr=np.asarray(pixel_pred,dtype=np.int32)
    parcel_true=[]
    parcel_pred=[]
    uniq_pids=np.unique(pid_arr)
    for pid in uniq_pids:
        idx=np.where(pid_arr==pid)[0]
        if idx.size==0:
            continue
        yt=int(lbl_arr[idx[0]])
        votes=np.bincount(pred_arr[idx],minlength=C)
        yp=int(votes.argmax())
        parcel_true.append(yt)
        parcel_pred.append(yp)
    rcm=confusion_matrix(parcel_true,parcel_pred,labels=list(range(C)))
    rr=classification_report(parcel_true,parcel_pred,output_dict=True,labels=list(range(C)))
    racc=accuracy_score(parcel_true,parcel_pred)
    with open(os.path.join(metrics_dir,'parcel_class_report.csv'),'w',encoding='utf-8') as f:
        f.write('class,precision,recall,f1-score,support,name\n')
        for i in range(C):
            k=str(i)
            if k in rr:
                r=rr[k]
                name=names[i] if names else str(i)
                f.write(f"{i},{r['precision']:.6f},{r['recall']:.6f},{r['f1-score']:.6f},{r['support']},{name}\n")
        f.write(f"overall_accuracy,{racc:.6f}\n")
    with open(os.path.join(metrics_dir,'parcel_confusion.csv'),'w',encoding='utf-8') as f:
        for row in rcm:
            f.write(','.join(map(str,row))+'\n')
    plt.figure(figsize=(6,5))
    plt.imshow(rcm,cmap='Greens')
    plt.title('Parcel Confusion')
    plt.colorbar()
    plt.xticks(range(C), names if names else [str(i) for i in range(C)], rotation=45, ha='right')
    plt.yticks(range(C), names if names else [str(i) for i in range(C)])
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir,'parcel_confusion.png'),dpi=150)
    plt.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--processed_zarr',type=str,default=None)
    p.add_argument('--zarr_path',type=str,default=None)
    p.add_argument('--vector_path',type=str,default=None)
    p.add_argument('--label_json',type=str,default=None)
    p.add_argument('--weights',type=str,required=True)
    p.add_argument('--out_path',type=str,default='outputs/predicted_parcels.shp')
    p.add_argument('--d_model',type=int,default=256)
    p.add_argument('--nhead',type=int,default=8)
    p.add_argument('--layers',type=int,default=6)
    p.add_argument('--ff',type=int,default=512)
    p.add_argument('--max_pixels',type=int,default=200)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--limit_parcels',type=int,default=None)
    p.add_argument('--batch_size_infer',type=int,default=256)
    p.add_argument('--label_field',type=str,default='label')
    p.add_argument('--metrics_dir',type=str,default='outputs')
    p.add_argument('--use_dora',action='store_true')
    p.add_argument('--dora_rank',type=int,default=32)
    p.add_argument('--num_bands',type=int,default=3)
    p.add_argument('--use_sar_norm',action='store_true')
    p.add_argument('--num_classes',type=int,default=None)
    a=p.parse_args()
    if a.processed_zarr:
        infer_processed(a.processed_zarr,a.weights,a.d_model,a.nhead,a.layers,a.ff,a.batch_size_infer,a.metrics_dir,a.vector_path,a.out_path,a.use_dora,a.dora_rank,a.num_bands,a.use_sar_norm,a.num_classes)
    else:
        infer(a.zarr_path,a.vector_path,a.label_json,a.weights,a.out_path,a.d_model,a.nhead,a.layers,a.ff,a.max_pixels,a.seed,a.limit_parcels,a.batch_size_infer,a.label_field,a.metrics_dir,a.use_dora,a.dora_rank,a.num_bands,a.use_sar_norm)

if __name__=='__main__':
    main()
