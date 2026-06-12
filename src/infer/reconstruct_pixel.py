import argparse
import os
import numpy as np
import zarr
import torch
import matplotlib.pyplot as plt
from src.models.bert_sar_encoder import SarBertEncoder

def load_pixel(zarr_path,h=None,w=None,mask_ratio=0.15,noise_scale=0.1,seed=0):
    g=zarr.open(zarr_path,mode='r')
    data=g['observations']
    T,H,W,_=data.shape
    rng=np.random.default_rng(seed)
    if h is None:
        h=rng.integers(0,H)
    if w is None:
        w=rng.integers(0,W)
    x=np.asarray(data[:,h,w,:],dtype=np.float32)
    std=np.std(x,axis=0)+1e-6
    m=np.zeros((T,),dtype=np.bool_)
    k=max(1,int(T*mask_ratio))
    pos=rng.choice(T,size=k,replace=False)
    m[pos]=True
    s=rng.choice([-1.0,1.0],size=k)
    eps=noise_scale*std
    xn=x.copy()
    for i,pi in enumerate(pos):
        xn[pi]=x[pi]+s[i]*eps
    ti=np.arange(T,dtype=np.int64)
    return x,xn,ti,m,pos,(h,w)

def run(zarr_path,weights,output,h,w,d_model,nhead,layers,ff,mask_ratio,noise_scale,seed):
    clean,noisy,ti,mask,pos,(h,w)=load_pixel(zarr_path,h,w,mask_ratio,noise_scale,seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SarBertEncoder(d_model=d_model,nhead=nhead,num_layers=layers,dim_feedforward=ff).to(device)
    sd=torch.load(weights,map_location=device)
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        pred=model(torch.from_numpy(noisy).unsqueeze(0).to(device),torch.from_numpy(ti).unsqueeze(0).to(device),torch.from_numpy(mask).unsqueeze(0).to(device))
    target=torch.from_numpy(clean[pos]).to(device)
    mse=torch.mean((pred-target)**2,dim=0).cpu().numpy()
    T=noisy.shape[0]
    yh=noisy.copy()
    yh[pos]=pred.cpu().numpy()
    t=np.arange(T)
    fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True)
    names=['vv','vh','vv/vh']
    for b in range(3):
        ax=axs[b]
        ax.plot(t,clean[:,b],color='C0',label='clean')
        ax.plot(t,noisy[:,b],color='0.7',label='noisy')
        ax.scatter(pos,clean[pos,b],color='C0')
        ax.scatter(pos,yh[pos,b],color='C3',label='pred')
        ax.set_ylabel(names[b])
        ax.legend(loc='upper right')
    axs[-1].set_xlabel('time index')
    fig.suptitle(f'h={h}, w={w}, MSE vv={mse[0]:.4f}, vh={mse[1]:.4f}, ratio={mse[2]:.4f}')
    os.makedirs(os.path.dirname(output),exist_ok=True)
    fig.savefig(output,dpi=150)
    return output,mse,(h,w)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zarr_path',type=str,required=True)
    p.add_argument('--weights',type=str,required=True)
    p.add_argument('--output',type=str,default='outputs/reconstruct.png')
    p.add_argument('--h',type=int,default=4002)
    p.add_argument('--w',type=int,default=5050)
    p.add_argument('--d_model',type=int,default=256)
    p.add_argument('--nhead',type=int,default=8)
    p.add_argument('--layers',type=int,default=6)
    p.add_argument('--ff',type=int,default=512)
    p.add_argument('--mask_ratio',type=float,default=0.1)
    p.add_argument('--noise_scale',type=float,default=0.1)
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    run(a.zarr_path,a.weights,a.output,a.h,a.w,a.d_model,a.nhead,a.layers,a.ff,a.mask_ratio,a.noise_scale,a.seed)

if __name__=='__main__':
    main()

