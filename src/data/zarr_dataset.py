import numpy as np
import zarr
import torch
from torch.utils.data import Dataset

class TemporalPixelDataset(Dataset):
    def __init__(self, zarr_path, samples_per_epoch, mask_ratio=0.15, noise_scale=0.1, seed=42, split='train', train_ratio=0.9, span_masking=False, span_len=3, physics_guided=False, span_physics=False, band_indices=None):
        self.store=zarr.open(zarr_path,mode='r')
        self.data=self.store['observations']
        self.t=self.store['timestamps']
        self.band_indices=band_indices
        self.T=self.data.shape[0]
        self.H=self.data.shape[1]
        self.W=self.data.shape[2]
        self.mask_ratio=mask_ratio
        self.noise_scale=noise_scale
        self.span_masking=span_masking
        self.span_len=span_len
        self.physics_guided=physics_guided
        self.span_physics=span_physics
        self.rng=np.random.default_rng(seed)
        pixels=[(h,w) for h in range(self.H) for w in range(self.W)]
        self.rng.shuffle(pixels)
        n_train=int(len(pixels)*train_ratio)
        self.pool=pixels[:n_train] if split=='train' else pixels[n_train:]
        self.samples=samples_per_epoch
    def __len__(self):
        return self.samples
    def __getitem__(self, idx):
        h,w=self.pool[self.rng.integers(0,len(self.pool))]
        x=np.asarray(self.data[:,h,w,:],dtype=np.float32)
        if self.band_indices is not None:
            x=x[:,self.band_indices]
        std=np.std(x,axis=0)+1e-6
        m=np.zeros((self.T,),dtype=np.bool_)
        if self.span_physics:
            m=self._compute_span_physics_mask(x)
        elif self.physics_guided:
            m=self._compute_physics_mask(x)
        elif self.span_masking:
            L=max(1,min(self.span_len,self.T))
            s=int(self.rng.integers(0,self.T-L+1))
            pos=np.arange(s,s+L,dtype=np.int64)
            m[pos]=True
        else:
            k=max(1,int(self.T*self.mask_ratio))
            pos=self.rng.choice(self.T,size=k,replace=False)
            m[pos]=True
        xn=x.copy()
        eps=self.noise_scale*std
        bands = x.shape[1] if x.ndim == 2 else 1
        masked_indices = torch.where(torch.from_numpy(m))[0]
        noise=self.rng.normal(0.0, eps, size=(len(masked_indices), bands))
        for i,pi in enumerate(masked_indices):
            xn[pi]=x[pi]+noise[i]
        ti=np.arange(self.T,dtype=np.int64)
        return {
            'noisy':torch.from_numpy(xn),
            'clean':torch.from_numpy(x),
            'time_idx':torch.from_numpy(ti),
            'mask':torch.from_numpy(m),
        }
    def _compute_physics_mask(self, x):
        """
        物理引导掩蔽：基于SAR后向散射时序变化率计算掩蔽概率
        变化率越大的时刻，掩蔽概率越高
        """
        m=np.zeros((self.T,),dtype=np.bool_)
        if x.shape[1] >= 2:
            diff_vv = np.abs(x[1:, 0] - x[:-1, 0])
            diff_vh = np.abs(x[1:, 1] - x[:-1, 1])
            change_rate = (diff_vv + diff_vh) / 2
            rate_min = change_rate.min()
            rate_max = change_rate.max()
            if rate_max > rate_min:
                mask_prob = (change_rate - rate_min) / (rate_max - rate_min + 1e-6)
            else:
                mask_prob = np.ones_like(change_rate) * 0.5
            rand_vals = self.rng.random(size=len(mask_prob))
            m[1:] = rand_vals < mask_prob * self.mask_ratio * 2
            k_actual = m.sum()
            k_target = max(1, int(self.T * self.mask_ratio))
            if k_actual < k_target:
                n_add = k_target - k_actual
                unmasked = np.where(~m[1:])[0] + 1
                if len(unmasked) > n_add:
                    add_idx = self.rng.choice(unmasked, size=min(n_add, len(unmasked)), replace=False)
                    m[add_idx] = True
            elif k_actual > k_target * 2:
                n_remove = k_actual - k_target
                masked = np.where(m[1:])[0] + 1
                if len(masked) > n_remove:
                    remove_idx = self.rng.choice(masked, size=min(n_remove, len(masked)), replace=False)
                    m[remove_idx] = False
        else:
            k=max(1,int(self.T*self.mask_ratio))
            pos=self.rng.choice(self.T,size=k,replace=False)
            m[pos]=True
        return m

    def _compute_span_physics_mask(self, x):
        m=np.zeros((self.T,),dtype=np.bool_)
        L=max(1,min(self.span_len,self.T))
        k_target=max(1,int(self.T*self.mask_ratio))
        n_spans=max(1,k_target//L)
        if x.shape[1]>=2:
            diff_vv=np.abs(x[1:,0]-x[:-1,0])
            diff_vh=np.abs(x[1:,1]-x[:-1,1])
            change_rate=(diff_vv+diff_vh)/2
            rate_min=change_rate.min()
            rate_max=change_rate.max()
            if rate_max>rate_min:
                center_prob=(change_rate-rate_min)/(rate_max-rate_min+1e-6)
            else:
                center_prob=np.ones_like(change_rate)*0.5
            span_prob=np.zeros(self.T)
            for t in range(1,self.T):
                half=L//2
                lo=max(0,t-half)
                hi=min(self.T,t+L-half)
                span_prob[lo:hi]+=center_prob[t-1]
            span_prob=span_prob/span_prob.max()+1e-6
            span_starts=np.arange(0,max(1,self.T-L+1))
            start_prob=span_prob[span_starts]
            start_prob=start_prob/start_prob.sum()
            chosen=self.rng.choice(span_starts,size=min(n_spans,len(span_starts)),replace=False,p=start_prob)
            for s in chosen:
                pos=np.arange(s,min(s+L,self.T))
                m[pos]=True
            k_actual=m.sum()
            if k_actual<k_target:
                unmasked=np.where(~m)[0]
                if len(unmasked)>0:
                    add_idx=self.rng.choice(unmasked,size=min(k_target-k_actual,len(unmasked)),replace=False)
                    m[add_idx]=True
            elif k_actual>k_target*2:
                masked=np.where(m)[0]
                n_remove=k_actual-k_target
                if len(masked)>n_remove:
                    remove_idx=self.rng.choice(masked,size=n_remove,replace=False)
                    m[remove_idx]=False
        else:
            for _ in range(n_spans):
                s=int(self.rng.integers(0,max(1,self.T-L+1)))
                pos=np.arange(s,min(s+L,self.T))
                m[pos]=True
        return m
