import math
import torch
import torch.nn as nn
from .dora import DoRAEncoderLayer, DualBranchDoRAEncoderLayer


class SARAdaptiveNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.speckle_scale = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps) * self.speckle_scale
        x_norm = (x - mean) / std
        return x_norm * self.gamma + self.beta


def make_norm(norm_type, num_features):
    if norm_type == 'sar_adaptive':
        return SARAdaptiveNorm(num_features)
    elif norm_type == 'batch':
        return nn.LayerNorm(num_features)
    elif norm_type == 'layer':
        return nn.LayerNorm(num_features)
    else:
        return None


class NormAblationEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, norm_type='layer'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        if norm_type == 'sar_adaptive':
            self.norm1 = SARAdaptiveNorm(d_model)
            self.norm2 = SARAdaptiveNorm(d_model)
        elif norm_type == 'batch':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

def positional_encoding(indices,dim):
    device=indices.device
    t=indices.float().unsqueeze(-1)
    div_term=torch.exp(torch.arange(0,dim,2,device=device).float()*(-math.log(10000.0)/dim))
    pe=torch.zeros(indices.shape[0],indices.shape[1],dim,device=device)
    pe[:,:,0::2]=torch.sin(t*div_term)
    pe[:,:,1::2]=torch.cos(t*div_term)
    return pe


class CrossModalFusion(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.branch_dim = d_model // 2

        self.cross_attn_s2p = nn.MultiheadAttention(
            self.branch_dim, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_p2s = nn.MultiheadAttention(
            self.branch_dim, nhead, dropout=dropout, batch_first=True
        )

        self.fusion = nn.Linear(self.branch_dim * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_s, h_p):
        attn_s, _ = self.cross_attn_s2p(h_s, h_p, h_p)
        attn_p, _ = self.cross_attn_p2s(h_p, h_s, h_s)

        fused = torch.cat([h_s + h_p, attn_s, attn_p], dim=-1)
        fused = self.fusion(fused)
        return self.norm(fused)


class SarBertEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1, num_bands=3, use_dora=False, dora_rank=32, norm_type=None):
        super().__init__()
        self.d_model=d_model
        self.num_bands=num_bands
        self.use_dora=use_dora
        self.obs_proj=nn.Linear(num_bands,d_model//2)
        
        if use_dora:
            encoder_layer=DoRAEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout,batch_first=True,dora_rank=dora_rank)
            self.encoder=nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        elif norm_type is not None:
            encoder_layer=NormAblationEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout,norm_type=norm_type)
            self.encoder=nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        else:
            encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout,batch_first=True)
            self.encoder=nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        
        self.head=nn.Linear(d_model,num_bands)
    
    def forward(self, noisy, time_idx, mask):
        pe=positional_encoding(time_idx,self.d_model//2)
        x_obs=self.obs_proj(noisy)
        x=torch.cat([x_obs,pe],dim=-1)
        h=self.encoder(x)
        y=self.head(h)
        b=torch.where(mask)
        pred=y[b[0],b[1]]
        return pred


class DualBranchSarBertEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=512,
                 dropout=0.1, num_bands=6, use_dora=False, dora_rank=32,
                 scatter_layers=2, physics_layers=2):
        super().__init__()
        self.d_model = d_model
        self.num_bands = num_bands
        self.use_dora = use_dora

        self.scatter_proj = nn.Linear(3, d_model // 4)
        self.physics_proj = nn.Linear(3, d_model // 4)

        self.pos_dim = d_model // 4

        if use_dora:
            scatter_layer = DualBranchDoRAEncoderLayer(
                d_model=d_model // 2, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, dora_rank=dora_rank, branch_name='scatter'
            )
            physics_layer = DualBranchDoRAEncoderLayer(
                d_model=d_model // 2, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, dora_rank=dora_rank, branch_name='physics'
            )
        else:
            scatter_layer = nn.TransformerEncoderLayer(
                d_model=d_model // 2, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True
            )
            physics_layer = nn.TransformerEncoderLayer(
                d_model=d_model // 2, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True
            )

        self.scatter_encoder = nn.TransformerEncoder(scatter_layer, num_layers=scatter_layers)
        self.physics_encoder = nn.TransformerEncoder(physics_layer, num_layers=physics_layers)

        remaining_layers = num_layers - scatter_layers - physics_layers
        if remaining_layers > 0:
            if use_dora:
                fusion_layer = DualBranchDoRAEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, batch_first=True, dora_rank=dora_rank, branch_name='fusion'
                )
            else:
                fusion_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, batch_first=True
                )
            self.fusion_encoder = nn.TransformerEncoder(fusion_layer, num_layers=remaining_layers)
        else:
            self.fusion_encoder = None

        self.cross_fusion = CrossModalFusion(d_model, nhead, dropout)

        self.head = nn.Linear(d_model, num_bands)

    def forward(self, noisy, time_idx, mask):
        scatter = noisy[..., 0:3]
        physics = noisy[..., 3:6]

        pe_s = positional_encoding(time_idx, self.pos_dim)
        pe_p = positional_encoding(time_idx, self.pos_dim)

        h_s = self.scatter_proj(scatter)
        h_s = torch.cat([h_s, pe_s], dim=-1)
        h_s = self.scatter_encoder(h_s)

        h_p = self.physics_proj(physics)
        h_p = torch.cat([h_p, pe_p], dim=-1)
        h_p = self.physics_encoder(h_p)

        h_fused = self.cross_fusion(h_s, h_p)

        if self.fusion_encoder is not None:
            h_fused = self.fusion_encoder(h_fused)

        y = self.head(h_fused)
        b = torch.where(mask)
        pred = y[b[0], b[1]]
        return pred

    def get_scatter_params(self):
        return list(self.scatter_encoder.parameters())

    def get_physics_params(self):
        return list(self.physics_encoder.parameters())

    def get_cross_attn_params(self):
        return list(self.cross_fusion.parameters())

    def get_fusion_encoder_params(self):
        if self.fusion_encoder is not None:
            return list(self.fusion_encoder.parameters())
        return []

    def get_dora_params(self):
        dora_params = []
        for name, param in self.scatter_encoder.named_parameters():
            if 'dora_adapter' in name:
                dora_params.append(param)
        for name, param in self.physics_encoder.named_parameters():
            if 'dora_adapter' in name:
                dora_params.append(param)
        if self.fusion_encoder is not None:
            for name, param in self.fusion_encoder.named_parameters():
                if 'dora_adapter' in name:
                    dora_params.append(param)
        return dora_params

    def get_encoder_core_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'dora_adapter' not in name and 'scatter_proj' not in name and 'physics_proj' not in name and 'cross_fusion' not in name and 'head' not in name:
                params.append(param)
        return params
