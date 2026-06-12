import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.dora import DoRAAdapter


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, indices):
        device = indices.device
        t = indices.float().unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device).float() * (-math.log(10000.0) / self.dim))
        pe = torch.zeros(indices.shape[0], indices.shape[1], self.dim, device=device)
        pe[:, :, 0::2] = torch.sin(t * div_term)
        pe[:, :, 1::2] = torch.cos(t * div_term)
        return pe


class MultiScaleTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5], dropout=0.1):
        super().__init__()
        self.num_scales = len(kernel_sizes)
        self.branch_channels = out_channels // self.num_scales
        self.residual_channels = out_channels - self.branch_channels * self.num_scales
        
        self.branches = nn.ModuleList()
        for i, ks in enumerate(kernel_sizes):
            padding = ks // 2
            branch_out = self.branch_channels + (self.residual_channels if i == 0 else 0)
            branch = nn.Sequential(
                nn.Conv1d(in_channels, branch_out, kernel_size=ks, padding=padding),
                nn.BatchNorm1d(branch_out),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.branches.append(branch)
        
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        x = torch.cat(branch_outputs, dim=1)
        x = self.fusion(x)
        x = x.permute(0, 2, 1)
        return x


class DilatedTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x


class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, feat1, feat2):
        concat = torch.cat([feat1, feat2], dim=-1)
        gate = self.gate(concat)
        return gate * feat1 + (1 - gate) * feat2


class HTF_MST_Encoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=512,
                 dropout=0.1, num_bands=3, tcn_layers=2, tcn_channels=128,
                 kernel_sizes=[1, 3, 5], use_dora=False, dora_rank=32):
        super().__init__()
        self.d_model = d_model
        self.num_bands = num_bands
        self.tcn_layers = tcn_layers
        self.use_dora = use_dora
        
        self.input_proj = nn.Linear(num_bands, tcn_channels)
        
        self.tcn_stages = nn.ModuleList()
        for i in range(tcn_layers):
            in_ch = tcn_channels if i == 0 else d_model
            out_ch = d_model
            self.tcn_stages.append(MultiScaleTCN(in_ch, out_ch, kernel_sizes, dropout))
        
        self.tcn_to_trans_proj = nn.Linear(d_model, d_model)
        
        self.obs_proj = nn.Linear(num_bands, d_model // 2)
        self.pe = PositionalEncoding(d_model // 2)
        
        trans_layers = num_layers - tcn_layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)
        
        if use_dora:
            self.dora_adapters = nn.ModuleList([
                DoRAAdapter(d_model, dora_rank, dropout) for _ in range(trans_layers)
            ])
        
        self.gated_fusion = GatedFusion(d_model)
        
        self.head = nn.Linear(d_model, num_bands)
    
    def forward(self, noisy, time_idx, mask):
        b, t, c = noisy.shape
        
        x = self.input_proj(noisy)
        
        for tcn_stage in self.tcn_stages:
            x = tcn_stage(x)
        
        tcn_feat = self.tcn_to_trans_proj(x)
        
        x_pe = self.obs_proj(noisy)
        pe = self.pe(time_idx)
        trans_input = torch.cat([x_pe, pe], dim=-1)
        
        trans_feat = trans_input
        for i, layer in enumerate(self.trans_encoder.layers):
            trans_feat = layer(trans_feat)
            if self.use_dora:
                trans_feat = self.dora_adapters[i](trans_feat)
        
        fused_feat = self.gated_fusion(tcn_feat, trans_feat)
        
        y = self.head(fused_feat)
        b_idx, t_idx = torch.where(mask)
        if len(b_idx) > 0:
            pred = y[b_idx, t_idx]
        else:
            pred = torch.zeros(0, self.num_bands, device=noisy.device)
        return pred
    
    def get_fused_features(self, noisy, time_idx):
        b, t, c = noisy.shape
        
        x = self.input_proj(noisy)
        for tcn_stage in self.tcn_stages:
            x = tcn_stage(x)
        tcn_feat = self.tcn_to_trans_proj(x)
        
        x_pe = self.obs_proj(noisy)
        pe = self.pe(time_idx)
        trans_input = torch.cat([x_pe, pe], dim=-1)
        
        trans_feat = trans_input
        for i, layer in enumerate(self.trans_encoder.layers):
            trans_feat = layer(trans_feat)
            if self.use_dora:
                trans_feat = self.dora_adapters[i](trans_feat)
        
        fused_feat = self.gated_fusion(tcn_feat, trans_feat)
        return fused_feat


class HTF_MST_Encoder_V2(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=512,
                 dropout=0.1, num_bands=3, tcn_layers=2, tcn_channels=128,
                 kernel_sizes=[1, 3, 5], use_dora=False, dora_rank=32):
        super().__init__()
        self.d_model = d_model
        self.num_bands = num_bands
        self.tcn_layers = tcn_layers
        self.use_dora = use_dora
        
        self.input_proj = nn.Linear(num_bands, tcn_channels)
        
        self.tcn_stages = nn.ModuleList()
        for i in range(tcn_layers):
            in_ch = tcn_channels if i == 0 else d_model
            out_ch = d_model
            self.tcn_stages.append(MultiScaleTCN(in_ch, out_ch, kernel_sizes, dropout))
        
        self.obs_proj = nn.Linear(num_bands, d_model // 2)
        self.pe = PositionalEncoding(d_model // 2)
        
        trans_layers = num_layers - tcn_layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)
        
        if use_dora:
            self.dora_adapters = nn.ModuleList([
                DoRAAdapter(d_model, dora_rank, dropout) for _ in range(trans_layers)
            ])
        
        self.stage_fusions = nn.ModuleList([
            GatedFusion(d_model) for _ in range(min(tcn_layers, trans_layers))
        ])
        
        self.head = nn.Linear(d_model, num_bands)
    
    def forward(self, noisy, time_idx, mask):
        b, t, c = noisy.shape
        
        x = self.input_proj(noisy)
        tcn_feats = []
        for tcn_stage in self.tcn_stages:
            x = tcn_stage(x)
            tcn_feats.append(x)
        
        x_pe = self.obs_proj(noisy)
        pe = self.pe(time_idx)
        trans_input = torch.cat([x_pe, pe], dim=-1)
        
        trans_feat = trans_input
        trans_feats = []
        for i, layer in enumerate(self.trans_encoder.layers):
            trans_feat = layer(trans_feat)
            if self.use_dora:
                trans_feat = self.dora_adapters[i](trans_feat)
            trans_feats.append(trans_feat)
        
        fused_feat = tcn_feats[-1]
        for i in range(min(len(tcn_feats), len(trans_feats))):
            fused_feat = self.stage_fusions[i](tcn_feats[i], trans_feats[i])
        
        y = self.head(fused_feat)
        b_idx, t_idx = torch.where(mask)
        if len(b_idx) > 0:
            pred = y[b_idx, t_idx]
        else:
            pred = torch.zeros(0, self.num_bands, device=noisy.device)
        return pred
    
    def get_fused_features(self, noisy, time_idx):
        b, t, c = noisy.shape
        
        x = self.input_proj(noisy)
        tcn_feats = []
        for tcn_stage in self.tcn_stages:
            x = tcn_stage(x)
            tcn_feats.append(x)
        
        x_pe = self.obs_proj(noisy)
        pe = self.pe(time_idx)
        trans_input = torch.cat([x_pe, pe], dim=-1)
        
        trans_feat = trans_input
        trans_feats = []
        for i, layer in enumerate(self.trans_encoder.layers):
            trans_feat = layer(trans_feat)
            if self.use_dora:
                trans_feat = self.dora_adapters[i](trans_feat)
            trans_feats.append(trans_feat)
        
        fused_feat = tcn_feats[-1]
        for i in range(min(len(tcn_feats), len(trans_feats))):
            fused_feat = self.stage_fusions[i](tcn_feats[i], trans_feats[i])
        
        return fused_feat
