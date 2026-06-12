import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.dora import DoRAEncoderLayer

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


class CNNBranch(nn.Module):
    def __init__(self, in_channels, cnn_channels=[64, 128], kernel_sizes=[3, 5], dropout=0.1):
        super().__init__()
        layers = []
        for i, (out_ch, ks) in enumerate(zip(cnn_channels, kernel_sizes)):
            layers.append(nn.Conv1d(in_channels, out_ch, kernel_size=ks, padding=ks // 2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_ch))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_ch
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key_value):
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        return self.norm(query + attn_out)


class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        concat_feat = torch.cat([feat1, feat2], dim=-1)
        gate = self.gate_fc(concat_feat)
        return gate * feat1 + (1 - gate) * feat2


class CT_AE_Encoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=512,
                 dropout=0.1, num_bands=3, cnn_channels=[64, 128], kernel_sizes=[3, 5],
                 fusion_type='cross_attn', use_dora=False, dora_rank=32):
        super().__init__()
        self.d_model = d_model
        self.num_bands = num_bands
        self.fusion_type = fusion_type
        self.use_dora = use_dora

        self.cnn_proj = nn.Linear(num_bands, cnn_channels[0])
        self.cnn_branch = CNNBranch(cnn_channels[0], cnn_channels[1:], kernel_sizes, dropout)
        self.cnn_out_proj = nn.Linear(cnn_channels[-1], d_model)

        self.obs_proj = nn.Linear(num_bands, d_model // 2)
        self.pe = PositionalEncoding(d_model // 2)

        if use_dora:
            encoder_layer = DoRAEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, dora_rank=dora_rank
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            )
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if fusion_type == 'cross_attn':
            self.cross_attn = CrossAttentionFusion(d_model, nhead, dropout)
        elif fusion_type == 'gate':
            self.gate_fusion = GatedFusion(d_model)

        self.head = nn.Linear(d_model, num_bands)

    def forward(self, noisy, time_idx, mask):
        b, t, c = noisy.shape

        cnn_feat = self.cnn_proj(noisy)
        cnn_feat = self.cnn_branch(cnn_feat)
        cnn_feat = self.cnn_out_proj(cnn_feat)

        x_pe = self.obs_proj(noisy)
        pe = self.pe(time_idx)
        x_pe = torch.cat([x_pe, pe], dim=-1)
        trans_feat = self.trans_encoder(x_pe)

        if self.fusion_type == 'cross_attn':
            fused_feat = self.cross_attn(cnn_feat, trans_feat)
        elif self.fusion_type == 'gate':
            fused_feat = self.gate_fusion(cnn_feat, trans_feat)
        elif self.fusion_type == 'add':
            fused_feat = cnn_feat + trans_feat
        elif self.fusion_type == 'concat':
            fused_feat = torch.cat([cnn_feat, trans_feat], dim=-1)
            fused_feat = F.relu(self.head(fused_feat)[:, :, :self.d_model])
        else:
            fused_feat = trans_feat

        y = self.head(fused_feat)
        b_idx, t_idx = torch.where(mask)
        if len(b_idx) > 0:
            pred = y[b_idx, t_idx]
        else:
            pred = torch.zeros(0, self.num_bands, device=noisy.device)
        return pred

    def get_fused_features(self, noisy, time_idx):
        b, t, c = noisy.shape

        cnn_feat = self.cnn_proj(noisy)
        cnn_feat = self.cnn_branch(cnn_feat)
        cnn_feat = self.cnn_out_proj(cnn_feat)

        x_pe = self.obs_proj(noisy)
        pe = self.pe(time_idx)
        x_pe = torch.cat([x_pe, pe], dim=-1)
        trans_feat = self.trans_encoder(x_pe)

        if self.fusion_type == 'cross_attn':
            fused_feat = self.cross_attn(cnn_feat, trans_feat)
        elif self.fusion_type == 'gate':
            fused_feat = self.gate_fusion(cnn_feat, trans_feat)
        else:
            fused_feat = trans_feat

        return fused_feat
