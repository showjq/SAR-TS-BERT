import torch
import torch.nn as nn
from src.models.bert_sar_encoder import SarBertEncoder, DualBranchSarBertEncoder, positional_encoding
from src.models.ct_ae_encoder import CT_AE_Encoder
from src.models.htf_mst_encoder import HTF_MST_Encoder, HTF_MST_Encoder_V2
from src.models.sar_ts_mamba import SARTSMambaEncoder, SARTSMambaEncoderV2

class ParcelClassifier(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1, use_dora=False, dora_rank=32, num_bands=3, use_dual_branch=False, scatter_layers=2, physics_layers=2, use_ct_ae=False, ct_ae_fusion='cross_attn', use_htf_mst=False, htf_version=1, tcn_layers=2, tcn_channels=128, use_mamba=False, mamba_version=1, d_state=16, d_conv=4, expand=2, use_sar_norm=True, temporal_layers=4, spectral_layers=2, use_mamba3=False, mimo_rank=2, norm_type=None):
        super().__init__()
        self.use_dual_branch = use_dual_branch
        self.use_ct_ae = use_ct_ae
        self.use_htf_mst = use_htf_mst
        self.use_mamba = use_mamba
        self.num_bands = num_bands

        if use_mamba:
            encoder_class = SARTSMambaEncoder if mamba_version == 1 else SARTSMambaEncoderV2
            encoder_kwargs = {
                'd_model': d_model,
                'nhead': nhead,
                'num_layers': num_layers,
                'dim_feedforward': dim_feedforward,
                'dropout': dropout,
                'num_bands': num_bands,
                'd_state': d_state,
                'd_conv': d_conv,
                'expand': expand,
                'use_dora': use_dora,
                'dora_rank': dora_rank,
                'use_sar_norm': use_sar_norm,
                'use_mamba3': use_mamba3,
                'mimo_rank': mimo_rank
            }
            if mamba_version == 2:
                encoder_kwargs.update({
                    'temporal_layers': temporal_layers,
                    'spectral_layers': spectral_layers
                })
            self.encoder = encoder_class(**encoder_kwargs)
            print(f"使用SAR-TS-Mamba编码器 (version={mamba_version}, d_state={d_state}, d_conv={d_conv}, mamba3={use_mamba3})")
        elif use_htf_mst:
            encoder_class = HTF_MST_Encoder if htf_version == 1 else HTF_MST_Encoder_V2
            self.encoder = encoder_class(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_bands=num_bands,
                tcn_layers=tcn_layers,
                tcn_channels=tcn_channels,
                use_dora=use_dora,
                dora_rank=dora_rank
            )
            print(f"使用HTF-MST编码器 (version={htf_version}, tcn_layers={tcn_layers})")
        elif use_ct_ae:
            self.encoder = CT_AE_Encoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_bands=num_bands,
                fusion_type=ct_ae_fusion,
                use_dora=use_dora,
                dora_rank=dora_rank
            )
            print(f"使用CNN-Transformer双分支交叉注意力编码器 (fusion={ct_ae_fusion})")
        elif use_dual_branch:
            self.encoder = DualBranchSarBertEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_bands=num_bands,
                use_dora=use_dora,
                dora_rank=dora_rank,
                scatter_layers=scatter_layers,
                physics_layers=physics_layers
            )
        else:
            self.encoder = SarBertEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_bands=num_bands,
                use_dora=use_dora,
                dora_rank=dora_rank,
                norm_type=norm_type
            )

        self.d_model = d_model
        self.clf_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, series, time_idx):
        if self.use_mamba:
            fused_feat = self.encoder.get_fused_features(series, time_idx)
            pooled = fused_feat.mean(dim=1)
            logits = self.clf_head(pooled)
            return logits
        elif self.use_htf_mst:
            fused_feat = self.encoder.get_fused_features(series, time_idx)
            pooled = fused_feat.mean(dim=1)
            logits = self.clf_head(pooled)
            return logits
        elif self.use_ct_ae:
            fused_feat = self.encoder.get_fused_features(series, time_idx)
            pooled = fused_feat.mean(dim=1)
            logits = self.clf_head(pooled)
            return logits
        elif self.use_dual_branch:
            pe = positional_encoding(time_idx, self.d_model // 4)
            scatter = series[..., 0:3]
            physics = series[..., 3:6]

            h_s = self.encoder.scatter_proj(scatter)
            h_s = torch.cat([h_s, pe], dim=-1)
            h_s = self.encoder.scatter_encoder(h_s)

            h_p = self.encoder.physics_proj(physics)
            h_p = torch.cat([h_p, pe], dim=-1)
            h_p = self.encoder.physics_encoder(h_p)

            h = self.encoder.cross_fusion(h_s, h_p)

            if self.encoder.fusion_encoder is not None:
                h = self.encoder.fusion_encoder(h)
        else:
            pe = positional_encoding(time_idx, self.encoder.d_model // 2)
            x_obs = self.encoder.obs_proj(series)
            x = torch.cat([x_obs, pe], dim=-1)
            h = self.encoder.encoder(x)

        pooled = h.mean(dim=1)
        logits = self.clf_head(pooled)
        return logits

    def get_dora_params(self):
        dora_params = []
        if self.use_mamba:
            for name, param in self.encoder.named_parameters():
                if 'dora_adapter' in name:
                    dora_params.append(param)
        elif self.use_htf_mst:
            for name, param in self.encoder.named_parameters():
                if 'dora_adapter' in name:
                    dora_params.append(param)
        elif self.use_ct_ae:
            for name, param in self.encoder.named_parameters():
                if 'dora_adapter' in name:
                    dora_params.append(param)
        elif self.use_dual_branch:
            for name, param in self.encoder.scatter_encoder.named_parameters():
                if 'dora_adapter' in name:
                    dora_params.append(param)
            for name, param in self.encoder.physics_encoder.named_parameters():
                if 'dora_adapter' in name:
                    dora_params.append(param)
            if self.encoder.fusion_encoder is not None:
                for name, param in self.encoder.fusion_encoder.named_parameters():
                    if 'dora_adapter' in name:
                        dora_params.append(param)
        else:
            for name, param in self.encoder.encoder.named_parameters():
                if 'dora_adapter' in name:
                    dora_params.append(param)
        return dora_params

    def get_encoder_core_params(self):
        core_params = []
        if self.use_mamba:
            for name, param in self.encoder.named_parameters():
                if 'dora_adapter' not in name and 'obs_proj' not in name and 'sar_norm' not in name:
                    core_params.append(param)
        elif self.use_htf_mst:
            for name, param in self.encoder.named_parameters():
                if 'dora_adapter' not in name and 'input_proj' not in name and 'obs_proj' not in name:
                    core_params.append(param)
        elif self.use_ct_ae:
            for name, param in self.encoder.named_parameters():
                if 'dora_adapter' not in name:
                    core_params.append(param)
        elif self.use_dual_branch:
            encoder_named_params = dict(self.encoder.named_parameters())
            dora_param_ids = {id(p) for n, p in encoder_named_params.items() if 'dora_adapter' in n}
            for n, p in encoder_named_params.items():
                if 'scatter_proj' not in n and 'physics_proj' not in n and 'dora_adapter' not in n:
                    core_params.append(p)
        else:
            for name, param in self.encoder.encoder.named_parameters():
                if 'dora_adapter' not in name:
                    core_params.append(param)
        return core_params

    def get_obs_proj_params(self):
        if self.use_mamba:
            params = list(self.encoder.obs_proj.parameters())
            if self.encoder.sar_norm is not None:
                params += list(self.encoder.sar_norm.parameters())
            return params
        if self.use_htf_mst:
            return list(self.encoder.input_proj.parameters()) + list(self.encoder.obs_proj.parameters())
        if self.use_ct_ae:
            return []
        elif self.use_dual_branch:
            return list(self.encoder.scatter_proj.parameters()) + list(self.encoder.physics_proj.parameters())
        return [self.encoder.obs_proj.weight, self.encoder.obs_proj.bias]

    def get_head_params(self):
        return list(self.clf_head.parameters())
