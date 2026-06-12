import torch
import torch.nn as nn
from src.models.bert_sar_encoder import SarBertEncoder, DualBranchSarBertEncoder, positional_encoding

class MonthlyParcelClassifier(nn.Module):
    def __init__(self, num_classes, num_months, d_model=256, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1, use_dora=False, dora_rank=32, num_bands=3, use_dual_branch=False, scatter_layers=2, physics_layers=2):
        super().__init__()
        self.use_dual_branch = use_dual_branch
        self.num_bands = num_bands

        if use_dual_branch:
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
                dora_rank=dora_rank
            )

        self.num_months = num_months
        self.num_classes = num_classes
        self.d_model = d_model

        self.month_embeddings = nn.Parameter(torch.randn(num_months, d_model))

        self.clf_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(d_model, num_classes)
            ) for _ in range(num_months)
        ])

    def forward(self, series, time_idx):
        b = series.shape[0]

        if self.use_dual_branch:
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

        all_logits = []
        for i in range(self.num_months):
            month_emb = self.month_embeddings[i].unsqueeze(0).expand(b, -1)
            x_month = pooled + month_emb
            logits = self.clf_heads[i](x_month)
            all_logits.append(logits)

        logits = torch.stack(all_logits, dim=1)
        return logits

    def get_dora_params(self):
        dora_params = []
        if self.use_dual_branch:
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
        if self.use_dual_branch:
            encoder_named_params = dict(self.encoder.named_parameters())
            for n, p in encoder_named_params.items():
                if 'scatter_proj' not in n and 'physics_proj' not in n and 'dora_adapter' not in n:
                    core_params.append(p)
        else:
            for name, param in self.encoder.encoder.named_parameters():
                if 'dora_adapter' not in name:
                    core_params.append(param)
        return core_params

    def get_obs_proj_params(self):
        if self.use_dual_branch:
            return list(self.encoder.scatter_proj.parameters()) + list(self.encoder.physics_proj.parameters())
        return [self.encoder.obs_proj.weight, self.encoder.obs_proj.bias]

    def get_head_params(self):
        head_params = []
        head_params.append(self.month_embeddings)
        for head in self.clf_heads:
            head_params.extend(list(head.parameters()))
        return head_params
