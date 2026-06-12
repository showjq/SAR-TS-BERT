"""
SAR-TS-Mamba: SAR Time Series Mamba Encoder
A state space model based encoder for SAR time series pixel-wise classification.

Key Features:
- Bidirectional Mamba for temporal modeling
- SAR-adaptive preprocessing (speckle-aware normalization)
- Temporal span masking for pretraining
- DoRA support for efficient fine-tuning
- Mamba-3 support with trapezoidal discretization, complex SSM, and MIMO
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


MAMBA_AVAILABLE = False


def _check_mamba_available():
    global MAMBA_AVAILABLE
    if not MAMBA_AVAILABLE:
        try:
            from mamba_ssm import Mamba
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            MAMBA_AVAILABLE = True
        except ImportError:
            MAMBA_AVAILABLE = False


def positional_encoding(indices: torch.Tensor, dim: int) -> torch.Tensor:
    device = indices.device
    t = indices.float().unsqueeze(-1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(indices.shape[0], indices.shape[1], dim, device=device)
    pe[:, :, 0::2] = torch.sin(t * div_term)
    pe[:, :, 1::2] = torch.cos(t * div_term)
    return pe


class SimplifiedMambaBlock(nn.Module):
    """Mamba-1/2 简化实现"""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner)
        nn.init.zeros_(self.conv1d.weight)
        
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        nn.init.xavier_uniform_(self.x_proj.weight, gain=0.5)
        
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        nn.init.zeros_(self.dt_proj.weight)
        nn.init.ones_(self.dt_proj.bias)
        
        A = torch.arange(1, self.d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner) * 0.5)
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        x_dbl = self.x_proj(x_conv)
        dt, B_proj = x_dbl.chunk(2, dim=-1)
        
        dt = self.dt_proj(dt)
        dt = F.softplus(dt + 0.5)
        dt = torch.clamp(dt, min=1e-4, max=10.0)
        
        A = -torch.exp(self.A_log.float().clamp(max=4.0))
        
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for t in range(L):
            x_t = x_conv[:, t]
            dt_t = dt[:, t]
            B_t = B_proj[:, t]
            
            deltaA = torch.exp(dt_t.unsqueeze(-1) * A)
            deltaA = torch.clamp(deltaA, max=1e6)
            
            deltaB_u = dt_t.unsqueeze(-1) * B_t.unsqueeze(1) * x_t.unsqueeze(-1)
            deltaB_u = torch.clamp(deltaB_u, min=-1e6, max=1e6)
            
            h = deltaA * h + deltaB_u
            h = torch.clamp(h, min=-1e6, max=1e6)
            
            y = torch.sum(h * A.unsqueeze(0), dim=-1) + self.D * x_t
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return self.norm(y + residual)


class Mamba3Block(nn.Module):
    """
    Mamba-3: 推理优先的SSM架构
    
    三大核心改进:
    1. 梯形离散化 (Trapezoidal Discretization): 二阶精度，隐式引入宽度为2的卷积
    2. 复数SSM (Complex-valued SSM): 通过RoPE技巧实现，解决状态追踪问题
    3. MIMO机制: 多输入多输出，提高算术强度
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, dropout: float = 0.1, mimo_rank: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_model * expand
        self.mimo_rank = mimo_rank
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        nn.init.xavier_uniform_(self.x_proj.weight, gain=0.5)
        
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        nn.init.zeros_(self.dt_proj.weight)
        nn.init.ones_(self.dt_proj.bias)
        
        A = torch.arange(1, self.d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner) * 0.5)
        
        self.alpha_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)
        nn.init.zeros_(self.alpha_proj.weight)
        
        self.theta_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)
        nn.init.zeros_(self.theta_proj.weight)
        
        self.mimo_proj = nn.Linear(self.d_inner, self.d_inner * mimo_rank, bias=False)
        nn.init.xavier_uniform_(self.mimo_proj.weight, gain=0.5)
        
        self.mimo_out = nn.Linear(self.d_inner * mimo_rank, self.d_inner, bias=False)
        nn.init.xavier_uniform_(self.mimo_out.weight, gain=0.5)
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        x_dbl = self.x_proj(x_proj)
        dt, B_proj = x_dbl.chunk(2, dim=-1)
        
        dt = self.dt_proj(dt)
        dt = F.softplus(dt + 0.5)
        dt = torch.clamp(dt, min=1e-4, max=10.0)
        
        A = -torch.exp(self.A_log.float().clamp(max=4.0))
        
        alpha = torch.sigmoid(self.alpha_proj(x_proj))
        theta = self.theta_proj(x_proj)
        
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        h_prev = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for t in range(L):
            x_t = x_proj[:, t]
            dt_t = dt[:, t]
            B_t = B_proj[:, t]
            alpha_t = alpha[:, t]
            theta_t = theta[:, t]
            
            deltaA = torch.exp(dt_t.unsqueeze(-1) * A)
            deltaA = torch.clamp(deltaA, max=1e6)
            
            deltaB_u = dt_t.unsqueeze(-1) * B_t.unsqueeze(1) * x_t.unsqueeze(-1)
            deltaB_u = torch.clamp(deltaB_u, min=-1e6, max=1e6)
            
            cos_theta = torch.cos(theta_t).unsqueeze(-1)
            sin_theta = torch.sin(theta_t).unsqueeze(-1)
            
            h_rotated = h * cos_theta + h_prev * sin_theta
            
            h_new = deltaA * h_rotated + deltaB_u
            h_new = torch.clamp(h_new, min=-1e6, max=1e6)
            
            h_trapezoid = alpha_t.unsqueeze(-1) * h_new + (1 - alpha_t.unsqueeze(-1)) * h
            h_prev = h
            h = h_trapezoid
            
            y = torch.sum(h * A.unsqueeze(0), dim=-1) + self.D * x_t
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        
        y_mimo = self.mimo_proj(y)
        y_mimo = self.mimo_out(y_mimo)
        y = y + y_mimo
        
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return self.norm(y + residual)


class BidirectionalMamba(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1, use_mamba3: bool = False, mimo_rank: int = 2):
        super().__init__()
        self.use_mamba3 = use_mamba3
        
        if use_mamba3:
            self.mamba_forward = Mamba3Block(d_model, d_state, expand, dropout, mimo_rank)
            self.mamba_backward = Mamba3Block(d_model, d_state, expand, dropout, mimo_rank)
            self.norm = nn.LayerNorm(d_model)
        else:
            _check_mamba_available()
            if MAMBA_AVAILABLE:
                from mamba_ssm import Mamba
                self.mamba_forward = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                self.mamba_backward = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                self.norm = nn.LayerNorm(d_model)
            else:
                self.mamba_forward = SimplifiedMambaBlock(d_model, d_state, d_conv, expand, dropout)
                self.mamba_backward = SimplifiedMambaBlock(d_model, d_state, d_conv, expand, dropout)
                self.norm = None
        
        self.gate = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_forward = self.mamba_forward(x)
        h_backward = self.mamba_backward(torch.flip(x, dims=[1]))
        h_backward = torch.flip(h_backward, dims=[1])
        
        h_cat = torch.cat([h_forward, h_backward], dim=-1)
        h_gate = torch.sigmoid(self.gate(h_cat))
        
        out = h_gate * h_forward + (1 - h_gate) * h_backward
        
        if self.norm is not None:
            out = self.norm(out + x)
        return out


class SARAdaptiveNorm(nn.Module):
    def __init__(self, num_bands: int, eps: float = 1e-6):
        super().__init__()
        self.num_bands = num_bands
        self.eps = eps
        
        self.gamma = nn.Parameter(torch.ones(num_bands))
        self.beta = nn.Parameter(torch.zeros(num_bands))
        
        self.speckle_scale = nn.Parameter(torch.ones(num_bands))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        
        std = torch.sqrt(var + self.eps)
        std = std * self.speckle_scale
        
        x_norm = (x - mean) / std
        x_norm = x_norm * self.gamma + self.beta
        
        return x_norm


class TemporalSpanMasking(nn.Module):
    def __init__(self, mask_ratio: float = 0.3, span_len: int = 3, random_ratio: float = 0.1):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.span_len = span_len
        self.random_ratio = random_ratio
        
    def forward(self, x: torch.Tensor, time_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        device = x.device
        
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        
        num_spans = int(T * self.mask_ratio / self.span_len)
        
        for b in range(B):
            span_starts = torch.randperm(T - self.span_len + 1, device=device)[:num_spans]
            for start in span_starts:
                mask[b, start:start + self.span_len] = True
        
        num_random = int(T * self.random_ratio)
        random_mask = torch.rand(B, T, device=device) < (num_random / T)
        mask = mask | random_mask
        
        noisy = x.clone()
        noise = torch.randn_like(x[mask]) * 0.1
        noisy[mask] = x[mask] + noise
        
        return noisy, mask, time_idx


class DoRAAdapter(nn.Module):
    def __init__(self, d_model: int, rank: int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        
        self.down_proj = nn.Linear(d_model, rank, bias=False)
        self.up_proj = nn.Linear(rank, d_model, bias=False)
        self.scaling = nn.Parameter(torch.ones(1) * 0.1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scaling * self.up_proj(F.silu(self.down_proj(self.dropout(x))))


class SARTSMambaEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_bands: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_dora: bool = False,
        dora_rank: int = 32,
        use_sar_norm: bool = True,
        use_mamba3: bool = False,
        mimo_rank: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.num_bands = num_bands
        self.use_dora = use_dora
        self.use_sar_norm = use_sar_norm
        self.use_mamba3 = use_mamba3
        
        if use_sar_norm:
            self.sar_norm = SARAdaptiveNorm(num_bands)
        else:
            self.sar_norm = None
        
        self.obs_proj = nn.Linear(num_bands, d_model // 2)
        
        self.mamba_layers = nn.ModuleList([
            BidirectionalMamba(d_model, d_state, d_conv, expand, dropout, use_mamba3, mimo_rank)
            for _ in range(num_layers)
        ])
        
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        if use_dora:
            self.dora_adapters = nn.ModuleList([
                DoRAAdapter(d_model, dora_rank, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.dora_adapters = None
        
        self.head = nn.Linear(d_model, num_bands)
        
    def forward(self, noisy: torch.Tensor, time_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.sar_norm is not None:
            noisy = self.sar_norm(noisy)
        
        pe = positional_encoding(time_idx, self.d_model // 2)
        x_obs = self.obs_proj(noisy)
        x = torch.cat([x_obs, pe], dim=-1)
        
        h = x
        for i, (mamba_layer, ffn_layer, norm) in enumerate(zip(self.mamba_layers, self.ffn_layers, self.norms)):
            h_mamba = mamba_layer(h)
            h_ffn = ffn_layer(norm(h_mamba))
            h = h + h_ffn
            
            if self.dora_adapters is not None:
                h = self.dora_adapters[i](h)
        
        y = self.head(h)
        b = torch.where(mask)
        pred = y[b[0], b[1]]
        
        return pred
    
    def get_fused_features(self, series: torch.Tensor, time_idx: torch.Tensor) -> torch.Tensor:
        if self.sar_norm is not None:
            series = self.sar_norm(series)
        
        pe = positional_encoding(time_idx, self.d_model // 2)
        x_obs = self.obs_proj(series)
        x = torch.cat([x_obs, pe], dim=-1)
        
        h = x
        for i, (mamba_layer, ffn_layer, norm) in enumerate(zip(self.mamba_layers, self.ffn_layers, self.norms)):
            h_mamba = mamba_layer(h)
            h_ffn = ffn_layer(norm(h_mamba))
            h = h + h_ffn
            
            if self.dora_adapters is not None:
                h = self.dora_adapters[i](h)
        
        return h


class SARTSMambaEncoderV2(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_bands: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_dora: bool = False,
        dora_rank: int = 32,
        use_sar_norm: bool = True,
        temporal_layers: int = 4,
        spectral_layers: int = 2,
        use_mamba3: bool = False,
        mimo_rank: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.num_bands = num_bands
        self.use_dora = use_dora
        self.use_sar_norm = use_sar_norm
        self.temporal_layers = temporal_layers
        self.spectral_layers = spectral_layers
        self.use_mamba3 = use_mamba3
        
        if use_sar_norm:
            self.sar_norm = SARAdaptiveNorm(num_bands)
        else:
            self.sar_norm = None
        
        self.obs_proj = nn.Linear(num_bands, d_model // 2)
        
        self.temporal_mamba = nn.ModuleList([
            BidirectionalMamba(d_model, d_state, d_conv, expand, dropout, use_mamba3, mimo_rank)
            for _ in range(temporal_layers)
        ])
        
        self.spectral_mamba = nn.ModuleList([
            BidirectionalMamba(d_model, d_state, d_conv, expand, dropout, use_mamba3, mimo_rank)
            for _ in range(spectral_layers)
        ])
        
        self.temporal_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(temporal_layers)
        ])
        
        self.spectral_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(spectral_layers)
        ])
        
        self.temporal_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(temporal_layers)])
        self.spectral_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(spectral_layers)])
        
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        
        if use_dora:
            self.dora_adapters = nn.ModuleList([
                DoRAAdapter(d_model, dora_rank, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.dora_adapters = None
        
        self.head = nn.Linear(d_model, num_bands)
        
    def forward(self, noisy: torch.Tensor, time_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.sar_norm is not None:
            noisy = self.sar_norm(noisy)
        
        pe = positional_encoding(time_idx, self.d_model // 2)
        x_obs = self.obs_proj(noisy)
        x = torch.cat([x_obs, pe], dim=-1)
        
        h = x
        for i, (mamba_layer, ffn_layer, norm) in enumerate(zip(self.temporal_mamba, self.temporal_ffn, self.temporal_norms)):
            h_mamba = mamba_layer(h)
            h_ffn = ffn_layer(norm(h_mamba))
            h = h + h_ffn
        
        h_spectral = h
        for i, (mamba_layer, ffn_layer, norm) in enumerate(zip(self.spectral_mamba, self.spectral_ffn, self.spectral_norms)):
            h_mamba = mamba_layer(h_spectral)
            h_ffn = ffn_layer(norm(h_mamba))
            h_spectral = h_spectral + h_ffn
        
        gate = torch.sigmoid(self.fusion_gate(torch.cat([h, h_spectral], dim=-1)))
        h = gate * h + (1 - gate) * h_spectral
        
        if self.dora_adapters is not None:
            for adapter in self.dora_adapters:
                h = adapter(h)
        
        y = self.head(h)
        b = torch.where(mask)
        pred = y[b[0], b[1]]
        
        return pred
    
    def get_fused_features(self, series: torch.Tensor, time_idx: torch.Tensor) -> torch.Tensor:
        if self.sar_norm is not None:
            series = self.sar_norm(series)
        
        pe = positional_encoding(time_idx, self.d_model // 2)
        x_obs = self.obs_proj(series)
        x = torch.cat([x_obs, pe], dim=-1)
        
        h = x
        for i, (mamba_layer, ffn_layer, norm) in enumerate(zip(self.temporal_mamba, self.temporal_ffn, self.temporal_norms)):
            h_mamba = mamba_layer(h)
            h_ffn = ffn_layer(norm(h_mamba))
            h = h + h_ffn
        
        h_spectral = h
        for i, (mamba_layer, ffn_layer, norm) in enumerate(zip(self.spectral_mamba, self.spectral_ffn, self.spectral_norms)):
            h_mamba = mamba_layer(h_spectral)
            h_ffn = ffn_layer(norm(h_mamba))
            h_spectral = h_spectral + h_ffn
        
        gate = torch.sigmoid(self.fusion_gate(torch.cat([h, h_spectral], dim=-1)))
        h = gate * h + (1 - gate) * h_spectral
        
        return h
