import math
import torch
import torch.nn as nn

class DoRAAdapter(nn.Module):
    """
    DoRA (Domain-adaptive Representations for Accelerated Fine-tuning) 适配器
    低秩适配器，用于高效微调预训练模型
    """
    def __init__(self, d_model, rank=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        
        # 低秩投影层
        self.down_proj = nn.Linear(d_model, rank)
        self.up_proj = nn.Linear(rank, d_model)
        
        # 激活函数
        self.activation = nn.GELU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02/math.sqrt(rank))
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x

class DoRAEncoderLayer(nn.TransformerEncoderLayer):
    """
    集成DoRA适配器的Transformer编码器层
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", 
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False, dora_rank=32):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, 
                        layer_norm_eps, batch_first, norm_first)
        
        # 添加DoRA适配器
        self.dora_adapter1 = DoRAAdapter(d_model, rank=dora_rank, dropout=dropout)
        self.dora_adapter2 = DoRAAdapter(d_model, rank=dora_rank, dropout=dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None, 
                is_causal=False):
        x = src
        
        if self.norm_first:
            # 第一层归一化
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
            # 添加DoRA适配器1
            x = self.dora_adapter1(x)
            # 第二层归一化
            x = x + self._ff_block(self.norm2(x))
            # 添加DoRA适配器2
            x = self.dora_adapter2(x)
        else:
            # 自注意力模块
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal))
            # 添加DoRA适配器1
            x = self.dora_adapter1(x)
            # 前馈网络
            x = self.norm2(x + self._ff_block(x))
            # 添加DoRA适配器2
            x = self.dora_adapter2(x)
        
        return x


class DualBranchDoRAEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False, dora_rank=32, branch_name='default'):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                        layer_norm_eps, batch_first, norm_first)

        self.branch_name = branch_name

        self.dora_adapter1 = DoRAAdapter(d_model, rank=dora_rank, dropout=dropout)
        self.dora_adapter2 = DoRAAdapter(d_model, rank=dora_rank, dropout=dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None,
                is_causal=False):
        x = src

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
            x = self.dora_adapter1(x)
            x = x + self._ff_block(self.norm2(x))
            x = self.dora_adapter2(x)
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal))
            x = self.dora_adapter1(x)
            x = self.norm2(x + self._ff_block(x))
            x = self.dora_adapter2(x)

        return x