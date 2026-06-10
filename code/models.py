import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLinear(nn.Module):
    """
    简易高效率 KAN (Kolmogorov-Arnold Network) 线性层实现
    使用基函数（如 SiLU）配合可学习的样条/多项式基底简化
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 基础线性映射 (Base function: sub-layer equivalent to silu(x) * w)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # 样条/非线性插值权重 (Spline component for flexible nonlinear compensation)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

    def forward(self, x):
        # x shape: [Batch, Length, In_features] 或 [Batch, In_features]
        base_output = F.linear(F.silu(x), self.base_weight)
        
        # 简化版多项式/样条基底计算（论文中提及作为非线性补偿器）
        # 这里用特征的多阶幂近似样条基底的行为
        x_expanded = x.unsqueeze(-1) # [B, L, In, 1]
        spline_basis = torch.cat([x_expanded ** i for i in range(1, self.spline_weight.size(-1) + 1)], dim=-1)
        
        spline_output = torch.einsum('blin,oin->blo', spline_basis, self.spline_weight)
        return base_output + spline_output


class PIKFormer(nn.Module):
    """
    PIKFormer: 基于 KAN 的时空风电预测网络
    """
    def __init__(self, seq_len, pred_len, num_features, embed_dim=64, num_heads=4):
        super(PIKFormer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 时空特征嵌入
        self.enc_embedding = nn.Linear(num_features, embed_dim)
        
        # KAN 增强的时空注意力层 (KAN-enhanced Spatio-Temporal Attention)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 用 KANLinear 替换传统的 MLP 层
        self.kan_layer1 = KANLinear(embed_dim, embed_dim * 2)
        self.kan_layer2 = KANLinear(embed_dim * 2, embed_dim)
        
        # 预测输出头
        self.head = KANLinear(seq_len * embed_dim, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_len, Num_features] (包含风速、历史功率等)
        B, L, F = x.shape
        
        # 1. 嵌入
        x_emb = self.enc_embedding(x) # [B, L, D]
        
        # 2. 时空注意力机制
        attn_out, _ = self.attention(x_emb, x_emb, x_emb)
        x_emb = x_emb + attn_out
        
        # 3. KAN 前馈网络 (取代传统 Transformer 的 Feed-Forward MLP)
        kan_out = self.kan_layer2(self.kan_layer1(x_emb))
        x_emb = x_emb + kan_out
        
        # 4. 映射输出预测功率值
        x_flat = x_emb.view(B, -1)
        pred_power = self.head(x_flat) # [Batch, Pred_len]
        
        return pred_power


class PIKDLinear(nn.Module):
    """
    PIKDLinear: 论文提出的轻量化变体 (无需 Transformer 主干)
    验证在不使用重度注意力机制下的物理和 KAN 层的有效性
    """
    def __init__(self, seq_len, pred_len):
        super(PIKDLinear, self).__init__()
        # 结合 DLinear 的时空趋势/季节性拆分思想，内部采用 KAN 进行非线性补偿
        self.kan_trend = KANLinear(seq_len, pred_len)
        self.kan_seasonal = KANLinear(seq_len, pred_len)

    def forward(self, x):
        # 假设 x 的最后一个通道是历史功率: [B, Seq_len, Features]
        # 这里简化抽取主特征对应的趋势和周期
        power_hist = x[:, :, -1] # [B, Seq_len]
        
        # 简易移动平均拆分（趋势项与季节项）
        trend_part = torch.mean(power_hist, dim=1, keepdim=True).repeat(1, power_hist.size(1))
        seasonal_part = power_hist - trend_part
        
        pred_trend = self.kan_trend(trend_part)
        pred_seasonal = self.kan_seasonal(seasonal_part)
        
        return pred_trend + pred_seasonal