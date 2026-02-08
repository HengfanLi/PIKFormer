import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from torch_geometric.data import Data, Batch
import dgl
import dgl.function as fn
import pandas as pd
from net import *
WIN = 3
DECOMP = 24


class UncertaintyHead(nn.Module):
    def __init__(self, num_losses):
        super(UncertaintyHead, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self):
        #return 0.5*torch.exp(-self.log_vars)
        return torch.exp(-torch.clamp(self.log_vars, min=-0.5, max=3))
    
#########################################################################################################
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=3,
        spline_order=2,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-3.5, 8.5],#[-1, 1]
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy)
    
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # 这里的判断语句主要是为了和Fedformer的部分进行区分，如果只考虑autoformer可以默认这里的判断全是True
        if type(self.kernel_size) == list:
            if len(self.kernel_size) == 1:
                self.kernel_size = self.kernel_size[0]
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """
#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0,1,3,2))#0,1,2,3-0,1,3,2
#         x = x.permute(0,1,3,2)
#         return x

# class SeriesDecomp2(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, kernel_size):
#         super(SeriesDecomp2, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)

#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean

# class SeriesDecomp(nn.Module):
#     def __init__(self, kernel_size):
#         super().__init__()
#         self.kernel_size = kernel_size

#     def forward(self, x):
#         t_x = torch.transpose(x, 1, 2)
#         # 手动计算填充值
#         padding = (self.kernel_size - 1) // 2
#         mean_x = F.avg_pool1d(
#             t_x, self.kernel_size, stride=1, padding=padding)
#         mean_x = torch.transpose(mean_x, 1, 2)

#         # 检查 x 和 mean_x 的尺寸是否一致
#         if x.size(1) != mean_x.size(1):
#             # 调整 mean_x 的尺寸使其与 x 匹配
#             if x.size(1) > mean_x.size(1):
#                 # 如果 x 的尺寸大于 mean_x，对 mean_x 进行填充
#                 diff = x.size(1) - mean_x.size(1)
#                 pad = (0, 0, 0, diff)
#                 mean_x = F.pad(mean_x, pad, "constant", 0)
#             else:
#                 # 如果 x 的尺寸小于 mean_x，对 mean_x 进行裁剪
#                 mean_x = mean_x[:, :x.size(1), :]

#         return x - mean_x, mean_x

class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder with Time Series Decomposition

    Ideas comes from AutoFormer

    Decoding trends and seasonal 
    Decompose a time series into trends and seasonal

    Refs:  https://arxiv.org/abs/2106.13008
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dims_feedforward,
                 dropout=0.1,
                 activation="gelu",
                 attn_dropout=None,
                 act_dropout=None,
                 trends_out=134,
                 grid_size=3,
                 spline_order=2):
        super(TransformerDecoderLayer, self).__init__()

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout

        self.decomp = SeriesDecomp(DECOMP)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        #self.linear1 = nn.Linear(d_model, dims_feedforward)
        self.linear1 =KANLinear(d_model, dims_feedforward, grid_size=grid_size, spline_order=spline_order)
        self.dropout = nn.Dropout(act_dropout)
        #self.linear2 = nn.Linear(dims_feedforward, d_model)
        self.linear2 = KANLinear(dims_feedforward, d_model, grid_size=grid_size, spline_order=spline_order)

        self.linear_trend = nn.Conv1d(
            d_model, trends_out, WIN, padding="same")

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, src, memory, src_mask=None, cache=None):
        residual = src
       # print('src2 ',src.shape)#([32, 432, 128])
        src, _ = self.self_attn(src, src, src, None)
        # src = self.self_attn(src, src, src, None)
        src = residual + self.dropout1(src)

        src, trend1 = self.decomp(src)

        residual = src
        src, _ = self.cross_attn(src, memory, memory, None)
        src = residual + self.dropout1(src)

        src, trend2 = self.decomp(src)

        residual = src
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)

        src, trend3 = self.decomp(src)
        res_trend = trend1 + trend2 + trend3
        #print('res_trend1 ',res_trend.shape)#res_trend1  torch.Size([32, 432, 128])
        # Change shape from [batch, seq_len, channels] to [batch, channels, seq_len]
        res_trend = torch.transpose(res_trend, 1, 2)
        res_trend = self.linear_trend(res_trend)
        #print('res_trend2 ',res_trend.shape)res_trend2  torch.Size([32, 134, 432])
        # Change shape back to [batch, seq_len, channels]
        res_trend = torch.transpose(res_trend, 1, 2)
        return src, res_trend



class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dims_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 grid_size=3,
                 spline_order=2):
        super(TransformerEncoderLayer, self).__init__()

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout

        self.decomp = SeriesDecomp(DECOMP)

        self.self_attn = nn.MultiheadAttention(d_model, nhead)

       #self.linear1 = nn.Linear(d_model, dims_feedforward)
        self.linear1 = KANLinear(d_model, dims_feedforward, grid_size=grid_size, spline_order=spline_order)
        self.dropout = nn.Dropout(act_dropout)
        #self.linear2 = nn.Linear(dims_feedforward, d_model)
        self.linear2 = KANLinear(dims_feedforward, d_model, grid_size=grid_size, spline_order=spline_order)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, src, src_mask=None, cache=None):
        residual = src
        #print('src ',src.shape)#([32, 144, 128])
        src, _ = self.self_attn(src, src, src, None)
        src = residual + self.dropout1(src)

        src, _ = self.decomp(src)

        residual = src

        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)

        src, _ = self.decomp(src)
        return src

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config_file = config
        self.var_len = config.var_len
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.hidden_dims = config.hidden_dims
        self.nhead = config.nhead
        self.num_encoder_layer = config.encoder_layers
        self.grid_size = getattr(config, 'grid_size', 3)
        self.spline_order = getattr(config, 'spline_order', 2)

        self.enc_lins = nn.ModuleList()
        self.dropout = config.dropout
        self.drop = nn.Dropout(self.dropout)
        for _ in range(self.num_encoder_layer):
            self.enc_lins.append(
                TransformerEncoderLayer(
                    d_model=self.hidden_dims,
                    nhead=self.nhead,
                    dropout=self.dropout,
                    activation="gelu",
                    attn_dropout=self.dropout,
                    act_dropout=self.dropout,
                    dims_feedforward=self.hidden_dims * 2,
                    grid_size=self.grid_size,
                    spline_order=self.spline_order))

    def forward(self, batch_x):
        for lin in self.enc_lins:
            batch_x = lin(batch_x)
        batch_x = self.drop(batch_x)
        return batch_x

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config_file = config
        self.var_len = config.var_len
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.hidden_dims = config.hidden_dims
        self.nhead = config.nhead
        self.num_decoder_layer = config.decoder_layers
        self.grid_size = getattr(config, 'grid_size', 3)
        self.spline_order = getattr(config, 'spline_order', 2)
        self.dec_lins = nn.ModuleList()
        self.dropout = config.dropout
        self.drop = nn.Dropout(self.dropout)
        self.capacity = config.capacity

        for _ in range(self.num_decoder_layer):
            self.dec_lins.append(
                TransformerDecoderLayer(
                    d_model=self.hidden_dims,
                    nhead=self.nhead,
                    dropout=self.dropout,
                    activation="gelu",
                    attn_dropout=self.dropout,
                    act_dropout=self.dropout,
                    dims_feedforward=self.hidden_dims * 2,
                    trends_out=self.capacity,
                    grid_size=self.grid_size,
                    spline_order=self.spline_order))

    def forward(self, season, trend, enc_output):
        for lin in self.dec_lins:
            season, trend_part = lin(season, enc_output)
            trend = trend + trend_part
        return season, trend


class SpatialTemporalConv(nn.Module):

    def __init__(self, id_len, input_dim, output_dim ,grid_size=3, spline_order=2):
        super(SpatialTemporalConv, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=id_len * input_dim,
            out_channels=output_dim,
            kernel_size=WIN,
            padding="same",
            bias=False)
        self.id_len = id_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.q = StableLinear(input_dim, output_dim, use_transformer_scale=True)
        # self.k = StableLinear(input_dim, output_dim, use_transformer_scale=True)
        #print('output_dim/2 ',output_dim/2)
        # self.q = nn.Linear(input_dim, output_dim)
        # self.k = nn.Linear(input_dim, output_dim)
        self.q = KANLinear(input_dim, output_dim, grid_size=grid_size, spline_order=spline_order)
        self.k = KANLinear(input_dim, output_dim, grid_size=grid_size, spline_order=spline_order)

        self.norm = nn.LayerNorm(input_dim, eps=1e-5)

    def _send_attention(self, edges):
        alpha = edges.src['k'] * edges.dst['q']
        alpha = torch.sum(alpha, dim=-1, keepdim=True)
        if torch.isnan(edges.src['k']).any() or torch.isnan(edges.dst['q']).any():
            print("edges.src['k'] NaN detected in attention computation")

        return {"alpha": alpha, "output_series": edges.src["v"]}
      
    def _reduce_attention(self, nodes):
        alpha = F.softmax(nodes.mailbox["alpha"], dim=1)
        output = torch.sum(nodes.mailbox["output_series"] * alpha, dim=1)
        return {"output": output}
    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            if hasattr(layer, 'weight') and isinstance(layer.weight, torch.Tensor):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            layer.eps = 1e-12
    def forward(self, x, graph):
        bz, seqlen, _ = x.shape
        x = x.view(bz, seqlen, self.id_len, self.input_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(-1, seqlen, self.input_dim)       
        mean_x = x.mean(dim=1)  # (bz*id_len, input_dim)      
        q_x = self.q(mean_x)
        q_x = q_x /math.sqrt(self.output_dim)        
        k_x = self.k(mean_x)
        x = x.view(-1, seqlen * self.input_dim)
        
        graph = graph.to(k_x.device)
        graph.ndata["k"] = k_x
        graph.ndata["q"] = q_x
        graph.ndata["v"] = x

        graph.apply_edges(self._send_attention)
        graph.update_all(self._send_attention, self._reduce_attention)

        output = graph.ndata["output"]
        x = output.view(bz, self.id_len, seqlen, self.input_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bz, seqlen, self.id_len * self.input_dim)
        return self.conv1(x.transpose(1, 2)).transpose(1, 2)


# class NeuralNetwork(nn.Module):

#     def __init__(
#         self,
#         t_in =144,
#         t_out=288,
#         f_in =9,
#         dropout_rate=0.2,
#         use_physics_feature=True,
#         v_index=2,        # 你的 V 在第3个特征 -> index=2
#         p_index=6,        # 你的 P 在第7个特征 -> index=6
#         etmp_index=3,     # 你之前用的 etmp 在第4个特征 -> index=3
#         out_last_dim=False,
#         alpha_init=1.0
#     ):
#         super().__init__()
#         self.t_in = t_in
#         self.t_out = t_out
#         self.f_in = f_in
#         self.use_physics_feature = use_physics_feature
#         self.v_index = v_index
#         self.p_index = p_index
#         self.etmp_index = etmp_index
#         self.out_last_dim = out_last_dim
#         f_eff = f_in #+ (1 if use_physics_feature else 0)
#         input_dim = t_in * f_eff
#         self.fc1 = nn.Linear(input_dim, 512)
#         self.norm1 = nn.LayerNorm(512)
#         self.fc2 = nn.Linear(512, 256)
#         self.norm2 = nn.LayerNorm(256)
#         self.fc5 = nn.Linear(256, 128)
        
#         # 2. 瓶颈段：连续两个 128 维残差块
#         self.bottleneck_res1 = IdentityBlock(128, dropout_rate=0.2)
#         self.bottleneck_res2 = IdentityBlock(128, dropout_rate=0.2)
        
#         # 3. 解码段
#         self.fc7 = nn.Linear(128, 256)
#         self.norm7 = nn.LayerNorm(256)
#         self.fc8 = nn.Linear(256, 512)
#         self.norm8 = nn.LayerNorm(512)
#         self.fc9 = nn.Linear(512, t_out)

#         self.gelu = nn.GELU()
#         self.relu = nn.ReLU()
#         self.dropout1 = nn.Dropout(0.1)
#         self.dropout2 = nn.Dropout(0.2)
#         self.dropout3 = nn.Dropout(0.3)
#         self.softplus = nn.Softplus()
#         self.c_scale = nn.Parameter(torch.tensor(1000.0))
#         self.alpha_scale = nn.Parameter(torch.tensor(alpha_init))
#         # self.learned_limit = nn.Parameter(torch.tensor(7500.0))
#         # self.limit_activator = nn.Softplus()
#     def normalize(self, x):
#         B, N, T, F = x.shape
#         #x_reshaped = x.view(-1,T, F)  # 形状变为 [32*134,144, 8]
#         mean = x.mean(dim=(0,1,2))  # [8]
#         std = x.std(dim=(0,1,2))    # [8]
#         return (x - mean) / (std + 1e-6)
#     # def normalize(self, x):
#     # # x 形状: [B, N, T, F]
#     #     original_shape = x.shape
#     #     F = original_shape[-1]
        
#     #     # 展平以便计算，只保留最后一维特征
#     #     flat_x = x.reshape(-1, F)
        
#     #     # 1. 计算中位数
#     #     median = torch.median(flat_x, dim=0).values # 形状: [8]
        
#     #     # 2. 计算分位数
#     #     # 注意: torch.quantile 在处理大规模数据时建议在 CPU 或 确保数据类型正确
#     #     q = torch.quantile(flat_x, torch.tensor([0.25, 0.75], device=x.device), dim=0)
#     #     iqr = q[1] - q[0] # 形状: [8]
        
#     #     # 3. 广播计算
#     #     # 将 median 和 iqr 调整为 [1, 1, 1, 8] 以便更清晰地与 [B, N, T, F] 对齐
#     #     # 虽然 PyTorch 会自动处理，但显式 reshape 增加可读性
#     #     median = median.view(1, 1, 1, F)
#     #     iqr = iqr.view(1, 1, 1, F)
        
#     #     return (x - median) / (iqr + 1e-6)
    
#     def forward(self, x):
#         x = self.normalize(x)

#         if self.training:
#             noise_level = 0.02 
#             x = x + torch.randn_like(x) * noise_level
#         #print(x.shape)
#         B,N, T, F = x.shape
#         x = x.reshape(B*N, -1)
        
#         x = self.fc1(x) 
#         res_outer = x  # 记录外层残差 (dim: 512)
#         x = self.dropout2(x)
#         # --- 第二层：降维 Block ---
#         x = self.norm1(x)  # Pre-Norm
#         x = self.gelu(x)
#         x = self.fc2(x)    # dim: (512, 256)
#         x = self.dropout3(x)
#         #res_inner = x  # 记录内层残差 ((512, 256))
        
#         # --- 第三层：中间瓶颈层 ---
#         # x = self.norm2(x)  # Pre-Norm
#         # x = self.fc5(x)    # dim: 128
#         # x = self.gelu(x)
#         # x = self.dropout2(x)
#         # x = self.bottleneck_res1(x)
#         # x = self.bottleneck_res2(x)
#         # --- 第四层：升维与内层残差相加 ---
#         # x = self.fc7(x)    # dim: (128, 256)
#         # x = x + res_inner  # 残差连接 (256 + 256)
        
#         # # --- 第五层：升维与外层残差相加 ---
#         x = self.norm7(x)  # Pre-Norm
#         x = self.gelu(x)
#         x = self.fc8(x)    # dim: 256 -> 512
#         x = x + res_outer  # 残差连接 (512 + 512)
        
#         # --- 第六层：输出映射 ---
#         x = self.norm8(x)  # 在最后一次映射前做一次归一化
#         x = self.gelu(x)
#         x = self.fc9(x)    # dim: 512 -> 288 (t_out)
#         c_pred = self.softplus(x) * self.c_scale
#         #c_pred = self.relu(x) * self.c_scale
#         #c_pred = torch.sigmoid(x) *self.c_scale
#         return c_pred

# class ResidualLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, dropout1=0.2, dropout2=0.2):
#         super(ResidualLayer, self).__init__()
#         self.linear1 = nn.Linear(input_dim, output_dim)
#         self.linear2 = nn.Linear(output_dim, output_dim)

#         #self.norm0 = nn.LayerNorm(input_dim)
#         self.norm1 = nn.LayerNorm(output_dim)
#         self.norm2 = nn.LayerNorm(output_dim)

#         self.relu = nn.ReLU()
#         self.gelu = nn.GELU()
#         self.dropout1 = nn.Dropout(dropout1)
#         self.dropout2 = nn.Dropout(dropout1)
#         self.residual_proj = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)

#     def forward(self, x): 
#         residual = self.residual_proj(x)
#         x = self.linear1(x)                        
#         x = self.norm1(x)
#         x = self.gelu(x)
#         x = self.dropout1(x)        
#         x = self.linear2(x)
#         x = self.dropout2(x)
#         x = x + residual #残差连接
#         x = self.norm2(x)
#         x = self.relu(x)
#         return x
#invalid_mask = (wspd < 1.0) | ((wspd > 3.0) & (patv <= 0))
def refine_wind(x, wspd_idx=2, patv_idx=-1):
    B, N, T, F = x.shape
    x_new = x.clone()
    wspd = x_new[:, :, :, wspd_idx]
    patv = x_new[:, :, :, patv_idx]

    device = wspd.device

    # 1. 定义参数
    v_start, v_end, step = 3.0, 12.0, 0.2
    # 计算 bin 的数量: (12.0 - 3.0) / 0.2 = 45
    num_bins = int((v_end - v_start) / step) + 1 

    # 2. 筛选 mask2 区域
    mask2 = (wspd >= v_start) & (wspd <= v_end)
    
    if mask2.any():
        # 计算每个点属于哪个 bin (索引从 0 到 44)
        # 映射公式: (wspd - 3.0) / 0.2
        bin_indices = torch.clamp(((wspd - v_start) / step).long(), 0, num_bins - 1)
        
        # 只针对 mask2 区域计算
        valid_bins = bin_indices[mask2]
        valid_patv = patv[mask2]

        # 3. 计算每个 bin 的均值 (替代 groupby.mean)
        # 使用 scatter_add_ 计算各 bin 总和与计数
        bin_sum = torch.zeros(num_bins, device=device)
        bin_count = torch.zeros(num_bins, device=device)
        
        bin_sum.scatter_add_(0, valid_bins, valid_patv)
        bin_count.scatter_add_(0, valid_bins, torch.ones_like(valid_patv))
        
        # 避免除以 0
        bin_mean = bin_sum / bin_count.clamp(min=1.0)

        # 4. 计算分位数（分桶全局或局部）
        # 注意：在 GPU 上对每个 bin 独立算分位数较复杂
        # 建议先用全局分位数或根据逻辑简化的 bin 分位数
        q_low = torch.quantile(valid_patv, 0.1)
        q_high = torch.quantile(valid_patv, 0.8)

        # 5. 替换逻辑
        # 映射回原形状
        batch_mean = bin_mean[bin_indices]
        
        # 异常定义：在 mask2 范围内，且超出分位数区间
        replace_mask = mask2 & ((patv < q_low) | (patv > q_high))
        
        # 执行替换：将异常值替换为对应 bin 的均值
        patv = torch.where(replace_mask, batch_mean, patv)
#####################################################################
    low_wind_mask = (wspd < 1.0)
    patv[low_wind_mask] = 0.0
    low_patv_mask = (patv < 0.0)
    patv[low_patv_mask] = 0.0
######################################################################
    mask1 = (wspd >= 1.0) & (wspd < 3.0)
    if mask1.any():
        p_range1 = patv[mask1]
        q11 = torch.quantile(p_range1, 0.1)
        q71 = torch.quantile(p_range1, 0.7)
        normal_mask1 = (p_range1 >= q11) & (p_range1 <= q71)
        if normal_mask1.any():
            mean1 = p_range1[normal_mask1].mean()
        else:
            mean1 = p_range1.mean() # 保底逻辑
        replace_mask1 = mask1 & ((patv < q11) | (patv > q71))
        patv = torch.where(replace_mask1, mean1, patv)
# #######################################################################
#     mask2 = (wspd >= 3.0) & (wspd < 4.0)
#     p_range2 = patv[mask2]
#     q22 = torch.quantile(p_range2, 0.2)
#     q92 = torch.quantile(p_range2, 0.9)
#     # 找到 20% 到 60% 之间的点并计算均值
#     normal_mask2 = (p_range2 >= q22) & (p_range2 <= q92)
#     if normal_mask2.any():
#         mean2 = p_range2[normal_mask2].mean()
#     else:
#         mean2 = p_range2.mean() # 保底逻辑  
#     replace_mask2 = mask2 & ((patv < q22) | (patv > q92))
#     patv = torch.where(replace_mask2, mean2, patv)
# #######################################################################       
#     mask3 = (wspd >= 4.0) & (wspd <=12.0)
#     if mask3.any():
#         p_range3 = patv[mask3]
#         q13 = torch.quantile(p_range3, 0.1)
#         q93 = torch.quantile(p_range3, 0.9)
#         normal_mask3 = (p_range3 >= q13) & (p_range3 <= max_val)
#         if normal_mask3.any():
#             mean3 = p_range3[normal_mask3].mean()
#         else:
#             mean3 = p_range3.mean() # 保底逻辑
#         #print("mean1: ", mean1)    
#         # 替换在该区间内，但不处于 [q20, q60] 范围内的点
#         replace_mask3 = mask3 & ((patv < q13) | (patv >=max_val))
#         patv = torch.where(replace_mask3, mean3, patv)
    # mask2 = (wspd > 3.0) & (wspd <= 4.0)
    # normal2 = patv[mask2 & (patv < 100.0)]
    # mean2 = normal2.mean() if normal2.numel() > 0 else 50.0 # 给个合理的默认值
    # patv[mask2 & (patv >= 100.0)] = mean2
    x_new[:, :, :, patv_idx] = patv
    return x_new
def air_density_etmp_positive(etmp_c: torch.Tensor,
                              p0_pa: float = 101325.0,
                              R: float = 287.05) -> torch.Tensor:
    T_k = etmp_c + 273.15
    rho_full = p0_pa / (R * T_k)
    #mask = (etmp_c > 0).float()
    # Etmp<=0 的位置 rho = 0
    rho = rho_full 
    return rho   
# def prepare_input(P, V,P_min,V_min, etmp):
#     # 1. 物理限制：防止分母过小导致的爆炸
#     V_safe = torch.clamp(V, min=V_min) # 假设 1m/s 以下视为无功
#     P_safe = torch.clamp(P, min=P_min)
#     # V_safe = V # 假设 1m/s 以下视为无功
#     # P_safe = P
#     ad = air_density_etmp_positive(etmp)
#     #print('ad ',ad.min())4
#     # 2. 计算 C (加入物理上限，例如贝兹极限的 2 倍作为容错)
#     C_raw = P_safe / (ad * V_safe.pow(3) + 1e-6)
    # with torch.no_grad():
    #     max_idx = torch.argmax(C_raw)
    #     c_max_val = C_raw.view(-1)[max_idx].item()
    #     v_at_max = V_safe.view(-1)[max_idx].item()
    #     p_at_max = P_safe.view(-1)[max_idx].item()
    #     ad_at_max = ad.view(-1)[max_idx].item()
        
    #     print("\n" + "="*40)
    #     print("DEBUG: C_raw 最大值点分析")
    #     print(f"C_max:      {c_max_val:.6f}")
    #     print(f"对应 V_safe: {v_at_max:.4f} m/s")
    #     print(f"对应 P_safe: {p_at_max:.4f}")
    #     print(f"对应 Air Density: {ad_at_max:.4f}")
    #     print("="*40 + "\n")

    #     min_idx = torch.argmin(C_raw)
    #     c_min_val = C_raw.view(-1)[min_idx].item()
    #     v_at_min = V_safe.view(-1)[min_idx].item()
    #     p_at_min = P_safe.view(-1)[min_idx].item()
    #     ad_at_min = ad.view(-1)[min_idx].item()

    #     print("\n" + "="*40)
    #     print("DEBUG: C_raw 最小值点分析")
    #     print(f"C_max:      {c_min_val:.6f}")
    #     print(f"对应 V_min: {v_at_min:.4f} m/s")
    #     print(f"对应 P_min: {p_at_min:.4f}")
    #     print(f"对应 Air Density: {ad_at_min:.4f}")
    # return C_raw
@torch.no_grad()    
def masked_mean_std(x: torch.Tensor, mask: torch.Tensor, dims=(0,1,2), eps=1e-6):
    if x.ndim == 4: x = x.squeeze(-1)
    if mask.ndim == 4: mask = mask.squeeze(-1)
    m = mask.float()
    denom = m.sum(dim=dims, keepdim=True).clamp_min(1.0)
    mean = (x * m).sum(dim=dims, keepdim=True) / denom
    var  = ((x - mean)**2 * m).sum(dim=dims, keepdim=True) / denom
    std  = torch.sqrt(var + eps)
    return mean, std

def prepare_input(
    P_kw: torch.Tensor,
    V_ms: torch.Tensor,
    patv_min: torch.Tensor,
    V_min: torch.Tensor,
    etmp_c: torch.Tensor,
    p_low_frac: float = 0.0,
    p_high_frac: float = 0.95,
    eps_rel: float = 1e-6,
):

    # 统一形状到广播友好
    P_kw = P_kw.squeeze(-1) if P_kw.dim() > 3 and P_kw.size(-1) == 1 else P_kw
    V_ms = V_ms.squeeze(-1) if V_ms.dim() > 3 and V_ms.size(-1) == 1 else V_ms
    etmp_c = etmp_c.squeeze(-1) if etmp_c.dim() > 3 and etmp_c.size(-1) == 1 else etmp_c

    rho = air_density_etmp_positive(etmp_c)  # kg/m^3
    p_low = 0
    p_high = 1550000
    mask_phys = (V_ms >= V_min) & (P_kw > p_low) & (P_kw <= p_high)

    # 分母：rho * V^3
    den = rho * V_ms.pow(3)

    # 相对epsilon：用 den 的均值做缩放，更稳
    den_mean = den.detach().mean().clamp_min(1.0)
    eps = eps_rel * den_mean

    K_eff = torch.zeros_like(P_kw)
    K_eff[mask_phys] = P_kw[mask_phys] / (den[mask_phys] + eps)
    mask2 = (V_ms <= 1)
    K_eff[mask2] = 0
    K_eff = torch.clamp(K_eff, 0.0, None)
    # with torch.no_grad():
    #     flat_idx = torch.nonzero(mask_phys.reshape(-1), as_tuple=False).squeeze(1)  # [M]
    #     K_valid = K_eff.reshape(-1)[flat_idx]

    #     # max/min 在有效集合里的位置
    #     i_max = torch.argmax(K_valid)
    #     i_min = torch.argmin(K_valid)

    #     idx_max = flat_idx[i_max].item()
    #     idx_min = flat_idx[i_min].item()

    #     print("\n" + "="*40)
    #     print("DEBUG: K_eff(=P/(rho V^3)) 最大值点分析")
    #     print(f"K_max: {K_eff.reshape(-1)[idx_max].item():.6f}")
    #     print(f"V:     {V_ms.reshape(-1)[idx_max].item():.4f} m/s")
    #     print(f"P:     {P_kw.reshape(-1)[idx_max].item():.1f} W")
    #     print(f"rho:   {rho.reshape(-1)[idx_max].item():.4f} kg/m^3")
    #     print("="*40)

    #     print("\n" + "="*40)
    #     print("DEBUG: K_eff 最小值点分析（有效集合内）")
    #     print(f"K_min: {K_eff.reshape(-1)[idx_min].item():.6f}")
    #     print(f"V:     {V_ms.reshape(-1)[idx_min].item():.4f} m/s")
    #     print(f"P:     {P_kw.reshape(-1)[idx_min].item():.1f} W")
    #     print(f"rho:   {rho.reshape(-1)[idx_min].item():.4f} kg/m^3")
    #     print("="*40)
    return K_eff, mask_phys
class WPFModel(nn.Module):
    """Models for Wind Power Prediction
    """
    def __init__(self, config,input_dim=128,output_dim=144,out_dim2=134,out_dim3=288):
        super(WPFModel, self).__init__()
        self.config_file = config
        self.var_len = config.var_len
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.hidden_dims = config.hidden_dims
        self.capacity = config.capacity

        #self.cp_predictor = fnn() 
        #self.cp_predictor = PINNModel()
        #self.cp_predictor = bf()
        self.cp_predictor = wc()
        #self.cp_predictor =ct()
        self.grid_size = getattr(config, 'grid_size', 3)
        self.spline_order = getattr(config, 'spline_order', 2)
        # 应用权重初始化
        self.decomp = SeriesDecomp(DECOMP)

        self.pos_dec_emb = nn.Parameter(torch.empty(1, self.input_len + self.output_len, self.hidden_dims),requires_grad=True)
        nn.init.normal_(self.pos_dec_emb, mean=0.0, std=0.02)

        self.pos_emb = nn.Parameter(torch.empty(1, self.input_len, self.hidden_dims), requires_grad=True)# （可选）显式初始化（与 Paddle 默认一致，正态分布 μ=0, σ=1）
       # self.pos_emb2 = nn.Parameter(torch.empty(1, self.output_len, self.hidden_dims), requires_grad=True)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        
        self.st_conv_encoder = SpatialTemporalConv(self.capacity, self.var_len,self.hidden_dims)#capacity 134 var_len 10 hidden_dims 128
        #self.st_conv_encoder2 = SpatialTemporalConv(self.capacity, 5,self.hidden_dims)
        self.st_conv_decoder = SpatialTemporalConv(self.capacity, self.var_len,self.hidden_dims)

        self.enc = Encoder(config)
        self.dec = Decoder(config)
        self.batch_switch = KANLinear(
            self.hidden_dims, 
            self.capacity, 
            grid_size=self.grid_size, 
            spline_order=self.spline_order
        )
        #self.batch_switch = nn.Linear(self.hidden_dims, self.capacity)
        #self.pred_nn = nn.Linear(self.hidden_dims, self.capacity)
        self.pred_nn = KANLinear(
            self.hidden_dims, #128
            self.capacity, #134
            grid_size=self.grid_size, 
            spline_order=self.spline_order
        )
        self.apply(self.init_weights)
        ####################################################################
        self.p = 1.225
        self.relu = nn.ReLU()
        ##############################################################
    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # PyTorch 中使用 nn.init 模块进行权重初始化
            if hasattr(layer, 'weight') and isinstance(layer.weight, torch.Tensor):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            # 如果有偏置项，也可以初始化
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)
                
        elif isinstance(layer, nn.LayerNorm):
            layer.eps = 1e-12  
    
    def forward(self, batch_x, batch_y, data_mean, data_scale,select_batch_xx,batch_yy, graph=None):
        
        bz, id_len, input_len, var_len = batch_x.shape#32,134,144,7
        batch_graph = dgl.batch([graph] * bz)
        bz1, id_len1, output_len, var_len1 = batch_y.shape     
        #batch_x  = refine_wind(batch_x)   
        batch_x1 = (batch_x - data_mean) / data_scale
        batch_x1 = batch_x1.permute(0, 2, 1, 3)
       # print('batch_x1.max(),batch_x1.min() ',batch_x1.max(),batch_x1.min())
        batch_pred_trend = torch.mean(batch_x1, dim=1, keepdim=True)[:, :, :, -1]
        batch_pred_trend = batch_pred_trend.repeat(1, output_len, 1)
        batch_pred_trend = torch.cat(
            [self.decomp(batch_x1[:, :, :, -1])[0], batch_pred_trend], dim=1)# batch_x (32,144,134,7) 
        ########################################[0]
        #print('batch_pred_trend.shape ',batch_pred_trend.shape)  
        batch_x1 = batch_x1.reshape(bz, input_len, var_len * id_len)#32,144,134*7
        _,season_init = self.decomp(batch_x1) #, season_init torch.Size([32, 144, 938]) torch.Size([32, 144, 938])
       
        batch_pred_season = torch.zeros(
            [bz, output_len, var_len * id_len], dtype=torch.float32)#32,288,134*7
        #print('season_init.shape,batch_pred_season.shape ',season_init.shape,batch_pred_season.shape)#([32, 144, 938]) torch.Size([32, 288, 938])
        device = next(self.parameters()).device
        season_init = season_init.to(device)
        #season_init0 = season_init0.to(device)
        batch_pred_season = batch_pred_season.to(device)
        batch_pred_season = torch.cat([season_init, batch_pred_season], dim=1)

        batch_x1 = self.st_conv_encoder(batch_x1, batch_graph) + self.pos_emb
       # print('batch_x1.shape ',batch_x1.shape)([32, 144, 128])
        batch_x1 = self.enc(batch_x1)   
        #print('batch_x1.shape2 ',batch_x1.shape)#([32, 144, 128])
        #####################################################################                 
        batch_pred_season = self.st_conv_decoder(
            batch_pred_season, batch_graph) + self.pos_dec_emb
        batch_x_pred, batch_x_trends = self.dec(batch_pred_season, batch_pred_trend, batch_x1)
       # print('batch_x_pred.shape ',batch_x_pred.shape)#batch_x_pred.shape  torch.Size([32, 432, 128])
        #print('batch_x_trends.shape ',batch_x_trends.shape)#batch_x_pred.shape  torch.Size([32, 432,134])
        ###################################################################       
        batch_x_pred = self.pred_nn(batch_x_pred)
       # print('batch_x_pred.shape ',batch_x_pred.shape)#432,32,134
       #print('batch_x_trends.shape ',batch_x_trends.shape)#432,32,134
        pred_y = batch_x_pred + batch_x_trends  
        pred_y = pred_y.permute(0, 2, 1)
        #print('pred_y  ',pred_y.shape)([32, 134, 432])
        pred_y1  = pred_y[:, :, -output_len:]#32,134,288 
        pred_y12 = pred_y[:, :, :input_len]      
        pred_y12 = pred_y12.unsqueeze(-1)
        pred_y12 = pred_y12.detach()
        # ########################################### 
        #select_batch_xx = impute_invalid_data(select_batch_xx)  
        # weekday = select_batch_xx[:,:,:,0]  
        # time    = select_batch_xx[:,:,:,1]  
        #select_batch_xx = aug_xx 
        wspd1   = select_batch_xx[:,:,:,2]#32,134,144
        etmp2   = select_batch_xx[:,:,:,3]  
        
        #patv2   = batch_yy[:,:,:,-1]
        #max_val = torch.maximum(patv1, patv2).max()
        #select_batch_xx = refine_wind(select_batch_xx)
        select_batch_xx = select_batch_xx.clone()
        select_batch_xx[:, :, :, -1] *= 1000
        patv1   = select_batch_xx[:,:,:,-1]

        wspd1_min = 2.5
        patv_min = 0 
        etmp2 = torch.clamp(etmp2, min=0.0)
        C_hist, mask_phys = prepare_input(patv1, wspd1, patv_min, wspd1_min, etmp2)
        #print('C_hist.max,C_hist.min ',C_hist.max(),C_hist.min())
        # masked mean/std（如果你要标准化 C_hist）
        c_mean, c_std = masked_mean_std(C_hist, mask_phys)   # dims=(0,1,2)
        C_hist_norm = (C_hist - c_mean) / (c_std + 1e-6)
       # print('c_mean: ',c_mean)
        C_hist_norm = C_hist_norm.unsqueeze(-1)
        input_x = torch.cat([select_batch_xx, C_hist_norm,pred_y12], dim=-1)#7+1+1
        #input_x = torch.cat([select_batch_xx, C_hist_norm], dim=-1)
        bz2,var_len2,input_len2 ,feature2= select_batch_xx.shape#[32,134,144,8]
        cp = self.cp_predictor(input_x)
        cp = F.softplus(cp*c_std+c_mean)
        #cp = self.relu(cp*c_std+c_mean)
        #cp = self.cp_predictor(select_batch_xx)
       # cp = cp.reshape(bz2,var_len2,output_len)#32,134,288
        print('Cp_max_min ',cp.max(),cp.min())
        #######################################
        #pred_y2 = pred_y[:, :, -output_len:]
       # input_x = self.normalize(input_x)
        #cp = self.cp_predictor(input_x)
        # cp = cp.permute(0, 1, 3, 2)
        # cp = cp[:, :,:,-1]
        ###################################
        #pred_y2= self.normalize(pred_y2)
        torch.set_printoptions(precision=4, sci_mode=False)
        #cp = self.cp_predictor(pred_y2)  # 输入为综合特征
    

       # print('###############################################')
       # print('cp[3,100,:10] ',cp[3,110,:10])
        #print(cp.shape)
        wspd  = batch_yy[:,:,:,2]#32,134,288
        #patv2 = batch_yy[:,:,:,-1]
        
        #print('patv2.shape ',patv2.shape)
        #     print("="*40 + "\n")
        etmp = batch_yy[:,:,:,4]
        etmp = torch.clamp(etmp, min=0.0)
        rho = air_density_etmp_positive(etmp)
       # eps = 1e-6
        #cp_test = patv2/(rho * wspd**3+eps)
        # with torch.no_grad():
        #     max_idx = torch.argmax(cp)           
        #     c_max_val = cp.view(-1)[max_idx].item()
        #     v_at_max = wspd.view(-1)[max_idx].item()

        #     min_idx = torch.argmin(cp)           
        #     c_min_val = cp.view(-1)[min_idx].item()
        #     v_at_min = wspd.view(-1)[min_idx].item()
        #     print("\n" + "="*40)
        #     print("DEBUG: cp_test 最大值点分析")
        #     print(f"cp_test:      {c_max_val:.6f}")
        #     print(f"对应 wspd: {v_at_max:.4f} m/s")
        #     print("DEBUG: cp_test 最小值点分析")
        #     print(f"cp_test:      {c_min_val:.6f}")
        #     print(f"对应 wspd2: {v_at_min:.4f} m/s")
        #     print("="*40 + "\n")
        pinn_patv = 1e-3* rho * cp * wspd**3
        pinn_patv = torch.clamp(pinn_patv, min=0.0, max=1550.0)
        #pinn_patv = 0.5 * self.p * cp * wspd**3 
        return pred_y1,pinn_patv

        ###################################
        
    # def normalize(self,x):
    #     mean = x.mean()
    #     std = x.std()
    #     return (x - mean) / (std + 1e-8)
    # def normalize_bn_t(self,x, eps=1e-6, std_min=0.3):
    #     mean = x.mean(dim=-1, keepdim=True)   # [B,N,1]
    #     std  = x.std(dim=-1, keepdim=True, unbiased=False)  # 先别用 unbiased=True
    #     std  = std.clamp_min(std_min)                       # 关键：避免除以极小数
    #     return (x - mean) / (std + eps)
