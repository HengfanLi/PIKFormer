# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.data import Data, Batch
import dgl
import dgl.function as fn
WIN = 3
DECOMP = 24


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        t_x = torch.transpose(x, 1, 2)
        # 手动计算填充值
        padding = (self.kernel_size - 1) // 2
        mean_x = F.avg_pool1d(
            t_x, self.kernel_size, stride=1, padding=padding)
        mean_x = torch.transpose(mean_x, 1, 2)

        # 检查 x 和 mean_x 的尺寸是否一致
        if x.size(1) != mean_x.size(1):
            # 调整 mean_x 的尺寸使其与 x 匹配
            if x.size(1) > mean_x.size(1):
                # 如果 x 的尺寸大于 mean_x，对 mean_x 进行填充
                diff = x.size(1) - mean_x.size(1)
                pad = (0, 0, 0, diff)
                mean_x = F.pad(mean_x, pad, "constant", 0)
            else:
                # 如果 x 的尺寸小于 mean_x，对 mean_x 进行裁剪
                mean_x = mean_x[:, :x.size(1), :]

        return x - mean_x, mean_x
# class SeriesDecomp(nn.Module):
#     def __init__(self, kernel_size):
#         super().__init__()
#         self.kernel_size = kernel_size

#     def forward(self, x):
#         t_x = torch.transpose(x, 1, 2)
#         padding = (self.kernel_size - 1) // 2
#         mean_x = F.avg_pool1d(
#             t_x, self.kernel_size, stride=1, padding=padding)
#         mean_x = torch.transpose(mean_x, 1, 2)
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
                 trends_out=134):
        super(TransformerDecoderLayer, self).__init__()

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout

        self.decomp = SeriesDecomp(DECOMP)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.linear1 = nn.Linear(d_model, dims_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(dims_feedforward, d_model)

        self.linear_trend = nn.Conv1d(
            d_model, trends_out, WIN, padding="same")

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, src, memory, src_mask=None, cache=None):
        residual = src
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
        # Change shape from [batch, seq_len, channels] to [batch, channels, seq_len]
        res_trend = torch.transpose(res_trend, 1, 2)
        res_trend = self.linear_trend(res_trend)
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
                 act_dropout=None):
        super(TransformerEncoderLayer, self).__init__()

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout

        self.decomp = SeriesDecomp(DECOMP)

        self.self_attn = nn.MultiheadAttention(d_model, nhead)

        self.linear1 = nn.Linear(d_model, dims_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(dims_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, src, src_mask=None, cache=None):
        residual = src
        src, _ = self.self_attn(src, src, src, None)
        src = residual + self.dropout1(src)

        src, _ = self.decomp(src)

        residual = src

        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)

        src, _ = self.decomp(src)
        return src
################


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
                    dims_feedforward=self.hidden_dims * 2))

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
                    trends_out=self.capacity))

    def forward(self, season, trend, enc_output):
        for lin in self.dec_lins:
            season, trend_part = lin(season, enc_output)
            trend = trend + trend_part
            #print('season ',season)
            #print('trend ',trend)
        return season, trend


# class SpatialTemporalConv(nn.Module):
#     """ Spatial Temporal Embedding
#     Apply GAT and Conv1D based on Temporal and Spatial Correlation
#     """

#     def __init__(self, id_len, input_dim, output_dim):
#         super(SpatialTemporalConv, self).__init__()
#         self.conv1 = nn.Conv1d(
#             id_len * input_dim,
#             output_dim,
#             kernel_size=WIN,
#             padding="same",
#             bias=False)
#         self.id_len = id_len
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.q = nn.Linear(input_dim, output_dim)
#         self.k = nn.Linear(input_dim, output_dim)

#     def _send_attention(self, edges):
#         alpha = edges.src['k'] * edges.dst['q']
#         alpha = torch.sum(alpha, dim=-1, keepdim=True)
#         return {'alpha': alpha, 'output_series': edges.src['v']}

#     def _reduce_attention(self, nodes):
#         alpha = torch.softmax(nodes.mailbox['alpha'], dim=1)
#         return {'output': torch.sum(nodes.mailbox['output_series'] * alpha, dim=1)}

#     def forward(self, x, graph):
#         device = next(self.parameters()).device
#         x = x.to(device)
#         bz, seqlen, total_dim = x.shape
#         # 计算实际的 input_dim
#         actual_input_dim = total_dim // self.id_len
        
#         # 使用计算得到的 input_dim 进行重塑
#         x = x.reshape(bz, seqlen, self.id_len, actual_input_dim)
#         x = x.permute(0, 2, 1, 3)
#         x = x.reshape(-1, seqlen, actual_input_dim)
        
#         mean_x = torch.mean(x, dim=1)
#         mean_x = mean_x.float()
#         q_x = self.q(mean_x) / math.sqrt(self.output_dim)
#         k_x = self.k(mean_x)
#         x = x.reshape(-1, seqlen * actual_input_dim)
        
#         graph = graph.to(k_x.device)
#         graph.ndata['k'] = k_x
#         graph.ndata['q'] = q_x
#         graph.ndata['v'] = x

#         graph.apply_edges(self._send_attention)
#         graph.update_all(message_func=lambda edges: {'alpha': edges.data['alpha'], 'output_series': edges.src['v']},
#                          reduce_func=self._reduce_attention)
        
#         output = graph.ndata['output']
#         x = output.reshape(bz, self.id_len, seqlen, actual_input_dim)
#         x = x.permute(0, 2, 1, 3)
#         x = x.reshape(bz, seqlen, self.id_len * actual_input_dim)
        
#         # PyTorch's Conv1d expects input shape (N, C, L)
#         x = x.float()
#         x = x.transpose(1, 2)
#         x = self.conv1(x)
#         x = x.transpose(1, 2)  # Back to (N, L, C)
#         return x
class SpatialTemporalConv(nn.Module):

    def __init__(self, id_len, input_dim, output_dim):
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
        self.q = nn.Linear(input_dim, output_dim)
        self.k = nn.Linear(input_dim, output_dim)

        self.norm = nn.LayerNorm(input_dim, eps=1e-5)
       # nn.init.uniform_(self.q.weight, -0.01, 0.01)  # 更保守的初始化
       # nn.init.zeros_(self.q.bias)
        #nn.init.uniform_(self.k.weight, -0.01, 0.01)  # 更保守的初始化
       # nn.init.zeros_(self.k.bias)
        # nn.init.xavier_normal_(self.q.weight)  # 显式初始化
        # nn.init.xavier_normal_(self.k.weight)
        # nn.init.constant_(self.q.weight, 1e-3)  # 使用常数初始化
        # nn.init.constant_(self.k.weight, 1e-3)
        #nn.init.xavier_uniform_(self.q.weight)
        nn.init.zeros_(self.q.bias)  # 偏置初始化为0
       # nn.init.xavier_uniform_(self.k.weight)
        nn.init.zeros_(self.k.bias)  # 偏置初始化为0
        nn.init.normal_(self.q.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.k.weight, mean=0.0, std=0.01)

        # nn.init.uniform_(self.q.weight, -0.01, 0.01)
        # nn.init.uniform_(self.k.weight, -0.01, 0.01)
        # nn.init.xavier_normal_(self.q.weight, gain=0.01)
        # nn.init.xavier_normal_(self.k.weight, gain=0.01)
        # 将偏置初始化为零
        # if self.q.bias is not None:
        #     nn.init.zeros_(self.q.bias)
        # if self.k.bias is not None:
        #     nn.init.zeros_(self.k.bias)
    def _send_attention(self, edges):
        alpha = edges.src['k'] * edges.dst['q']
        a= edges.src['k']
        b=edges.dst['q']
        #print('a ',a[100])
        #print('b ',b)
        alpha = torch.sum(alpha, dim=-1, keepdim=True)
        if torch.isnan(edges.src['k']).any() or torch.isnan(edges.dst['q']).any():
            print("edges.src['k'] NaN detected in attention computation")

        return {"alpha": alpha, "output_series": edges.src["v"]}
      
    def _reduce_attention(self, nodes):
        alpha = F.softmax(nodes.mailbox["alpha"], dim=1)
        output = torch.sum(nodes.mailbox["output_series"] * alpha, dim=1)
        return {"output": output}
        # alpha = F.softmax(torch.nan_to_num(nodes.mailbox['alpha'], nan=-1e9), dim=1)
        # v = torch.nan_to_num(nodes.mailbox['v'], nan=0.0)
        # return {'output': (alpha * v).sum(dim=1)}
   
    def forward(self, x, graph):
        bz, seqlen, _ = x.shape
        x = x.view(bz, seqlen, self.id_len, self.input_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(-1, seqlen, self.input_dim)
         # 新增数据验证
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     print("输入 x 包含 NaN 或 Inf 值")
        #     x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        mean_x = x.mean(dim=1)  # (bz*id_len, input_dim)
        #mean_x = self.norm(mean_x)
        #mean_x = mean_x.to(torch.float32)  # 确保使用float32精度
       # print("mean_x range:", mean_x.min().item(), mean_x.max().item())
        #mean_x = torch.clamp(mean_x, min=-3.0, max=3.0)
       # print("mean_x range2:", mean_x.min().item(), mean_x.max().item())
        #mean_x = self.norm(mean_x)
        # if torch.isnan(mean_x).any() or torch.isinf(mean_x).any():
        #     print("mean_x contains NaN or Inf before passing to self.q")
        #     mean_x = torch.nan_to_num(mean_x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        q_x = self.q(mean_x)
        # if torch.isnan(q_x).any() or torch.isinf(q_x).any():
        #       print("q_x contains NaN or Inf before passing to self.q")
       # print('q_x ',q_x)# nan #print("q_x range:", q_x.min().item(), q_x.max().item())  # 检查输出范围       
        scaling_factor = max(1.0, math.sqrt(float(self.output_dim)+ 1e-6))
        q_x = q_x /math.sqrt(self.output_dim)
        
        k_x = self.k(mean_x)
       # print("k_x range:", k_x.min().item(), k_x.max().item())
        x = x.view(-1, seqlen * self.input_dim)
        
       # print('k_x ',k_x)# nan
       # print('x ',x)
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

# class SpatialTemporalConv(nn.Module):
#     """ Spatial Temporal Embedding

#     Apply GAT and Conv1D based on Temporal and Spatial Correlation
#     """

#     def __init__(self, id_len, input_dim, output_dim, win=3):
#         super(SpatialTemporalConv, self).__init__()
#         self.WIN = win
#         self.conv1 = nn.Conv1d(
#             id_len * input_dim,
#             output_dim,
#             kernel_size=self.WIN,
#             padding="same",  # "SAME" padding
#             bias=False)
#         self.id_len = id_len
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
#         # Initialize linear layers with Xavier initialization to prevent NaN values
#         self.q = nn.Linear(input_dim, output_dim)
#         self.k = nn.Linear(input_dim, output_dim)
        
#         # Initialize weights to prevent NaN outputs
#         nn.init.xavier_uniform_(self.q.weight)
#         nn.init.xavier_uniform_(self.k.weight)
#         if self.q.bias is not None:
#             nn.init.zeros_(self.q.bias)
#         if self.k.bias is not None:
#             nn.init.zeros_(self.k.bias)

#     def forward(self, x, graph):
#         bz, seqlen, _ = x.shape
#         x = x.reshape(bz, seqlen, self.id_len, self.input_dim)
#         x = x.permute(0, 2, 1, 3)
#         x = x.reshape(-1, seqlen, self.input_dim)
        
#         # Calculate mean across sequence length
#         mean_x = torch.mean(x, dim=1)
        
#         # Apply linear transformations with NaN prevention
#         q_x = self.q(mean_x) / math.sqrt(self.output_dim)
#         k_x = self.k(mean_x)
        
#         # Reshape for message passing
#         x_reshaped = x.reshape(-1, seqlen * self.input_dim)
        
#         # DGL message passing
#         with graph.local_scope():
#             # Set node features
#             graph = graph.to(k_x.device)
#             graph.ndata['q'] = q_x
#             graph.ndata['k'] = k_x
#             graph.ndata['v'] = x_reshaped
            
#             # Compute attention scores
#             graph.apply_edges(lambda edges: {'alpha': torch.sum(edges.src['k'] * edges.dst['q'], dim=-1, keepdim=True)})
            
#             # Normalize attention scores using softmax
#             graph.edata['alpha'] = dgl.nn.functional.edge_softmax(graph, graph.edata['alpha'])
            
#             # Message passing
#             graph.update_all(
#                 message_func=dgl.function.u_mul_e('v', 'alpha', 'm'),
#                 reduce_func=dgl.function.sum('m', 'output')
#             )
            
#             # Get output
#             output = graph.ndata['output']
        
#         # Reshape output
#         x = output.reshape(bz, self.id_len, seqlen, self.input_dim)
#         x = x.permute(0, 2, 1, 3)
#         x = x.reshape(bz, seqlen, self.id_len * self.input_dim)
        
#         # Apply 1D convolution (need to transpose for PyTorch's Conv1d)
#         x = x.transpose(1, 2)  # [bz, channels, seq_len]
#         x = self.conv1(x)
#         x = x.transpose(1, 2)  # [bz, seq_len, output_dim]
        
#         return x
    
#     def _message_func(self, edges):
#         # Compute attention scores
#         alpha = edges.src['k'] * edges.dst['q']
#         alpha = torch.sum(alpha, dim=-1, keepdim=True)
#         return {'alpha': alpha, 'output_series': edges.src['v']}
    
#     def _message_aggregation(self, edges):
#         return {'alpha': edges.data['alpha'], 'output_series': edges.src['v']}
    
#     def _reduce_func(self, nodes):
#         # Weighted sum based on attention scores
#         return {'output': nodes.mailbox['output_series'] * nodes.mailbox['alpha']}

class WPFModel(nn.Module):
    """Models for Wind Power Prediction
    """

    def __init__(self, config):
        super(WPFModel, self).__init__()
        self.config_file = config
        self.var_len = config.var_len
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.hidden_dims = config.hidden_dims
        self.capacity = config.capacity
        #self.norm = nn.LayerNorm(hidden_size, eps=1e-12)  # 直接在创建时设置 eps
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 应用权重初始化
        self.decomp = SeriesDecomp(DECOMP)

        self.t_emb = nn.Embedding(300, self.hidden_dims)
        self.w_emb = nn.Embedding(300, self.hidden_dims)

        self.t_dec_emb = nn.Embedding(300, self.hidden_dims)
        self.w_dec_emb = nn.Embedding(300, self.hidden_dims)

        self.pos_dec_emb = nn.Parameter(torch.empty(1, self.input_len + self.output_len, self.hidden_dims),requires_grad=True)
        nn.init.normal_(self.pos_dec_emb, mean=0.0, std=0.02)
        #self.pos_emb = nn.Parameter(torch.empty(1, self.input_len, self.hidden_dims),requires_grad=True)
        #self.pos_emb = nn.Parameter(torch.zeros(1, self.input_len, self.hidden_dims, dtype=torch.float32))
        self.pos_emb = nn.Parameter(torch.empty(1, self.input_len, self.hidden_dims), requires_grad=True)# （可选）显式初始化（与 Paddle 默认一致，正态分布 μ=0, σ=1）
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        self.st_conv_encoder = SpatialTemporalConv(self.capacity, self.var_len,self.hidden_dims)#capacity 134 var_len 10 hidden_dims 128
        self.st_conv_decoder = SpatialTemporalConv(self.capacity, self.var_len,self.hidden_dims)

        self.enc = Encoder(config)
        self.dec = Decoder(config)

        self.pred_nn = nn.Linear(self.hidden_dims, self.capacity)
        self.apply(self.init_weights)
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
            # PyTorch 中 LayerNorm 的 epsilon 是通过 eps 参数设置的
            # 需要在创建 LayerNorm 时设置，而不是这样直接修改
            # 如果已经创建了 LayerNorm，可以这样修改
            layer.eps = 1e-12

    # def init_weights(self, layer):
    #     """ Initialization hook """
    #     if isinstance(layer, (nn.Linear, nn.Embedding)):
    #         nn.init.normal_(layer.weight, mean=0.0, std=0.02)
    #     elif isinstance(layer, nn.LayerNorm):
    #         layer.eps = 1e-12
    
    def forward(self, batch_x, batch_y, data_mean, data_scale, graph=None):
        bz, id_len, input_len, var_len = batch_x.shape
        
       #print("pos_emb初始化值：", self.pos_emb.mean(), self.pos_emb.std())
        # 这里假设 graph 可以处理成适合 PyTorch Geometric 的格式
        # 由于不清楚 pgl.Graph.batch 的具体功能，这里无法完全准确替换
        # 假设已经有适合 PyTorch 的批量处理方式
        # 你需要根据实际情况修改
        #batch_graph = Batch.from_data_list([graph] * bz)
        batch_graph = dgl.batch([graph] * bz)
        _, _, output_len, _ = batch_y.shape
        var_len = var_len - 2

        time_id = batch_x[:, 0, :, 1].long()
        weekday_id = batch_x[:, 0, :, 0].long()

        batch_x = batch_x[:, :, :, 2:]
        batch_x = (batch_x - data_mean) / data_scale

        y_weekday_id = batch_y[:, 0, :, 0].long()
        y_time_id = batch_y[:, 0, :, 1].long()

        # batch_x_time_emb = self.t_emb(time_id)
        # batch_y_time_emb = self.t_dec_emb(
        #     torch.cat([time_id, y_time_id], dim=1))

        # batch_x_weekday_emb = self.w_emb(weekday_id)
        # batch_y_weekday_emb = self.w_dec_emb(
        #     torch.cat([weekday_id, y_weekday_id], dim=1))

        batch_x = batch_x.permute(0, 2, 1, 3)

        batch_pred_trend = torch.mean(batch_x, dim=1, keepdim=True)[:, :, :, -1]
        batch_pred_trend = batch_pred_trend.repeat(1, output_len, 1)
        batch_pred_trend = torch.cat(
            [self.decomp(batch_x[:, :, :, -1])[0], batch_pred_trend], dim=1)

        batch_x = batch_x.reshape(bz, input_len, var_len * id_len)
        _, season_init = self.decomp(batch_x)

        batch_pred_season = torch.zeros(
            [bz, output_len, var_len * id_len], dtype=torch.float32)
        device = next(self.parameters()).device
        season_init = season_init.to(device)
        batch_pred_season = batch_pred_season.to(device)
        batch_pred_season = torch.cat([season_init, batch_pred_season], dim=1)
        #print('batch_pred_season ',batch_pred_season) #yes
        # print('batch_x ',batch_x)
        batch_x = self.st_conv_encoder(batch_x, batch_graph) + self.pos_emb
        #print('tets1',batch_x)
        #print('self.pos_emb',self.pos_emb)
        batch_pred_season = self.st_conv_decoder(
            batch_pred_season, batch_graph) + self.pos_dec_emb
       # print('self.pos_dec_emb',self.pos_dec_emb)  
        #print('batch_pred_season ',batch_pred_season)
        #print('batch_x ',batch_x)
        #print('batch_pred_season ',batch_pred_season)
        batch_x = self.enc(batch_x)

        batch_x_pred, batch_x_trends = self.dec(batch_pred_season,
                                                batch_pred_trend, batch_x)
        batch_x_pred = self.pred_nn(batch_x_pred)
        #print('batch_x_pred,batch_x_trends ',batch_x_pred,batch_x_trends)
        pred_y = batch_x_pred + batch_x_trends
        
        pred_y = pred_y.permute(0, 2, 1)
        pred_y = pred_y[:, :, -output_len:]
        #print('pred_y ',pred_y)
        return pred_y
