import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class WindPowerDataset(Dataset):
    """
    WPF / SDWPF 风电时空预测数据集加载器
    """
    def __init__(self, dataframe, seq_len=144, pred_len=72, target_col='Passable_Power'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 提取特征和目标值
        self.data = dataframe.to_numpy(dtype=np.float32)
        self.target_idx = dataframe.columns.get_loc(target_col)
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        # 历史观测序列 (X)
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data[s_begin:s_end] # [Seq_len, Num_features]
        # 未来真实的功率标签 (Y)
        seq_y = self.data[r_begin:r_end, self.target_idx] # [Pred_len]
        
        # 为了物理信息约束，这里同时返回未来的风速特征（假设风速在特征的前几列）
        # 用于计算空气动力学方程残差
        future_features = self.data[r_begin:r_end] # [Pred_len, Num_features]
        
        return torch.tensor(seq_x), torch.tensor(seq_y), torch.tensor(future_features)

def get_dataloader(file_path, batch_size, seq_len, pred_len):
    df = pd.read_csv(file_path)
    # 此处省略缺失值填充、归一化等预处理步骤...
    dataset = WindPowerDataset(df, seq_len=seq_len, pred_len=pred_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader