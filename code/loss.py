
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ["FilterMSELOSS", "MSELoss", "HuberLoss", "MAELoss", "SmoothMSELoss", "FilterHuberLoss"]


class FilterMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(FilterMSELoss, self).__init__()

    def forward(self, pred, gold, raw, col_names):
        # Remove bad input
        cond1 = raw[:, :, :, col_names["Patv"]] < 0

        cond2 = raw[:, :, :, col_names["Pab1"]] > 89
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Pab2"]] > 89)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Pab3"]] > 89)

        cond2 = torch.logical_or(cond2,
                               raw[:, :, :, col_names["Wdir"]] < -180)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Wdir"]] > 180)
        cond2 = torch.logical_or(cond2,
                               raw[:, :, :, col_names["Ndir"]] < -720)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Ndir"]] > 720)
        cond2 = torch.logical_or(cond2, cond1)

        cond3 = raw[:, :, :, col_names["Patv"]] == 0
        cond3 = torch.logical_and(cond3,
                                raw[:, :, :, col_names["Wspd"]] > 2.5)
        cond3 = torch.logical_or(cond3, cond2)

        cond = torch.logical_not(cond3)
        cond = cond.float()
       # print('cond ',cond)
        return torch.mean(F.mse_loss(pred, gold, reduction='none') * cond)
class FilterHuberLoss(nn.Module):
    def __init__(self, delta=5, **kwargs):
        super(FilterHuberLoss, self).__init__()
        self.delta = delta
        print('FilterHuberLoss', 'delta = {}'.format(self.delta))

    def forward(self, pred, gold, raw, col_names):
        # Remove bad input
        cond1 = raw[:, :, :, col_names["Patv"]] < 0

        cond2 = raw[:, :, :, col_names["Pab1"]] > 89
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Pab2"]] > 89)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Pab3"]] > 89)

        cond2 = torch.logical_or(cond2,
                                  raw[:, :, :, col_names["Wdir"]] < -180)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Wdir"]] > 180)
        cond2 = torch.logical_or(cond2,
                                  raw[:, :, :, col_names["Ndir"]] < -720)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Ndir"]] > 720)
        cond2 = torch.logical_or(cond2, cond1)

        cond3 = raw[:, :, :, col_names["Patv"]] == 0
        cond3 = torch.logical_and(cond3,
                                   raw[:, :, :, col_names["Wspd"]] > 2.5)
        cond3 = torch.logical_or(cond3, cond2)

        cond = torch.logical_not(cond3)
        cond = cond.float()
        # cond = torch.cast(cond, "float32")
        #print('cond ',cond)
        return torch.mean(F.smooth_l1_loss(pred, gold, reduction='none', beta=self.delta) * cond)

class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()

    def forward(self, pred, gold, raw, col_names):
       # print('pred ',pred)
       # print('gold ',gold)
        return F.mse_loss(pred, gold)


class MAELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MAELoss, self).__init__()

    def forward(self, pred, gold, raw, col_names):
        loss = F.l1_loss(pred, gold)
        return loss


class HuberLoss(nn.Module):
    def __init__(self, delta=5, **kwargs):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, gold, raw, col_names):
        loss = F.huber_loss(pred, gold, reduction='mean', delta=self.delta)
        return loss


class SmoothMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(SmoothMSELoss, self).__init__()
        self.smooth_win = kwargs["smooth_win"]

    def forward(self, pred, gold, raw, col_names):
        # PyTorch's avg_pool1d expects input of shape (N, C, L)
        # Reshape gold if needed to match the expected input shape
        orig_shape = gold.shape
        gold = gold.reshape(gold.shape[0], -1, gold.shape[-1])
        gold = F.avg_pool1d(
            gold,
            kernel_size=self.smooth_win,
            stride=1,
            padding=self.smooth_win//2)
        # Reshape back to original shape
        gold = gold.reshape(orig_shape)
        loss = F.mse_loss(pred, gold)
        return loss

