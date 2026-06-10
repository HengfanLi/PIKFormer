import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """
    集成空气动力学控制方程残差与涡轮运行边界条件的 PINN 损失函数
    """
    def __init__(self, cut_in_speed=3.0, rated_speed=12.0, rated_power=1.0):
        super(PhysicsInformedLoss, self).__init__()
        self.cut_in_speed = cut_in_speed
        self.rated_speed = rated_speed
        self.rated_power = rated_power

    def forward(self, pred_power, future_features):
        # 假设 future_features 中的第 0 列是风速 (Wind Speed)
        wind_speed = future_features[:, :, 0] # [Batch, Pred_len]
        
        # 1. 边界条件约束 1：当风速低于切入风速时，理论功率应为 0
        loss_cut_in = torch.mean(torch.relu(3.0 - wind_speed) * (pred_power ** 2))
        
        # 2. 边界条件约束 2：当风速大于额定风速时，功率应进入额定功率高原 (Rated-power plateau)
        loss_rated = torch.mean(torch.relu(wind_speed - 12.0) * torch.relu(pred_power - self.rated_power))
        
        # 3. 空气动力学功率追踪残差 (部分负荷区域的非线性三次曲线约束简化版)
        # P = 0.5 * rho * A * Cp * v^3
        # 论文提到在切入和额定速度之间呈立方趋势
        mask = (wind_speed >= self.cut_in_speed) & (wind_speed <= self.rated_speed)
        cubic_speed = (wind_speed ** 3) * mask.float()
        # 建立某种理论比例关系残差（此处简化为预测值与风速立方的正则化差距）
        loss_aerodynamic = torch.mean(mask.float() * torch.abs(pred_power - 0.0005 * cubic_speed))
        
        physics_loss = loss_cut_in + loss_rated + loss_loss_aerodynamic
        return physics_loss


class GSLBLossWeighter(nn.Module):
    """
    GSLB (Gradient-guided Spatial-temporal Loss Balancing) 
    动态调节监督损失和物理损失的权重，规避梯度冲突与不确定性
    """
    def __init__(self, num_losses=2):
        super(GSLBLossWeighter, self).__init__()
        # 使用可学习的参数来表征各损失任务的不确定性 (Uncertainty weighting)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, loss_supervised, loss_physics, model_params=None):
        # 1. 基于不确定性的多任务学习权重平衡 (Uncertainty-based core strategy)
        weight_sub = torch.exp(-self.log_vars[0])
        weight_phy = torch.exp(-self.log_vars[1])
        
        balanced_loss = (weight_sub * loss_supervised + self.log_vars[0] + 
                         weight_phy * loss_physics + self.log_vars[1])
        
        # 注：论文中还考虑了梯度冲突（Gradient conflicts）
        # 如果需要实现 PCGrad (Projecting Conflicting Gradients)，通常需要在训练循环内手动对 backward 进行投影修改。
        return balanced_loss