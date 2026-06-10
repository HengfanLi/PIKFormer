import torch
import torch.nn as nn
from models import PIKFormer
from dataset import get_dataloader
from loss import PhysicsInformedLoss, GSLBLossWeighter

def train_and_evaluate():
    # 参数设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 144
    pred_len = 72
    num_features = 5
    batch_size = 32
    epochs = 10
    
    # 1. 实例化模型、损失函数和数据加载器
    model = PIKFormer(seq_len, pred_len, num_features).to(device)
    train_loader = get_dataloader('train_wpf.csv', batch_size, seq_len, pred_len)
    test_loader = get_dataloader('test_wpf.csv', batch_size, seq_len, pred_len)
    
    criterion_supervised = nn.L1Loss() # MAE 作为基本监督损失
    criterion_physics = PhysicsInformedLoss().to(device)
    gslb_weighter = GSLBLossWeighter(num_losses=2).to(device)
    
    # 优化器（需同时优化模型参数与GSLB自身的权重参数）
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(gslb_weighter.parameters()), 
        lr=0.001
    )
    
    # --- 训练循环 (Training Loop) ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq_x, seq_y, future_features in train_loader:
            seq_x = seq_x.to(device)
            seq_y = seq_y.to(device)
            future_features = future_features.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            pred_y = model(seq_x)
            
            # 计算各项原始损失
            loss_sub = criterion_supervised(pred_y, seq_y)
            loss_phy = criterion_physics(pred_y, future_features)
            
            # 通过 GSLB 机制进行动态权重融合
            loss_total = gslb_weighter(loss_sub, loss_phy)
            
            loss_total.backward()
            optimizer.step()
            
            total_loss += loss_total.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Combined Balanced Loss: {total_loss/len(train_loader):.4f}")
        
    # --- 测试与评估 (Testing/Evaluation Loop) ---
    model.eval()
    mae_list, rmse_list = [], []
    with torch.no_grad():
        for seq_x, seq_y, _ in test_loader:
            seq_x = seq_x.to(device)
            seq_y = seq_y.to(device)
            
            pred_y = model(seq_x)
            
            # 计算指标并保存以便统计 mean ± std（对应表8、表9）
            mae = torch.mean(torch.abs(pred_y - seq_y))
            rmse = torch.sqrt(torch.mean((pred_y - seq_y) ** 2))
            
            mae_list.append(mae.item())
            rmse_list.append(rmse.item())
            
    import numpy as np
    print("\n--- Final Test Results ---")
    print(f"MAE: {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f}")
    print(f"RMSE: {np.mean(rmse_list):.2f} ± {np.std(rmse_list):.2f}")

if __name__ == "__main__":
    train_and_evaluate()