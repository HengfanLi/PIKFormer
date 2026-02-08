import os
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import yaml
import numpy as np
from logging import getLogger
from torch.utils.data import DataLoader
import random
from easydict import EasyDict as edict
import loss as loss_factory
from loss import FilterHuberLoss,FilterMSELoss,FilterHuberLoss2
from wpf_dataset import PGL4WPFDataset
from wpf_model3 import WPFModel,UncertaintyHead#,
import optimization as optim
from metrics import regressor_scores, regressor_detailed_scores
from utils import save_model, get_logger
import matplotlib
import torch.distributed as dist
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
torch.set_printoptions(precision=2, sci_mode=False)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) 
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

def data_augment(X, y, p=0.8, alpha=0.5, beta=0.5, seed=None):

    device = X.device
    fix_X, X = X[:, :, :, :2], X[:, :, :, 2:]
    fix_y, y = y[:, :, :, :2], y[:, :, :, 2:]
    batch_size = X.shape[0]
    random_values = torch.rand(batch_size, device=device)
    idx_to_change = random_values < p

    # ensure that first element to switch has probability > 0.5
    np_betas = np.random.beta(alpha, beta, batch_size) / 2 + 0.5
    random_betas = torch.tensor(
        np_betas, dtype=torch.float32, device=device).reshape(-1, 1, 1, 1)
    index_permute = torch.randperm(batch_size, device=device)

    X[idx_to_change] = random_betas[idx_to_change] * X[idx_to_change]
    X[idx_to_change] += (
        1 - random_betas[idx_to_change]) * X[index_permute][idx_to_change]

    y[idx_to_change] = random_betas[idx_to_change] * y[idx_to_change]
    y[idx_to_change] += (
        1 - random_betas[idx_to_change]) * y[index_permute][idx_to_change]

    return torch.cat([fix_X, X], -1), torch.cat([fix_y, y], -1)
def build_optimizer(config, log, train_data_loader,params):
    #log.info('You select `{}` optimizer.'.format(config.learner.lower()))
    if config.learner.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=1e-4)# 1e-50
        
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=3, 
    verbose=True) 
    # elif config.learner.lower() == 'sgd':
    #     optimizer = torch.optim.SGD(params, lr=config.lr,
    #                                     momentum=config.lr_momentum, weight_decay=config.weight_decay)
    # elif config.learner.lower() == 'adamw':
    #     optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=0.00001,betas=(0.9, 0.999))    
    return optimizer,scheduler

# def build_optimizer(config,params):
#     epochs = 25
#     warmup_epochs = 2 # 预热周期，默认 5
#     optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=1e-3
#     )

#     # 2. 方案 A: 使用 SequentialLR 串联 (推荐)
#     # 第一阶段: 线性预热
#     # 学习率从 lr * start_factor 线性增长到 lr
#     # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
#     #     optimizer, 
#     #     start_factor=0.01, # 初始 LR 为 1% 的设置值
#     #     end_factor=0.1, 
#     #     total_iters=warmup_epochs)#lr: 0.001

#     # # 第二阶段: 余弦退火
#     # # 在 warmup 结束后开始下降
#     # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     #     optimizer, 
#     #     T_max=epochs - warmup_epochs, 
#     #     eta_min=1e-6
#     # )

#     # # 将两个调度器拼接
#     # # milestone 表示在哪个时刻切换到下一个调度器
#     # scheduler = torch.optim.lr_scheduler.SequentialLR(
#     #     optimizer, 
#     #     schedulers=[warmup_scheduler, cosine_scheduler], 
#     #     milestones=[warmup_epochs]
#     # )

#     return optimizer#, scheduler
def combined_loss(loss1, loss2, loss_cons,epoch,use_pinn, uncertainty_head=None, reg_coef=0.001):#0.001
    #weights = uncertainty_head()
    weighted_loss1 =  loss1 
    weighted_loss2 =  loss2 
    
    #log_vars = torch.clamp(uncertainty_head.log_vars, -3, 3)
    # weighted_loss1 = weights[0] * loss1 + log_vars[0]
    # weighted_loss2 = weights[1] * loss2 + log_vars[1]
    #print('weights[1] ',weights[1])
     # weighted_loss2 = 0.00005* weights[1] * loss2 + 0.00005  *uncertainty_head.log_vars[1]
    #reg_loss = reg_coef * torch.sum(torch.square(uncertainty_head.log_vars))
    w=10
    #w=2
    # if use_pinn:
    #     weighted_loss2 = w * weighted_loss2
    #     total_loss = weighted_loss1 + weighted_loss2
    # else:
    #     total_loss = weighted_loss1
    alpha = 10
    loss_cons = alpha * loss_cons
    weighted_loss2=w*weighted_loss2
    totall_loss = weighted_loss1 + weighted_loss2+loss_cons
    #total_loss = weighted_loss2
    # if epoch>3:        
    #     weighted_loss2=w*weighted_loss2
    #     totall_loss = weighted_loss1 + weighted_loss2
    # else:
    #     totall_loss = weighted_loss1 
        
    print('weighted_loss1,weighted_loss2,loss_cons: ',weighted_loss1,weighted_loss2,loss_cons)
    return totall_loss
    #return weighted_loss1 + weighted_loss2+ reg_loss

# def wind_gate(v, v_in=3.0, v_rated=12.0, s=0.8):
#     return torch.sigmoid((v - v_in)/s) * torch.sigmoid((v_rated - v)/s)
criterion = nn.MSELoss()
# #criterion = FilterHuberLoss2()

def pinn_loss(pinn_patv, batch_yyy,batch_yy,batch_xx, col_names):
            # 预测pinn功率，未处理x，未处理y，增强x
    #Patv =input_y[:, : , :, 6]#[:, : , :, 11]#32,134,144
    wspd = batch_yy[:, : , :, 2]
    Patv = batch_yyy
    #Patv = batch_xx[:, : , :, -1]
    #Patv = batch_yy_p
    # Wspd = batch_yy[:, : , :, 2]
    # Patv=batch_yy_p
    #mask = (Patv > 1).bool()
    capacity=1550
    mask= (wspd >= 2.5) & (Patv > 0) & (Patv < 1550) 
    pinn_patv = pinn_patv / capacity
    Patv = Patv / capacity
    #print('pinn_patv ',pinn_patv[3,110,10:20])
    #print('Patv '     ,Patv[3,110,10:20])
    # print('#####################################')
    #physical_loss1 = criterion(pinn_patv[mask],Patv[mask])
    physical_loss1 = ((pinn_patv - Patv)**2 * mask).sum() / mask.sum()

    #physical_loss1 = criterion(pinn_patv,Patv)
    #physical_loss1 = criterion(pinn_patv,Patv,batch_yy , col_names)
    total_loss = physical_loss1
    return total_loss
# criterion = nn.MSELoss(reduction="none")  # 关键：先不做mean，后面自己加权

#     return loss
def train_and_evaluate(config, train_data, valid_data, test_data=None):
    name2id = {
        'weekday': 0,
        'time': 1,
        'Wspd': 2,
        'Wdir': 3,
        'Etmp': 4,
        'Itmp': 5,
        'Ndir': 6,
        'Pab1': 7,
        'Pab2': 8,
        'Pab3': 9,
        'Prtv': 10,
        'Patv': 11
    }
    #[0, 1, 2, 4, 5, 10, 11]
    # select useful features
    select = config.select
    select_ind = [name2id[name] for name in select]
    log = getLogger()
    data_mean = torch.FloatTensor(train_data.data_mean).to(config.device)  # (1, 134, 1, 1)
    data_scale = torch.FloatTensor(train_data.data_scale).to(config.device)  # (1, 134, 1, 1)
    graph = train_data.graph  # (134, 134)
    train_data_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        )

    valid_data_loader = DataLoader(
        valid_data,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers)


    model = WPFModel(config=config).to(config.device)
    #uncertainty_head = SelfAdaptiveLoss(num_terms=2)
    uncertainty_head = UncertaintyHead(num_losses=2)
    loss_fn = getattr(loss_factory, config.loss)()
    opt,scheduler = build_optimizer(config, log,train_data_loader ,list(model.parameters()) )
    # opt,scheduler = build_optimizer(config, log,train_data_loader ,list(model.parameters()) + list(uncertainty_head.parameters()))
    # opt,scheduler = build_optimizer(config, list(model.parameters()) + list(uncertainty_head.parameters()))
   # opt = build_optimizer(config, log, model)
    #opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    grad_accmu_steps = config.gsteps
    opt.zero_grad()
    global_step = 0
    best_score = np.inf
    patient = 0

    col_names = dict(
        [(v, k) for k, v in enumerate(train_data.get_raw_df()[0].columns)])

    valid_records = []
    pinn_enabled = False          # 当前epoch是否用PINN
    pinn_enable_next = False      # 下一epoch是否启用PINN（第一次下降后置True）
    has_triggered = False         # 确保“只触发一次”

    for epoch in range(config.epoch):
        model.train()
        pinn_enabled = pinn_enable_next

        losses = []
        for batch_x,batch_xx, batch_y,batch_yy in tqdm(train_data_loader, 'train'):#########batch_xx和batch_yy没处理
            #########x是过去一天，y是未来两天
            batch_x = batch_x.to(config.device)
            batch_y= batch_y.to(config.device)
            batch_xx = batch_xx.to(config.device)
            select_batch_xx = batch_xx[:, :, :, select_ind]
            batch_yy = batch_yy.to(config.device)

            batch_x, batch_y = data_augment(batch_x, batch_y)           
            # aug_xx,_= data_augment(batch_xx, batch_yy)
            # aug_xx  = aug_xx[:, :, :, select_ind]
            input_x = batch_x[:, :, :, select_ind]
            input_x = input_x.to(config.device)            
            # input_y = batch_y[:, :, :, select_ind]
            # input_y = input_y.to(config.device)
            batch_y1 = batch_y[:, :, :, -1]  # (B,N,T)
            batch_y1 = (batch_y1 - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1]
            
           # select_batch_xx = select_batch_xx[:, :, :, -1]  # (B,N,T)
           # select_batch_xx = (select_batch_xx - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1]
            pred_y,pinn_patv = model(input_x, batch_y, data_mean, data_scale,select_batch_xx,batch_yy,graph)  # (B,N,T) 
                                       #增强选择x，增强y，正常y        
            #pinn_loss1= pinn_loss(pinn_patv,pinn_patv,batch_xx,batch_yy) 
           # pinn_loss1= pinn_loss(pinn_patv,batch_xx,batch_yy)      
           # pinn_patv = (pinn_patv - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1]
            batch_yy_p = batch_yy[:, :, :, -1]  # (B,N,T)
           # batch_yy_p = (batch_yy_p - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1]
            pinn_loss1= pinn_loss(pinn_patv,batch_yy_p,batch_yy,batch_xx, col_names)  
                             #预测pinn功率，未处理x，未处理y，增强x            
            ori_loss = loss_fn(pred_y, batch_y1,batch_y, col_names)
                               #预测y，归一化y，增强y
            #loss = combined_loss(ori_loss,pinn_loss1,epoch,uncertainty_head)
            #loss = combined_loss(ori_loss, pinn_loss1, epoch,use_pinn=pinn_enabled, uncertainty_head=uncertainty_head)
            pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])
            #print('pred_y ',pred_y)
           # print('pinn_patv,pred_y ',pinn_patv[3,10,:10],pred_y[3,10,:10])
            P_rated = 1550.0
            #pinn_loss1 = pinn_loss1 / (P_rated**2)
            mask_teacher = (pinn_patv.detach() < 1549.0).float()
            # loss_cons = (mask_teacher * (pred_y- (pinn_patv.detach()))**2).mean()
            loss_cons = (mask_teacher * (pred_y/P_rated - (pinn_patv.detach()/P_rated))**2).mean()
            #loss_cons = loss_cons / (P_rated**2)
            loss = combined_loss(ori_loss, pinn_loss1,loss_cons ,epoch,use_pinn=pinn_enabled)
            loss = loss / grad_accmu_steps
            #opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            if (global_step + 1) % grad_accmu_steps == 0:
                opt.step()
                opt.zero_grad()
            global_step += 1

            # if global_step % grad_accmu_steps == 0:
            #     opt.step()
            #     opt.zero_grad()
            # opt.step()
            #global_step += 1
            losses.append(loss.item())

            # if global_step % config.log_per_steps == 0:
            #     log.info("Step %s Train Loss: %s" % (global_step, loss.item()))
        log.info("Epoch=%s, exp_id=%s, Train Loss: %s" % (epoch, config.exp_id, np.mean(losses)))
        # with torch.no_grad():
        # # log_vars 是 [log(sigma1^2), log(sigma2^2)]
        #     log_vars = uncertainty_head.log_vars.detach()
        #     # 计算实际应用在 Loss 上的系数: 1 / (2 * exp(log_var))
        #     w1 = torch.exp(-log_vars[0])
        #     w2 = torch.exp(-log_vars[1])
            
        #     log.info(f"Step {global_step} | Loss: {loss.item():.4f}")
        #     log.info(f"Weights -> Task1(Ori): {w1:.4f}, Task2(PINN): {w2:.4f}")
        #     log.info(f"LogVars -> {log_vars.cpu().numpy()}")

        valid_r = evaluate(
                valid_data_loader,
                valid_data.get_raw_df(),
                model,
                loss_fn,
                config,
                data_mean,
                data_scale,
           
                tag="val",
                graph=graph,
             select_ind=select_ind)
        valid_records.append(valid_r)

        log.info("Epoch={}, exp_id={}, Valid ".format(epoch, config.exp_id) + str(dict(valid_r)))
        #scheduler.step()
        
        current_score = valid_r['score']
        prev_best_score = best_score  # 记录更新前的best

        scheduler.step(current_score)
        lr = opt.param_groups[0]['lr']
        logger.info(f"Epoch {epoch} LR: {lr:.6f}")

        # ---- 关键：第一次出现验证集性能下降 -> 下一epoch启用PINN ----
        # 你的score越小越好，所以“下降(变差)”是 current_score > prev_best_score
        # if (not has_triggered) and (current_score > prev_best_score):
        #     has_triggered = True
        #     pinn_enable_next = True
        #     logger.info(f"[Trigger] Val score got worse at epoch {epoch}: "
        #                 f"{current_score:.6f} > best {prev_best_score:.6f}. "
        #                 f"Enable PINN loss from next epoch.")
##########################################################            
        best_score = min(valid_r['score'], best_score)

        if best_score == valid_r['score']:
            patient = 0
            save_model(config.output_path+config.exp_id+'_'+config.model, model, opt=opt, steps=epoch, log=log)
        else:
            patient += 1
            if patient > config.patient:
                break

    best_epochs = min(enumerate(valid_records), key=lambda x: x[1]["score"])[0]
    log.info("exp_id={} model finish".format(config.exp_id))
    log.info("Best valid Epoch %s" % best_epochs)
    log.info("Best valid score %s" % valid_records[best_epochs])
###########################
def evaluate(valid_data_loader,
             valid_raw_df,
             model,
             loss_fn,
             config,
             data_mean,
             data_scale,
             tag="train",
             graph=None,
             select_ind=None):
    with torch.no_grad():
        col_names = dict([(v, k) for k, v in enumerate(valid_raw_df[0].columns)])
        model.eval()
        step = 0
        pred_batch = []
        gold_batch = []
        input_batch = []
        pred_batch2 =[]
        losses = []
        for batch_x, batch_y in tqdm(valid_data_loader, tag):
            # weekday,time,Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv,Patv
            # 0       1    2    3    4    5    6    7    8    9    10   11
            # if config.only_useful:
            batch_x  = batch_x.to(config.device)
            batch_x1 = batch_x[:, :, :, select_ind]
           # batch_x1 = batch_x1.to(config.device)

            batch_y  = batch_y.to(config.device)
            batch_y1 = batch_y[:, :, :, select_ind]
           # batch_y1 = batch_y1.to(config.device)

            pred_y,pinn_patv = model(batch_x1, batch_y, data_mean, data_scale,batch_x1,batch_y,graph)

            batch_yy = batch_y[:, :, :, -1]  # (B,N,T)
            scaled_batch_y = (batch_yy - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1]
            loss = loss_fn(pred_y, scaled_batch_y, batch_y, col_names)
            pinn_loss1= pinn_loss(pinn_patv,batch_yy,batch_y,batch_x1, col_names)  
            # 
            # losses.append(pinn_loss1.item())
            losses.append(loss.item())
            
            
            #pinn_patv = F.relu(pinn_patv * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])
            pinn_patv = pinn_patv.cpu().numpy()  # (B,N,T)
            pred_batch2.append(pinn_patv)

            pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])
            pred_y = pred_y.cpu().numpy()

            batch_y = batch_y[:, :, :, -1].cpu().numpy()  # (B,N,T)
            input_batch.append(batch_x[:, :, :, -1].cpu().numpy())  # (B,N,T)
            pred_batch.append(pred_y)           
            gold_batch.append(batch_y)

            step += 1
        model.train()

        pred_batch = np.concatenate(pred_batch, axis=0)  # (B,N,T)
        pred_batch2 = np.concatenate(pred_batch2, axis=0)
        gold_batch = np.concatenate(gold_batch, axis=0)  # (B,N,T)
        input_batch = np.concatenate(input_batch, axis=0)  # (B,N,T)

        pred_batch = np.expand_dims(pred_batch, -1)  # (B,N,T,1)
        pred_batch2 = np.expand_dims(pred_batch2, -1) 
        gold_batch = np.expand_dims(gold_batch, -1)  # (B,N,T,1)
        input_batch = np.expand_dims(input_batch, -1)  # (B,N,T,1)

        pred_batch = np.transpose(pred_batch, [1, 0, 2, 3])  # (N,B,T,1)
        pred_batch2 = np.transpose(pred_batch2, [1, 0, 2, 3])
        gold_batch = np.transpose(gold_batch, [1, 0, 2, 3])  # (N,B,T,1)
        input_batch = np.transpose(input_batch, [1, 0, 2, 3])  # (N,B,T,1)

        _mae, _rmse = regressor_detailed_scores(pred_batch, gold_batch,
                                                valid_raw_df, config.capacity,
                                                config.output_len)
        _mae2, _rmse2 = regressor_detailed_scores(pred_batch2, gold_batch,
                                                valid_raw_df, config.capacity,
                                                config.output_len)  
        _farm_mae, _farm_rmse = regressor_scores(
            np.sum(pred_batch, 0) / 1000., np.sum(gold_batch, 0) / 1000.)
        _farm_mae2, _farm_rmse2 = regressor_scores(
            np.sum(pred_batch2, 0) / 1000., np.sum(gold_batch, 0) / 1000.)
        output_metric = {
            'mae': _mae,
            'score': (_mae + _rmse) / 2,
            'rmse': _rmse,
            'farm_mae': _farm_mae,
            'farm_score': (_farm_mae + _farm_rmse) / 2,
            'farm_rmse': _farm_rmse,
            'loss': np.mean(losses),
            'mae2': _mae2,
            'score2': (_mae2 + _rmse2) / 2,
            'rmse2': _rmse2,
            'farm_mae2': _farm_mae2,
            'farm_score2': (_farm_mae2 + _farm_rmse2) / 2,
            'farm_rmse2': _farm_rmse2,
            'loss2': np.mean(losses)}

         
        # output_metric = {
        #     'mae': _mae2,
        #     'score': (_mae2 + _rmse2) / 2,
        #     'rmse': _rmse2,
        #     'farm_mae': _farm_mae2,
        #     'farm_score': (_farm_mae2 + _farm_rmse2) / 2,
        #     'farm_rmse': _farm_rmse2,
        #     'loss': np.mean(losses)}

        return output_metric                                                                    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    parser.add_argument("--select", nargs='+', type=str,
                        default=['weekday', 'time', 'Wspd', 'Etmp', 'Itmp', 'Prtv', 'Patv'])
    args = parser.parse_args()
    dict_args = vars(args)

    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.update(dict_args)
    set_seed(config.seed)
    exp_id = config.get('exp_id', None)
    if exp_id is None:
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = str(exp_id)
    logger = get_logger(config)
    logger.info(config)
    size = [config.input_len, config.output_len]
    train_data = PGL4WPFDataset(
        config.data_path,
        filename=config.filename,
        size=size,
        flag='train',
        total_days=config.total_days,
        train_days=config.train_days,
        val_days=config.val_days,
        test_days=config.test_days)
    valid_data = PGL4WPFDataset(
        config.data_path,
        filename=config.filename,
        size=size,
        flag='val',
        total_days=config.total_days,
        train_days=config.train_days,
        val_days=config.val_days,
        test_days=config.test_days)
    test_data = PGL4WPFDataset(
        config.data_path,
        filename=config.filename,
        size=size,
        flag='test',
        total_days=config.total_days,
        train_days=config.train_days,
        val_days=config.val_days,
        test_days=config.test_days)
    gpu_id = config.gpu_id
    if gpu_id != -1:
        device = torch.device('cuda:{}'.format(gpu_id))
    else:
        device = torch.device('cpu')
    config['device'] = device    
    train_and_evaluate(config, train_data, valid_data, test_data)