import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import numpy as np
from logging import getLogger
from torch.utils.data import DataLoader
import random
from easydict import EasyDict as edict
import loss as loss_factory
from wpf_dataset import PGL4WPFDataset
from wpf_model import WPFModel
import optimization as optim
from metrics import regressor_scores, regressor_detailed_scores
from utils import save_model, _create_if_not_exist, load_model, str2bool, ensure_dir, get_logger
import matplotlib
import torch.distributed as dist
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def set_seed(seed):
    """Set seed for reproduction.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) 
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 


def data_augment(X, y, p=0.8, alpha=0.5, beta=0.5):
    """Regression SMOTE
    """
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
def build_optimizer(config, log, model):
    """
    select optimizer
    """
    log.info('You select `{}` optimizer.'.format(config.learner.lower()))
    if config.learner.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.learner.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                        momentum=config.lr_momentum, weight_decay=config.weight_decay)
    elif config.learner.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr,
                                            eps=config.lr_epsilon, weight_decay=config.weight_decay)
    elif config.learner.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr,
                                            alpha=config.lr_alpha, eps=config.lr_epsilon,
                                            momentum=config.lr_momentum, weight_decay=config.weight_decay)
    elif config.learner.lower() == 'sparse_adam':
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=config.lr,
                                               eps=config.lr_epsilon, betas=config.lr_betas)
    elif config.learner.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        config.log.warning('Received unrecognized optimizer, set default Adam optimizer')
        optimizer = torch.optim.Adam(config.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return optimizer
def build_lr_scheduler(config, log, optimizer):
    """
    select lr_scheduler
    """
    if config.lr_decay:
        log.info('You select `{}` lr_scheduler.'.format(config.lr_scheduler_type.lower()))
        if config.lr_scheduler_type.lower() == 'multisteplr':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=config.milestones, gamma=config.lr_decay_ratio)
        elif config.lr_scheduler_type.lower() == 'steplr':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=config.step_size, gamma=config.lr_decay_ratio)
        elif config.lr_scheduler_type.lower() == 'exponentiallr':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=config.lr_decay_ratio)
        elif config.lr_scheduler_type.lower() == 'cosineannealinglr':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config.lr_T_max, eta_min=config.lr_eta_min)
        elif config.lr_scheduler_type.lower() == 'lambdalr':
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=config.lr_lambda)
        elif config.lr_scheduler_type.lower() == 'reducelronplateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', patience=config.lr_patience,
                    factor=config.lr_decay_ratio, threshold=config.lr_threshold)
        else:
            log.warning('Received unrecognized lr_scheduler, please check the parameter `lr_scheduler`.')
            lr_scheduler = None
    else:
        lr_scheduler = None
    return lr_scheduler

def train_and_evaluate(config, train_data, valid_data, test_data=None):

    log = getLogger()

    data_mean = torch.FloatTensor(train_data.data_mean).to(config.device)  # (1, 134, 1, 1)
    data_scale = torch.FloatTensor(train_data.data_scale).to(config.device)  # (1, 134, 1, 1)

    graph = train_data.graph  # (134, 134)

    train_data_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers)

    valid_data_loader = DataLoader(
        valid_data,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers)

    # test_data_loader = DataLoader(
    #     test_data,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=config.num_workers)

    model = WPFModel(config=config).to(config.device)
    # log.info(model)
    # for name, param in model.named_parameters():
    #     log.info(str(name) + '\t' + str(param.shape) + '\t' +
    #                       str(param.device) + '\t' + str(param.requires_grad))
    # total_num = sum([param.nelement() for param in model.parameters()])
    # log.info('Total parameter numbers: {}'.format(total_num))

    loss_fn = getattr(loss_factory, config.loss)()

    opt = build_optimizer(config, log, model)
    #opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    grad_accmu_steps = config.gsteps
    opt.zero_grad()
    lr_scheduler = build_lr_scheduler(config, log, opt)
    _create_if_not_exist(config.output_path)
    global_step = 0

    best_score = np.inf
    patient = 0

    col_names = dict(
        [(v, k) for k, v in enumerate(train_data.get_raw_df()[0].columns)])

    valid_records = []
    # test_records = []

    for epoch in range(config.epoch):
        model.train()
        losses = []
        for batch_x, batch_y in tqdm(train_data_loader, 'train'):
            
            batch_x, batch_y = data_augment(batch_x, batch_y)
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)

            input_y = batch_y  # (B,N,T,F)
            batch_y = batch_y[:, :, :, -1]  # (B,N,T)
            batch_y = (batch_y - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1]
            pred_y = model(batch_x, input_y, data_mean, data_scale,graph)  # (B,N,T)  
            
            #print('pred_y ',pred_y)
            #print('batch_y ',batch_y)
            loss = loss_fn(pred_y, batch_y, input_y, col_names)
            loss = loss / grad_accmu_steps
           
            # if any(torch.isnan(p.grad).any() for p in model.parameters()):
            #     print(f"Epoch {epoch}: 检测到NaN梯度，跳过更新")
            #     opt.zero_grad()
            #     continue
            #opt.step()
           # print('loss ',loss.item())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            #print('test ')
            opt.step()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
           # optimizer.step()
           # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)            
            # if global_step % grad_accmu_steps == 0:
            #     opt.step()
            #     opt.zero_grad()            
            # opt.zero_grad()
            global_step += 1
            losses.append(loss.item())

            if global_step % config.log_per_steps == 0:
                log.info("Step %s Train Loss: %s" % (global_step, loss.item()))
        log.info("Epoch=%s, exp_id=%s, Train Loss: %s" % (epoch, config.exp_id, np.mean(losses)))

        valid_r = evaluate(
                valid_data_loader,
                valid_data.get_raw_df(),
                model,
                loss_fn,
                config,
                data_mean,
                data_scale,
                tag="val",
                graph=graph)
        valid_records.append(valid_r)

        log.info("Epoch={}, exp_id={}, Valid ".format(epoch, config.exp_id) + str(dict(valid_r)))

        if lr_scheduler is not None:
            if config.lr_scheduler_type.lower() == 'reducelronplateau':
                lr_scheduler.step(valid_r['loss'])
                print('test0')
            else:
                lr_scheduler.step()
                print('test1')

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

def visualize_prediction(input_batch, pred_batch, gold_batch, tag):
    plt.figure()
    for i in range(1, 5):
        ax = plt.subplot(2, 2, i)
        ax.plot(
            np.concatenate(
                [input_batch[288 * (i - 1)], gold_batch[288 * (i - 1)]]),
            label="gold")
        ax.plot(
            np.concatenate(
                [input_batch[288 * (i - 1)], pred_batch[288 * (i - 1)]]),
            label="pred")
        ax.legend()
    plt.savefig(tag + "_vis.png")
    plt.close()
###########################

def evaluate(valid_data_loader,
             valid_raw_df,
             model,
             loss_fn,
             config,
             data_mean,
             data_scale,
             tag="train",
             graph=None):
    with torch.no_grad():
        col_names = dict([(v, k) for k, v in enumerate(valid_raw_df[0].columns)])
        model.eval()
        step = 0
        pred_batch = []
        gold_batch = []
        input_batch = []
        losses = []
        for batch_x, batch_y in tqdm(valid_data_loader, tag):
            # weekday,time,Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv,Patv
            # 0       1    2    3    4    5    6    7    8    9    10   11
            # if config.only_useful:
            #     batch_x = batch_x[:, :, :, select_ind]
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)

            pred_y = model(batch_x, batch_y, data_mean, data_scale,graph)

            scaled_batch_y = batch_y[:, :, :, -1]  # (B,N,T)
            scaled_batch_y = (scaled_batch_y - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1]
            loss = loss_fn(pred_y, scaled_batch_y, batch_y, col_names)
            losses.append(loss.item())

            pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])
            pred_y = pred_y.cpu().numpy()  # (B,N,T)

            batch_y = batch_y[:, :, :, -1].cpu().numpy()  # (B,N,T)
            input_batch.append(batch_x[:, :, :, -1].cpu().numpy())  # (B,N,T)
            pred_batch.append(pred_y)
            gold_batch.append(batch_y)

            step += 1
        model.train()

        pred_batch = np.concatenate(pred_batch, axis=0)  # (B,N,T)
        gold_batch = np.concatenate(gold_batch, axis=0)  # (B,N,T)
        input_batch = np.concatenate(input_batch, axis=0)  # (B,N,T)

        pred_batch = np.expand_dims(pred_batch, -1)  # (B,N,T,1)
        gold_batch = np.expand_dims(gold_batch, -1)  # (B,N,T,1)
        input_batch = np.expand_dims(input_batch, -1)  # (B,N,T,1)

        pred_batch = np.transpose(pred_batch, [1, 0, 2, 3])  # (N,B,T,1)
        gold_batch = np.transpose(gold_batch, [1, 0, 2, 3])  # (N,B,T,1)
        input_batch = np.transpose(input_batch, [1, 0, 2, 3])  # (N,B,T,1)

        _mae, _rmse = regressor_detailed_scores(pred_batch, gold_batch,
                                                valid_raw_df, config.capacity,
                                                config.output_len)

        _farm_mae, _farm_rmse = regressor_scores(
            np.sum(pred_batch, 0) / 1000., np.sum(gold_batch, 0) / 1000.)

        output_metric = {
            'mae': _mae,
            'score': (_mae + _rmse) / 2,
            'rmse': _rmse,
            'farm_mae': _farm_mae,
            'farm_score': (_farm_mae + _farm_rmse) / 2,
            'farm_rmse': _farm_rmse,
            'loss': np.mean(losses),
        }

        return output_metric                                                                     


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    args = parser.parse_args()
    dict_args = vars(args)

    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.update(dict_args)
    exp_id = config.get('exp_id', None)
    if exp_id is None:
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = str(exp_id)
    #print(config)
    logger = get_logger(config)
    logger.info(config)
    set_seed(config.seed)
    #torch.use_deterministic_algorithms(True)
    ensure_dir(config.output_path)
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