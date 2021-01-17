from ipdb import set_trace as st
import pandas as pd
from fastprogress import progress_bar
from loguru import logger
import configuration as C
from criterion import mixup_criterion
from utils import mixup_data

import numpy as np
import torch
from sklearn.metrics import accuracy_score


def train_cv(config):
    # config
    n_fold = config['split']['n_fold']
    n_epoch = config['globals']['num_epochs']
    path_trn_tp = config['path']['path_train_tp']

    # load data
    trn_tp = pd.read_csv(path_trn_tp)

    for i_fold in progress_bar(range(n_fold)):
        logger.info("-" * 30)
        logger.info(f'\tFold {i_fold + 1}/{n_fold}')
        logger.info("-" * 30)

        epochs = []
        losses_trn = []
        losses_val = []
        accs_val = []
        for epoch in progress_bar(range(1, n_epoch+1)):
            # 学習を行う
            model, loss_trn, loss_val, acc_val = train_fold(
                                                    i_fold, trn_tp, config)
            logger.info(f'[fold({i_fold+1})epoch({epoch})]'
                        f'loss_trn={loss_trn:.4f} '
                        f'loss_val={loss_val:.4f} '
                        f'acc_val={acc_val:.4f}')

            epochs.append(epoch)
            losses_trn.append(loss_trn)
            losses_val.append(loss_val)
            accs_val.append(acc_val)


def train_fold(i_fold, trn_tp, config):
    mixup = config['globals']['mixup']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trn_idxs, val_idxs = C.get_index_fold(trn_tp, i_fold, config)
    trn_tp_trn = trn_tp.iloc[trn_idxs].reset_index(drop=True)
    trn_loader = C.get_trn_val_loader(trn_tp_trn, 'train', config)
    trn_tp_val = trn_tp.iloc[val_idxs].reset_index(drop=True)
    val_loader = C.get_trn_val_loader(trn_tp_val, 'valid', config)

    # C
    model = C.get_model(config).to(device)
    criterion = C.get_criterion(config)
    optimizer = C.get_optimizer(model, config)
    scheduler = C.get_scheduler(optimizer, config)

    # train
    model.train()
    epoch_train_loss = 0
    for batch_idx, (data, target) in enumerate(trn_loader):
        data, target = data.to(device), target.to(device)
        if mixup:
            data, targets_a, targets_b, lam = mixup_data(data,
                                                         target,
                                                         alpha=1.0)
        optimizer.zero_grad()
        output = model(data)
        if mixup:
            loss = mixup_criterion(criterion, output,
                                   targets_a, targets_b, lam)
        else:
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()*data.size(0)
    scheduler.step()
    loss_trn = epoch_train_loss / len(trn_loader.dataset)
    del data

    # eval valid
    loss_val, score_val = get_loss_score(model, val_loader, criterion, device)

    return model, loss_trn, loss_val, score_val


def get_loss_score(model, val_loader, criterion, device):
    model.eval()
    epoch_valid_loss = 0
    y_pred_list = []
    y_true_list = []
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        epoch_valid_loss += loss.item()*data.size(0)

#         out_numpy = output.detach().cpu().numpy()
        output = output['output']
        _y_pred = output.detach().cpu().numpy().argmax(axis=1)
        y_pred_list.append(_y_pred)
        _y_true = target.detach().cpu().numpy().argmax(axis=1)
        y_true_list.append(_y_true)

    loss_val = epoch_valid_loss / len(val_loader.dataset)
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    accuracy_val = accuracy_score(y_true, y_pred)
    del data
    return loss_val, accuracy_val
