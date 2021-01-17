from ipdb import set_trace as st
from icecream import ic
import gc
import pandas as pd
from fastprogress import progress_bar
from loguru import logger
import numpy as np
import torch
from sklearn.metrics import accuracy_score

import utils as U
import configuration as C
import result_handler as rh
from criterion import mixup_criterion
from early_stopping import EarlyStopping


def train_cv(config):
    # config
    debug = config['globals']['debug']
    n_fold = config['split']['n_fold']
    n_epoch = config['globals']['num_epochs']
    path_trn_tp = config['path']['path_train_tp']
    n_classes = config['model']['params']['n_classes']
    _model = C.get_model(config)
    dir_save_exp, dir_save_ignore_exp, _ = U.get_save_dir_exp(config)

    # load data
    trn_tp = pd.read_csv(path_trn_tp)

    # init
    acc_val_folds = []
    if debug:
        oof_sig = np.zeros([n_classes*n_fold, n_classes])
    else:
        oof_sig = np.zeros([len(trn_tp), n_classes])
    for i_fold in progress_bar(range(n_fold)):
        logger.info("-" * 18)
        logger.info(f'\tFold {i_fold + 1}/{n_fold}')
        logger.info("-" * 18)

        epochs = []
        losses_trn = []
        losses_val = []
        accs_val = []
        best_acc_val = 0
        best_loss_val = 0
        best_output_sig = 0
        save_path = f'{dir_save_ignore_exp}/'\
                    f'{_model.__class__.__name__}_fold{i_fold}.pth'
        early_stopping = EarlyStopping(patience=12,
                                       verbose=True,
                                       path=save_path,
                                       trace_func=logger.info)
        for epoch in range(1, n_epoch+1):
            # 学習を行う
            result_dict = train_fold(i_fold, trn_tp, config)
            val_idxs = result_dict['val_idxs']
            output_sig = result_dict['output_sig']
            loss_trn = result_dict['loss_trn']
            loss_val = result_dict['loss_val']
            acc_val = result_dict['acc_val']
            logger.info(f'[fold({i_fold+1})epoch({epoch})]'
                        f'loss_trn={loss_trn:.6f} '
                        f'loss_val={loss_val:.6f} '
                        f'acc_val={acc_val:.6f}')

            # 格納
            epochs.append(epoch)
            losses_trn.append(loss_trn)
            losses_val.append(loss_val)
            accs_val.append(acc_val)

            # best model ?
            is_update = early_stopping(loss_val, result_dict['model'], debug)
            if is_update:
                best_loss_val = loss_val
                best_acc_val = acc_val
                best_output_sig = output_sig

            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
            # result
            rh.save_loss_figure(i_fold, epochs, losses_trn,
                                losses_val, dir_save_exp)
            rh.save_result_csv(i_fold, best_loss_val,
                               best_acc_val, dir_save_exp, config)

        # --- fold end ---
        # oof_sig
        acc_val_folds.append(best_acc_val)
        if debug:
            oof_sig[i_fold*n_classes:(i_fold+1)*n_classes] = best_output_sig
        else:
            oof_sig[val_idxs, :] = best_output_sig
        logger.info(f'best_loss_val: {best_loss_val:.6f}, '
                    f'best_acc_val: {best_acc_val:.6f}')

    oof = np.argmax(oof_sig, axis=1)
    if debug:
        # 適当な値を答えとする
        acc_oof = accuracy_score(np.zeros(len(oof)), oof)
    else:
        acc_oof = accuracy_score(trn_tp['species_id'].values, oof)

    # acc_val_folds
    acc_val_folds_mean = np.mean(acc_val_folds)
    acc_val_folds_std = np.std(acc_val_folds)
    logger.info(f'acc_folds(mean, std): '
                f'{acc_val_folds_mean:.6f} +- {acc_val_folds_std:6f}')
    logger.info(f'acc_oof: {acc_oof:6f}')


def train_fold(i_fold, trn_tp, config):
    mixup = config['globals']['mixup']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trn_idxs, val_idxs = C.get_index_fold(trn_tp, i_fold, config)
    trn_tp_trn = trn_tp.iloc[trn_idxs].reset_index(drop=True)
    trn_tp_val = trn_tp.iloc[val_idxs].reset_index(drop=True)
    trn_loader = C.get_trn_val_loader(trn_tp_trn, 'train', config)
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
            data, targets_a, targets_b, lam = U.mixup_data(data,
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
    loss_val, acc_val, output_sig = get_loss_score(model, val_loader,
                                                   criterion, device)

    result_dict = {
            'model': model,
            'val_idxs': val_idxs,
            'output_sig': output_sig,
            'loss_trn': loss_trn,
            'loss_val': loss_val,
            'acc_val': acc_val,
            }
    # return model, loss_trn, loss_val, acc_val, output_sig
    # 開放
    del trn_loader
    del val_loader
    del model
    del optimizer
    del scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return result_dict


def get_loss_score(model, val_loader, criterion, device):
    model.eval()
    epoch_valid_loss = 0
    y_pred_list = []
    y_true_list = []
    output_sig_list = []
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        epoch_valid_loss += loss.item()*data.size(0)

        output_ = output['output']
        output_sig = output['output_sigmoid']
        output_sig = output_sig.detach().cpu().numpy()
        _y_pred = output_.detach().cpu().numpy().argmax(axis=1)
        _y_true = target.detach().cpu().numpy().argmax(axis=1)
        y_pred_list.append(_y_pred)
        y_true_list.append(_y_true)
        output_sig_list.append(output_sig)

    loss_val = epoch_valid_loss / len(val_loader.dataset)
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    output_sig = np.concatenate(output_sig_list, axis=0)
    acc_val = accuracy_score(y_true, y_pred)
    del data
    return loss_val, acc_val, output_sig
