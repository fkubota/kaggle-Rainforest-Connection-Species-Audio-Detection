from ipdb import set_trace as st
import os
import time
import random
import numpy as np
from loguru import logger
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def get_save_dir_exp(config):
    _dir = os.path.dirname(os.path.abspath(__file__))
    exp_name = _dir.split('/')[-1]
    dir_save_exp = f'{config["path"]["dir_save"]}{exp_name}'
    dir_save_ignore_exp = f'{config["path"]["dir_save_ignore"]}{exp_name}'
    return dir_save_exp, dir_save_ignore_exp, exp_name


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def get_debug_idx(trn_tp, trn_idxs, val_idxs, config):
    n_classes = config['model']['params']['n_classes']

    trn_tp_trn = trn_tp.iloc[trn_idxs].copy()
    trn_tp_val = trn_tp.iloc[val_idxs].copy()
    trn_tp_trn['idx_'] = trn_idxs
    trn_tp_val['idx_'] = val_idxs

    trn_idxs_debug = []
    val_idxs_debug = []
    for idx in range(n_classes):
        bools = trn_tp_trn.species_id == idx
        trn_idxs_debug.append(trn_tp_trn[bools]['idx_'].values[0])

        bools = trn_tp_val.species_id == idx
        val_idxs_debug.append(trn_tp_val[bools]['idx_'].values[0])

    return trn_idxs_debug, val_idxs_debug


def set_debug_config(config):
    if config['globals']['debug']:
        logger.info(':: debug mode ::')
        config['globals']['num_epochs'] = 2
        config['split']['n_fold'] = 2
        config['loader']['train']['batch_size'] = 1
        config['loader']['valid']['batch_size'] = 1
        return config
    else:
        return config


def sec2time(sec):
    hour = int(sec//3600)
    minute = int((sec - 3600*hour)//60)
    second = int(sec - 3600*hour - 60*minute)

    hour = str(hour).zfill(2)
    minute = str(minute).zfill(2)
    second = str(second).zfill(2)
    str_time = f'{hour}:{minute}:{second}'

    return str_time
