import os
import torch
import random
import numpy as np
from loguru import logger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


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
