from ipdb import set_trace as st
import os
import yaml
import subprocess
import random
import numpy as np
from loguru import logger
import torch


def init_exp(config, config_update, run_name):
    '''
    - git hashの取得
    - dir_saveの作成と、dir_saveの取得
    - configのupdate
    '''
    logger.info(':: in ::')

    # git の hash値を取得
    cmd = "git rev-parse --short HEAD"
    hash_ = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    logger.info(f'hash: {hash_}')

    # 保存ディレクトリの用意
    dir_save, dir_save_ignore, exp_name = get_save_dir_exp(config, run_name)
    logger.info(f'exp_name: {exp_name}_{run_name}')
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    if not os.path.exists(dir_save_ignore):
        os.makedirs(dir_save_ignore)

    # configのupdateとconfig_updateの保存
    deepupdate(config, config_update)
    with open(f'{dir_save}/config_update.yml', 'w') as path:
        yaml.dump(config_update, path)
    logger.info(f'config_update: {config_update}')

    # set_seed
    set_seed(config['globals']['seed'])

    logger.info(':: out ::')
    return dir_save, dir_save_ignore, config


def deepupdate(dict_base, other):
    '''
    ディクショナリを再帰的に更新する
    ref: https://www.greptips.com/posts/1242/
    '''
    for k, v in other.items():
        if isinstance(v, dict) and k in dict_base:
            deepupdate(dict_base[k], v)
        else:
            dict_base[k] = v


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_save_dir_exp(config, run_name):
    _dir = os.path.dirname(os.path.abspath(__file__))
    exp_name = _dir.split('/')[-1]
    dir_save_exp = f'{config["path"]["dir_save"]}{exp_name}/{run_name}'
    dir_save_ignore_exp = f'{config["path"]["dir_save_ignore"]}'\
                          f'{exp_name}/{run_name}'
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


def LWLRAP(preds, labels):
    '''
    https://github.com/yuki-a4/rfcx-species-audio-detection/blob/main/yuki/notebook/ex_059_resnest_changeLoss_lr_0.15_aug0.3_seed239.ipynb
    '''
    # st()
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, labels = preds.to(device), labels.to(device)

    # Ranks of the predictions
    ranked_classes = torch.argsort(preds, dim=-1, descending=True)
    # i, j corresponds to rank of prediction in row i
    class_ranks = torch.zeros_like(ranked_classes)
    for i in range(ranked_classes.size(0)):
        for j in range(ranked_classes.size(1)):
            class_ranks[i, ranked_classes[i][j]] = j + 1
    # Mask out to only use the ranks of relevant GT labels
    ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
    # All the GT ranks are in front now
    sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks,
                                              dim=-1, descending=False)
    # Number of GT labels per instance
    # num_labels = labels.sum(-1)
    pos_matrix = torch.tensor(
            np.array([i+1 for i in range(labels.size(-1))])).unsqueeze(0)
    pos_matrix = pos_matrix.to(device)
    sorted_ground_truth_ranks = sorted_ground_truth_ranks.to(device)
    score_matrix = pos_matrix / sorted_ground_truth_ranks
    score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
    scores = score_matrix * score_mask_matrix
    score = scores.sum() / labels.sum()
    return score.item()
