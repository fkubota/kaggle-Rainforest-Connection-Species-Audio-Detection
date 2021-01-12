import os
import yaml
import subprocess
import numpy as np
import pandas as pd
from loguru import logger
import get_funcs as gf
from ipdb import set_trace as st


def init_exp(config):
    '''
    dir_saveの作成と、dir_saveの取得
    '''
    logger.info(':: in ::')

    # git の hash値を取得
    cmd = "git rev-parse --short HEAD"
    hash_ = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    logger.info(f'hash: {hash_}')

    _dir = os.path.dirname(os.path.abspath(__file__))
    exp_name = _dir.split('/')[-1]
    logger.info(f'exp_name: {exp_name}')
    dir_save = f'{config["globals"]["dir_save"]}{exp_name}'
    dir_save_ignore = f'{config["globals"]["dir_save_ignore"]}{exp_name}'
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    if not os.path.exists(dir_save_ignore):
        os.makedirs(dir_save_ignore)

    logger.info(':: out ::')
    return dir_save, dir_save_ignore


def main():
    logger.remove()
    logger.add('exp.log', mode='w')
    logger.info('='*20)
    logger.info('::: exp start :::')
    logger.info('='*20)
    with open('config.yml', 'r') as yml:
        config = yaml.safe_load(yml)

    # init
    dir_save, dir_save_ignore = init_exp(config)

    # config
    n_fold = config['split']['n_fold']
    path_trn_tp = config['globals']['path_train_tp']

    # load data
    trn_tp = pd.read_csv(path_trn_tp)

    for i_fold in range(n_fold):
        # 学習を行う
        logger.info(f'fold {i_fold + 1} - start training')

        # データセットの用意
        trn_idxs, val_idxs = gf.get_index_fold(trn_tp, i_fold, config)
        trn_tp_trn = trn_tp.iloc[trn_idxs].reset_index(drop=True)
        trn_tp_val = trn_tp.iloc[val_idxs].reset_index(drop=True)
        trn_loader = gf.get_trn_val_loader(trn_tp_trn, 'train', config)
        val_loader = gf.get_trn_val_loader(trn_tp_val, 'valid', config)


if __name__ == "__main__":
    main()
