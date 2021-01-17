from ipdb import set_trace as st
import utils as U
import configuration as C
import trainner
import os
import yaml
import result_handler as rh
import subprocess
import pandas as pd
from loguru import logger


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
    dir_save = f'{config["path"]["dir_save"]}{exp_name}'
    dir_save_ignore = f'{config["path"]["dir_save_ignore"]}{exp_name}'
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
    config = U.set_debug_config(config)
    dir_save, dir_save_ignore = init_exp(config)
    rh.save_model_architecture(dir_save, C.get_model(config))

    # config
    n_fold = config['split']['n_fold']
    path_trn_tp = config['path']['path_train_tp']

    # load data
    trn_tp = pd.read_csv(path_trn_tp)

    for i_fold in range(n_fold):
        # 学習を行う
        logger.info(f'fold {i_fold + 1}/{n_fold} - start training')

        model, loss_trn, loss_val, accuracy_val = trainner.train_fold(
                                                i_fold, trn_tp, config)
        logger.info(f'[fold {i_fold+1}]accuracy_val={accuracy_val:.4f}')

    logger.info('::: exp end :::')

if __name__ == "__main__":
    main()
