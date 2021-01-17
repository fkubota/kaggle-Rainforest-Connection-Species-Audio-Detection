from ipdb import set_trace as st
import utils as U
import trainner
import configuration as C
import result_handler as rh
import os
import time
import yaml
import subprocess
from fastprogress import progress_bar
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
    start = time.time()
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

    # train
    trainner.train_cv(config)

    end = time.time()
    time_all = end - start
    logger.info(f'elapsed time: {U.sec2time(time_all)}')
    logger.info('::: exp end :::')


if __name__ == "__main__":
    main()
