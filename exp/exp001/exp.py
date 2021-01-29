from icecream import ic
from ipdb import set_trace as st
import os
import time
import yaml
import wandb
import subprocess
from loguru import logger
import utils as U
import trainner
import configuration as C
import result_handler as rh


def init_exp(config):
    '''
    dir_saveの作成と、dir_saveの取得
    '''
    logger.info(':: in ::')

    # git の hash値を取得
    cmd = "git rev-parse --short HEAD"
    hash_ = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    logger.info(f'hash: {hash_}')

    # 保存ディレクトリの用意
    dir_save, dir_save_ignore, exp_name = U.get_save_dir_exp(config)
    logger.info(f'exp_name: {exp_name}')
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    if not os.path.exists(dir_save_ignore):
        os.makedirs(dir_save_ignore)

    # set_seed
    U.set_seed(config['globals']['seed'])

    logger.info(':: out ::')
    return dir_save, dir_save_ignore


def main():
    # start
    start = time.time()
    logger.remove()
    logger.add('exp.log', mode='w')
    logger.info('='*20)
    logger.info('::: exp start :::')
    logger.info('='*20)
    pwd = os.path.dirname(os.path.abspath(__file__))
    with open(f'{pwd}/config.yml', 'r') as yml:
        config = yaml.safe_load(yml)

    # init
    config = U.set_debug_config(config)
    dir_save, dir_save_ignore = init_exp(config)
    rh.save_model_architecture(dir_save, C.get_model(config))

    # train
    trainner.train_cv(config)

    # end
    end = time.time()
    time_all = end - start
    logger.info(f'elapsed time: {U.sec2time(time_all)}')
    logger.info('::: exp end :::')


if __name__ == "__main__":
    main()
