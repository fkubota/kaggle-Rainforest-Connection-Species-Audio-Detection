import os
import yaml
import subprocess
import numpy as np
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
    dir_save_exp = f'{config["globals"]["dir_save"]}{exp_name}'
    if not os.path.exists(dir_save_exp):
        os.makedirs(dir_save_exp)

    logger.info(':: out ::')
    return dir_save_exp


def main():
    logger.remove()
    logger.add('exp.log', mode='w')
    logger.info('='*20)
    logger.info('::: exp start :::')
    logger.info('='*20)
    with open('config.yml', 'r') as yml:
        config = yaml.safe_load(yml)

    # init
    dir_save_exp = init_exp(config)
    print(dir_save_exp)


if __name__ == "__main__":
    main()
