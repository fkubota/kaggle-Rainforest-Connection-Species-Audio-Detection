from icecream import ic
from ipdb import set_trace as st
import os
import time
import yaml
from loguru import logger

import utils as U
import trainner
import configuration as C
import result_handler as rh


def run_exp(run_name, config_update):
    # start
    start = time.time()
    pwd = os.path.dirname(os.path.abspath(__file__))
    with open(f'{pwd}/config.yml', 'r') as yml:
        config = yaml.safe_load(yml)

    # init
    dir_save, _, config = U.init_exp(config, config_update, run_name)
    config = U.set_debug_config(config)
    _, _, exp_name = U.get_save_dir_exp(config, run_name)
    rh.save_model_architecture(dir_save, C.get_model(config))

    # train
    logger.info('='*30)
    logger.info(f'::: {exp_name}_{run_name} start :::')
    logger.info('='*30)
    trainner.train_cv(config, run_name)

    # end
    end = time.time()
    time_all = end - start
    logger.info(f'elapsed time: {U.sec2time(time_all)}')
    logger.info(f'::: {exp_name}_{run_name} end :::\n\n')


def start_sweep_dict(list_config_str):
    '''
    dictのリスト分の掃引
    '''
    logger.remove()
    logger.add('exp.log', mode='w')

    for i_run, config_str in enumerate(list_config_str, 1):
        config_update = yaml.safe_load(config_str)
        run_name = f'run{str(i_run).zfill(3)}'
        run_exp(run_name, config_update)


def main():
    list_config_str = [
            '''
            globals:
                debug: True
            ''',
            ]

    start_sweep_dict(list_config_str)


if __name__ == "__main__":
    main()
