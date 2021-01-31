from icecream import ic
from ipdb import set_trace as st
import os
import time
import yaml
import subprocess
from loguru import logger

import utils as U
import trainner
import configuration as C
import result_handler as rh


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
    dir_save, dir_save_ignore = U.init_exp(config)
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
