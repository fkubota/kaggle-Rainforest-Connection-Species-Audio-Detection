from ipdb import set_trace as st
from loguru import logger


def save_model_architecture(dir_save, model):
    logger.info(':: in ::')
    logger.info(f'model name: {model.__class__.__name__}')
    with open(f'{dir_save}/model.txt', 'w') as f:
        f.write(str(model.__repr__))
    logger.info(':: out ::')
