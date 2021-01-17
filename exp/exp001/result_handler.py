from ipdb import set_trace as st
from loguru import logger
import pandas as pd
import matplotlib.pylab as plt
import configuration as C


def save_model_architecture(dir_save, model):
    logger.info(f'model name: {model.__class__.__name__}')
    with open(f'{dir_save}/model.txt', 'w') as f:
        f.write(str(model.__repr__))


def save_loss_figure(fold_i, epochs, losses_train,
                     losses_valid, save_dir):
    fig = plt.figure()
    plt.plot(epochs, losses_train, '-x', label='train')
    plt.plot(epochs, losses_valid, '-x', label='valid')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend()
    fig.savefig(f'{save_dir}/loss_fold{fold_i}.png')


def save_result_csv(fold_i, best_loss, best_acc, save_dir, config):
    debug = config['globals']['debug']
    loss_name = config['loss']['name']
    model = C.get_model(config)
    df = pd.DataFrame({
        'debug': [debug],
        'fold': [fold_i],
        'model_name': [model.__class__.__name__],
        'loss_name': [loss_name],
        'best_loss_val': [round(best_loss, 6)],
        'best_acc_val': [round(best_acc, 6)],
        })
    df.to_csv(f'{save_dir}/result_fold{fold_i}.csv', index=False)
