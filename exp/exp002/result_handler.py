from ipdb import set_trace as st
from loguru import logger
import pandas as pd
import matplotlib.pylab as plt
import configuration as C


def save_model_architecture(dir_save, model):
    logger.info(f'model name: {model.__class__.__name__}')
    with open(f'{dir_save}/model.txt', 'w') as f:
        f.write(str(model.__repr__))


def save_plot_figure(fold_i, epochs, losses_train, accs_val,
                     losses_valid, save_dir):
    fig, ax1 = plt.subplots()

    # ax1
    ax1.plot(epochs, losses_train, '-x', label='loss_trn')
    ax1.plot(epochs, losses_valid, '-x', label='loss_val')
    ax1.set_xlabel('epoch')
    ax1.set_xlabel('loss')
    ax1.grid()
    ax1.legend()

    # ax2
    ax2 = ax1.twinx()
    ax2.plot(epochs, accs_val, '-x', label='acc_val')
    ax2.set_xlabel('acc')

    # savefig
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
