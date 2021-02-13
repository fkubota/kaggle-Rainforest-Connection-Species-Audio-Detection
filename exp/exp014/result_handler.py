from ipdb import set_trace as st
from icecream import ic
from loguru import logger
import pandas as pd
import matplotlib.pylab as plt
import configuration as C
import utils as U


def save_model_architecture(dir_save, model):
    logger.info(f'model name: {model.__class__.__name__}')
    with open(f'{dir_save}/model.txt', 'w') as f:
        f.write(str(model.__repr__))


def save_plot_figure(fold_i, epochs, losses_train, accs_val, lwlraps_val,
                     losses_valid, save_dir):
    fig, ax1 = plt.subplots()

    # ax1
    ax1.plot(epochs, losses_train, '-x', label='loss_trn', color='blue')
    ax1.plot(epochs, losses_valid, '-o', label='loss_val', color='blue')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.grid()
    # ax1.legend()

    # ax2
    ax2 = ax1.twinx()
    ax2.plot(epochs, accs_val, '-.', label='acc_val', color='red')
    ax2.plot(epochs, lwlraps_val, '-.', label='lwlrap_val', color='orange')
    ax2.set_ylabel('score')

    # savefig
    fig.legend()
    fig.savefig(f'{save_dir}/loss_fold{fold_i}.png')


def save_result_csv(fold_i, best_loss, best_acc, best_lwlrap,
                    save_dir, config):
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
        'best_lwlrap_val': [round(best_lwlrap, 6)]
        })
    df.to_csv(f'{save_dir}/result_fold{fold_i}.csv', index=False)


def save_confusion_matrix(truth, pred, n_classes, save_dir):
    '''
    confusion matrix を保存する
    '''
    save_path = f'{save_dir}/confusion_matrix.png'
    fig = U.plot_confusion_matrix(truth, pred, n_classes)
    fig.savefig(save_path)
