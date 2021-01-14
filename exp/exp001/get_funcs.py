import datasets
import criterion
import model_list
import numpy as np
from torch import nn
from loguru import logger
from ipdb import set_trace as st
import torch.utils.data as data
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold


def get_trn_val_loader(df, phase, config):
    dataset_config = config["dataset"]
    name = dataset_config['name']
    # params = dataset_config['params']
    loader_config = config["loader"][phase]
    dir_data = config['globals']['dir_train']

    dataset = datasets.__getattribute__(name)(
            df,
            dir_data=dir_data,
            phase=phase,
            config=dataset_config)
    # 動作確認
    # dataset.__getitem__(2)  # single label
    # dataset.__getitem__(14)   # multi labels

    loader = data.DataLoader(dataset, **loader_config)
    return loader


def get_index_fold(trn_tp, i_fold, config):
    """
    recording_idでわける
    """
    # config
    config_split = config['split']
    n_fold = config_split['n_fold']
    seed = config_split['seed']

    recording_ids = trn_tp['recording_id'].values
    dummy_x = np.zeros(len(recording_ids))
    dummy_x_shf, recording_ids_shf = shuffle(
            dummy_x, recording_ids, random_state=seed)
    splitter = GroupKFold(n_splits=n_fold)
    return list(
            splitter.split(
                X=dummy_x,
                groups=recording_ids))[i_fold]


def get_criterion(config):
    loss_config = config["loss"]
    loss_name = loss_config["name"]
    loss_params = loss_config["params"]
    if (loss_params is None) or (loss_params == ""):
        loss_params = {}

    if hasattr(nn, loss_name):
        criterion_ = nn.__getattribute__(loss_name)(**loss_params)
    else:
        criterion_cls = criterion.__getattribute__(loss_name)
        if criterion_cls is not None:
            criterion_ = criterion_cls(**loss_params)
        else:
            raise NotImplementedError

    return criterion_


def get_model(config):
    logger.info(':: in ::')
    model_config = config["model"]
    model_name = model_config["name"]
    model_params = model_config["params"]

    model = model_list.__getattribute__(model_name)(model_params)

    # model = eval(model_name)(model_params)
    logger.info(':: out ::')
    return model

