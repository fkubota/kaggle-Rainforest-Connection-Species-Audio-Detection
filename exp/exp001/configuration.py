from ipdb import set_trace as st
from loguru import logger
import os
import numpy as np
from torch import nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold
import utils as U
import datasets
import criterion
import model_list


def get_trn_val_loader(df, phase, config):
    pwd = os.path.dirname(os.path.abspath(__file__))
    dataset_config = config["dataset"]
    name = dataset_config['name']
    # params = dataset_config['params']
    loader_config = config["loader"][phase]
    dir_data = f"{pwd}/{config['path']['dir_train']}"

    dataset = datasets.__getattribute__(name)(
            df,
            dir_data=dir_data,
            phase=phase,
            config=dataset_config)
    # 動作確認
    # dataset.__getitem__(3)  # single label
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
    debug = config['globals']['debug']

    recording_ids = trn_tp['recording_id'].values

    dummy_x = np.zeros(len(recording_ids))
    dummy_x_shf, recording_ids_shf = shuffle(
            dummy_x, recording_ids, random_state=seed)

    splitter = GroupKFold(n_splits=n_fold)
    trn_idxs, val_idxs = list(splitter.split(
        X=dummy_x_shf, groups=recording_ids_shf
        ))[i_fold]

    if debug:
        trn_idxs, val_idxs = U.get_debug_idx(
                trn_tp, trn_idxs, val_idxs, config)

    return trn_idxs, val_idxs


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
    model_config = config["model"]
    model_name = model_config["name"]
    model_params = model_config["params"]

    model = model_list.__getattribute__(model_name)(model_params)

    # model = eval(model_name)(model_params)
    return model


def get_optimizer(model, config):
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name")

    return optim.__getattribute__(optimizer_name)(model.parameters(),
                                                  **optimizer_config["params"])


def get_scheduler(optimizer, config):
    scheduler_config = config["scheduler"]
    scheduler_name = scheduler_config.get("name")

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **scheduler_config["params"])
