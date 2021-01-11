import datasets
import numpy as np
from ipdb import set_trace as st
import torch.utils.data as data
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold


def get_trn_val_loader(df, phase, config):
    dataset_config = config["dataset"]
    name = dataset_config['name']
    loader_config = config["loader"][phase]

    dataset = datasets.__getattribute__(name)(
            df,
            datadir=datadir,
            phase=phase,
            config=dataset_config['params'])
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
