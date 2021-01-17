from icecream import ic
from ipdb import set_trace as st
from loguru import logger
import cv2
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import torch.utils.data as data


def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


class SpectrogramDataset(data.Dataset):
    def __init__(self,
                 df_trn_tp,
                 dir_data,
                 phase,
                 config):
        self.df_trn_tp = df_trn_tp
        self.unique_rec = df_trn_tp['recording_id'].unique()
        self.dir_data = dir_data
        self.img_size = config['params']['img_size']
        self.period = config['params']['period']
        self.shift_duration = config['params']['shift_duration']
        self.melspec_params = config['melspec_params']
        # self.melspectrogram_parameters = config['melspectrogram_parameters']

    def __len__(self):
        return len(self.df_trn_tp)

    def __getitem__(self, idx):
        series = self.df_trn_tp.iloc[idx, :]
        rec = series['recording_id']
        path_flac = f'{self.dir_data}{rec}.flac'
        species_id = series["species_id"]
        t_min = series['t_min']

        # load
        y, sr = sf.read(path_flac)

        # ランダムに切り取る範囲の左端
        effective_length = sr * self.period
        t_left = t_min - self.shift_duration
        if t_left < 0:
            # 0~t_minの間でスタート
            start = int(np.random.randint(sr * t_min))
        else:
            # t_leftからshift_durationの間でスタート
            start = int(
                    t_left * sr + np.random.randint(sr * self.shift_duration))

        if start/sr >= len(y)/sr - self.period:
            # start が右端に行き過ぎてeffective_length以下になってしまった時の処理
            start = int((len(y)/sr - self.period) * sr)

        # spectrogramの計算
        y_crop = y[start:start+effective_length].astype(np.float32)

        # logger.info(f'y_crop: {len(y_crop)}')
        melspec = librosa.feature.melspectrogram(
                y_crop,
                sr=sr,
                **self.melspec_params)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        # ----- 可視化 -----
        # t_max = df_rec['t_max'].values[idx_choice]
        # librosa.display.specshow(
        #         # melspec, sr=sr, x_axis='time', y_axis='mel')
        #         melspec, sr=sr, x_axis='time', y_axis='mel')
        # import matplotlib.pyplot as plt
        # plt.title(f'{rec} [{t_min}~{t_max}],[{start/sr:.1f}~{(start+effective_length)/sr:.1f}]')
        # plt.show()
        # -----

        # 入力画像の加工
        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(
                image,
                (int(width * self.img_size / height), self.img_size)
                )
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        # ラベルの作成
        labels = np.zeros(self.df_trn_tp.species_id.nunique(), dtype=int)
        labels[species_id] = 1
        # logger.info(f't_left: {t_left}')
        # logger.info(f'len(y_crop): {len(y_crop)}')
        # logger.info(f'start_sec: {start/sr:.3f}')
        # logger.info(f'image.shape: {image.shape}\n')
        return image, labels
