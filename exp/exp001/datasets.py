import cv2
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from ipdb import set_trace as st
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
                 df,
                 dir_data,
                 phase,
                 config):
        self.df = df
        self.unique_rec = df['recording_id'].unique()
        self.dir_data = dir_data
        self.img_size = config['params']['img_size']
        self.period = config['params']['period']
        self.shift_duration = config['params']['shift_duration']
        self.melspec_params = config['melspec_params']
        # self.melspectrogram_parameters = config['melspectrogram_parameters']

    def __len__(self):
        return len(self.unique_rec)

    def __getitem__(self, idx):
        rec = self.unique_rec[idx]
        path_flac = f'{self.dir_data}{rec}.flac'
        df_rec = self.df.query('recording_id == @rec')
        n_label = len(df_rec)

        # どの labelを使うか選ぶ
        idx_choice = np.random.randint(n_label)

        species_id = df_rec["species_id"].values[idx_choice]
        t_min = df_rec['t_min'].values[idx_choice]
        # t_center = t_min + (t_max - t_min)/2

        # load
        y, sr = sf.read(path_flac)

        # ランダムに切り取る範囲の左恥
        # len_y = len(y)
        t_left = t_min - self.shift_duration
        effective_length = sr * self.period

        start = int(t_left * sr + np.random.randint(sr * self.shift_duration))
        y_crop = y[start:start+effective_length].astype(np.float32)

        melspec = librosa.feature.melspectrogram(
                y_crop,
                sr=sr,
                **self.melspec_params)
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        # -----
        # t_max = df_rec['t_max'].values[idx_choice]
        # librosa.display.specshow(
        #         # melspec, sr=sr, x_axis='time', y_axis='mel')
        #         melspec, sr=sr, x_axis='time', y_axis='mel')
        # import matplotlib.pyplot as plt
        # plt.title(f'{rec} [{t_min}~{t_max}], [{start/sr:.1f}~{(start+effective_length)/sr:.1f}]  ')
        # plt.show()
        # -----

        image = mono_to_color(melspec)
        height, width, _ = image.shape
        st()
        image = cv2.resize(
                image,
                (int(width * self.img_size / height), self.img_size)
                )
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        labels = np.zeros(len(BIRD_CODE), dtype=int)
        labels[BIRD_CODE[ebird_code]] = 1
        return image, labels
