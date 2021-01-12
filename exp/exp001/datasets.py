import cv2
import librosa
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
        self.img_size = config['img_size']
        self.period = config['period']
        # self.melspectrogram_parameters = config['melspectrogram_parameters']

    def __len__(self):
        return len(self.unique_rec)

    def __getitem__(self, idx):
        rec = self.unique_rec[idx]
        df_rec = self.df.query('recording_id == @rec')
        species_id = df_rec["species_id"].values[0]
        path_flac = f'{self.dir_data}{rec}.flac'

        y, sr = sf.read(path_flac)
        st()

        len_y = len(y)
        effective_length = sr * self.period

        start = np.random.randint(len_y - effective_length)
        y = y[start:start + effective_length].astype(np.float32)

        melspec = librosa.feature.melspectrogram(
                y, sr=sr, **self.melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        labels = np.zeros(len(BIRD_CODE), dtype=int)
        labels[BIRD_CODE[ebird_code]] = 1
        return image, labels
