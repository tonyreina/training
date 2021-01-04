import os
import glob
from typing import Tuple

import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset, DataLoader

from data_loading.dali_loader import get_dali_loader


def cross_validation(arr: np.ndarray, fold_idx: int, n_folds: int) -> Tuple[np.array, np.array]:
    """ Split data into folds for training and evaluation
    :param arr: Collection items to split
    :param fold_idx: Index of crossvalidation fold
    :param n_folds: Total number of folds
    :return: Train and Evaluation folds
    """
    if fold_idx < 0 or fold_idx >= n_folds:
        raise ValueError('Fold index has to be [0, n_folds). Received index {} for {} folds'.format(fold_idx, n_folds))
    _folders = np.array_split(arr, n_folds)
    return np.concatenate(_folders[:fold_idx] + _folders[fold_idx + 1:]), _folders[fold_idx]


def list_files_with_pattern(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def get_data_split(path: str, fold: int, num_folds: int) -> Tuple[list, list, list, list]:
    imgs = list_files_with_pattern(path, "*_x.npy")
    lbls = list_files_with_pattern(path, "*_y.npy")
    assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"
    train_imgs, val_imgs = cross_validation(np.array(imgs), fold_idx=fold, n_folds=num_folds)
    train_labels = np.array([f.replace("_x", "_y") for f in train_imgs])
    val_labels = np.array([f.replace("_x", "_y") for f in val_imgs])
    print("Found {} files. Training set: {}. Validation set: {}.".format(len(imgs), len(train_imgs), len(val_imgs)))
    return train_imgs.tolist(), train_labels.tolist(), val_imgs.tolist(), val_labels.tolist()


def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def get_split(data, train_idx, val_idx):
    train = list(np.array(data)[train_idx])
    val = list(np.array(data)[val_idx])
    return train, val


def get_nn_data_split(path: str, fold: int, num_folds: int):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=12345)
    imgs = load_data(path, "*_x.npy")
    lbls = load_data(path, "*_y.npy")
    assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"

    train_idx, val_idx = list(kfold.split(imgs))[fold]
    imgs_train, imgs_val = get_split(imgs, train_idx, val_idx)
    lbls_train, lbls_val = get_split(lbls, train_idx, val_idx)
    return imgs_train, imgs_val, lbls_train, lbls_val


class SyntheticDataset(Dataset):
    def __init__(self, channels_in=1, channels_out=3, shape=(128, 128, 128),
                 device="cpu", layout="NCDHW", scalar=False):
        shape = tuple(shape)
        x_shape = (channels_in,) + shape if layout == "NCDHW" else shape + (channels_in,)
        self.x = torch.rand((32, *x_shape), dtype=torch.float32, device=device, requires_grad=False)
        if scalar:
            self.y = torch.randint(low=0, high=channels_out - 1, size=(32, *shape), dtype=torch.int32,
                                   device=device, requires_grad=False)
            self.y = torch.unsqueeze(self.y, dim=1 if layout == "NCDHW" else -1)
        else:
            y_shape = (channels_out,) + shape if layout == "NCDHW" else shape + (channels_out,)
            self.y = torch.rand((32, *y_shape), dtype=torch.float32, device=device, requires_grad=False)

    def __len__(self):
        return 64

    def __getitem__(self, idx):
        return self.x[idx % 32], self.y[idx % 32]


def get_data_loaders(flags, num_shards, device_id):
    if flags.loader == "synthetic":
        dataset = SyntheticDataset(scalar=True, shape=flags.input_shape, layout=flags.layout)
        train_dataloader = DataLoader(dataset,
                                      batch_size=flags.batch_size,
                                      shuffle=flags.benchmark is False,
                                      num_workers=flags.num_workers,
                                      pin_memory=True,
                                      drop_last=True)

        val_dataloader = None

    elif "dali" in flags.loader:
        x_train, x_val, y_train, y_val = get_nn_data_split(flags.data_dir, flags.fold, flags.num_folds)
        train_dataloader = get_dali_loader(flags, x_train, y_train, mode="train",
                                           num_shards=num_shards, device_id=device_id)
        val_dataloader = get_dali_loader(flags, x_val, y_val, mode="validation",
                                         num_shards=num_shards, device_id=device_id)

    else:
        raise ValueError("Loader {} unknown.".format(flags.loader))

    return train_dataloader, val_dataloader
