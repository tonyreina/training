import glob
import os

import numpy as np
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from utils.utils import is_main_process

from data_loading.data_loader import MSDTrain, MSDVal


def cross_validation(arr: np.ndarray, fold_idx: int, n_folds: int):
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

def get_data_split(path: str, fold: int, num_folds: int):
    imgs = list_files_with_pattern(path, "*_x.npy")
    lbls = list_files_with_pattern(path, "*_y.npy")
    assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"
    train_imgs, val_imgs = cross_validation(np.array(imgs), fold_idx=fold, n_folds=num_folds)
    train_labels = np.array([f.replace("_x", "_y") for f in train_imgs])
    val_labels = np.array([f.replace("_x", "_y") for f in val_imgs])
    return train_imgs.tolist(), train_labels.tolist(), val_imgs.tolist(), val_labels.tolist()


class DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data = None
        self.data_kwargs = None
        self.kfold = KFold(n_splits=self.args.nfolds, shuffle=True, random_state=12345)
        self.loader_kwargs = {
            "batch_size": self.args.batch_size,
            "shuffle": True,
            "pin_memory": True,
            "num_workers": self.args.num_workers,
            "drop_last": True,
        }
        self.val_loader_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "pin_memory": True,
            "num_workers": self.args.num_workers,
        }

    def setup(self, stage=None):
        img_lr_train, img_lr_val, lbl_lr_train, lbl_lr_val = 4 * [None]
        img_train, img_val, lbl_train, lbl_val = self.get_3d_data_split(self.args.data)

        self.data = {
            "img_train": img_train,
            "img_val": img_val,
            "label_train": lbl_train,
            "label_val": lbl_val,
            "lowres_data": {
                "img_train": img_lr_train,
                "img_val": img_lr_val,
                "label_train": lbl_lr_train,
                "label_val": lbl_lr_val,
            },
        }

        self.data_kwargs = {
            "data": self.data,
            "patch_size": self.args.input_shape,
            "oversampling": self.args.oversampling,
            "seed": self.args.seed,
        }

        if is_main_process():
            print(f"Number of examples: Train {len(img_train)} - Val {len(img_val)}")

    def train_dataloader(self):
        return DataLoader(MSDTrain(**self.data_kwargs), **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(MSDVal(**self.data_kwargs), **self.val_loader_kwargs)

    def test_dataloader(self):
        return DataLoader(MSDVal(**self.data_kwargs), **self.val_loader_kwargs)

    @staticmethod
    def get_split(data, train_idx, val_idx=None):
        train = list(np.array(data)[train_idx])
        if val_idx is None:
            return train
        val = list(np.array(data)[val_idx])
        return train, val

    @staticmethod
    def load_data(path, files_pattern):
        data = sorted(glob.glob(os.path.join(path, files_pattern)))
        assert len(data) > 0, f"Found no data at {path}"
        return data

    def get_3d_data_split(self, path):
        imgs = self.load_data(path, "*_x.npy")
        lbls = self.load_data(path, "*_y.npy")
        assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"

        train_idx, val_idx = list(self.kfold.split(imgs))[self.args.fold]
        imgs_train, imgs_val = self.get_split(imgs, train_idx, val_idx)
        lbls_train, lbls_val = self.get_split(lbls, train_idx, val_idx)
        return imgs_train, imgs_val, lbls_train, lbls_val
