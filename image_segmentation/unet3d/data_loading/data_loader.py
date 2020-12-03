import random

import monai.transforms as transforms
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset


def get_train_transforms(**kwargs):
    rand_flip = RandFlip()
    rand_smooth = transforms.RandGaussianSmoothd(keys=["image"], prob=0.1)
    rand_noise = transforms.RandGaussianNoised(keys=["image"], std=0.1, prob=0.1)
    rand_scale = transforms.RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.1)
    mode = "trilinear"
    rand_zoom = transforms.RandZoomd(
        keys=["image", "label"], max_zoom=1.2, mode=(mode, "nearest"), align_corners=(True, None)
    )
    rand_rotate = transforms.RandRotated(keys=["image", "label"], prob=0.1, mode=["bilinear", "nearest"],
                                         range_x=15.0, range_y=15.0, range_z=15.0,
                                         keep_size=True, dtype=(np.float32, np.uint8))
    cast = transforms.CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8))
    train_transforms = transforms.Compose([rand_flip, rand_rotate, rand_zoom, cast, rand_noise, rand_smooth, rand_scale])
    train_transforms.set_random_state(seed=kwargs["seed"])
    return train_transforms


def get_data(data, training=True):
    img_key, lbl_key = ("img_train", "label_train") if training else ("img_val", "label_val")
    return data[img_key], data[lbl_key]


class MSDTrain(Dataset):
    def __init__(self, **kwargs):
        self.images, self.labels = get_data(data=kwargs["data"])
        self.train_transforms = get_train_transforms(cascade=False, **kwargs)
        patch_size, oversampling = kwargs["patch_size"], kwargs["oversampling"]
        self.rand_crop = RandBalancedCrop(patch_size=patch_size, oversampling=oversampling)
        self.pad = transforms.SpatialPadd(["image", "label"], patch_size, mode="reflect")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = {"image": np.load(self.images[idx]), "label": np.load(self.labels[idx])}
        data = self.pad(data)
        data = self.rand_crop(data)
        return self.train_transforms(data)


class MSDVal(Dataset):
    def __init__(self, **kwargs):
        self.images, self.labels = get_data(data=kwargs["data"], training=False)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": np.load(self.images[idx]), "label": np.load(self.labels[idx]), "name": self.labels[idx]}


class RandFlip(transforms.MapTransform):
    def __init__(self):
        self.axis = [1, 2, 3]
        self.prob = 1 / len(self.axis)

    def flip(self, data, axis):
        data["image"] = np.flip(data["image"], axis=axis).copy()
        data["label"] = np.flip(data["label"], axis=axis).copy()
        return data

    def __call__(self, data):
        for axis in self.axis:
            if random.random() < self.prob:
                data = self.flip(data, axis)
        return data


class RandBalancedCrop(transforms.MapTransform):
    def __init__(self, patch_size, oversampling, cascade=False):
        self.patch_size = patch_size
        self.oversampling = oversampling
        self.cascade = cascade

    def __call__(self, data):
        image, label = data["image"], data["label"]
        if random.random() < self.oversampling:
            image, label, cords = self.rand_foreg_cropd(image, label)
        else:
            image, label, cords = self._rand_crop(image, label)
        data.update({"image": image, "label": label})
        if self.cascade:
            return data, cords
        return data

    @staticmethod
    def randrange(max_range):
        return 0 if max_range == 0 else random.randrange(max_range)

    def get_cords(self, cord, idx):
        return cord[idx], cord[idx] + self.patch_size[idx]

    def _rand_crop(self, image, label):
        ranges = [s - p for s, p in zip(image.shape[1:], self.patch_size)]
        cord = [self.randrange(x) for x in ranges]
        low_x, high_x = self.get_cords(cord, 0)
        low_y, high_y = self.get_cords(cord, 1)
        low_z, high_z = self.get_cords(cord, 2)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]

    def rand_foreg_cropd(self, image, label):
        def adjust(foreg_slice, patch_size, label, idx):
            diff = patch_size[idx - 1] - (foreg_slice[idx].stop - foreg_slice[idx].start)
            sign = -1 if diff < 0 else 1
            diff = abs(diff)
            ladj = self.randrange(diff)
            hadj = diff - ladj
            low = max(0, foreg_slice[idx].start - sign * ladj)
            high = min(label.shape[idx], foreg_slice[idx].stop + sign * hadj)
            diff = patch_size[idx - 1] - (high - low)
            if diff > 0 and low == 0:
                high += diff
            elif diff > 0:
                low -= diff
            return low, high

        foreg_slices = ndimage.find_objects(label)
        foreg_slices = [x for x in foreg_slices if x is not None]
        if not foreg_slices:
            return self._rand_crop(image, label)
        foreg_slice = foreg_slices[random.randrange(len(foreg_slices))]
        low_x, high_x = adjust(foreg_slice, self.patch_size, label, 1)
        low_y, high_y = adjust(foreg_slice, self.patch_size, label, 2)
        low_z, high_z = adjust(foreg_slice, self.patch_size, label, 3)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]
