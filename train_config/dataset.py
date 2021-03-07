import json
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from albumentations import Compose, GaussNoise

from pietoolbelt.datasets.common import BasicDataset, get_root_by_env
from pietoolbelt.datasets.utils import AugmentedDataset

from train.config import DATASET_LABELS


class Dataset:
    def __init__(self):
        root = get_root_by_env("CLOTHING_DATASET")

        self._items = []
        for file in os.listdir(os.path.join(root, 'images_original')):
            self._items.append(os.path.join(root, 'images_original', file))

        self._items.sort(key=lambda x: os.path.splitext(x)[0])

    def get_items(self) -> List[str]:
        return self._items


def rle2mask(rle: [int], shape: Tuple[int, int]):
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    array = np.asarray(rle)
    starts = array[0::2]
    lengths = array[1::2]

    for index, start in enumerate(starts):
        mask[start:start + lengths[index]] = 255

    return mask.reshape(shape)


class LabeledDataset(BasicDataset, Dataset):
    def __init__(self):
        Dataset.__init__(self)
        BasicDataset.__init__(self, self._items)

        with open(DATASET_LABELS, 'r') as labels_file:
            labels = json.load(labels_file)

        self._labels = {k: [int(vav) for vav in v.split(' ')] for k, v in labels.items()}

    def _interpret_item(self, item) -> any:
        img = cv2.imread(item)
        index = os.path.splitext(os.path.basename(item))[0]
        mask = rle2mask(self._labels[index], (img.shape[0], img.shape[1]))
        return {'data': img, 'target': mask.astype(np.uint8)}


class Augmentations:
    def __init__(self, is_train: bool, to_pytorch: bool):
        if is_train:
            self._aug = Compose([
                GaussNoise(p=0.5),

            ], p=1)
        else:
            self._aug = None

        self._need_to_pytorch = to_pytorch

    def augmentate(self, data: {}):
        if self._aug is not None:
            augmented = self._aug(image=data['data'])
            img = augmented['image']
        else:
            img = data['data']

        if self._need_to_pytorch:
            img = self.img_to_pytorch(img)

        return {'data': img, 'target': self.data_to_pytorch(data['target'])}

    @staticmethod
    def img_to_pytorch(image):
        return torch.from_numpy(np.moveaxis(image, -1, 0).astype(np.float32) / 128 - 1)

    @staticmethod
    def data_to_pytorch(target):
        return torch.from_numpy(target.astype(np.float32))


def create_dataset(indices_path: str = None) -> 'LabeledDataset':
    dataset = LabeledDataset()

    if indices_path is not None:
        dataset.load_indices(indices_path).remove_unused_data()
    return dataset


def create_augmented_dataset(is_train: bool, to_pytorch: bool = True, indices_path: str = None) -> 'AugmentedDataset':
    dataset = create_dataset(indices_path)
    augs = Augmentations(is_train, to_pytorch)

    return AugmentedDataset(dataset).add_aug(augs.augmentate)
