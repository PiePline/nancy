import json
import os

import cv2
import numpy as np
import torch
from torch.nn import Module
from tqdm import tqdm

from train.config import ARTIFACTS_DIR, DATASET_LABELS
from train.train_config.dataset import Dataset
from inference.u2net import U2NET
from torchvision import transforms
from skimage import transform, color


def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)

    return dn


class RescaleT(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample["imidx"], sample["image"], sample["label"]

        h, w = image.shape[:2]

        img = transform.resize(
            image, (self.output_size, self.output_size), mode="constant"
        )
        lbl = transform.resize(
            label,
            (self.output_size, self.output_size),
            mode="constant",
            order=0,
            preserve_range=True,
        )

        return {"imidx": imidx, "image": img, "label": lbl}


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):

        imidx, image, label = sample["imidx"], sample["image"], sample["label"]

        tmpLbl = np.zeros(label.shape)

        if np.max(label) < 1e-6:
            label = label
        else:
            label = label / np.max(label)

        # change the color space
        if self.flag == 2:  # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
                    np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0])
            )
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
                    np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1])
            )
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
                    np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2])
            )
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
                    np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0])
            )
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
                    np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1])
            )
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
                    np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2])
            )

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(
                tmpImg[:, :, 0]
            )
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(
                tmpImg[:, :, 1]
            )
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(
                tmpImg[:, :, 2]
            )
            tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(
                tmpImg[:, :, 3]
            )
            tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(
                tmpImg[:, :, 4]
            )
            tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(
                tmpImg[:, :, 5]
            )

        elif self.flag == 1:  # with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
                    np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0])
            )
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
                    np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1])
            )
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
                    np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2])
            )

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(
                tmpImg[:, :, 0]
            )
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(
                tmpImg[:, :, 1]
            )
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(
                tmpImg[:, :, 2]
            )

        else:  # with rgb color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {
            "imidx": torch.from_numpy(imidx),
            "image": torch.from_numpy(tmpImg),
            "label": torch.from_numpy(tmpLbl),
        }


def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})

    return sample


def process_img(model: Module, img: np.ndarray) -> np.ndarray:
    sample = preprocess(img)

    with torch.no_grad():

        if torch.cuda.is_available():
            inputs_test = torch.cuda.FloatTensor(sample["image"].cuda().unsqueeze(0).float())
        else:
            inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())

        d1, d2, d3, d4, d5, d6, d7 = model(inputs_test)

        pred = d1[:, 0, :, :]
        predict = norm_pred(pred)

        predict = predict.squeeze()
        mask = predict.cpu().detach().numpy()

        del d1, d2, d3, d4, d5, d6, d7, pred, predict, inputs_test, sample

    return cv2.resize(mask, (img.shape[1], img.shape[0]))


def mask2rle(img: np.ndarray):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.copy().flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


if __name__ == '__main__':
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)

    model = U2NET(3, 1)
    model.load_state_dict(torch.load('u2net.pth'))
    model = model.eval().cuda()

    dataset = Dataset()

    with open(DATASET_LABELS, 'w') as labels_file:
        result = {}
        for path in tqdm(dataset.get_items(), desc="Dataset labeling"):
            img = cv2.imread(path)
            mask = process_img(model, img)
            mask = (mask > 0.7).astype(np.uint8)
            result[os.path.splitext(os.path.basename(path))[0]] = mask2rle(mask)

        json.dump(result, labels_file, indent=4)
