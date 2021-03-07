import numpy as np
from piepline.train_config.metrics import AbstractMetric
from torch import Tensor
import torch


def rmse(predict: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean((predict - target) ** 2, axis=0))))


def amad(predict: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.mean(np.abs(predict - target), axis=0)))


def relative(predict: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.mean(np.abs(predict - target) / (target + 1e-6), axis=1)))


def rmse_torch(predict: Tensor, target: Tensor) -> float:
    return float(torch.mean(torch.sqrt(torch.mean((predict - target) ** 2, axis=0))).cpu())


def amad_torch(predict: Tensor, target: Tensor) -> float:
    return float(torch.mean(torch.mean(torch.abs(predict - target), axis=0)).cpu())


def relative_torch(predict: Tensor, target: Tensor) -> float:
    return float(torch.mean(torch.mean(torch.abs(predict - target) / (target + 1e-6), axis=0)).cpu())


class AMADMetric(AbstractMetric):
    def __init__(self):
        super().__init__("AMAD")

    def calc(self, output: Tensor, target: Tensor) -> np.ndarray or float:
        return amad_torch(output, target)


class RMSEMetric(AbstractMetric):
    def __init__(self):
        super().__init__("RMSE")

    def calc(self, output: Tensor, target: Tensor) -> np.ndarray or float:
        return rmse_torch(output, target)


class RelativeMetric(AbstractMetric):
    def __init__(self):
        super().__init__("relative")

    def calc(self, output: Tensor, target: Tensor) -> np.ndarray or float:
        return relative_torch(output, target)
