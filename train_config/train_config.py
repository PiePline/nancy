import os
from abc import ABCMeta, abstractmethod

from pietoolbelt.models.decoders.unet import UNetDecoder
from torch import nn
from torch.nn import Module
from torch.optim import Adam

from piepline.data_producer import DataProducer
from piepline.train_config.stages import TrainStage, ValidationStage
from piepline.train_config.train_config import BaseTrainConfig
from pietoolbelt.datasets.utils import DatasetsContainer
from pietoolbelt.losses.regression import RMSELoss
from pietoolbelt.models import ResNet18, ModelsWeightsStorage, ModelWithActivation, ResNet34

__all__ = ['TrainConfig', 'ResNet18SegmentationTrainConfig', 'ResNet34SegmentationTrainConfig']

from train_config.dataset import create_augmented_dataset
from config import INDICES_DIR


class TrainConfig(BaseTrainConfig, metaclass=ABCMeta):
    experiment_dir = 'train'
    batch_size = int('4')
    folds_num = 3

    def __init__(self, fold_indices: {}):
        model = self.create_model().cuda()

        train_dts = []
        for indices in fold_indices['train']:
            train_dts.append(create_augmented_dataset(is_train=True, indices_path=os.path.join(INDICES_DIR, indices + '.npy')))

        val_dts = create_augmented_dataset(is_train=True, indices_path=os.path.join(INDICES_DIR, fold_indices['val'] + '.npy'))

        workers_num = int('6')
        self._train_data_producer = DataProducer(DatasetsContainer(train_dts), batch_size=self.batch_size, num_workers=workers_num). \
            global_shuffle(True).pin_memory(True)
        self._val_data_producer = DataProducer(val_dts, batch_size=self.batch_size, num_workers=workers_num). \
            global_shuffle(True).pin_memory(True)

        self.train_stage = TrainStage(self._train_data_producer)
        self.val_stage = ValidationStage(self._val_data_producer)

        loss = RMSELoss().cuda()
        optimizer = Adam(params=model.parameters(), lr=1e-4)

        super().__init__(model, [self.train_stage, self.val_stage], loss, optimizer)

    @staticmethod
    @abstractmethod
    def create_model(pretrained: bool = True) -> Module:
        pass


class ResNet18SegmentationTrainConfig(TrainConfig):
    model_name = 'resnet18'
    experiment_dir = os.path.join(TrainConfig.experiment_dir, model_name)

    @staticmethod
    def create_model(pretrained: bool = True) -> Module:
        """
        It is better to init model by separated method
        :return:
        """
        enc = ResNet18(in_channels=3)
        if pretrained:
            ModelsWeightsStorage().load(enc, 'imagenet')
        model = UNetDecoder(enc, classes_num=1)
        return ModelWithActivation(model, activation='sigmoid')


class ResNet34SegmentationTrainConfig(TrainConfig):
    model_name = 'resnet34'
    experiment_dir = os.path.join(TrainConfig.experiment_dir, model_name)

    @staticmethod
    def create_model(pretrained: bool = True) -> Module:
        """
        It is better to init model by separated method
        :return:
        """
        enc = ResNet34(in_channels=3)
        if pretrained:
            ModelsWeightsStorage().load(enc, 'imagenet')
        model = UNetDecoder(enc, classes_num=1)
        return ModelWithActivation(model, activation='sigmoid')
