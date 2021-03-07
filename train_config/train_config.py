import os
from abc import ABCMeta, abstractmethod

from pietoolbelt.datasets.utils import DatasetsContainer
from pietoolbelt.losses.regression import RMSELoss
from pietoolbelt.models import ResNet18, ModelsWeightsStorage, ModelWithActivation, ResNet34, ClassificationModel
from piepline import TrainConfig, DataProducer, TrainStage, ValidationStage, MetricsProcessor, MetricsGroup
from torch import nn
from torch.nn import Module
from torch.optim import Adam

from dataset import create_augmented_dataset

__all__ = ['MyTrainConfig', 'ResNet18SegmentationTrainConfig', 'ResNet34SegmentationTrainConfig']

from .metrics import AMADMetric, RelativeMetric


class MyTrainConfig(TrainConfig, metaclass=ABCMeta):
    experiment_dir = 'train'
    batch_size = int('4')
    folds_num = 3

    def __init__(self, fold_indices: {}):
        model = self.create_model().cuda()

        dir = os.path.join('data', 'indices')

        train_dts = []
        for indices in fold_indices['train']:
            train_dts.append(create_augmented_dataset(is_train=True, indices_path=os.path.join(dir, indices + '.npy')))

        val_dts = create_augmented_dataset(is_train=False, indices_path=os.path.join(dir, fold_indices['val'] + '.npy'))

        workers_num = int('6')
        self._train_data_producer = DataProducer(DatasetsContainer(train_dts), batch_size=self.batch_size, num_workers=workers_num). \
            global_shuffle(True).pin_memory(True)
        self._val_data_producer = DataProducer(val_dts, batch_size=self.batch_size, num_workers=workers_num). \
            global_shuffle(True).pin_memory(True)

        train_metrics_proc = MetricsProcessor()
        val_metrics_proc = MetricsProcessor()
        train_metrics_proc.add_metrics_group(MetricsGroup('train').add(AMADMetric()).add(RelativeMetric()))
        val_metrics_proc.add_metrics_group(MetricsGroup('validation').add(AMADMetric()).add(RelativeMetric()))

        self.train_stage = TrainStage(self._train_data_producer, train_metrics_proc)
        self.val_stage = ValidationStage(self._val_data_producer, val_metrics_proc)

        loss = RMSELoss().cuda()
        optimizer = Adam(params=model.parameters(), lr=1e-4)

        super().__init__(model, [self.train_stage, self.val_stage], loss, optimizer)

    @staticmethod
    @abstractmethod
    def create_model(pretrained: bool = True) -> Module:
        pass


class ResNet18SegmentationTrainConfig(MyTrainConfig):
    model_name = 'resnet18'
    experiment_dir = os.path.join(MyTrainConfig.experiment_dir, model_name)

    @staticmethod
    def create_model(pretrained: bool = True) -> Module:
        """
        It is better to init model by separated method
        :return:
        """
        enc = ResNet18(in_channels=2)
        if pretrained:
            ModelsWeightsStorage().load(enc, 'imagenet', params={'cin': 2})
        model = ClassificationModel(enc, pool=nn.AdaptiveAvgPool2d((1, 1)), in_features=512, classes_num=98)
        return ModelWithActivation(model, activation='sigmoid')


class ResNet34SegmentationTrainConfig(MyTrainConfig):
    model_name = 'resnet34'
    experiment_dir = os.path.join(MyTrainConfig.experiment_dir, model_name)

    @staticmethod
    def create_model(pretrained: bool = True) -> Module:
        """
        It is better to init model by separated method
        :return:
        """
        enc = ResNet34(in_channels=2)
        if pretrained:
            ModelsWeightsStorage().load(enc, 'imagenet', params={'cin': 2})
        model = ClassificationModel(enc, pool=nn.AdaptiveAvgPool2d((1, 1)), in_features=512, classes_num=98)
        return ModelWithActivation(model, activation='sigmoid')
