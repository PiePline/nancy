import argparse
import os
import sys

import torch
import numpy as np
from piepline.builtin.monitors.tensorboard import TensorboardMonitor
from piepline.monitoring.monitors import FileLogMonitor, ConsoleLossMonitor
from piepline.train import Trainer
from piepline.utils.checkpoints_manager import CheckpointsManager
from piepline.utils.fsm import FileStructManager

from pietoolbelt.metrics.torch.segmentation import SegmentationMetricsProcessor
from pietoolbelt.steps.common.train import FoldedTrainer
from piepline.monitoring.hub import MonitorHub

from src.train.config import folds_indices_files
from src.train.train_config.train_config import TrainConfig, ResNet18SegmentationTrainConfig


def init_trainer(config_type: type(TrainConfig), folds: dict, fsm: FileStructManager) -> Trainer:
    config = config_type(folds)

    train_metrics_proc = SegmentationMetricsProcessor(stage_name='train').subscribe_to_stage(config.train_stage)
    val_metrics_proc = SegmentationMetricsProcessor(stage_name='validation').subscribe_to_stage(config.val_stage)

    trainer = Trainer(config, fsm, device=torch.device('cuda'))

    file_log_monitor = FileLogMonitor(fsm).write_final_metrics()
    console_monitor = ConsoleLossMonitor()
    tensorboard_monitor = TensorboardMonitor(fsm, is_continue=False)
    mh = MonitorHub(trainer).subscribe2metrics_processor(train_metrics_proc)\
        .subscribe2metrics_processor(val_metrics_proc)\
        .add_monitor(file_log_monitor).add_monitor(console_monitor).add_monitor(tensorboard_monitor)

    config.train_stage._stage_end_event.add_callback(lambda stage: mh.update_losses({'train': config.train_stage.get_losses()}))
    config.val_stage._stage_end_event.add_callback(lambda stage: mh.update_losses({'validation': config.val_stage.get_losses()}))

    def get_m():
        return np.mean(val_metrics_proc.get_metrics()['groups'][0].metrics()[1].get_values())

    trainer.set_epoch_num(70)
    trainer.enable_lr_decaying(coeff=0.5, patience=10, target_val_clbk=get_m)

    CheckpointsManager(fsm=fsm).subscribe2trainer(trainer)

    # trainer.enable_best_states_saving(get_m)
    # trainer.add_stop_rule(lambda: trainer.data_processor().get_lr() < 1e-6)

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('-m', '--model', type=str, help='Train one model', required=True, choices=['resnet18', 'resnet34'])

    if len(sys.argv) < 2:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)
    args = parser.parse_args()

    folds_dict = {'fold_{}.npy'.format(i): 'fold_{}'.format(i) for i in range(TrainConfig.folds_num)}

    if args.model == "resnet18":
        folded_trainer = FoldedTrainer(folds=list(folds_indices_files.keys()))
        folded_trainer.run(init_trainer=lambda fsm, folds: init_trainer(ResNet18SegmentationTrainConfig, folds, fsm),
                           model_name='resnet18', out_dir=os.path.join('artifacts', 'train'))
