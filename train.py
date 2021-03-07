import argparse
import sys

import torch
import numpy as np

from piepline import Trainer, FileStructManager
from piepline.builtin.monitors.tensorboard import TensorboardMonitor

from train.train_config.train_config import MyTrainConfig


def init_trainer(config_type: type(MyTrainConfig), folds: dict, fsm: FileStructManager) -> Trainer:
    config = config_type(folds)

    trainer = Trainer(config, fsm, device=torch.device('cuda'))
    tensorboard = TensorboardMonitor(fsm, is_continue=False)
    trainer.monitor_hub.add_monitor(tensorboard)


    def get_m():
        return np.mean(config.val_stage.metrics_processor().get_metrics()['groups'][0].metrics()[1].get_values())


    trainer.set_epoch_num(int('300'))
    trainer.enable_lr_decaying(coeff=0.5, patience=10, target_val_clbk=get_m)
    trainer.add_on_epoch_end_callback(lambda: tensorboard.update_scalar('params/lr', trainer.data_processor().get_lr()))

    trainer.enable_best_states_saving(get_m)
    trainer.add_stop_rule(lambda: trainer.data_processor().get_lr() < 1e-6)


    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('-m', '--model', type=str, help='Train one model', required=True, choices=['resnet18', 'resnet34'])

    if len(sys.argv) < 2:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)
    args = parser.parse_args()

    if args.model == "resnet18":
    

    if args.model == "resnet34":
    

