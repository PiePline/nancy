import argparse
import json
import os
import sys

import numpy as np
import torch
from pietoolbelt.metrics.cpu.regression import amad, rmse, relative
from piepline import FileStructManager, Predictor
from tqdm import tqdm

from train.train_config.dataset import create_dataset
from train.train_config.train_config import ResNet18SegmentationTrainConfig, ResNet34SegmentationTrainConfig, MyTrainConfig


def predict(config_type: type(MyTrainConfig)):
    output_path = 'predicts'
    model = config_type.model_name

    meta_file_path = os.path.join(output_path, 'meta.json')
    meta_info = []
    if os.path.exists(meta_file_path):
        with open(meta_file_path, 'r') as meta_file:
            meta_info = json.load(meta_file)

    for fold in os.listdir(config_type.experiment_dir):
        dataset = create_dataset(indices_path='data/indices/test.npy')

        output_dir = os.path.join(output_path, model, fold)
        if not os.path.exists(output_dir) and not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        fsm = FileStructManager(base_dir=os.path.join(config_type.experiment_dir, fold), is_continue=True)
        predictor = Predictor(config_type.create_model().cuda(), fsm=fsm)

        predicted_mes, target_mes = [dataset.mes_names()], [dataset.mes_names()]
        for dat in tqdm(dataset):
            img = torch.from_numpy(np.expand_dims(np.moveaxis(dat['data'], -1, 0).astype(np.float32), axis=0) / 128 - 1).cuda()
            predict = predictor.predict({'data': img}).data.cpu().numpy()
            predicted_mes.append(dataset.target_to_array(np.squeeze(predict), dat['target']['m12']))
            target_mes.append([float(dat['target'][m]) for m in dataset.mes_names()])

        with open(os.path.join(output_dir, 'predicted_mes.csv'), 'w') as out_file:
            np.savetxt(out_file, predicted_mes, delimiter=',', fmt='%s')
        with open(os.path.join(output_dir, 'target_mes.csv'), 'w') as out_file:
            np.savetxt(out_file, target_mes, delimiter=',', fmt='%s')

        predicted_mes, target_mes = np.array(predicted_mes[1:], dtype=np.float32), np.array(target_mes[1:], dtype=np.float32)
        meta_info.append({'model': config_type.model_name, 'fold': fold, 'path': os.path.join(model, fold),
                          'metrics': {'amad': amad(predicted_mes, target_mes),
                                      'rmse': rmse(predicted_mes, target_mes),
                                      'relative': relative(predicted_mes, target_mes)}})

    with open(meta_file_path, 'w') as meta_file:
        json.dump(meta_info, meta_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('-m', '--model', type=str, help='Model to predict', required=True, choices=['resnet18', 'resnet34'])

    if len(sys.argv) < 3:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)
    args = parser.parse_args()

    if args.model == 'resnet18':
        predict(ResNet18SegmentationTrainConfig)
    elif args.model == 'resnet34':
        predict(ResNet34SegmentationTrainConfig)
    else:
        raise Exception("Train pipeline doesn't implemented for model '{}'".format(args.model))
