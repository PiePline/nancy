import json
import os
from shutil import rmtree, copyfile

if __name__ == '__main__':
    output_path = 'result'
    remove_existing = True
    config_path = 'bagging/meta.json'
    train_dir = ''

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    if os.path.exists(output_path) and remove_existing:
        rmtree(output_path)

    res_meta = []
    for model in config:
        checkpoint = os.path.join(train_dir, model['model'], model['fold'], 'checkpoints', 'best', 'best_checkpoint.zip')

        model_file = os.path.join(model['model'], model['fold'], 'checkpoints', 'last')
        dst_path = os.path.join(output_path, model_file)

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        copyfile(checkpoint, os.path.join(dst_path, 'last_checkpoint.zip'))
        res_meta.append({'model': model['model'], 'path': model_file})

    with open(os.path.join(output_path, 'meta.json'), 'w') as meta_file:
        json.dump(res_meta, meta_file, indent=4)
