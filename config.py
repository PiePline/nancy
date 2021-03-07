import os
from train_config.config import folds_num

ARTIFACTS_DIR = 'artifacts'
DATASET_ROOT = os.path.join(ARTIFACTS_DIR, 'dataset')
DATASET_LABELS = os.path.join(DATASET_ROOT, 'labels.json')
INDICES_DIR = os.path.join(ARTIFACTS_DIR, 'indices')

folds_indices_files = {'fold_{}'.format(i): 'fold_{}.npy'.format(i) for i in range(folds_num)}
