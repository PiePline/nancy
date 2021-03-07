import os
from pietoolbelt.steps.stratification import DatasetStratification

from train_config.dataset import create_dataset
from train_config.train_config import TrainConfig
from config import INDICES_DIR

if __name__ == '__main__':
    if not os.path.exists(INDICES_DIR):
        os.makedirs(INDICES_DIR)

    test_part = 0.1
    folds_dict = {'fold_{}.npy'.format(i): (1 - test_part) / TrainConfig.folds_num for i in range(TrainConfig.folds_num)}

    strat = DatasetStratification(create_dataset(), lambda x: 1)
    strat.run(dict(folds_dict, **{'test.npy': test_part}), INDICES_DIR)
