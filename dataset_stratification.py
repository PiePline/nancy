import os
from pietoolbelt.steps.stratification import DatasetStratification

from train.train_config.dataset import create_dataset
from train.train_config.train_config import MyTrainConfig

if __name__ == '__main__':
    out_dir = os.path.join('data', 'indices')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    test_part = 0.1
    folds_dict = {'fold_{}.npy'.format(i): (1 - test_part) / MyTrainConfig.folds_num for i in range(MyTrainConfig.folds_num)}

    strat = DatasetStratification(create_dataset(), lambda x: int(x['m16'] // 100))
    strat.run(dict(folds_dict, **{'test.npy': test_part}), out_dir)
