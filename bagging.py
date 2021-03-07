import argparse
import sys

from pietoolbelt.metrics.cpu.regression import relative, rmse, amad
from pietoolbelt.steps.regression.bagging import Bagging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('-p', '--predicts_path', type=str, help='Path to predicts', required=True)

    if len(sys.argv) < 2:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)

    args = parser.parse_args()
    bagging = Bagging(args.predicts_path, main_metric={'relative': relative}, workers_num=4, predicts_with_headers=True,
                      predicts_names='predicted_mes.csv', targets_names='target_mes.csv')

    bagging.add_metric(rmse, 'rmse').add_metric(amad, 'amad').run('bagging')
