import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype, load_breast_cancer

from dataset_utils import load_voxel
import custom_utils as cut
import federated_feature_selection as ffs


def main():
    parser = argparse.ArgumentParser(description='Federated Feature Selection')
    parser.add_argument('-r', '--rounds', type=int, default=300,
                        help='Maximum number of communication rounds (default: 300)')
    args = parser.parse_args()

    X_all, y_all = load_voxel(return_Xy=True)

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X_all, y_all, test_size=0.20)

    y_train = np.array(ytrain.sort_values(by='timestamps').iloc[:, 1])
    y_test = np.array(ytest.sort_values(by='timestamps').iloc[:, 1])
    X_train = np.array(Xtrain.sort_values(by='timestamps').iloc[:, 1:])
    X_test = np.array(Xtest.sort_values(by='timestamps').iloc[:, 1:])
    num_classes = np.unique(y_train).shape[0]

    num_workers = 10
    n_train = X_train.shape[0]
    nums = [n_train // num_workers] * num_workers
    idxs = np.array([np.random.choice(np.arange(n_train),
                                      num, replace=False) for num in nums])

    print("{} workers holding {} patterns, respectively".format(
        idxs.shape[0], idxs.shape[1]))

    perc_nodes = [1.]
    perc_data = [1.]

    tolerance = 1e-6
    ce_max_samples = 100
    ce_max_steps = 10
    ce_tol = 1e-5
    selection_prob = 0.99
    alpha_smooth = 1.0
    MAX_ROUND = args.rounds
    alpha_step = (alpha_smooth) / (MAX_ROUND * ce_max_steps)

    njobs = 1

    base_dir = Path(__file__).resolve().parent.parent
    log_path = str(base_dir / 'stats' / 'fedfs-mav') + '/'
    if not os.path.isdir(log_path):
        print('Creating log dir: {}'.format(log_path))
        os.mkdir(log_path)
    print('Running Federated Crossentropy with alpha {}'.format(alpha_smooth))
    for frac_nodes in perc_nodes:
        for frac_data in perc_data:
            print(
                '*** New Configuration: [nodes {}, data {}]'.format(frac_nodes, frac_data))
            fs, net, prob, conv_round, fed_running_fs, fed_running_prob, dwl_data_before_conv, max_rounds, alpha_per_round = ffs.federated_feature_selection(
                X_train, y_train, idxs,
                fraction_selected_workers=frac_nodes,
                fraction_local_subsample=frac_data,
                early_stop_tolerance=tolerance,
                ce_selection_prob=selection_prob,
                ce_tolerance=ce_tol,
                alpha=alpha_smooth,
                alpha_step=alpha_step,
                max_sample_ce=ce_max_samples,
                max_step_ce=ce_max_steps,
                fed_max_round=MAX_ROUND,
                verbose=True,
                verbose_scenario=False,
                njobs=njobs)

            print('Saving logs...')
            sfx = '-nodes({})_data({})-alpha_({}).csv'.format(
                frac_nodes, frac_data, alpha_smooth)
            cut.write_tocsv(log_path + 'fed_fs' + sfx,
                            map(lambda x: [x], fs))
            cut.write_tocsv(log_path + 'fed_prob' + sfx,
                            map(lambda x: [x], prob))
            cut.write_tocsv(log_path + 'fed_dwl' + sfx,
                            map(lambda x: [x], dwl_data_before_conv))
            cut.write_tocsv(log_path + 'fed_running_prob' + sfx,
                            fed_running_prob)
            cut.write_tocsv(log_path + 'fed_running_fs' + sfx, fed_running_fs)

            cut.write_tocsv(log_path + 'fed_running_alpha' +
                            sfx, map(lambda x: [x], alpha_per_round))
            cut.write_tocsv(log_path + 'fed_running_stats' + sfx,
                            [[conv_round, net, max_rounds, tolerance, ce_max_samples,
                              ce_max_steps, int(num_workers * frac_nodes)]],
                            header=['Convergence Round', 'Net. Overhead', 'Max Rounds',
                                    'Tolerance', 'CE max samples', 'CE max steps',
                                    'Active workers'])


if __name__ == "__main__":
    main()
