import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
STATS_DIR = BASE_DIR / 'stats' / 'fedfs-mav'
PLOTS_DIR = BASE_DIR / 'plots'
sys.path.insert(0, str(BASE_DIR / 'src'))


def load_stats():
    sfx = 'nodes(1.0)_data(1.0)-alpha_(1.0).csv'

    running_stats = pd.read_csv(STATS_DIR / f'fed_running_stats-{sfx}')
    conv_round = int(running_stats['Convergence Round'].iloc[0])
    net_overhead = int(running_stats['Net. Overhead'].iloc[0])
    max_rounds = int(running_stats['Max Rounds'].iloc[0])
    active_workers = int(running_stats['Active workers'].iloc[0])

    p_final = np.loadtxt(STATS_DIR / f'fed_prob-{sfx}', delimiter=',')
    alpha_per_round = np.loadtxt(STATS_DIR / f'fed_running_alpha-{sfx}', delimiter=',')

    selected_features = np.where(p_final >= 0.99)[0]

    fs_per_round = []
    with open(STATS_DIR / f'fed_running_fs-{sfx}', 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                fs_per_round.append([int(x) for x in line.split(',')])
            else:
                fs_per_round.append([])

    num_feats_per_round = [len(x) for x in fs_per_round]

    return {
        'conv_round': conv_round,
        'net_overhead': net_overhead,
        'max_rounds': max_rounds,
        'active_workers': active_workers,
        'p_final': p_final,
        'alpha_per_round': alpha_per_round,
        'selected_features': selected_features,
        'num_feats_per_round': num_feats_per_round,
    }


def load_dataset():
    from dataset_utils import load_voxel
    X_all, y_all = load_voxel(return_Xy=True)
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X_all, y_all, test_size=0.20, random_state=42)

    y_train = np.array(ytrain.sort_values(by='timestamps').iloc[:, 1])
    y_test = np.array(ytest.sort_values(by='timestamps').iloc[:, 1])
    X_train = np.array(Xtrain.sort_values(by='timestamps').iloc[:, 1:])
    X_test = np.array(Xtest.sort_values(by='timestamps').iloc[:, 1:])

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    return X_train, y_train, X_test, y_test


def table_v(stats):
    print("\n" + "=" * 80)
    print("TABLE V - SELECTED FEATURES")
    print("=" * 80)
    sel = stats['selected_features']
    print(f"\nFeatures selected (p_i >= 0.99): {sel}")
    print(f"Total selected: {len(sel)}")
    print(f"\nProbabilities of selected features:")
    for idx in sel:
        print(f"  Feature {idx}: p = {stats['p_final'][idx]:.6f}")


def table_vi(stats, X_train, y_train, X_test, y_test):
    print("\n" + "=" * 80)
    print("TABLE VI - PERFORMANCE OF FFS")
    print("=" * 80)

    m = X_train.shape[1]
    sel = stats['selected_features']
    FS = len(sel)

    compression = 100 * (m - FS) / m if m > 0 else 0

    Rc = len(stats['num_feats_per_round'])
    for i in range(len(stats['num_feats_per_round'])):
        if stats['num_feats_per_round'][i:] == \
           [stats['num_feats_per_round'][i]] * (len(stats['num_feats_per_round']) - i):
            Rc = i + 1
            break

    if FS > 0:
        X_train_sel = X_train[:, sel]
        X_test_sel = X_test[:, sel]
    else:
        X_train_sel = X_train
        X_test_sel = X_test
        print("\nWARNING: No features selected, using all features.")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X_train_sel, y_train, cv=5)
    acc_mean = scores.mean() * 100
    acc_std = scores.std() * 100

    clf_all = RandomForestClassifier(n_estimators=100, random_state=42)
    scores_all = cross_val_score(clf_all, X_train, y_train, cv=5)
    acc_all_mean = scores_all.mean() * 100
    acc_all_std = scores_all.std() * 100

    results = pd.DataFrame({
        'Dataset': ['MAV'],
        'rho': [1.0],
        'FS (#)': [FS],
        'Accuracy (% std)': [f'{acc_mean:.2f} +/- {acc_std:.2f}'],
        'Rc (#)': [Rc],
        'C (%)': [round(compression, 2)],
    })

    print()
    print(results.to_string(index=False))

    print(f"\nBaseline (all {m} features): {acc_all_mean:.2f} +/- {acc_all_std:.2f}%")


def table_vii(stats):
    print("\n" + "=" * 80)
    print("TABLE VII - COMMUNICATION OVERHEAD")
    print("=" * 80)

    d = len(stats['p_final'])
    N = 2910
    dws = 2910
    net = stats['net_overhead']
    conv = stats['conv_round']
    max_r = stats['max_rounds']

    overhead_up = d + 68
    overhead_down = d + 68
    total_per_round = overhead_up + overhead_down
    total_overhead = total_per_round * conv * stats['active_workers']

    print(f"\n  d (features):              {d}")
    print(f"  N (total samples):         {N}")
    print(f"  Bitmap size:               68 bytes")
    print(f"  Overhead uplink per AV:    {overhead_up} bytes")
    print(f"  Overhead downlink per AV:  {overhead_down} bytes")
    print(f"  Total per round per AV:    {total_per_round} bytes")
    print(f"  Active workers:            {stats['active_workers']}")
    print(f"  Convergence round:         {conv}")
    print(f"  Max rounds:                {max_r}")
    print(f"  Net overhead (cumulative): {net} bytes")
    print(f"  Total downloaded data:     {N} x {conv} rounds = {N * conv}")


def plot_convergence(stats):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    rounds = np.arange(len(stats['num_feats_per_round']))

    axes[0].plot(rounds, stats['num_feats_per_round'], 'bo-', markersize=4)
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Number of Selected Features')
    axes[0].set_title('Feature Selection Convergence')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, stats['alpha_per_round'], 'rs-', markersize=4)
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Alpha')
    axes[1].set_title('Alpha Decay')
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(np.arange(len(stats['p_final'])), stats['p_final'], width=1.0)
    axes[2].axhline(y=0.99, color='r', linestyle='--', label='Threshold (0.99)')
    axes[2].set_xlabel('Feature Index')
    axes[2].set_ylabel('Probability (p_i)')
    axes[2].set_title('Final Feature Probabilities')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PLOTS_DIR / 'convergence.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {out_path}")


def main():
    PLOTS_DIR.mkdir(exist_ok=True)

    print("Loading stats...")
    stats = load_stats()

    print("Loading dataset...")
    X_train, y_train, X_test, y_test = load_dataset()

    table_v(stats)
    table_vi(stats, X_train, y_train, X_test, y_test)
    table_vii(stats)

    print("\nGenerating plots...")
    plot_convergence(stats)

    print("\nDone.")


if __name__ == '__main__':
    main()
