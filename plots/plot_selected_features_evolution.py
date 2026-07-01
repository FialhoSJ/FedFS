import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
STATS_DIR = BASE_DIR / 'stats' / 'fedfs-mav'
PLOTS_DIR = BASE_DIR / 'plots'

sfx = 'nodes(1.0)_data(1.0)-alpha_(1.0).csv'

history = []
with open(STATS_DIR / f'fed_running_prob-{sfx}', 'r') as f:
    for line in f:
        vals = np.array([float(x) for x in line.strip().split(',')])
        history.append(vals)

threshold = 0.85
selected_per_round = [np.sum(p > threshold) for p in history]

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(selected_per_round) + 1), selected_per_round,
         marker='*', label='Num. selected features')
plt.xlabel('Communication Round')
plt.ylabel('Num. Features')
plt.title('Selected Features Evolution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
out_path = PLOTS_DIR / 'selected_features_evolution.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved: {out_path}")
