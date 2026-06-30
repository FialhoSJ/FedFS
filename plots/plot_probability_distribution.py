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
p_final = np.loadtxt(STATS_DIR / f'fed_prob-{sfx}', delimiter=',')

threshold = 0.99

plt.figure(figsize=(10, 5))
plt.scatter(np.arange(len(p_final)), p_final, s=8, label='Features')
plt.axhline(y=threshold, linestyle='--', color='r', label=f'Threshold = {threshold}')
plt.xlabel('Features')
plt.ylabel('Probability')
plt.title('FFS Probability Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
out_path = PLOTS_DIR / 'probability_distribution.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved: {out_path}")
