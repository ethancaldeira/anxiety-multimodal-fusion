"""
explain.reporting — plotting helper for the saliency report: a sorted bar
chart of network-level attribution scores. The glass-brain and stat-map
views are rendered in src/report_saliency.py.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def network_bar(network_scores,
                out_path: str,
                title: str = '',
                color: str = '#2266aa') -> None:
    """Sorted horizontal bar chart of Yeo network attribution scores.

    Accepts a pandas Series (index = network name) or a dict.
    """
    s = pd.Series(network_scores).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 0.6 * len(s) + 1.2))
    ax.barh(s.index, s.values, color=color, alpha=0.85)
    ax.set_xlabel('Mean |attribution|')
    ax.set_title(title or 'Network attribution')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'  [plot] {out_path}')
