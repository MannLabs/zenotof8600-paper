"""One quantity binned against another, drawn as a median line with an IQR band."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._style import (PALETTE_SINGLE, _hide_top_right_spines,
                     _resolve_panel_size)
from ..stats import binned_stats


def plot_binned_median(
    data, x, y, *, group_col=None, group_order=None, palette=None,
    n_bins=10, mode='quantile', min_count=8, show_iqr=True,
    threshold=None, threshold_color='#d62728',
    x_label=None, y_label=None, legend_title=None, legend_loc='upper right',
    ylim=(0, None), figsize=None, title=None, label_fontsize=10,
    tick_fontsize=10, legend_fontsize=8, ax=None,
):
    """Median of `y` against binned `x`, one line per group, with an IQR band."""
    import matplotlib.pyplot as plt

    if group_col is None:
        data = data.assign(_group='all')
        group_col = '_group'
    groups = (list(group_order) if group_order is not None
              else list(pd.Series(data[group_col]).dropna().unique()))
    palette = PALETTE_SINGLE if palette is None else palette
    if isinstance(palette, (list, tuple)):
        palette = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    else:
        fig = ax.figure

    rows = []
    for group in groups:
        sub = data[data[group_col] == group]
        stats = binned_stats(sub[x], sub[y], bins=n_bins, mode=mode,
                             min_count=min_count)
        if stats.empty:
            continue
        ax.plot(stats['x'], stats['median'], marker='o', markersize=3.5,
                lw=1.6, color=palette[group], label=str(group), zorder=3)
        if show_iqr:
            ax.fill_between(stats['x'], stats['q_lo'], stats['q_hi'],
                            color=palette[group], alpha=0.15, lw=0, zorder=1)
        rows.extend({'group': group, 'bin': int(i), 'x': float(r['x']),
                     'median': float(r['median']), 'q_lo': float(r['q_lo']),
                     'q_hi': float(r['q_hi']), 'n': int(r['n'])}
                    for i, (_, r) in enumerate(stats.iterrows()))

    if threshold is not None:
        ax.axhline(threshold, ls=':', color=threshold_color, lw=1.2, zorder=0)
    ax.set_xlabel(x_label if x_label is not None else x,
                  fontsize=label_fontsize)
    ax.set_ylabel(y_label if y_label is not None else y,
                  fontsize=label_fontsize)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 1, fontweight='bold')
    ax.tick_params(labelsize=tick_fontsize)
    if rows and group_col != '_group':
        ax.legend(frameon=False, fontsize=legend_fontsize, loc=legend_loc,
                  title=legend_title, title_fontsize=legend_fontsize)
    _hide_top_right_spines(ax)
    if created:
        fig.tight_layout()
    return fig, ax, pd.DataFrame(
        rows, columns=['group', 'bin', 'x', 'median', 'q_lo', 'q_hi', 'n'])
