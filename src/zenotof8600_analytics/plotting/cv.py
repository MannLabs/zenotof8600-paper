"""CV-distribution plots: violin, ECDF, and combined stacked bars."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._style import (
    PALETTE_SINGLE, PALETTE_CV_CATEGORIES, _hide_top_right_spines, _LEVEL_COLS, _resolve_panel_size,
)
from ..stats import _compute_cv_table
from .binned import plot_binned_median


def plot_cv_violin(
    df,
    sample_info,
    *,
    level='protein',
    group_col='condition2',
    hue_col=None,
    group_order=None,
    cv_threshold=0.2,
    min_values_for_cv=3,
    figsize=(10, 6),
    palette=None,
    ylim=None,
    as_percent=False,
    title=None,
    y_label=None,
    show_threshold=False,
    show_median_label=True,
    median_label_loc='below',
    median_label_color='black',
    median_label_fontsize=10,
    box_facecolor='#4d4d4d',
    box_edgecolor='black',
    box_width=0.12,
    median_color='white',
    bold_xticklabels=False,
    legend_fontsize=10,
    ax=None,
):
    """Violin plot of CV distribution per replicate group."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    palette = palette if palette is not None else PALETTE_SINGLE
    palette_is_dict = isinstance(palette, dict)
    cv_table = _compute_cv_table(df, sample_info, level, group_col,
                                 min_values_for_cv, hue_col=hue_col)

    stats_keys = ['group'] + (['hue'] if hue_col else [])
    cv_stats_rows = []
    for keys, sub in cv_table.groupby(stats_keys, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(sub)
        if n == 0:
            continue
        row = {'group': keys[0]}
        if hue_col:
            row['hue'] = keys[1]
        row.update({
            'n_total': n,
            'n_CV<10%': int((sub['cv'] < 0.10).sum()),
            'n_CV<20%': int((sub['cv'] < cv_threshold).sum()),
            '%_CV<20%': float((sub['cv'] < cv_threshold).mean() * 100),
            'median_CV': float(sub['cv'].median()),
            'mean_CV': float(sub['cv'].mean()),
        })
        cv_stats_rows.append(row)
    cv_stats = pd.DataFrame(cv_stats_rows)

    scale = 100.0 if as_percent else 1.0
    cv_table = cv_table.copy()
    cv_table['cv_plot'] = cv_table['cv'] * scale
    if ylim is None:
        ylim = (0, 80) if as_percent else (0, 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    if hue_col is None:
        groups = (group_order if group_order is not None
                  else list(cv_table['group'].unique()))
        if palette_is_dict:
            color_map = {g: palette[g] for g in groups}
        else:
            color_map = {g: palette[i % len(palette)]
                         for i, g in enumerate(groups)}
        sns.violinplot(
            data=cv_table, x='group', y='cv_plot', order=groups,
            hue='group', hue_order=groups, palette=color_map,
            ax=ax, inner=None, cut=0, density_norm='width', legend=False,
        )
        sns.boxplot(
            data=cv_table, x='group', y='cv_plot', order=groups, ax=ax,
            width=box_width, showcaps=False,
            boxprops={'facecolor': box_facecolor, 'edgecolor': box_edgecolor,
                      'linewidth': 0.8},
            whiskerprops={'color': box_edgecolor, 'linewidth': 1},
            medianprops={'color': median_color, 'linewidth': 2},
            showfliers=False,
        )
    else:
        hues = list(cv_table['hue'].drop_duplicates())
        groups = (group_order if group_order is not None
                  else list(cv_table['group'].unique()))
        if palette_is_dict:
            color_map = {h: palette[h] for h in hues}
        else:
            color_map = {h: palette[i % len(palette)]
                         for i, h in enumerate(hues)}
        sns.violinplot(
            data=cv_table, x='group', y='cv_plot', order=groups,
            hue='hue', palette=color_map,
            ax=ax, inner=None, cut=0, density_norm='width',
            hue_order=hues, dodge=True,
        )
        slot = 0.8 / len(hues)
        for group_index, grp in enumerate(groups):
            for hue_index, hue_value in enumerate(hues):
                cell = cv_table.loc[(cv_table['group'] == grp)
                                    & (cv_table['hue'] == hue_value),
                                    'cv_plot'].dropna()
                if cell.empty:
                    continue
                offset = (hue_index - (len(hues) - 1) / 2) * slot
                ax.boxplot(
                    [cell.to_numpy()], positions=[group_index + offset],
                    widths=box_width * slot / 0.8, patch_artist=True,
                    showcaps=False, showfliers=False, manage_ticks=False,
                    boxprops={'facecolor': box_facecolor,
                              'edgecolor': box_edgecolor, 'linewidth': 0.8},
                    whiskerprops={'color': box_edgecolor, 'linewidth': 1},
                    medianprops={'color': median_color, 'linewidth': 2},
                )

    if show_threshold:
        ax.axhline(cv_threshold * scale, color='red', linestyle='--',
                   linewidth=1.5, label=f'CV = {cv_threshold * 100:.0f}%')

    if show_median_label and hue_col is None and median_label_loc != 'none':
        y_below = ylim[0] - 0.06 * (ylim[1] - ylim[0])
        for i, grp in enumerate(groups):
            row = cv_stats[cv_stats['group'] == grp]
            if row.empty:
                continue
            median_cv = float(row['median_CV'].iloc[0])
            med = median_cv * scale
            med_pct = median_cv * 100.0
            if median_label_loc == 'below':
                ax.annotate(
                    f'{med_pct:.1f}%', xy=(i, y_below), ha='center', va='top',
                    fontsize=median_label_fontsize, fontweight='bold',
                    color=median_label_color, annotation_clip=False,
                )
            else:
                ax.annotate(
                    f'{med_pct:.1f}%', xy=(i + 0.15, med), ha='left', va='center',
                    fontsize=median_label_fontsize, fontweight='bold',
                    color=median_label_color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='none', alpha=0.8),
                )

    if y_label is None:
        y_label = 'CV [%]' if as_percent else 'Coefficient of Variation'
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_xlabel('')
    ax.set_ylim(ylim)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             fontweight='bold' if bold_xticklabels else 'normal')
    ax.tick_params(labelsize=10)
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight='bold')

    if hue_col is not None:
        ax.legend(fontsize=legend_fontsize, frameon=False,
                  loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    elif show_threshold:
        ax.legend(fontsize=legend_fontsize, loc='upper right', frameon=False)

    sns.despine(ax=ax)
    if created_fig:
        plt.tight_layout()
    return fig, ax, cv_stats


def plot_cv_ecdf(
    df,
    sample_info,
    *,
    level='protein',
    group_col='condition2',
    group_order=None,
    palette=None,
    cv_threshold=0.2,
    min_values_for_cv=3,
    figsize=(3.5, 3),
    xlim=(0, 60),
    title=None,
    y_label='Cumulative fraction',
    x_label='CV [%]',
    linestyle='-',
    linewidth=1.5,
    annotate_median=True,
    legend_loc='lower right',
    legend_fontsize=8,
    ax=None,
):
    """Empirical CDF of per-feature CVs, one step curve per replicate group."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    palette = palette if palette is not None else PALETTE_SINGLE
    palette_is_dict = isinstance(palette, dict)
    cv_table = _compute_cv_table(df, sample_info, level, group_col,
                                 min_values_for_cv)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    groups = (group_order if group_order is not None
              else list(cv_table['group'].unique()))
    cv_stats_rows = []
    for i, grp in enumerate(groups):
        vals = (cv_table.loc[cv_table['group'] == grp, 'cv']
                .dropna().sort_values().to_numpy() * 100.0)
        if vals.size == 0:
            continue
        med = float(np.median(vals))
        y = np.arange(1, vals.size + 1) / vals.size
        color = palette[grp] if palette_is_dict else palette[i % len(palette)]
        label = f'{grp} ({med:.1f}%)' if annotate_median else str(grp)
        ax.plot(vals, y, drawstyle='steps-post', color=color,
                linewidth=linewidth, linestyle=linestyle, label=label)
        cv_stats_rows.append({
            'group': grp, 'median_CV': med / 100.0, 'n_total': int(vals.size),
            '%_CV<20%': float((vals < cv_threshold * 100).mean() * 100),
        })

    ax.axvline(cv_threshold * 100, color='red', linestyle=':', linewidth=1)
    ax.set_xlim(xlim)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.tick_params(labelsize=10)
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=legend_fontsize, frameon=False, loc=legend_loc)
    _hide_top_right_spines(ax)
    if created_fig:
        plt.tight_layout()
    return fig, ax, pd.DataFrame(cv_stats_rows)


