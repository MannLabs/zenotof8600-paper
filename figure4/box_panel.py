"""Shared construction for the figure-4 identification box panels."""

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import zenotof8600_analytics as core

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import figure_style as fs

UNIT_IN = 0.62

POINT_DIAM = math.sqrt(fs.POINT_SIZE)

MIN_BOX_N = 3


def points_with_median(ax, x, values, color, *, half_width=0.25, seed=0):
    """A small group: every value as a point, a black line at the median."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return
    rng = np.random.default_rng(seed)
    ax.hlines(np.median(v), x - half_width, x + half_width, color='black',
              linewidth=1.5, zorder=3)
    ax.scatter(x + rng.uniform(-0.6, 0.6, v.size) * half_width * 0.5, v,
               s=POINT_DIAM ** 2, color=color, alpha=0.85, edgecolor='black',
               linewidth=0.4, zorder=5)


LEVELS = [
    ('protein_groups', 'protein_groups', '# protein groups'),
    ('precursors', 'precursors', '# precursors'),
    ('peptides', 'peptides', '# peptides'),
]


def run_names(features_csv):
    """Every run in a PEAKS features table, from its per-run `<run> m/z` columns."""
    return [c[:-len(' m/z')] for c in pd.read_csv(features_csv, nrows=0).columns
            if c.endswith(' m/z') and c != 'm/z']


def identification_counts(experiments, cond_order, *, strip=r'.*_TiHe_'):
    """One row per run: precursor, peptide and protein-group identifications."""
    df, sample_info = core.load_experiments(experiments)
    ids = (df[df['precursor_intensity'] > 0]
           .groupby('run')
           .agg(precursors=('precursor_id', 'nunique'),
                peptides=('peptide_id', 'nunique'),
                protein_groups=('protein_group', 'nunique'))
           .reset_index()
           .merge(sample_info[['run', 'condition2']], on='run'))
    ids['well'] = ids['run'].str.replace(strip, '', regex=True)
    ids['condition2'] = pd.Categorical(ids['condition2'],
                                       categories=cond_order, ordered=True)
    return ids.sort_values(['condition2', 'protein_groups'])


def draw(ids, *, name, col, ylabel, cond_order, color, out_dir,
         unit_in=UNIT_IN, xlabel='', show_excluded=False):
    """One box per condition, every included run drawn as a point on it."""
    shown = ids[ids['included']]
    n_per = shown.groupby('condition2', observed=True)['run'].count()
    small = [c for c in cond_order if n_per.get(c, 0) < MIN_BOX_N]
    boxed = shown.copy()
    boxed[col] = boxed[col].where(~boxed['condition2'].isin(small))
    fig, ax = core.plot_boxplot_with_points(
        boxed, x_col='condition2', y_col=col, hue_col='condition2',
        palette=[color[c] for c in cond_order],
        figsize=(fs.PANEL_W_IN, 4), y_label=ylabel, x_label='',
        show_points=True, point_size=POINT_DIAM, point_alpha=0.85,
        label_fontsize=fs.FONTSIZE, tick_fontsize=fs.FONTSIZE,
        legend_fontsize=fs.LEGEND_FONTSIZE,
        xtick_rotation=0, ymin=0,
    )
    for i, c in enumerate(cond_order):
        if c in small:
            points_with_median(ax, i, shown.loc[shown['condition2'] == c, col],
                               color[c], seed=i)
    if show_excluded:
        rng = np.random.default_rng(7)
        for i, c in enumerate(cond_order):
            v = ids.loc[(ids['condition2'] == c) & ~ids['included'], col]
            if len(v):
                ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v,
                           s=POINT_DIAM ** 2, facecolor='white',
                           edgecolor=color[c], linewidth=0.8, alpha=0.6,
                           zorder=4)
    lo, hi = ax.get_ylim()
    top = float(shown[col].max())
    if top * 1.03 > hi:
        ax.set_ylim(lo, top * 1.05)
    fs.tidy(ax)
    ax.tick_params(labelsize=fs.FONTSIZE)
    ax.yaxis.label.set_fontsize(fs.FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=fs.FONTSIZE)

    n = shown.groupby('condition2', observed=True)['run'].count()
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels(list(cond_order), fontsize=fs.FONTSIZE)
    print('    n per condition: '
          + '  '.join(f'{c} {int(n[c])}' for c in cond_order))

    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f'{v / 1000:,.0f}k' if v >= 1000 else f'{v:,.0f}'))

    med = shown.groupby('condition2', observed=True)[col].median()
    print('    ' + '  '.join(f'{c} {med[c]:,.0f}' for c in cond_order))

    source = ids[['run', 'well', 'condition2', col, 'included',
                  'exclusion_reason']].rename(
        columns={'condition2': 'sample_type', col: 'value'})
    source['quantity'] = col
    fig.tight_layout()
    fs.set_categorical_width_inches(fig, ax, unit_in=unit_in)
    w, _ = fs.save_panel(fig, out_dir, name, source, dpi=300)
    if w > 4.43:
        print(f'    WARNING: {name} is {w:.2f} in wide; past 4.43 in a two-up '
              f'row drops 10 pt text below the 8 pt printed floor')
    plt.close(fig)
    return w


def report_small_groups(ids, cond_order, *, minimum=3):
    """Say out loud which boxes rest on too few runs to be a distribution."""
    for c in cond_order:
        k = ids[(ids['condition2'] == c) & ids['included']]
        if len(k) < minimum:
            print(f'  NOTE: "{c}" is drawn from {len(k)} run(s) -- the box '
                  f'carries no distribution')
