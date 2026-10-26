"""Supplementary figure 4, panels c-e — discovery mode on against off, H032_E387."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(cfg.DATA_ROOT, 'H032_E387')
OUT = cfg.panels(__file__)

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

CONDITIONS = [('disco off', 'disco_off', ['A5', 'A6', 'A7', 'A8']),
              ('disco on', 'disco_on', ['A1', 'A2', 'A3', 'A4'])]
ORDER = [c[0] for c in CONDITIONS]

COLOR = {'disco off': core.PALETTE_SINGLE[2],
         'disco on': core.PALETTE_SINGLE[0]}

AXES_W_IN = 1.38
IDS_AXES_W_IN = 1.38

AXIS_INK = '#5A5A5A'


def load():
    """The two searches as one long frame, with `condition2` on every row."""
    exps = [{'path': os.path.join(DATA, d, 'lfq.dia.features.csv'),
             'protein_path': os.path.join(DATA, d, 'lfq.dia.proteins.csv'),
             'file_tags': tags, 'condition1': 'E387 discovery mode',
             'condition2': lab} for lab, d, tags in CONDITIONS]
    df, si = core.load_experiments(exps)
    si['replicate'] = si.groupby('condition2').cumcount() + 1
    df = df.merge(si[['run', 'condition2']], on='run', how='left')
    return df[df['precursor_intensity'] > 0].copy(), si


def _n_bins(x, per_bin=250, lo=20, hi=100):
    """House rule: bins capped by COUNTS per bin, never at a fixed number."""
    x = np.asarray(x, float)
    q75, q25 = np.nanpercentile(x, [75, 25])
    fd = 2 * (q75 - q25) / max(len(x), 1) ** (1 / 3)
    n_fd = int(np.ceil((np.nanmax(x) - np.nanmin(x)) / fd)) if fd > 0 else hi
    return int(np.clip(min(n_fd, len(x) // per_bin), lo, hi))


def _paired(df):
    """Mean intensity per precursor per condition, precursors seen in both."""
    piv = (df.drop_duplicates(['run', 'precursor_id'])
           .pivot_table(index='precursor_id', columns='condition2',
                        values='precursor_intensity', aggfunc='mean'))
    return piv


def panel_ratio(df, letter):
    """c — the distribution of per-precursor log2(on / off)."""
    piv = _paired(df).dropna()
    r = np.log2(piv['disco on'] / piv['disco off'])
    x = np.log2(piv.mean(axis=1))

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 3.4))
    bins = _n_bins(r)
    span = (np.floor(r.quantile(0.001) * 2) / 2,
            np.ceil(r.quantile(0.999) * 2) / 2)
    counts, edges = np.histogram(r, bins=bins, range=span)
    centres = (edges[:-1] + edges[1:]) / 2
    c_on = COLOR['disco on']
    ax.fill_between(centres, counts, color=c_on, alpha=0.18, lw=0)
    ax.plot(centres, counts, color=c_on, lw=2.2)
    ax.axvline(0, color='#888888', lw=1.0, ls=':', zorder=1)
    ax.axvline(r.median(), color=c_on, lw=1.2, ls='--', zorder=2)

    ax.set_xlabel(r'$\log_2$ (on / off)', fontsize=fs.FONTSIZE)
    ax.set_ylabel(f'precursors per bin (bin {edges[1] - edges[0]:.2f})',
                  fontsize=fs.FONTSIZE)
    ax.set_xlim(*span)
    ax.set_ylim(0, counts.max() * 1.10)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)

    print(f'    shared precursors {len(r):,}   median {r.median():+.3f} log2 '
          f'({2 ** r.median():.2f}x)   IQR {r.quantile(.25):+.3f} to '
          f'{r.quantile(.75):+.3f}')
    q = pd.qcut(x, 5, labels=False)
    for i in range(5):
        m = r[q == i].median()
        print(f'      abundance quintile {i + 1} (weakest first): '
              f'{m:+.3f} log2 ({2 ** m:.2f}x)')

    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=AXES_W_IN, h_in=fs.AXES_H_IN)
    fs.save_panel(fig, OUT, f'supplementary_figure4{letter}_disco_ratio',
                  pd.DataFrame({'log2_bin_centre': centres,
                                'precursors': counts,
                                'bin_width': edges[1] - edges[0],
                                'median_log2': r.median(),
                                'n_shared': len(r)}), dpi=300)
    plt.close(fig)


def panel_level(agg, letter, level):
    """d / e -- one level, overlapping bars as in panels a and b."""
    total_col, cv_col, ylabel, name = {
        'precursor': ('total_precursors', 'Pr20', '# precursors',
                      'precursor_ids'),
        'protein group': ('total_protein_groups', 'PG20', '# protein groups',
                          'protein_group_ids')}[level]
    fig, ax, summary = core.plot_overlapping_bars(
        agg, x_col='condition2', higher_y_col=total_col, lower_y_col=cv_col,
        higher_label='IDs', lower_label='< 20% CV',
        palette=[COLOR[c] for c in ORDER],
        figsize=(fs.PANEL_W_IN, 4), y_label=ylabel,
        label_fontsize=fs.FONTSIZE, tick_fontsize=fs.FONTSIZE,
        legend_fontsize=fs.LEGEND_FONTSIZE, value_fontsize=fs.FONTSIZE,
        point_size=fs.POINT_SIZE)
    if not fs.SHOW_VALUE_LABELS:
        fs.strip_value_labels(ax)
    ax.set_xticklabels([t.get_text().replace('disco ', '')
                        for t in ax.get_xticklabels()], rotation=0,
                       ha='center', fontsize=fs.FONTSIZE)
    ax.set_xlabel('Enhanced sensitivity\nmode', fontsize=fs.FONTSIZE)
    ax.yaxis.label.set_fontsize(fs.FONTSIZE)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    tot = agg.groupby('condition2')[total_col].first().reindex(ORDER)
    q20 = agg.groupby('condition2')[cv_col].first().reindex(ORDER)
    print(f'    {level:13s} union {tot.iloc[0]:,.0f} -> {tot.iloc[1]:,.0f} '
          f'({(tot.iloc[1] / tot.iloc[0] - 1) * 100:+.1f}%)   <20% CV '
          f'{q20.iloc[0]:,.0f} -> {q20.iloc[1]:,.0f} '
          f'({(q20.iloc[1] / q20.iloc[0] - 1) * 100:+.1f}%)')
    fig.tight_layout()
    fs.set_bar_geometry_inches(fig, ax)
    fs.save_panel(fig, OUT, f'supplementary_figure4{letter}_disco_{name}',
                  summary, dpi=300)
    plt.close(fig)


LEVEL_AXES_W_IN = 1.30


def main(letters=('c', 'd', 'e')):
    fs.init()
    fs.start_build(script=__file__, inputs=[DATA], out_dir=OUT)
    print(f'loading {DATA} ...')
    df, si = load()
    print(f'  {df.shape[0]:,} rows, {len(si)} runs')
    agg = core.process_experiment(df.drop(columns=['condition2']), si,
                                  group_col='condition2')

    print(f'panel {letters[0]} -- relative intensity ...')
    panel_ratio(df, letters[0])
    print(f'panels {letters[1]}, {letters[2]} -- identifications ...')
    panel_level(agg, letters[1], 'precursor')
    panel_level(agg, letters[2], 'protein group')
    fs.report_stale(OUT, prefix='supplementary_figure')


if __name__ == '__main__':
    main(tuple(sys.argv[1:]) or ('c', 'd', 'e'))
