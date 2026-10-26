"""Figure 6, panel g -- pS129 occupancy of alpha-synuclein, MRM-HR."""

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

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(cfg.DATA_ROOT, 'H032_E397', 'mrmhr')
OUT = cfg.panels(__file__)

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

E397_DIR = cfg.data('H032_E397')

ORDER = ('PFF-', 'PFF+')
COLOR = [core.PALETTE_SINGLE[2], core.PALETTE_SINGLE[0]]

TEST = None


QUANT_SET = (('z', 4), ('y', 4), ('z', 5), ('z', 6))
MEASUREMENT = os.path.join(DATA, os.pardir, 'transition_check', 'without_y3',
                           'e397_lib_measurements.csv')


def table():
    """Per-run readouts with E397's definition (`plot_E397.table`), from the y4/z4/z5/z6 measurement."""
    if E397_DIR not in sys.path:
        sys.path.insert(0, E397_DIR)
    import pathlib
    import plot_E397 as p
    p.SOURCE = pathlib.Path(MEASUREMENT)
    t = p.table().reset_index(drop=True)
    t['condition'] = pd.Categorical(t['condition'].astype(str),
                                    categories=ORDER, ordered=True)
    return t.sort_values(['condition', 'run']).reset_index(drop=True)


def panel(t, letter):
    """g -- pS129 occupancy, four injections of one digest per arm."""
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    core.plot_bar(
        t, x_col='condition', y_col='occupancy', ax=ax,
        palette=COLOR, y_label='pS129 occupancy [%]', x_label='',
        show_points=True, value_format='float',
        value_fontsize=fs.FONTSIZE, label_fontsize=fs.FONTSIZE,
        tick_fontsize=fs.FONTSIZE, significance_test=TEST)
    if not fs.SHOW_VALUE_LABELS:
        fs.strip_value_labels(ax, keep_annotations=True)
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    fs.tidy(ax)
    ax.tick_params(labelsize=fs.FONTSIZE)
    ax.set_ylim(0, None)

    fig.tight_layout()
    fs.set_bar_geometry_inches(fig, ax)
    core.scale_replicate_points(ax, ax.collections)
    source = (t[['run', 'condition', 'occupancy', 'lh_ps129', 'lh_unmod',
                 'light_ps129']]
              .assign(condition=lambda d: d['condition'].astype(str)))
    fs.save_panel(fig, OUT, f'figure6{letter}_ps129_occupancy', source,
                  dpi=300)
    plt.close(fig)
    _report(t)


def _report(t):
    for cond in ORDER:
        v = t.loc[t['condition'] == cond, 'occupancy'].to_numpy()
        print(f'    {cond:5s} n={len(v)}  occupancy {np.mean(v):5.2f}% '
              f'(sd {np.std(v, ddof=1):.2f}, range {v.min():.2f}-{v.max():.2f})')
    a = t.loc[t['condition'] == ORDER[0], 'occupancy'].mean()
    b = t.loc[t['condition'] == ORDER[1], 'occupancy'].mean()
    print(f'    PFF+ over PFF- : {b / a:.2f}x')
    ua = t.loc[t['condition'] == ORDER[0], 'lh_unmod'].mean()
    ub = t.loc[t['condition'] == ORDER[1], 'lh_unmod'].mean()
    pa = t.loc[t['condition'] == ORDER[0], 'lh_ps129'].mean()
    pb = t.loc[t['condition'] == ORDER[1], 'lh_ps129'].mean()
    print(f'    for context: total aSyn peptide {ub / ua:.2f}x, '
          f'heavy-normalised pS129 {pb / pa:.2f}x')


def main(letter='g'):
    fs.init()
    fs.start_build(script=__file__, inputs=[DATA], out_dir=OUT)
    print(f'loading {DATA} ...')
    t = table()
    print(f'  {len(t)} runs')
    print(f'\npanel {letter} ...')
    panel(t, letter)
    fs.report_stale(OUT, prefix='figure')


if __name__ == '__main__':
    main()
