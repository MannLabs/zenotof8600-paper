"""Figure 6 b, c, d -- the Promega 6x5 standard measured by MRM-HR."""

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
DATA = os.path.join(cfg.DATA_ROOT, 'H032_E396')
OUT = cfg.panels(__file__)

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

PANEL_B_PEPTIDE = 'LGFTDLFSK'

ACC_BAND = (80.0, 120.0)
XLIM_AMOL = (1.2, 33000.0)
CV_REF = 20.0

CURVE_COLOR = core.PALETTE_SINGLE[sorted(
    ['LASVSVSR', 'LGFTDLFSK', 'LLSLGAGEFK', 'VVGGLVALR', 'YVYVADVAAK']
).index(PANEL_B_PEPTIDE)]
BELOW_COLOR = '#BFBFBF'
LIMIT_COLOR = '#888888'
HIST_COLOR = core.PALETTE_SINGLE[2]

PANEL_AXES_W_IN = 1.70
HALF_AXES_H_IN = 1.05


def load():
    levels = pd.read_csv(os.path.join(DATA, '6x5_levels_sourcedata.csv'))
    cal = pd.read_csv(os.path.join(DATA, '6x5_calibration.csv'))
    acc = pd.read_csv(os.path.join(DATA, '6x5_accuracy_sourcedata.csv'))
    resp = pd.read_csv(os.path.join(DATA, '6x5_response_sourcedata.csv'))
    return levels, cal, acc, resp


def panel_curve(levels, cal, letter):
    pep = PANEL_B_PEPTIDE
    d = levels[levels['peptide'] == pep].copy()
    row = cal[cal['peptide'] == pep].iloc[0]
    used = set(float(x) for x in str(row['levels_amol']).split(','))
    lod, loq, slope = row['LOD_amol'], row['LOQ_amol'], row['slope']

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 3.4))
    inside = d[d['amol'].isin(used)]
    outside = d[~d['amol'].isin(used)]

    ax.scatter(outside['amol'], outside['ions'], s=22, color=BELOW_COLOR,
               edgecolor='black', linewidth=0.4, zorder=3)
    ax.scatter(inside['amol'], inside['ions'], s=26, color=CURVE_COLOR,
               edgecolor='black', linewidth=0.5, zorder=4)

    fit = core.raw.calibration_fit(inside['amol'].to_numpy(),
                                   inside['ions'].to_numpy())
    assert np.isclose(fit['slope'], slope, rtol=1e-3), (fit['slope'], slope)
    x = np.geomspace(min(used), max(used), 200)
    ax.plot(x, fit['slope'] * x + fit['intercept'], color='#444444', lw=1.2,
            zorder=2)

    marks = []
    for v, style, lab in ((lod, ':', 'LOD'), (loq, '--', 'LOQ')):
        ax.axvline(v, color=LIMIT_COLOR, lw=1.0, ls=style, zorder=1)
        marks.append(plt.Line2D([], [], color=LIMIT_COLOR, lw=1.0, ls=style,
                                label=f'{lab} {v:.1f} amol'))

    ax.legend(handles=marks, title=f'weighted R$^2$ = {row["r2"]:.4f}',
              loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, fontsize=fs.LEGEND_FONTSIZE,
              title_fontsize=fs.LEGEND_FONTSIZE, handlelength=1.5,
              handletextpad=0.4, borderpad=0.0, columnspacing=1.0,
              labelspacing=0.25)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Spiked amount [amol]', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Targeted peptide intensity', fontsize=fs.FONTSIZE)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    print(f'  b  {pep}: {int(row["n_levels"])} levels {row["levels_amol"]}, '
          f'R2 {row["r2"]:.4f}, LOD {lod:.2f} amol, LOQ {loq:.2f} amol')

    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=PANEL_AXES_W_IN, h_in=fs.AXES_H_IN)
    src = d[['peptide', 'run', 'level', 'amol', 'ions']].assign(
        in_fit=d['amol'].isin(used), lod_amol=lod, loq_amol=loq,
        slope=slope, r2=row['r2'])
    fs.save_panel(fig, OUT, f'figure6{letter}_calibration_curve', src, dpi=300)
    plt.close(fig)


def _median_limits(ax, cal):
    """The MEDIAN LOD (dotted) and LOQ (dashed) over the five peptides."""
    fitted = cal.dropna(subset=['LOD_amol'])
    for v, style in ((fitted['LOD_amol'].median(), ':'),
                     (fitted['LOQ_amol'].median(), '--')):
        ax.axvline(v, color=LIMIT_COLOR, lw=1.0, ls=style, zorder=1)


def panel_accuracy(acc, letter, cal=None):
    """c -- back-calculated against spiked amount, every point of the standard."""
    d = acc[np.isfinite(acc['accuracy_pct']) & (acc['accuracy_pct'] > 0)
            & np.isfinite(acc['amol'])].copy()

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 2.0))
    colors = {p: core.PALETTE_SINGLE[i % len(core.PALETTE_SINGLE)]
              for i, p in enumerate(sorted(d['peptide'].unique()))}
    ax.axhspan(*ACC_BAND, color='#D62728', alpha=0.08, zorder=1)
    ax.axhline(100.0, color='#D62728', lw=1.0, ls=':', zorder=2)
    for pep, sub_ in d[d['used']].groupby('peptide'):
        ax.scatter(sub_['amol'], sub_['accuracy_pct'], s=18,
                   color=colors[pep], edgecolor='black', linewidth=0.4,
                   zorder=4)

    ax.set_xscale('log')
    ax.set_xlim(*XLIM_AMOL)
    if cal is not None:
        _median_limits(ax, cal)
    ax.set_ylim(75, 130)
    ax.set_yticks([80, 100, 120])
    ax.set_ylabel('Accuracy [%]', fontsize=fs.FONTSIZE)
    ax.set_xlabel('')
    ax.tick_params(labelsize=fs.FONTSIZE, labelbottom=False)
    fs.tidy(ax)

    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', markersize=5,
                                  markerfacecolor=colors[p], markeredgecolor='black',
                                  markeredgewidth=0.4, label=p)
                       for p in sorted(colors)],
              loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, fontsize=fs.LEGEND_FONTSIZE, handlelength=0.9,
              handletextpad=0.4, borderpad=0.0, columnspacing=1.0,
              labelspacing=0.25)

    v = d.loc[d['used'], 'accuracy_pct']
    inside = int(((v >= ACC_BAND[0]) & (v <= ACC_BAND[1])).sum())
    print(f'  c  accuracy: n={len(v)} quantified, median {v.median():.1f}%, '
          f'range {v.min():.1f}-{v.max():.1f}%, '
          f'{inside}/{len(v)} within {ACC_BAND[0]:g}-{ACC_BAND[1]:g}%')
    below = d.loc[~d['used'], 'accuracy_pct']
    if len(below):
        print(f'       below the quantified floor: n={len(below)}, '
              f'median {below.median():,.0f}% -- interference, not peptide')

    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=PANEL_AXES_W_IN, h_in=HALF_AXES_H_IN)
    fs.save_panel(fig, OUT, f'figure6{letter}_accuracy',
                  d[['peptide', 'run', 'level', 'amol', 'back_amol',
                     'accuracy_pct', 'used']], dpi=300)
    plt.close(fig)


def panel_precision(resp, levels, letter, cal=None):
    """d -- replicate CV against spiked amount, every point of the standard."""
    amol = (levels.drop_duplicates(['peptide', 'level'])
            .set_index(['peptide', 'level'])['amol'])
    d = resp.copy()
    d['amol'] = [amol.get((p, l), np.nan)
                 for p, l in zip(d['peptide'], d['level'])]
    d = d[np.isfinite(d['amol']) & np.isfinite(d['cv_percent'])]

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 2.0))
    colors = {p: core.PALETTE_SINGLE[i % len(core.PALETTE_SINGLE)]
              for i, p in enumerate(sorted(d['peptide'].unique()))}
    for pep, sub_ in d.groupby('peptide'):
        c = colors[pep]
        on = sub_[sub_['measured']]
        off = sub_[~sub_['measured']]
        ax.scatter(on['amol'], on['cv_percent'], s=18, color=c,
                   edgecolor='black', linewidth=0.4, zorder=4)
        ax.scatter(off['amol'], off['cv_percent'], s=18, facecolor='white',
                   edgecolor=c, linewidth=0.8, zorder=3)

    ax.axhline(CV_REF, color='#D62728', lw=1.0, ls=':', zorder=2)
    ax.annotate(f'{CV_REF:g}% CV', xy=(0.99, CV_REF), xycoords=('axes fraction',
                'data'), xytext=(0, 2), textcoords='offset points',
                ha='right', va='bottom', fontsize=fs.LEGEND_FONTSIZE,
                color='#D62728')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*XLIM_AMOL)
    if cal is not None:
        _median_limits(ax, cal)
    ax.set_xlabel('Spiked amount [amol]', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Replicate CV [%]', fontsize=fs.FONTSIZE)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)

    v = d.loc[d['measured'], 'cv_percent']
    print(f'  d  precision: n={len(v)} quantified, median {v.median():.1f}%, '
          f'range {v.min():.1f}-{v.max():.1f}%, '
          f'{int((v < CV_REF).sum())}/{len(v)} below {CV_REF:g}% CV')
    for lvl, g in d[d['measured']].groupby('level'):
        print(f'       level {int(lvl)} ({g["amol"].iloc[0]:>7,.0f} amol): '
              f'median CV {g["cv_percent"].median():5.1f}%  n={len(g)}')

    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=PANEL_AXES_W_IN, h_in=HALF_AXES_H_IN)
    fs.save_panel(fig, OUT, f'figure6{letter}_precision',
                  d[['peptide', 'level', 'amol', 'mean_ions', 'cv_percent',
                     'measured']], dpi=300)
    plt.close(fig)


def main(letter_b='b', letter_c='c', letter_d='d'):
    fs.init()
    fs.start_build(script=__file__, inputs=[DATA], out_dir=OUT)
    levels, cal, acc, resp = load()
    panel_curve(levels, cal, letter_b)
    panel_accuracy(acc, letter_c, cal=cal)
    panel_precision(resp, levels, letter_d, cal=cal)
    fs.report_stale(OUT, prefix='figure')


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['b', 'c', 'd']))
