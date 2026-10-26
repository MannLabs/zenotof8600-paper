"""Supplementary figure 2 — what figure 2 leaves out."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import math
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = cfg.panels(__file__)
DATA = cfg.data('supplementary_figure2')
E376 = cfg.data('H032_E376')

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

CONDITIONS = [
    {'label': '40 SPD\n1.48 s', 'dir': '40SPD_1Th_1p48s'},
    {'label': '20 SPD\n2.4 s', 'dir': '20SPD_1Th_2p4s'},
]
COND_ORDER = [c['label'] for c in CONDITIONS]
COLOR = {c: core.PALETTE_SINGLE[i] for i, c in enumerate(COND_ORDER)}

IQR_TO_SIGMA = 1.349


def save(fig, ax, name, source_df, *, w_in=2.60, dpi=300):
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=w_in, h_in=fs.AXES_H_IN)
    return fs.save_panel(fig, OUT, name, source_df, dpi=dpi)


def panel_a():
    """a — 40 SPD against 20 SPD, identifications and the CV<20% subset."""
    experiments = []
    for c in CONDITIONS:
        d = os.path.join(E376, c['dir'])
        f = os.path.join(d, 'lfq.dia.features.csv')
        runs = [x[:-len(' m/z')] for x in pd.read_csv(f, nrows=0).columns
                if x.endswith(' m/z') and x != 'm/z']
        experiments.append({
            'path': f, 'protein_path': os.path.join(d, 'lfq.dia.proteins.csv'),
            'file_tags': runs, 'condition1': '250 ng HEK, 1 Th ZT Scan',
            'condition2': c['label']})
    df, si = core.load_experiments(experiments)
    si['replicate'] = si.groupby('condition2').cumcount() + 1
    agg = core.process_experiment(df, si, group_col='condition2')

    fig, ax, summary = core.plot_overlapping_bars(
        agg, x_col='condition2',
        higher_y_col='total_protein_groups',
        lower_y_col='PG20',
        higher_label='IDs', lower_label='< 20% CV',
        palette=[COLOR[c] for c in COND_ORDER],
        figsize=(fs.PANEL_W_IN, 4), y_label='# protein groups',
        label_fontsize=fs.FONTSIZE, tick_fontsize=fs.FONTSIZE,
        legend_fontsize=fs.LEGEND_FONTSIZE,
        value_fontsize=fs.FONTSIZE, point_size=fs.POINT_SIZE)
    if not fs.SHOW_VALUE_LABELS:
        fs.strip_value_labels(ax)
    ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    fs.tidy(ax)
    ax.tick_params(labelsize=fs.FONTSIZE)
    ax.yaxis.label.set_fontsize(fs.FONTSIZE)

    tot = (agg.groupby('condition2')['total_protein_groups'].first().reindex(COND_ORDER))
    gain = (tot.iloc[1] - tot.iloc[0]) / tot.iloc[0] * 100
    fs.label_pct(ax, [(1, float(tot.iloc[1]), gain)])
    summary['gain_pct_20SPD_over_40SPD'] = gain
    print(f'    40 SPD {tot.iloc[0]:,.0f} -> 20 SPD {tot.iloc[1]:,.0f}  '
          f'({gain:+.1f}% for half the throughput)')

    fig.tight_layout()
    fs.set_bar_geometry_inches(fig, ax)
    fs.save_panel(fig, OUT, 'supplementary_figure2e_20spd_vs_40spd', summary,
                  dpi=300)
    plt.close(fig)


ESTIMATORS = [
    ('parab', 'parabolic vertex', True),
    ('interp', 'half-max midpoint', False),
    ('grid', 'grid weighted mean', False),
]


AXES = [('A5', '1 Th', 'q1_axis_A5_precursors.csv'),
        ('A9', '2.8 Th', 'q1_axis_A9_precursors.csv')]

CAL_BINS = 12

def panel_b():
    """b -- the Q1 axis calibration against m/z, before and after."""
    import zlib
    from zenotof8600_analytics.ztscan.calibration import fit_q1_axis
    from zenotof8600_analytics.ztscan.axis import q1_correction
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'figure2'))
    import figure2 as f2

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    ax.axhline(0, color='0.35', lw=0.8, ls=':', zorder=0)
    rows = []
    for i, (tag, label, _f) in enumerate(AXES):
        d = f2.wholefile_placements(tag)
        held = d['seq'].map(lambda s_: zlib.crc32(s_.encode()) % 2 == 1)
        tr, te = d[~held], d[held]
        axis = fit_q1_axis(tr['q1'], tr['mz'], tr['rt'], loess_frac=f2.Q1_AXIS_LOESS_FRAC,
                           n_knots=80)
        dev = te['q1'] - te['mz']
        res = dev - q1_correction(axis, te['q1'], te['rt'])
        col = core.PALETTE_SINGLE[i * 2]
        for stage, y, style in [('uncalibrated', dev, '-'),
                                ('calibrated', res, '--')]:
            bb = core.binned_stats(te['mz'], y, bins=CAL_BINS)
            ax.fill_between(bb['x'], bb['q_lo'], bb['q_hi'], color=col,
                            alpha=0.10 if stage == 'uncalibrated' else 0.18,
                            lw=0, zorder=1)
            ax.plot(bb['x'], bb['median'], style, color=col, lw=2.0,
                    zorder=3 if stage == 'uncalibrated' else 2)
            rows.append(bb.assign(run=tag, isolation_width=label, stage=stage,
                                  n_train=len(tr), n_test=len(te)))
        print(f'    {tag} ({label}): fit on {len(tr):,}, drawn from {len(te):,}'
              f' held out; sigma_IQR {core.sigma_iqr(dev):.3f} -> '
              f'{core.sigma_iqr(res):.3f} Th')

    ax.set_xlabel('precursor m/z', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Q1 centroid $-$ precursor m/z [Th]', fontsize=fs.FONTSIZE)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)

    handles = ([plt.Line2D([], [], color=core.PALETTE_SINGLE[i * 2], lw=2.0,
                           label=lab) for i, (_, lab, _) in enumerate(AXES)]
               + [plt.Line2D([], [], color=fs.LEGEND_DARK, lw=2.0,
                             label='uncalibrated'),
                  plt.Line2D([], [], color=fs.LEGEND_DARK, lw=2.0, ls='--',
                             label='calibrated')])
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=False, fontsize=fs.LEGEND_FONTSIZE,
              handlelength=1.8, handletextpad=0.5, columnspacing=1.4,
              labelspacing=0.3, borderpad=0.15)

    save(fig, ax, 'supplementary_figure2c_q1_calibration',
         pd.concat(rows, ignore_index=True), w_in=2.60)
    plt.close(fig)


def _window_only(mz, w):
    """(median, min, max) over grid phases -- see `window_only_check.py`."""
    from window_only_check import window_only
    v = window_only(mz, w)
    return float(np.median(v)), float(v.min()), float(v.max())


def panel_c():
    """c -- Q1 centroid precision (sigma_IQR) against isolation width."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'figure2'))
    import figure2 as f2
    t = f2.q1_wholefile().copy()
    t = t[t['err'].notna()]
    widths = sorted(t['width_th'].unique())

    def sig(e):
        q25, q75 = np.quantile(e, [0.25, 0.75], axis=-1)
        return (q75 - q25) / IQR_TO_SIGMA

    rng = np.random.default_rng(0)
    rows = []
    print('    width      n   sigma_IQR  95% CI')
    for w in widths:
        e = t.loc[t['width_th'] == w, 'err'].to_numpy()
        boot = np.array([sig(rng.choice(e, len(e))) for _ in range(500)])
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append({'width_th': w, 'n': len(e), 'sigma_iqr_th': sig(e),
                     'ci_lo_th': lo, 'ci_hi_th': hi,
                     'window_only_analytical_th': w / 2 / IQR_TO_SIGMA,
                     **dict(zip(('window_only_th', 'window_only_lo_th',
                                 'window_only_hi_th'),
                                _window_only(t.loc[t['width_th'] == w, 'mz']
                                             .to_numpy(), w))),
                     'median_th': float(np.median(e))})
        print(f'    {w:>5g} Th {len(e):>6}   {sig(e):.3f}   {lo:.3f}-{hi:.3f}')
    r = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    wl = np.array([min(widths) * 0.85, max(widths) * 1.15])
    ax.fill_between(r['width_th'], r['window_only_lo_th'],
                    r['window_only_hi_th'], color='0.45', alpha=0.15, lw=0,
                    zorder=0)
    ax.plot(r['width_th'], r['window_only_th'], color='0.45', lw=1.2,
            ls=':', zorder=1, label='Q1 window only')
    c = core.PALETTE_SINGLE[2]
    ax.plot(r['width_th'], r['sigma_iqr_th'], color=c, lw=1.6, zorder=2)
    ax.errorbar(r['width_th'], r['sigma_iqr_th'],
                yerr=[r['sigma_iqr_th'] - r['ci_lo_th'],
                      r['ci_hi_th'] - r['sigma_iqr_th']],
                fmt='o', color=c, mec='black', mew=0.6, ms=6, capsize=2,
                ecolor='black', elinewidth=0.8, zorder=3,
                label='Q1 centroid')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(*wl)
    ax.xaxis.set_major_locator(mticker.FixedLocator(widths))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    yt = [0.1, 0.2, 0.5, 1, 2]
    ax.set_ylim(0.08, 3.4)
    ax.yaxis.set_major_locator(mticker.FixedLocator(yt))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xlabel('Q1 isolation width [Th]', fontsize=fs.FONTSIZE)
    ax.set_ylabel('m/z locating error, $\\sigma_{IQR}$ [Th]',
                  fontsize=fs.FONTSIZE)
    ax.legend(loc='upper left', frameon=False, fontsize=fs.LEGEND_FONTSIZE,
              handlelength=1.6, borderpad=0.2)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    save(fig, ax, 'supplementary_figure2a_q1_precision_by_width', r)
    plt.close(fig)


def main():
    fs.init()
    fs.start_build(script=__file__, inputs=[DATA, E376], out_dir=OUT)
    print('panel a ...')
    panel_a()
    print('\npanel b ...')
    panel_b()
    print('\npanel c ...')
    panel_c()
    fs.report_printed_sizes()
    fs.report_stale(OUT, prefix='supplementary_figure')


if __name__ == '__main__':
    main()
