"""Supplementary figure 8 -- the variable-window Zeno SWATH DIA scheme."""

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
OUT = cfg.panels(__file__)
DATA = cfg.data('supplementary_figure8')
sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

MZ_RANGE = (359.0, 961.0)
N_WINDOWS = 100
ROW_W_IN = 2 * fs.PANEL_W_IN
COLOR = core.PALETTE_SINGLE[2]
BAND = ('0.80', '0.91')


def load():
    mz = core.load_precursor_mz(cfg.data('H032_E293', 'H032_E293_GPF_precursors.csv'))
    acq = pd.read_csv(cfg.data('supplementary_figure8', 'acquired_windows_E352_A1.csv'))
    sc = core.design_windows(mz, mz_range=MZ_RANGE, n_windows=N_WINDOWS)
    t = sc.table
    d = max(np.abs(t['window_start'].to_numpy() - acq['isolation_lower_mz'].to_numpy()).max(),
            np.abs(t['window_end'].to_numpy() - acq['isolation_upper_mz'].to_numpy()).max())
    if len(t) != len(acq) or d > 0.001:
        raise SystemExit(f'regenerated scheme differs from the acquired one by {d:.4f} Th')
    print(f'  {len(mz):,} GPF precursors; regenerated scheme matches the acquired '
          f'windows to {d:.4f} Th')
    return np.asarray(mz), acq


def _bands(ax, acq):
    for k, r in enumerate(acq.itertuples()):
        ax.axvspan(r.isolation_lower_mz, r.isolation_upper_mz,
                   color=BAND[k % 2], lw=0, zorder=0)


def panel_a(mz, acq):
    """a -- precursor m/z distribution, windows as alternating bands."""
    lo, hi = 300.0, 1500.0
    edges = np.arange(lo, hi + 2, 2.0)
    counts, _ = np.histogram(mz, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(ROW_W_IN, 3))
    _bands(ax, acq)
    ax.fill_between(centres, counts, color=COLOR, alpha=0.25, lw=0, zorder=2)
    ax.plot(centres, counts, color=COLOR, lw=1.4, zorder=3)
    inside = (mz >= acq['isolation_lower_mz'].min()) & (mz <= acq['isolation_upper_mz'].max())
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, None)
    ax.set_xlabel('Precursor m/z', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Precursors per 2 Th', fontsize=fs.FONTSIZE)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=ROW_W_IN - 1.0, h_in=fs.AXES_H_IN)
    fs.save_panel(fig, OUT, 'supplementary_figure8a_precursor_mz', pd.DataFrame(
        {'mz_bin_centre': centres, 'precursors': counts, 'bin_width': 2.0}),
        dpi=300)
    plt.close(fig)


def panel_b(mz, acq):
    """b -- precursors per window; bar width = window width."""
    lo_e = acq['isolation_lower_mz'].to_numpy()
    hi_e = acq['isolation_upper_mz'].to_numpy()
    n = np.array([((mz >= a) & (mz < b)).sum() for a, b in zip(lo_e, hi_e)])
    fig, ax = plt.subplots(figsize=(ROW_W_IN, 3))
    ax.bar(lo_e, n, width=hi_e - lo_e, align='edge', color=COLOR, alpha=0.55,
           edgecolor='white', lw=0.4, zorder=2)
    ax.axhline(np.median(n), color='0.35', lw=0.9, ls='--', zorder=3)
    ax.set_xlim(300.0, 1500.0)
    ax.set_xlabel('Isolation window (m/z)', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Precursors per window', fontsize=fs.FONTSIZE)
    w = hi_e - lo_e
    ax.set_ylim(0, n.max() * 1.25)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=ROW_W_IN - 1.0, h_in=fs.AXES_H_IN)
    print(f'  windows {w.min():.2f}-{w.max():.2f} Th; precursors per window '
          f'{n.min()}-{n.max()} (median {np.median(n):.0f})')
    fs.save_panel(fig, OUT, 'supplementary_figure8b_precursors_per_window',
                  acq.assign(width_th=w, precursors=n), dpi=300)
    plt.close(fig)


def main():
    fs.init()
    fs.start_build(script=__file__, inputs=[DATA], out_dir=OUT)
    mz, acq = load()
    panel_a(mz, acq)
    panel_b(mz, acq)
    fs.report_printed_sizes(ROW_W_IN, 1)
    fs.report_stale(OUT, prefix='supplementary_figure')


if __name__ == '__main__':
    main()
