"""Supplementary figure 2, d and e -- the Q1 dimension and its abundance cutoff."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys

import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = cfg.panels(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import figure_style as fs
sys.path.insert(0, os.path.join(ROOT, 'figure2'))
import figure2 as f2

E376 = os.path.join(ROOT, 'figure2', 'data', 'H032_E376')
Q1_TOP30 = os.path.join(E376, 'q1_precision_by_width_p32.csv')

LETTER = {'q1_abundance': 'b', 'occupancy': 'd'}

PCT_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
MIN_PER_BIN = 8
Y_MAX_TH = 0.40

Q1_STEP = 0.1
CYCLE_MIN = 1.48 / 60


def sigma_iqr(x):
    x = pd.Series(x).dropna()
    return (float((x.quantile(.75) - x.quantile(.25)) / 1.349)
            if len(x) >= MIN_PER_BIN else np.nan)


def width_label(w):
    return f'{w:g}'


Y_CAP_TH = 1.5


def panel_q1_abundance():
    """d -- Q1 sigma against abundance over the WHOLE range, whole file."""
    sys.path.insert(0, os.path.join(ROOT, 'figure2'))
    d = f2.q1_wholefile(min_pct=0.0)
    mids = [(a + b) / 2 for a, b in zip(PCT_BINS[:-1], PCT_BINS[1:])]
    d = d.assign(pct_mid=pd.cut(d['area_pct'], PCT_BINS,
                                labels=mids).astype(float))
    src = (d.groupby(['width_th', 'pct_mid'])
           .agg(n=('err', 'size'), sigma_iqr_th=('err', sigma_iqr))
           .reset_index())
    print(src.pivot(index='pct_mid', columns='width_th',
                    values='sigma_iqr_th').round(3).to_string())

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    ax.axvline(f2.Q1_MIN_PCT, color='0.35', lw=1.0, ls=(0, (5, 3)), zorder=1)
    for w in sorted(src.width_th.unique()):
        s_ = src[src.width_th == w]
        ax.plot(s_.pct_mid, s_.sigma_iqr_th, '-o', color=f2.COLOR[width_label(w)],
                lw=1.8, ms=4.5, label=f'{w:g} Th', zorder=3)
    ax.set_ylim(0, Y_CAP_TH)
    ax.set_xlim(0, 100)
    ax.set_ylabel(r'Q1 $\sigma_{IQR}$ [Th]', fontsize=fs.FONTSIZE)
    ax.set_xlabel('Precursor abundance percentile', fontsize=fs.FONTSIZE)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3,
              frameon=False, fontsize=fs.LEGEND_FONTSIZE, handlelength=1.2,
              handletextpad=0.4, columnspacing=0.8, borderpad=0.1,
              labelspacing=0.25)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=2.60, h_in=fs.AXES_H_IN)
    fs.save_panel(fig, OUT, f'supplementary_figure2{LETTER["q1_abundance"]}'
                  f'_q1_precision_vs_abundance', src, dpi=300)
    plt.close(fig)


def reference_set():
    """The abundant union precursor set of supplementary 2d -- one definition."""
    df, si = f2.load()
    return f2.abundant_reference(df, si)


def interpolated_quantile(n, q=0.5):
    """Grouped-data quantile of an integer count (class k spans k +- 0.5)."""
    n = np.asarray(n, dtype=np.int64)
    c = np.bincount(n)
    F = np.cumsum(c)
    target = q * len(n)
    k = int(np.searchsorted(F, target))
    below = F[k - 1] if k else 0
    return float(k - 0.5 + (target - below) / c[k])


def interpolated_median(n):
    return interpolated_quantile(n, 0.5)


def window_counts(mz, rt, half_width, rt_tol, mz_lo, mz_hi, rt_lo, rt_hi):
    """Count of precursors in every (Q1 centre, cycle) window, by box sums."""
    qe = np.arange(mz_lo - half_width - Q1_STEP, mz_hi + half_width
                   + 2 * Q1_STEP, Q1_STEP)
    te = np.arange(rt_lo - rt_tol - CYCLE_MIN, rt_hi + rt_tol
                   + 2 * CYCLE_MIN, CYCLE_MIN)
    H, _, _ = np.histogram2d(mz, rt, bins=[qe, te])
    C = np.zeros((H.shape[0] + 1, H.shape[1] + 1))
    C[1:, 1:] = H.cumsum(0).cumsum(1)
    qc = np.arange(mz_lo, mz_hi + 1e-9, Q1_STEP)
    tc = np.arange(rt_lo, rt_hi + 1e-9, CYCLE_MIN)
    i0 = np.searchsorted(qe, qc - half_width)
    i1 = np.searchsorted(qe, qc + half_width)
    j0 = np.searchsorted(te, tc - rt_tol)
    j1 = np.searchsorted(te, tc + rt_tol)
    return (C[i1][:, j1] - C[i0][:, j1] - C[i1][:, j0]
            + C[i0][:, j0]).ravel()


def panel_occupancy():
    """e -- co-isolated precursors per precursor, isolation window vs Q1 centroid."""
    ref = reference_set()
    mz, rt = ref['mz'].to_numpy(), ref['rt'].to_numpy()
    print(f'    reference set {len(ref):,} precursors, co-elution '
          f'+-{f2.COELUTION_MIN} min')
    sig = f2.q1_sigma_by_width()
    rows = []
    for w in sorted(sig):
        for kind, half in (('window', w / 2),
                           ('q1_element', f2.K_SIGMA * sig[w])):
            n = f2.n_within_tol(mz, rt, half, f2.COELUTION_MIN) - 1
            rows.append({'width_th': w, 'kind': kind, 'element_th': 2 * half,
                         'mean': float(n.mean()), 'median': float(np.median(n)),
                         'q25': float(np.percentile(n, 25)),
                         'q75': float(np.percentile(n, 75)),
                         'frac_none': float((n == 0).mean()),
                         'n_precursors': len(n)})
            print(f'      {w:>4} Th  {kind:<10} +-{half:5.3f} Th  mean '
                  f'{n.mean():5.2f}  median {np.median(n):3.0f}  IQR '
                  f'{np.percentile(n, 25):.0f}-{np.percentile(n, 75):.0f}  '
                  f'none {100 * (n == 0).mean():4.1f}%')
    src = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    style = {'window': (core.PALETTE_SINGLE[0], 'in isolation window'),
             'q1_element': (core.PALETTE_SINGLE[1],
                            rf'within $\pm${f2.K_SIGMA:g}$\sigma$ of Q1 centroid')}
    for kind, (col, lab) in style.items():
        d = src[src.kind == kind]
        ax.fill_between(d.width_th, d.q25, d.q75, color=col, alpha=0.18, lw=0)
        ax.plot(d.width_th, d['mean'], '-o', color=col, lw=2.2, ms=5,
                label=lab)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=1,
              frameon=False, fontsize=fs.LEGEND_FONTSIZE, handlelength=1.6,
              handletextpad=0.5, borderpad=0.15, labelspacing=0.3)
    ax.set_xlabel('Q1 isolation width [Th]', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Co-isolated precursors', fontsize=fs.FONTSIZE)
    ax.set_ylim(0, None)
    ax.set_xticks([1, 2.8, 4.3, 5.7, 7.2])
    ax.set_xticklabels(['1', '2.8', '4.3', '5.7', '7.2'])
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=2.60, h_in=fs.AXES_H_IN)
    fs.save_panel(fig, OUT, f'supplementary_figure2{LETTER["occupancy"]}'
                  f'_precursors_per_window', src, dpi=300)
    plt.close(fig)


def main(*letters):
    wanted = set(letters) or set(LETTER.values())
    unknown = wanted - set(LETTER.values())
    if unknown:
        raise SystemExit(f'unknown panel(s) {sorted(unknown)}')
    fs.init()
    fs.start_build(script=__file__, inputs=[E376], out_dir=OUT)
    if LETTER['q1_abundance'] in wanted:
        print('panel e -- Q1 precision vs abundance ...')
        panel_q1_abundance()
    if LETTER['occupancy'] in wanted:
        print('panel f -- precursors per window ...')
        panel_occupancy()
    fs.report_stale(OUT, prefix='supplementary_figure')


if __name__ == '__main__':
    main(*sys.argv[1:])
