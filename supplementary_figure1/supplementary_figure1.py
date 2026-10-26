"""Supplementary figure 1 — the sensitivity and quant case behind figure 1."""

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

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'figure1'))
import figure1 as f1

COND_ORDER = f1.COND_ORDER
COLOR = f1.COLOR

TOP_Q = 0.90

W_IN = 1.70
H_IN = 2.05


def save(fig, ax, name, source_df, *, w_in=W_IN, dpi=300):
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=w_in, h_in=H_IN)
    return fs.save_panel(fig, OUT, name, source_df, dpi=dpi)


RASTERIZE_OVER = 500
RASTER_DPI = 600


def _fonts(ax):
    ax.tick_params(labelsize=fs.FONTSIZE)
    ax.xaxis.label.set_fontsize(fs.FONTSIZE)
    ax.yaxis.label.set_fontsize(fs.FONTSIZE)
    fs.tidy(ax)


def panel_a(df, sample_info):
    """a — protein-group CV as an ECDF, one curve per arm."""
    fig, ax, cv_stats = core.plot_cv_ecdf(
        df, sample_info, level='protein', group_order=COND_ORDER,
        palette=COLOR, figsize=(fs.PANEL_W_IN, 4), xlim=(0, 30),
        annotate_median=False, legend_loc='lower right',
        legend_fontsize=fs.LEGEND_FONTSIZE)
    _fonts(ax)
    print(cv_stats.to_string(index=False))
    save(fig, ax, 'supplementary_figure1a_pg_cv_ecdf', cv_stats)
    plt.close(fig)


def panel_b(ratio):
    """b — per-precursor intensity gain, with the top decile shaded."""
    r = ratio['log2_ratio']
    med, cut = r.median(), r.quantile(TOP_Q)
    top_med = r[r >= cut].median()

    span = (np.floor(r.quantile(0.005)), np.ceil(r.quantile(0.995)))
    bins = f1._n_bins(r)
    counts, edges = np.histogram(r, bins=bins, range=span)
    centres = (edges[:-1] + edges[1:]) / 2
    color = COLOR[COND_ORDER[1]]

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    ax.fill_between(centres, counts, color=color, alpha=0.18, linewidth=0)
    ax.fill_between(centres, counts, where=centres >= cut, color=color,
                    alpha=0.45, linewidth=0)
    ax.plot(centres, counts, color=color, lw=2.2)
    ax.axvline(0, color='0.35', lw=0.8, ls=':')
    ax.axvline(med, color=color, lw=1.2, ls='--')

    ticks = [t for t in (1, 4, 16, 64)
             if span[0] <= np.log2(t) <= span[1]]
    top_ax = ax.secondary_xaxis('top', functions=(lambda v: v, lambda v: v))
    top_ax.set_xticks([np.log2(t) for t in ticks])
    top_ax.set_xticklabels([f'{t}$\\times$' for t in ticks],
                           fontsize=fs.FONTSIZE)
    top_ax.tick_params(length=2)


    ax.set_xlabel(f'$\\log_2$ ({COND_ORDER[1]} / {COND_ORDER[0]})',
                  fontsize=fs.FONTSIZE)
    ax.set_ylabel(f'precursors per bin (bin {edges[1] - edges[0]:.2f})',
                  fontsize=fs.FONTSIZE)
    ax.set_xlim(*span)
    ax.set_ylim(0, counts.max() * 1.12)
    _fonts(ax)

    source = pd.DataFrame({'log2_ratio_bin_centre': centres,
                           'precursors': counts,
                           'bin_width': edges[1] - edges[0],
                           'in_top_decile': centres >= cut,
                           'n_precursors_total': len(r),
                           'median_log2_ratio': med,
                           'top_decile_cutoff_log2': cut})
    save(fig, ax, 'supplementary_figure1c_intensity_gain', source)
    plt.close(fig)
    print(f'    median {2 ** med:.2f}x   top decile >={2 ** cut:.1f}x '
          f'(median {2 ** top_med:.1f}x)')


def panel_c(df, sample_info):
    """c — precursor CV against a COMMON abundance axis."""
    d = df[df['precursor_intensity'] > 0].merge(
        sample_info[['run', 'condition2']], on='run')
    g = d.groupby(['precursor_id', 'condition2'])['precursor_intensity']
    stat = g.agg(['mean', 'std', 'count']).reset_index()
    stat = stat[stat['count'] == f1.MIN_REPLICATES]
    stat['cv'] = stat['std'] / stat['mean'] * 100

    wide = stat.pivot(index='precursor_id', columns='condition2',
                      values='mean').reindex(columns=COND_ORDER).dropna()
    common = np.log10(wide).mean(axis=1).rename('common_abundance')

    tidy = (stat.set_index('precursor_id')
            .join(common, how='inner').reset_index())

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    out = core.plot_binned_median(
        tidy, x='common_abundance', y='cv',
        group_col='condition2', group_order=COND_ORDER, palette=COLOR,
        n_bins=10, show_iqr=True, threshold=20.0,
        x_label='$\\log_{10}$ mean intensity',
        y_label='Precursor CV [%]', legend_title=None, ylim=(0, None),
        label_fontsize=fs.FONTSIZE, tick_fontsize=fs.FONTSIZE,
        legend_fontsize=fs.LEGEND_FONTSIZE, ax=ax)
    source = out[-1] if isinstance(out, tuple) else out
    _fonts(ax)
    med = tidy.groupby('condition2', observed=True)['cv'].median()
    print('    ' + '  '.join(f'{c} median CV {med[c]:.1f}%'
                             for c in COND_ORDER)
          + f'   (n={len(common):,} shared, complete)')
    save(fig, ax, 'supplementary_figure1d_cv_vs_abundance', source)
    plt.close(fig)


RAW_DIAG = os.path.join(cfg.DATA_ROOT, 'H032_E352')
RAW_DIAG_ARMS = {'7600+': os.path.join(cfg.DATA_ROOT, 'H032_E303'),
                 '8600': RAW_DIAG}
LEVEL_STYLE = {'MS1 precursor': '-', 'MS2 fragment': '--'}
SERIES_COLOR = {('7600+', 'MS1 precursor'): core.PALETTE_SINGLE[0],
                ('7600+', 'MS2 fragment'): core.PALETTE_SINGLE[5],
                ('8600', 'MS1 precursor'): core.PALETTE_SINGLE[1],
                ('8600', 'MS2 fragment'): core.PALETTE_SINGLE[2]}
LEVEL_COLOR = {'MS1 precursor': core.PALETTE_SINGLE[2],
               'MS2 fragment': core.PALETTE_SINGLE[3]}
LEVEL_LABEL = {'MS1 precursor': 'MS1', 'MS2 fragment': 'MS2'}


def raw_diagnostics_arms():
    """Both arms' measurement tables, filtered as raw_diagnostics, + 'arm'."""
    return pd.concat([raw_diagnostics(folder).assign(arm=arm)
                      for arm, folder in RAW_DIAG_ARMS.items()],
                     ignore_index=True)


def raw_diagnostics(folder=None):
    import glob
    folder = RAW_DIAG if folder is None else folder
    f = [p for p in glob.glob(os.path.join(folder, '*.rawdiag.parquet'))
         if '.top30_' not in os.path.basename(p)]
    if len(f) != 1:
        raise SystemExit(f'expected one rawdiag table in {folder}, got {f}')
    d = pd.read_parquet(f[0])
    ms1_iso = (d['level'] == 'MS1 precursor') & (d['label'] != 'precursor')
    ms2_absent = (d['level'] == 'MS2 fragment') & ~(d['sn'] >= MS2_MIN_SN)
    return d[~ms1_iso & ~ms2_absent].reset_index(drop=True)


MS2_MIN_SN = 5.0


def _fwhm_ppm(e, lim=20.0):
    """FWHM of the error distribution from 1 ppm bins -- unlike sigma_IQR it does not depend on how far the search window lets the tails run, and it is what can be read off PEAKS' own histogram."""
    h, edges = np.histogram(e, bins=np.arange(-lim, lim + 0.01, 1.0))
    c = (edges[:-1] + edges[1:]) / 2
    above = c[h >= h.max() / 2]
    return float(above.max() - above.min() + 1.0)


def panel_f(d):
    """f -- calibrated mass error, MS1 precursors and MS2 fragments."""
    lim = 25.0
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    rows = []
    n_small = d.groupby(['arm', 'level']).size().min()
    n_bins = int(np.clip(n_small / 250, 20, 60))
    edges = np.linspace(-lim, lim, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    for arm, lvl in [(a, l) for l in LEVEL_STYLE for a in COND_ORDER]:
        col, ls = SERIES_COLOR[(arm, lvl)], LEVEL_STYLE[lvl]
        e = d.loc[(d['arm'] == arm) & (d['level'] == lvl), 'ppm'].dropna()
        counts, _ = np.histogram(e, bins=edges)
        frac = counts / len(e)
        s = core.sigma_iqr(e)
        fwhm = _fwhm_ppm(e)
        lab = f'{arm} {LEVEL_LABEL[lvl]}'
        ax.plot(centres, frac, color=col, lw=2.0, ls=ls, label=lab)
        rows.append(pd.DataFrame({'arm': arm, 'level': lvl,
                                  'ppm_bin_centre': centres,
                                  'fraction_of_peaks': frac,
                                  'median_abs_error_ppm':
                                      float(e.abs().median()),
                                  'sigma_iqr_ppm': s, 'fwhm_ppm': fwhm,
                                  'median_ppm': e.median(),
                                  'n_peaks': len(e),
                                  'fraction_beyond_20ppm':
                                      float((e.abs() > lim).mean())}))
        print(f'    {arm} {lvl}: median {e.median():+.2f} ppm, median |error| '
              f'{e.abs().median():.2f}, sigma_IQR {s:.2f}, '
              f'within +-10 ppm {float((e.abs() <= 10).mean()):.1%}, n {len(e):,}')
        g = d[(d['arm'] == arm) & (d['level'] == lvl)].dropna(
            subset=['ppm', 'height'])
        q = pd.qcut(np.log10(g['height']), 5, labels=False)
        print('      sigma_IQR by height quintile: '
              + ', '.join(f'{core.sigma_iqr(x):.1f}'
                          for _, x in g.groupby(q)['ppm']))
    ax.axvline(0, color='0.35', lw=0.8, ls=':')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, None)
    ax.set_xlabel('Mass error [ppm]')
    ax.set_ylabel(f'Fraction of peaks (bin {edges[1] - edges[0]:.1f} ppm)')
    ax.legend(loc='upper left', frameon=False, fontsize=fs.LEGEND_FONTSIZE,
              handlelength=1.0, borderpad=0.2, labelspacing=0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.8)
    _fonts(ax)
    save(fig, ax, 'supplementary_figure1g_mass_error', pd.concat(rows))
    plt.close(fig)


def panel_g(d):
    """g -- resolution (m/z / FWHM) against m/z, MS1 and MS2."""
    t = d[np.isfinite(d['resolution'])].copy()
    t['series'] = t['arm'] + ' ' + t['level'].map(LEVEL_LABEL)
    order = [f'{a} {LEVEL_LABEL[l]}' for l in LEVEL_STYLE for a in COND_ORDER]
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    out = core.plot_binned_median(
        t, x='mz_observed', y='resolution', group_col='series',
        group_order=order,
        palette={f'{a} {LEVEL_LABEL[l]}': c
                 for (a, l), c in SERIES_COLOR.items()},
        n_bins=16, mode='width', min_count=30, show_iqr=True, x_label='m/z',
        y_label='Resolution (m/z / FWHM)', legend_title=None,
        ylim=(0, None), label_fontsize=fs.FONTSIZE,
        tick_fontsize=fs.FONTSIZE, legend_fontsize=fs.LEGEND_FONTSIZE, ax=ax)
    source = out[-1] if isinstance(out, tuple) else out
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f'{v / 1000:.0f}k' if v else '0'))
    _fonts(ax)
    for line in ax.get_lines():
        if str(line.get_label()).endswith('MS2'):
            line.set_linestyle('--')
    ax.legend(*ax.get_legend_handles_labels(), loc='lower right',
              frameon=False, fontsize=fs.LEGEND_FONTSIZE, handlelength=1.6,
              labelspacing=0.25, borderpad=0.1)
    for lvl, g in t.groupby('series'):
        near = g[(g['mz_observed'] > 780) & (g['mz_observed'] < 880)]
        print(f'    {lvl}: median R {g.resolution.median():,.0f}; m/z 780-880 '
              f'{near.resolution.median():,.0f} (n {len(near)})')
    save(fig, ax, 'supplementary_figure1h_resolution', source)
    plt.close(fig)


def completeness(df, sample_info):
    """Fraction of each arm's union retained at 'detected in >= k of N runs'."""
    d = df[df['precursor_intensity'] > 0].merge(
        sample_info[['run', 'condition2']], on='run')
    rows = []
    for level, col in [('precursor', 'precursor_id'),
                       ('protein group', 'protein_group')]:
        for c in COND_ORDER:
            sub = d[d['condition2'] == c]
            n_runs = sub['run'].nunique()
            seen = sub.groupby(col)['run'].nunique()
            union = len(seen)
            for k in range(1, n_runs + 1):
                rows.append({'arm': c, 'level': level, 'k': k,
                             'n_runs': n_runs, 'entities': int((seen >= k).sum()),
                             'fraction_of_union': float((seen >= k).mean()),
                             'union': union})
    return pd.DataFrame(rows)


def panel_d(df, sample_info):
    """d — completeness, as the fraction of the union retained at k of N."""
    comp = completeness(df, sample_info)
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    for c in COND_ORDER:
        for level, style in [('precursor', '--'), ('protein group', '-')]:
            s = comp[(comp['arm'] == c) & (comp['level'] == level)]
            ax.plot(s['k'], s['fraction_of_union'], style, color=COLOR[c],
                    lw=2.0, marker='o', ms=4)
    handles = [plt.Line2D([], [], color=COLOR[c], lw=2.0) for c in COND_ORDER]
    handles.append(plt.Line2D([], [], color='0.45', lw=2.0, ls='--'))
    handles.append(plt.Line2D([], [], color='0.45', lw=2.0, ls='-'))
    ax.legend(handles, list(COND_ORDER) + ['precursor', 'protein group'],
              loc='lower left', frameon=False, fontsize=fs.LEGEND_FONTSIZE,
              handlelength=1.4, handletextpad=0.5, borderpad=0.2,
              labelspacing=0.35)
    ax.set_xticks(sorted(comp['k'].unique()))
    ax.set_xlabel('detected in $\\geq$ k of 4',
                  fontsize=fs.FONTSIZE)
    ax.set_ylabel('fraction of the arm\'s union', fontsize=fs.FONTSIZE)
    ax.set_ylim(0, 1.04)
    _fonts(ax)
    save(fig, ax, 'supplementary_figure1e_completeness', comp)
    plt.close(fig)
    full = comp[comp['k'] == comp['n_runs']]
    for _, r in full.iterrows():
        print(f'    {r["arm"]:6s} {r["level"]:13s} 4/4: '
              f'{r["fraction_of_union"]:.3f} of {r["union"]:,}')


def panel_e(df, sample_info):
    """e — protein-group rank / dynamic range."""
    fig, ax, source = core.plot_rank(
        df, sample_info, level='protein', group_col='condition2',
        figsize=(fs.PANEL_W_IN, 4), palette=[COLOR[c] for c in COND_ORDER],
        point_size=4, alpha=0.6)
    ax.set_xlabel('protein group rank', fontsize=fs.FONTSIZE)
    ax.set_ylabel('$\\log_{10}$ intensity', fontsize=fs.FONTSIZE)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(False)
        for t in leg.get_texts():
            t.set_fontsize(fs.LEGEND_FONTSIZE)
            t.set_text(t.get_text().split(' (n=')[0])
    _fonts(ax)
    n_r = 0
    for coll in ax.collections:
        if len(coll.get_offsets()) > RASTERIZE_OVER:
            coll.set_rasterized(True)
            n_r += len(coll.get_offsets())
    print(f'    rasterized {n_r:,} points at {RASTER_DPI} dpi')
    save(fig, ax, 'supplementary_figure1f_protein_rank', source,
         dpi=RASTER_DPI)
    plt.close(fig)


def main(*letters):
    fs.init()
    if letters and set(letters) <= {'g', 'h'}:
        fs.start_build(script=__file__, inputs=list(RAW_DIAG_ARMS.values()),
                       out_dir=OUT)
        d = raw_diagnostics_arms()
        if 'g' in letters:
            panel_f(d)
        if 'h' in letters:
            panel_g(d)
        fs.report_printed_sizes()
        fs.report_stale(OUT, prefix='supplementary_figure')
        return
    fs.start_build(script=__file__, inputs=[f1.DATA], out_dir=OUT)
    df, sample_info = f1.load()
    print(f'  {df.shape[0]:,} rows, {len(sample_info)} runs')
    for name, fn in [('a', lambda: panel_a(df, sample_info)),
                     ('d', lambda: panel_c(df, sample_info)),
                     ('e', lambda: panel_d(df, sample_info)),
                     ('f', lambda: panel_e(df, sample_info))]:
        print(f'\npanel {name} ...')
        fn()
    print('\npanel b ...')
    f1.panel_intensity(df, sample_info, OUT,
                       'supplementary_figure1b_intensity_distributions',
                       w_in=W_IN, h_in=H_IN, legend_ncol=2)
    print('\npanel c ...')
    ratio = f1.precursor_ratios(df, sample_info)
    panel_b(ratio)
    print()
    f1.report_gain(ratio)
    d = raw_diagnostics_arms()
    panel_f(d)
    panel_g(d)
    fs.report_printed_sizes()
    fs.report_stale(OUT, prefix='supplementary_figure')


if __name__ == '__main__':
    main(*sys.argv[1:])
