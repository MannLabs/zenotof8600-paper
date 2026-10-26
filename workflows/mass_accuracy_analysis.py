"""Precursor mass accuracy, per condition, as a KDE."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from zenotof8600_analytics import sigma_iqr
import matplotlib.pyplot as plt

import technical_qc_analysis as tqc
from lc_peak_width_analysis import _n_bins


CALIB_MODEL = 'offset'

CALIB_BINS = 40

KDE_MAX_POINTS = 50_000
KDE_GRID = 400

BW_FACTOR = 0.9

X_COVERAGE = 0.99

N_INTENSITY_BINS = 30

EXTRA_CONDITIONS = {}

_RNG_SEED = 0


def _register_extra_conditions(extra):
    """Add out-of-root conditions to tqc's registry, so colours, order and the legend helper keep working off a single source of truth."""
    for label, path in extra.items():
        if os.path.isdir(path):
            picked = tqc._pick_output_file(path)
            if picked is None:
                raise FileNotFoundError(f'no search output in {path} (for {label!r})')
            path = picked
        elif not os.path.isfile(path):
            raise FileNotFoundError(f'not a file or folder: {path} (for {label!r})')
        tqc.CONDITIONS[label] = path
        if label not in tqc.COND_ORDER:
            tqc.COND_ORDER.append(label)
    tqc.COLOR = {c: tqc.PALETTE[i % len(tqc.PALETTE)]
                 for i, c in enumerate(tqc.COND_ORDER)}


def configure(*, calib_bins=None, kde_max_points=None, calib_model=None,
              extra_conditions=None, **tqc_kwargs):
    """Set this script's config; everything else is passed to tqc.configure."""
    global CALIB_BINS, KDE_MAX_POINTS, CALIB_MODEL, EXTRA_CONDITIONS
    if extra_conditions is not None:
        EXTRA_CONDITIONS = dict(extra_conditions)
    if calib_model is not None:
        if calib_model not in ('offset', 'linear'):
            raise ValueError(f"calib_model must be 'offset' or 'linear': {calib_model}")
        CALIB_MODEL = calib_model
    if calib_bins is not None:
        CALIB_BINS = int(calib_bins)
    if kde_max_points is not None:
        KDE_MAX_POINTS = int(kde_max_points)
    result = tqc.configure(**tqc_kwargs)
    if EXTRA_CONDITIONS:
        _register_extra_conditions(EXTRA_CONDITIONS)
    return result


def _robust_sd(x):
    """IQR/1.349 - see ``core.sigma_iqr``, which this now defers to."""
    return sigma_iqr(x)


def _read_peaks(path, condition):
    df = pd.read_csv(path, low_memory=False)
    if 'm/z' not in df.columns:
        raise ValueError(f'no theoretical "m/z" column in {path}')
    run_cols = [c for c in df.columns if c.endswith(' m/z') and c != 'm/z']
    if not run_cols:
        raise ValueError(f'no per-run "<run> m/z" columns in {path}. PEAKS only '
                         f'writes those in features.csv, not proteins.csv.')
    out = []
    for c in run_cols:
        run = c[:-len(' m/z')]
        area = pd.to_numeric(df.get(f'{run} Normalized Area'), errors='coerce')
        part = pd.DataFrame({
            'condition': condition,
            'run': run,
            'mz_theoretical': pd.to_numeric(df['m/z'], errors='coerce'),
            'mz_measured': pd.to_numeric(df[c], errors='coerce'),
            'rt': pd.to_numeric(df.get(f'{run} RT mean'), errors='coerce'),
            'intensity': area,
        })
        out.append(part)
    return pd.concat(out, ignore_index=True)


def _read_diann(path, condition):
    """UNVERIFIED — see the module docstring."""
    cols = ['Run', 'Precursor.Mz', 'Ms1.Apex.Mz.Delta', 'RT',
            'Precursor.Normalised']
    df = pd.read_parquet(path, columns=[c for c in cols if c])
    mz = pd.to_numeric(df['Precursor.Mz'], errors='coerce')
    delta = pd.to_numeric(df['Ms1.Apex.Mz.Delta'], errors='coerce')
    return pd.DataFrame({
        'condition': condition,
        'run': df['Run'].astype(str),
        'mz_theoretical': mz,
        'mz_measured': mz + delta,
        'rt': pd.to_numeric(df.get('RT'), errors='coerce'),
        'intensity': pd.to_numeric(df.get('Precursor.Normalised'), errors='coerce'),
    })


def load():
    """Long frame of per-run precursor mass errors, raw and calibrated."""
    if tqc.CONDITIONS is None:
        configure()
    print(f'loading from {tqc.DATA_ROOT} ...')
    parts = []
    for cond in tqc.COND_ORDER:
        path = tqc.condition_path(cond)
        reader = _read_diann if path.lower().endswith('.parquet') else _read_peaks
        part = reader(path, cond)
        only = tqc.condition_runs(cond)
        if only is not None:
            part = part[part['run'].isin(set(only))]
        n_runs = part['run'].nunique()
        print(f'  {cond}: {n_runs} runs  ({os.path.basename(path)})')
        parts.append(part)

    d = pd.concat(parts, ignore_index=True)
    d = d.dropna(subset=['mz_theoretical', 'mz_measured'])
    d = d[(d['mz_measured'] > 0) & (d['mz_theoretical'] > 0)]
    d['ppm_raw'] = (d['mz_measured'] - d['mz_theoretical']) / d['mz_theoretical'] * 1e6

    d['ppm_calibrated'] = np.nan
    d['ppm_detrended'] = np.nan
    fits = []
    for run, g in d.groupby('run', sort=False):
        slope, intercept = _fit_calibration(g['mz_theoretical'].values,
                                            g['ppm_raw'].values)
        offset = float(np.median(g['ppm_raw']))
        d.loc[g.index, 'ppm_detrended'] = _detrend_flexible(
            g['mz_theoretical'].values, g['ppm_raw'].values).values
        d.loc[g.index, 'ppm_calibrated'] = (
            g['ppm_raw'] - (slope * g['mz_theoretical'] + intercept)
            if CALIB_MODEL == 'linear' else g['ppm_raw'] - offset)
        fits.append({'run': run, 'condition': g['condition'].iloc[0],
                     'calib_model': CALIB_MODEL,
                     'offset_ppm': offset,
                     'slope_ppm_per_1000Th': slope * 1000, 'intercept_ppm': intercept,
                     'n_precursors': len(g),
                     'median_raw_ppm': float(np.median(g['ppm_raw'])),
                     'robust_sd_raw_ppm': _robust_sd(g['ppm_raw']),
                     'robust_sd_detrended_ppm': _robust_sd(
                         d.loc[g.index, 'ppm_detrended'])})
    for f in fits:
        cal = d.loc[d['run'] == f['run'], 'ppm_calibrated']
        f['median_calibrated_ppm'] = float(np.median(cal))
        f['robust_sd_calibrated_ppm'] = _robust_sd(cal)
    fit_df = pd.DataFrame(fits)

    print(f'  {len(d):,} precursor measurements over {d["run"].nunique()} runs')
    _report_residual_trends(d)
    return d, fit_df


def _detrend_flexible(mz, ppm, n_bins=60):
    """Remove the m/z trend WITHOUT assuming its shape: subtract the median error of each precursor's own m/z bin."""
    frame = pd.DataFrame({'mz': np.asarray(mz, dtype=float),
                          'ppm': np.asarray(ppm, dtype=float)})
    n = min(n_bins, max(2, len(frame) // 200))
    if n < 2:
        return frame['ppm'] - frame['ppm'].median()
    bins = pd.qcut(frame['mz'], n, duplicates='drop')
    return frame['ppm'] - frame.groupby(bins, observed=True)['ppm'].transform('median')


def _fit_calibration(mz, ppm, n_bins=None):
    """Least squares on per-m/z-bin medians — resistant to the heavy tails."""
    n_bins = CALIB_BINS if n_bins is None else n_bins
    frame = pd.DataFrame({'mz': mz, 'ppm': ppm})
    bins = pd.qcut(frame['mz'], min(n_bins, max(2, len(frame) // 50)),
                   duplicates='drop')
    g = frame.groupby(bins, observed=True).median()
    if len(g) < 2:
        return 0.0, float(np.median(ppm))
    slope, intercept = np.polyfit(g['mz'], g['ppm'], 1)
    return float(slope), float(intercept)


def _report_residual_trends(d):
    """After an m/z-only calibration, is there anything left against RT?"""
    if d['rt'].notna().sum() == 0:
        print('  no RT column — residual RT trend not checked')
        return
    print('  residual after removing the m/z trend (span of binned medians):')
    for cond, g in d.groupby('condition', sort=False):
        g = g.dropna(subset=['rt'])
        if g.empty:
            continue
        b = pd.qcut(g['rt'], 8, duplicates='drop')
        med = g.groupby(b, observed=True)['ppm_detrended'].median()
        span_rt = float(med.max() - med.min())
        b2 = pd.qcut(g['mz_theoretical'], 8, duplicates='drop')
        span_mz = float(g.groupby(b2, observed=True)['ppm_detrended'].median().agg(
            lambda s: s.max() - s.min()))
        print(f'    {cond}: RT span {span_rt:.3f} ppm, residual m/z span '
              f'{span_mz:.3f} ppm  (both should be << the raw m/z trend)')


def _hist_counts(values, edges, *, normalize=True, per=1):
    """Binned counts as a FRACTION of the series' own values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 20:
        return None
    counts, _ = np.histogram(v, bins=edges)
    if normalize:
        return counts / v.size
    return counts / max(per, 1)


def _dist_panel(ax, d, value_col, *, title, xlabel, rows, panel_name,
                coverage=None, show_ci=False, xlim=None):
    """Filled histogram per condition, thin per-run curves behind it."""
    coverage = X_COVERAGE if coverage is None else coverage
    vals = d[value_col].dropna()
    if xlim is not None:
        lo, hi = xlim
    else:
        lo, hi = np.quantile(vals, [(1 - coverage) / 2, 1 - (1 - coverage) / 2])
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad

    inside = vals[vals.between(lo, hi)]
    smallest = min((d[d['condition'] == c][value_col].dropna().between(lo, hi).sum()
                    for c in tqc.COND_ORDER if (d['condition'] == c).any()),
                   default=len(inside))
    n_bins = _n_bins(inside, lo, hi, max(int(smallest), 1))
    edges = np.linspace(lo, hi, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    bin_w = edges[1] - edges[0]

    labels = []
    for cond in tqc.COND_ORDER:
        g = d[d['condition'] == cond]
        if g.empty:
            continue
        color = tqc.COLOR[cond]
        n_runs = g['run'].nunique()

        for run, gr in g.groupby('run', sort=False):
            counts = _hist_counts(gr[value_col], edges)
            if counts is None:
                continue
            ax.plot(centres, counts, color=color, lw=0.6, alpha=0.35, zorder=2)
            rows.append(pd.DataFrame({
                'panel': panel_name, 'series': f'{cond}|{run}', 'condition': cond,
                'run': run, 'x_ppm': centres, 'fraction': counts}))

        v = g[value_col].dropna()
        counts = _hist_counts(v, edges)
        if counts is None:
            continue
        ax.plot(centres, counts, color=color, lw=2.2, zorder=3)
        ax.fill_between(centres, counts, color=color, alpha=0.18, zorder=1)
        med, rsd = float(np.median(v)), _robust_sd(v)
        ax.axvline(med, color=color, ls='--', lw=1.0, alpha=0.9, zorder=4)
        p95 = float(np.quantile(np.abs(v - med), 0.95))
        off_c = float((~v.between(lo, hi)).mean() * 100)
        labels.append(f'{cond}  {med:+.2f} ± {rsd:.2f}'
                      + (f'   ±{p95:.1f}' if show_ci else '')
                      + f'   {off_c:.1f}%')
        rows.append(pd.DataFrame({
            'panel': panel_name, 'series': cond, 'condition': cond, 'run': 'all',
            'x_ppm': centres, 'fraction': counts, 'median_ppm': med,
            'robust_sd_ppm': rsd, 'p95_abs_dev_ppm': p95,
            'pct_off_axis': off_c, 'n': len(v)}))

    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(f'Fraction of precursors (bin {bin_w:.3f} ppm)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    key = ('median ± robust sd · 95% of |dev| · % off axis' if show_ci
           else 'median ± robust sd · % off axis   [ppm]')
    tqc._compact_legend(ax, [c for c in tqc.COND_ORDER
                             if (d['condition'] == c).any()],
                        labels=labels, loc='upper right', fontsize=9.5,
                        title=key, title_fontsize=9.5, alpha=0.18)
    ax.spines[['top', 'right']].set_visible(False)
    return lo, hi


def _trend_panel(ax, d, rows):
    """CALIBRATED mass error against m/z, per run."""
    for cond in tqc.COND_ORDER:
        g = d[d['condition'] == cond]
        if g.empty:
            continue
        color = tqc.COLOR[cond]
        for run, gr in g.groupby('run', sort=False):
            b = pd.qcut(gr['mz_theoretical'], 20, duplicates='drop')
            agg = gr.groupby(b, observed=True).agg(
                mz=('mz_theoretical', 'median'),
                cal=('ppm_calibrated', 'median'))
            ax.plot(agg['mz'], agg['cal'], color=color, lw=1.2, alpha=0.8,
                    zorder=3)
            rows.append(pd.DataFrame({
                'panel': 'error_vs_mz', 'series': f'{cond}|{run}',
                'condition': cond, 'run': run, 'mz': agg['mz'].values,
                'median_ppm_calibrated': agg['cal'].values}))
    ax.axhline(0, color='black', lw=0.8, zorder=2)
    ax.set_xlabel('Precursor m/z (theoretical)', fontsize=10)
    ax.set_ylabel('Median calibrated mass error [ppm]', fontsize=10)
    ax.set_title('Calibrated mass error vs m/z\n(residual trend after the offset)',
                 fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    tqc._compact_legend(ax, [c for c in tqc.COND_ORDER if (d['condition'] == c).any()],
                        loc='best', fontsize=9.5,
                        title='one line per run', title_fontsize=9.5)


def _intensity_panel(ax, d, rows):
    """Calibrated scatter against abundance."""
    if d['intensity'].notna().sum() == 0:
        ax.set_visible(False)
        return
    for cond in tqc.COND_ORDER:
        g = d[(d['condition'] == cond) & d['intensity'].notna()
              & (d['intensity'] > 0)]
        if len(g) < N_INTENSITY_BINS * 50:
            continue
        color = tqc.COLOR[cond]
        per_run = []
        for run, gr in g.groupby('run', sort=False):
            b = pd.qcut(gr['intensity'], N_INTENSITY_BINS, labels=False,
                        duplicates='drop')
            agg = gr.groupby(b, observed=True).agg(
                x=('intensity', lambda s: np.log10(np.median(s))),
                sd=('ppm_detrended', _robust_sd))
            ax.plot(agg['x'], agg['sd'], color=color, lw=0.6, alpha=0.35, zorder=2)
            per_run.append(agg.assign(run=run))
            rows.append(pd.DataFrame({
                'panel': 'scatter_vs_intensity', 'series': f'{cond}|{run}',
                'condition': cond, 'run': run,
                'log10_median_intensity': agg['x'].values,
                'robust_sd_ppm': agg['sd'].values}))
        if not per_run:
            continue
        allr = pd.concat(per_run)
        agg = allr.groupby(level=0).agg(x=('x', 'mean'), sd=('sd', 'mean'))
        ax.plot(agg['x'], agg['sd'], color=color, lw=2.2, marker='o', ms=2.8,
                zorder=3)
        rows.append(pd.DataFrame({
            'panel': 'scatter_vs_intensity', 'series': cond, 'condition': cond,
            'run': 'mean_of_runs', 'log10_median_intensity': agg['x'].values,
            'robust_sd_ppm': agg['sd'].values}))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f'log$_{{10}}$ median precursor intensity '
                  f'({N_INTENSITY_BINS} quantile bins)', fontsize=10)
    ax.set_ylabel('Robust sd, m/z-detrended error [ppm]', fontsize=10)
    ax.set_title('Mass scatter vs abundance\n(ion statistics alone would fall)',
                 fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    tqc._compact_legend(ax, [c for c in tqc.COND_ORDER if (d['condition'] == c).any()],
                        loc='best', fontsize=9.5,
                        title='thin = per run', title_fontsize=9.5)


def mass_accuracy(d=None, fit_df=None):
    """The 2x2 mass-accuracy figure."""
    if d is None:
        d, fit_df = load()

    rows = []

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    _dist_panel(axes[0, 0], d, 'ppm_raw', rows=rows,
                panel_name='hist_uncalibrated',
                title='Precursor mass error\n(as reported — before calibration)',
                xlabel='Mass error [ppm]')
    widest = max(_robust_sd(g['ppm_calibrated'].dropna())
                 for _, g in d.groupby('condition') if g['ppm_calibrated'].notna().any())
    half = max(6 * widest, 1.5)
    _dist_panel(axes[0, 1], d, 'ppm_calibrated', rows=rows,
                panel_name='hist_calibrated', xlim=(-half, half), show_ci=True,
                title='Precursor mass error\n(calibrated — per-run offset removed)',
                xlabel='Calibrated mass error [ppm]')
    _trend_panel(axes[1, 0], d, rows)
    _intensity_panel(axes[1, 1], d, rows)

    for ax in axes.ravel():
        tqc._unbold_axis_labels(ax)
    fig.tight_layout()

    source = pd.concat(rows, ignore_index=True)
    if fit_df is not None:
        source = pd.concat(
            [source, fit_df.assign(panel='calibration_fit', series=fit_df['run'])],
            ignore_index=True)
    tqc.save(fig, 'mass_accuracy', source)

    print('\nper condition (calibrated):')
    for cond in tqc.COND_ORDER:
        g = d[d['condition'] == cond]
        if g.empty:
            continue
        raw, cal = g['ppm_raw'], g['ppm_calibrated']
        print(f'  {cond}: raw median {np.median(raw):+.3f} ppm '
              f'(robust sd {_robust_sd(raw):.3f})  ->  calibrated '
              f'{np.median(cal):+.3f} ppm (robust sd {_robust_sd(cal):.3f}), '
              f'95% within ±{np.quantile(np.abs(cal), 0.95):.2f} ppm, n={len(g):,}')
    if fit_df is not None and len(fit_df):
        print('\ncalibration fits (per run):')
        print(fit_df.to_string(index=False))
    return d, fit_df


def main(**configure_kwargs):
    configure(**configure_kwargs)
    d, fit_df = load()
    mass_accuracy(d, fit_df)
    print(f'\ndone -> {tqc.OUT_DIR}')


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('data_root', nargs='?', default=None)
    ap.add_argument('--conditions', default=None,
                    help='comma-separated subset of condition folder names')
    ap.add_argument('--calib-bins', type=int, default=None)
    ap.add_argument('--calib-model', choices=('offset', 'linear'), default=None)
    ap.add_argument('--extra', action='append', default=None, metavar='LABEL=PATH',
                    help='condition from outside the data root, e.g. '
                         '--extra "8600=Z:\\path\\to\\folder". Repeatable.')
    ap.add_argument('--show', action='store_true')
    a = ap.parse_args()

    kw = {}
    if a.data_root:
        kw['data_root'] = a.data_root
    if a.conditions:
        kw['include_only'] = [c.strip() for c in a.conditions.split(',')]
    if a.calib_bins:
        kw['calib_bins'] = a.calib_bins
    if a.calib_model:
        kw['calib_model'] = a.calib_model
    if a.extra:
        extra = {}
        for item in a.extra:
            if '=' not in item:
                ap.error(f'--extra needs LABEL=PATH, got {item!r}')
            label, path = item.split('=', 1)
            extra[label.strip()] = path.strip()
        kw['extra_conditions'] = extra
    if a.show:
        kw['show_figures'] = True
    main(**kw)
