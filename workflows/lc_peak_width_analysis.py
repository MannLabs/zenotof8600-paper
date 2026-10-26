"""LC peak width (FWHM) measured from the raw MS1, per condition."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import zenotof8600_analytics as core
import technical_qc_analysis as tqc


RUNS_PER_CONDITION = 1

TOP_FRACTION = 0.50
MAX_TARGETS = None

MZ_TOL_PPM = 20.0
RT_WINDOW = 'auto'
RT_WINDOW_START = 0.05
MIN_SN = 5.0

MIN_XIC_POINTS = 5

MIN_SCANS_IN_PROBE = 12

USE_FRAGMENTS = True
ACTIVATION = 'cid'
FRAGMENT_ION_TYPES = ('b', 'y')
FRAGMENT_MZ_MARGIN = 0.0
MIN_FRAGMENTS = 3
MAX_APEX_DELTA_S = 2.0


HIST_BINS = None
MIN_COUNTS_PER_BIN = 250
HIST_BINS_RANGE = (20, 100)

CACHE_DIR = None


def _ms_cache_dir():
    """Directory for alpharaw's HDF caches of the raw files."""
    if CACHE_DIR is not None:
        return CACHE_DIR
    base = (os.environ.get('LOCALAPPDATA')
            or os.path.join(os.path.expanduser('~'), '.cache'))
    return os.path.join(base, 'zenotof8600_analytics', 'ms_data_cache')


_LOADED = {}


def _load_run(raw_path, *, verbose=False):
    key = (os.path.abspath(raw_path), _ms_cache_dir())
    if key not in _LOADED:
        from zenotof8600_analytics.raw import load_ms_data
        cache_dir = _ms_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        _LOADED.clear()
        _LOADED[key] = load_ms_data(raw_path, cache_dir=cache_dir,
                                    verbose=verbose)
    return _LOADED[key]


def configure(*, runs_per_condition=None, top_fraction=None, max_targets=None,
              rt_window=None,
              cache_dir=None, use_fragments=None, **tqc_kwargs):
    """Set this script's config; everything else is passed to tqc.configure."""
    global RUNS_PER_CONDITION, TOP_FRACTION, MAX_TARGETS, CACHE_DIR, RT_WINDOW
    global USE_FRAGMENTS
    if use_fragments is not None:
        USE_FRAGMENTS = use_fragments
    if runs_per_condition is not None:
        RUNS_PER_CONDITION = (None if runs_per_condition == 'all'
                              else int(runs_per_condition))
    if top_fraction is not None:
        TOP_FRACTION = float(top_fraction)
        if not 0 < TOP_FRACTION <= 1:
            raise ValueError(f'top_fraction must be in (0, 1]: {TOP_FRACTION}')
    if rt_window is not None:
        RT_WINDOW = rt_window
    if max_targets is not None:
        MAX_TARGETS = int(max_targets)
    if cache_dir is not None:
        CACHE_DIR = cache_dir
    return tqc.configure(**tqc_kwargs)


def load(**kwargs):
    """Load the long frame exactly as the QC script does."""
    return tqc.load(**kwargs)


def _targets(df, run):
    """The top TOP_FRACTION of one run's precursors, by abundance."""
    sub = df[(df['run'] == run) & df['precursor_intensity'].notna()]
    if 'rt' not in sub.columns:
        return sub.iloc[:0]
    sub = sub.drop_duplicates('precursor_id')
    n = int(round(TOP_FRACTION * len(sub)))
    if MAX_TARGETS is not None:
        n = min(n, MAX_TARGETS)
    return (sub.sort_values('precursor_intensity', ascending=False).head(n)
               .dropna(subset=['mz', 'rt']))


def _fragments_by_target(targets):
    """{positional target index: its fragments above the precursor m/z}."""
    from zenotof8600_analytics.raw.targeted import (target_fragments,
                                                   y_ions_above_precursor)
    need = ('sequence', 'mods', 'mod_sites', 'charge')
    missing = [c for c in need if c not in targets.columns]
    if missing:
        raise ValueError(f'fragment mode needs {missing} in the long frame')
    frags = target_fragments(targets[list(need)], activation=ACTIVATION)
    frags = y_ions_above_precursor(frags, ion_types=FRAGMENT_ION_TYPES,
                                   margin=FRAGMENT_MZ_MARGIN)
    return {k: v for k, v in frags.groupby('target_idx')}


def _cycle_time_s(raw_path):
    """Median MS1 scan-to-scan spacing, in seconds — the acquisition cycle."""
    ms = _load_run(raw_path)
    rt = np.sort(ms.spectrum_df.loc[ms.spectrum_df['ms_level'] == 1,
                                    'rt'].to_numpy())
    return float(np.median(np.diff(rt))) * 60 if rt.size > 1 else np.nan


def measure_run(raw_path, targets, *, verbose=True, rt_window=None):
    """Measure LC FWHM for one run, one chromatogram per target."""
    from zenotof8600_analytics.raw.extraction import (extract_xic, pick_peak,
                                                     points_in_window)

    ms = _load_run(raw_path, verbose=verbose)

    ms1_rt = np.sort(ms.spectrum_df.loc[ms.spectrum_df['ms_level'] == 1,
                                        'rt'].to_numpy())
    cycle_s = float(np.median(np.diff(ms1_rt))) * 60 if ms1_rt.size > 1 else np.nan

    targets = targets.reset_index(drop=True)
    by_target = _fragments_by_target(targets) if USE_FRAGMENTS else {}

    rows = []
    win = RT_WINDOW if rt_window is None else rt_window
    if not isinstance(win, (int, float)):
        raise ValueError("measure_run needs a numeric window; call "
                         "resolve_rt_window() first when RT_WINDOW is 'auto'")
    for i, t in enumerate(targets.itertuples()):
        if USE_FRAGMENTS:
            frag = by_target.get(i)
            if frag is None or len(frag) < MIN_FRAGMENTS:
                continue
            xic = extract_xic(ms, frag['product_mz'].to_numpy(),
                              mz_tol=MZ_TOL_PPM, ms_level=2, precursor_mz=t.mz,
                              rt_range=(t.rt - win, t.rt + win))
        else:
            xic = extract_xic(ms, t.mz, mz_tol=MZ_TOL_PPM, ms_level=1,
                              rt_range=(t.rt - win, t.rt + win))
        if len(xic) < MIN_XIC_POINTS:
            continue
        rt_grid = xic['rt'].to_numpy()
        peak = pick_peak(rt_grid, xic['intensity'].to_numpy(),
                         baseline='offpeak',
                         interpolate_points=500, min_sn=MIN_SN)
        if not np.isfinite(peak['apex_rt']) or peak['fwhm'] <= 0:
            continue
        if abs(peak['apex_rt'] - t.rt) * 60 > MAX_APEX_DELTA_S:
            continue
        rows.append({
            'sequence': getattr(t, 'sequence', ''),
            'charge': getattr(t, 'charge', pd.NA),
            'mz': t.mz, 'reported_rt_min': t.rt,
            'n_fragments': len(by_target[i]) if USE_FRAGMENTS else 0,
            'apex_rt_min': peak['apex_rt'],
            'apex_intensity': peak['apex_intensity'],
            'fwhm_s': peak['fwhm'] * 60,
            'n_points_fwhm': points_in_window(rt_grid, peak['apex_rt'],
                                              peak['fwhm'] / 2),
            'points_per_fwhm': (peak['fwhm'] * 60 / cycle_s
                                if np.isfinite(cycle_s) and cycle_s > 0
                                else np.nan),
            'sn': peak['sn'],
        })
    return pd.DataFrame(rows), cycle_s


def resolve_rt_window(raw_path, targets, *, n_probe=200, verbose=True):
    """The extraction half-window, chosen once from the peaks and reused."""
    if RT_WINDOW != 'auto':
        return float(RT_WINDOW)
    probe = targets.head(n_probe)

    def measure(w):
        res, _cyc = measure_run(raw_path, probe, verbose=False, rt_window=w)
        if len(res) < 10:
            return np.nan, {}
        return float(res['fwhm_s'].median()) / 60.0, {
            'ok_frac': len(res) / max(len(probe), 1)}

    start = core.seed_rt_window(_cycle_time_s(raw_path),
                                start=float(RT_WINDOW_START),
                                min_scans=MIN_SCANS_IN_PROBE,
                                verbose=verbose, label='RT_WINDOW')

    win, _trace = core.auto_rt_window(measure, start=start,
                                      verbose=verbose, label='RT_WINDOW')
    if not np.isfinite(win):
        raise RuntimeError(
            f'could not resolve an RT window on {os.path.basename(raw_path)}: '
            f'no window held measurable peaks for at least 10 of '
            f'{len(probe)} probe targets. Pin RT_WINDOW / --rt-window to a '
            f'float to measure anyway.')
    return win


def _runs_to_measure(sample_info):
    """(condition, run, raw path) for the runs to measure, in COND_ORDER."""
    raw_paths = tqc._raw_file_map()
    chosen = []
    for cond in tqc.COND_ORDER:
        runs = list(sample_info.loc[sample_info['condition2'] == cond, 'run'])
        if 'acq_time' in sample_info.columns:
            times = sample_info.set_index('run')['acq_time']
            runs = sorted(runs, key=lambda r: (pd.isna(times.get(r)),
                                               times.get(r)))
        if RUNS_PER_CONDITION is not None:
            runs = runs[:RUNS_PER_CONDITION]
        chosen += [(cond, r, raw_paths[r]) for r in runs if r in raw_paths]
    return chosen


def _n_bins(values, lo, hi, n_smallest):
    """Bin count over [lo, hi]: Freedman-Diaconis, capped by counts per bin."""
    if HIST_BINS is not None:
        return HIST_BINS
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return HIST_BINS_RANGE[0]
    iqr = float(np.subtract(*np.percentile(v, [75, 25])))
    fd = (hi - lo) / (2 * iqr / v.size ** (1 / 3)) if iqr > 0 else np.inf
    by_counts = n_smallest / MIN_COUNTS_PER_BIN
    return int(np.clip(round(min(fd, by_counts)), *HIST_BINS_RANGE))


_LOG_TICK_CANDIDATES = (0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30)


def _short_labels(conditions):
    """Drop the prefix every condition shares, at an underscore boundary."""
    parts = [c.split('_') for c in conditions]
    n = 0
    while (len(parts) > 1 and all(len(p) > n + 1 for p in parts)
           and len({p[n] for p in parts}) == 1):
        n += 1
    return ['_'.join(p[n:]) for p in parts]


def _plain_log_ticks(ax, xlim):
    """Label a log x-axis with plain numbers over `xlim`."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ticks = [t for t in _LOG_TICK_CANDIDATES if xlim[0] <= t <= xlim[1]]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _histogram_by_condition(ax, data, value_col, groups, *, xlabel,
                            median_fmt='{:.2f}', legend_loc='upper right',
                            log_x=False, xlim=None):
    """Overlapping per-condition distributions as filled frequency polygons."""
    present = [g for g in groups if (data['condition'] == g).any()]
    if xlim is not None:
        lo, hi = xlim
    else:
        lo = float(data[value_col].quantile(0.002))
        hi = float(data[value_col].quantile(0.998))
    n_smallest = min(int(data.loc[data['condition'] == g, value_col].notna().sum())
                     for g in present)
    if log_x:
        nb = _n_bins(np.log10(data[value_col]), np.log10(lo), np.log10(hi),
                     n_smallest)
        edges = np.logspace(np.log10(lo), np.log10(hi), nb + 1)
        centres = np.sqrt(edges[:-1] * edges[1:])
    else:
        nb = _n_bins(data[value_col], lo, hi, n_smallest)
        edges = np.linspace(lo, hi, nb + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])

    short = dict(zip(present, _short_labels(present)))
    labels, medians, ymax = [], {}, 0.0
    for g in present:
        sub = data.loc[data['condition'] == g]
        vals = sub[value_col].to_numpy()
        n_runs = max(sub['run'].nunique(), 1)
        counts, _ = np.histogram(vals, bins=edges)
        counts = counts / n_runs
        ymax = max(ymax, counts.max())
        ax.fill_between(centres, 0, counts, color=tqc.COLOR[g], alpha=0.18,
                        linewidth=0, zorder=2)
        ax.plot(centres, counts, color=tqc.COLOR[g], linewidth=2.2, zorder=3)
        med = float(np.median(vals))
        medians[g] = med
        ax.axvline(med, color=tqc.COLOR[g], linestyle='--', linewidth=1.4,
                   alpha=0.9, zorder=5)
        labels.append(f'{short[g]}   {median_fmt.format(med)}')

    ax.set_xlabel(xlabel, fontsize=10)
    binning = (f'{len(centres)} log-spaced bins' if log_x
               else f'bin {edges[1] - edges[0]:.3g} s')
    ax.set_ylabel(f'precursors per bin (mean / run, {binning})', fontsize=10)
    if log_x:
        ax.set_xscale('log')
    ax.set_xlim(edges[0], edges[-1])
    ax.tick_params(labelsize=10)
    ax.set_ylim(0, ymax * 1.30)
    tqc._compact_legend(ax, present, loc=legend_loc, labels=labels,
                        fontsize=9.5, title_fontsize=9.5, alpha=0.45,
                        title='median (dashed)')
    return medians


def lc_peak_width(df, sample_info):
    """Measure and plot LC peak width."""
    print('LC peak width ...')
    try:
        import zenotof8600_analytics.raw
    except Exception as exc:
        print(f'  skipped: alpharaw not available ({exc}); '
              f'pip install -e C:\\Github\\Analytics[raw]')
        return None

    chosen = _runs_to_measure(sample_info)
    if not chosen:
        print(f'  skipped: no raw file found for any run in '
              f'{", ".join(tqc.RAW_SEARCH_DIRS)}')
        return None

    print(f'  cache: {_ms_cache_dir()}')
    per_target, per_run = [], []
    win = None
    for cond, run, path in chosen:
        targets = _targets(df, run)
        if targets.empty:
            print(f'  {run}: no targets with both mz and rt — skipped')
            continue
        if win is None:
            win = resolve_rt_window(path, targets)
        meas, cycle_s = measure_run(path, targets, rt_window=win)
        if meas.empty:
            print(f'  {run}: no peak cleared S/N {MIN_SN}')
            continue
        meas.insert(0, 'run', run)
        meas.insert(0, 'condition', cond)
        meas['ms1_cycle_s'] = cycle_s
        meas['rt_window_min'] = win
        per_target.append(meas)
        per_run.append({'condition': cond, 'run': run, 'ms1_cycle_s': cycle_s,
                        'n_measured': len(meas), 'n_targets': len(targets),
                        'median_fwhm_s': meas['fwhm_s'].median(),
                        'median_points_per_fwhm':
                            meas['points_per_fwhm'].median()})
        print(f'  {run}: {len(meas):,}/{len(targets):,} peaks, cycle '
              f'{cycle_s:.2f} s, median FWHM {meas["fwhm_s"].median():.2f} s, '
              f'{meas["points_per_fwhm"].median():.1f} scans across it')
    if not per_target:
        print('  skipped: nothing measurable')
        return None

    data = pd.concat(per_target, ignore_index=True)
    runs_df = pd.DataFrame(per_run)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    _histogram_by_condition(
        axes[0], data, 'fwhm_s', tqc.COND_ORDER,
        xlabel='LC peak FWHM [s]', median_fmt='{:.2f} s')
    source = ('MS2 fragments above the precursor' if USE_FRAGMENTS
              else 'MS1 precursor XIC')
    conds = [c for c in tqc.COND_ORDER if (data['condition'] == c).any()]
    dropped = conds[0][:len(conds[0]) - len(_short_labels(conds)[0])]
    prefix = f'{dropped}* conditions, ' if dropped else ''
    fig.suptitle(f'LC peak width — {prefix}{source}, top {TOP_FRACTION:.0%} of '
                 f'precursors by abundance', fontsize=11, fontweight='bold')
    axes[0].set_title('Peak width (FWHM)', fontsize=10)

    ppf = data['points_per_fwhm']
    xlim = (float(ppf.quantile(0.002)), float(ppf.quantile(0.998)))
    _histogram_by_condition(
        axes[1], data, 'points_per_fwhm', tqc.COND_ORDER,
        xlabel='datapoints per FWHM (cycles)', median_fmt='{:.1f}',
        log_x=True, xlim=xlim, legend_loc='upper right')
    axes[1].set_title('Sampling of the LC peak', fontsize=10)
    _plain_log_ticks(axes[1], xlim)

    for ax in axes:
        tqc._unbold_axis_labels(ax)
    fig.tight_layout()

    src = pd.concat([data, runs_df.assign(level='per_run_summary')],
                    ignore_index=True)
    tqc.save(fig, 'lc_peak_width', src)

    with pd.option_context('display.width', 200, 'display.max_columns', 50):
        print(runs_df.to_string(index=False))
    print('  read this as an LC readout, not a condition comparison: the runs '
          'share one LC, so a width difference between conditions is the MS1 '
          'cycle time sampling the same peak differently. A measured FWHM is '
          'biased upwards as the cycle time approaches it, so the '
          'fastest-sampled condition carries the best estimate of the true '
          'width.')
    print(f'  measured over the top {TOP_FRACTION:.0%} of each run by '
          f'abundance. That depth is part of the number: sampling deeper reads '
          f'narrower, because weak precursors are undersampled rather than '
          f'genuinely sharp. Quote it with the width.')
    return runs_df


def main(**configure_kwargs):
    """Configure (this script's keywords and tqc's), load, measure, plot."""
    configure(**configure_kwargs)
    df, sample_info = load()
    lc_peak_width(df, sample_info)
    return df, sample_info


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('data_root', nargs='?', default=None,
                    help='folder with one subfolder per condition')
    ap.add_argument('--out-dir', default=None, help='default <data_root>/analysis')
    ap.add_argument('--conditions', default=None,
                    help='comma-separated subfolders to include, in display order')
    ap.add_argument('--runs', default=None,
                    help="runs per condition to measure (default 1, 'all' for "
                         'every run)')
    ap.add_argument('--top-fraction', type=float, default=None,
                    help='fraction of each run\'s precursors to measure, from '
                         f'the top of the abundance ranking (default '
                         f'{TOP_FRACTION:g}). Deeper is smoother but reads '
                         'narrower — see TOP_FRACTION in the source.')
    ap.add_argument('--rt-window', default=None,
                    help="XIC half-window in minutes, or 'auto' (default) to "
                         "derive it from the peaks - a fixed window barely "
                         "wider than the peak biases every width narrow")
    ap.add_argument('--max-targets', type=int, default=None,
                    help='absolute cap on targets per run (default none)')
    ap.add_argument('--ms1', action='store_true',
                    help='measure on the precursor MS1 XIC instead of on '
                         'summed fragments above the precursor. Reads ~35%% '
                         'narrower on H032_E350 — the MS1 trace is too noisy '
                         'to carry a width here.')
    ap.add_argument('--cache-dir', default=None,
                    help=f'raw-file HDF caches (default: {_ms_cache_dir()})')
    a = ap.parse_args()
    order = a.conditions.split(',') if a.conditions else None
    main(rt_window=(a.rt_window if a.rt_window in (None, 'auto')
                     else float(a.rt_window)),
         runs_per_condition=a.runs, top_fraction=a.top_fraction,
         max_targets=a.max_targets, cache_dir=a.cache_dir,
         use_fragments=False if a.ms1 else None,
         data_root=a.data_root, out_dir=a.out_dir,
         include_only=order, cond_order=order)
    print(f'\ndone -> {tqc.OUT_DIR}')
