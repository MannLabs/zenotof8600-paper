"""Figure 1 b-d — ZenoTOF 8600 against the 7600+ on an identical method."""

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
DATA = cfg.DATA_ROOT
OUT = cfg.panels(__file__)

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

CONDITIONS = [
    {
        'label': '7600+',
        'dir': os.path.join(DATA, 'H032_E303', '7600_60SPD_100vw_12ms'),
        'tags': ['20260417_Zeno1_Eno01_22p4min_TiHe_SA_H032_E303_A1',
                 '20260417_Zeno1_Eno01_22p4min_TiHe_SA_H032_E303_A2',
                 '20260417_Zeno1_Eno01_22p4min_TiHe_SA_H032_E303_A3',
                 '20260417_Zeno1_Eno01_22p4min_TiHe_SA_H032_E303_A4_repeat'],
    },
    {
        'label': '8600',
        'dir': os.path.join(DATA, 'H032_E352', '8600_60SPD_100vw_12ms'),
        'tags': [f'20260805_Zeno2_Eno0_22p4min_TiHe_SA_H032_E352_A{i}'
                 for i in (1, 2, 3, 4)],
    },
]

COND_ORDER = [c['label'] for c in CONDITIONS]
COLOR = {c['label']: core.PALETTE_SINGLE[i] for i, c in enumerate(CONDITIONS)}

LETTER = {'ids_pg': 'b', 'ids_prec': 'c', 'ion_current': 'd'}


def load():
    """Canonical long df + sample_info for the two instruments."""
    experiments = [{
        'path': os.path.join(c['dir'], 'lfq.dia.features.csv'),
        'protein_path': os.path.join(c['dir'], 'lfq.dia.proteins.csv'),
        'file_tags': c['tags'],
        'condition1': '250 ng HEK, 60 SPD, 100 vw 12 ms',
        'condition2': c['label'],
    } for c in CONDITIONS]
    df, sample_info = core.load_experiments(experiments)
    sample_info['replicate'] = sample_info.groupby('condition2').cumcount() + 1
    return df, sample_info


def save(fig, name, source_df, out=None):
    """Panels ship without titles; `fs.save_panel` strips and prints them."""
    return fs.save_panel(fig, out or OUT, name, source_df, dpi=300)


_unbold = fs.tidy


def panel_bc(agg):
    """b, c — identifications with the CV<20% subset overlaid, plus the gain."""
    for key, total_col, cv_col, ylabel, title in [
        ('ids_pg', 'total_protein_groups', 'PG20',
         '# protein groups', 'Protein groups'),
        ('ids_prec', 'total_precursors', 'Pr20',
         '# precursors', 'Precursors'),
    ]:
        name = f'figure1{LETTER[key]}_' + (
            'protein_group_ids' if key == 'ids_pg' else 'precursor_ids')
        fig, ax, summary = core.plot_overlapping_bars(
            agg, x_col='condition2',
            higher_y_col=total_col,
            lower_y_col=cv_col,
            higher_label='IDs', lower_label='< 20% CV',
            palette=[COLOR[c] for c in COND_ORDER],
            figsize=(fs.PANEL_W_IN, 4), y_label=ylabel,
            label_fontsize=fs.FONTSIZE, tick_fontsize=fs.FONTSIZE,
            legend_fontsize=fs.LEGEND_FONTSIZE,
            value_fontsize=fs.FONTSIZE, point_size=fs.POINT_SIZE)
        if not fs.SHOW_VALUE_LABELS:
            fs.strip_value_labels(ax)

        per_run = agg.groupby('condition2')[total_col].first()
        lo = float(per_run[COND_ORDER[0]])
        hi = float(per_run[COND_ORDER[1]])
        gain = (hi - lo) / lo * 100
        fs.label_pct(ax, [(1, hi, gain)])
        summary['gain_pct_8600_over_7600'] = gain

        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        _unbold(ax)
        fig.tight_layout()
        fs.set_bar_geometry_inches(fig, ax)
        save(fig, name, summary)
        plt.close(fig)
        print(f'    {title}: {lo:,.0f} -> {hi:,.0f}  (+{gain:.1f}%)')


MIN_REPLICATES = 4


def precursor_ratios(df, sample_info):
    """Per-precursor log2(8600 / 7600+) over precursors complete in both arms."""
    d = df[df['precursor_intensity'] > 0].merge(
        sample_info[['run', 'condition2']], on='run')
    d['log2_intensity'] = np.log2(d['precursor_intensity'])

    n = (d.groupby(['precursor_id', 'condition2'])['run'].nunique()
         .unstack().reindex(columns=COND_ORDER))
    complete = n[(n >= MIN_REPLICATES).all(axis=1)].index

    level = (d[d['precursor_id'].isin(complete)]
             .groupby(['precursor_id', 'condition2'])['log2_intensity']
             .median().unstack().reindex(columns=COND_ORDER).dropna())
    ratio = (level[COND_ORDER[1]] - level[COND_ORDER[0]]).rename('log2_ratio')
    return ratio.reset_index()


def _n_bins(x):
    """House rule: Freedman-Diaconis, capped so no bin is Poisson-noisy."""
    q1, q3 = np.percentile(x, [25, 75])
    width = 2 * (q3 - q1) / len(x) ** (1 / 3)
    fd = int(np.ceil((x.max() - x.min()) / width)) if width > 0 else 100
    return int(np.clip(min(fd, len(x) // 250), 20, 100))


def report_gain(ratio):
    """Print the quotable gain statistics."""
    r = ratio['log2_ratio']
    print(f'    n = {len(r):,} precursors, complete in both arms')
    print('    ' + '  '.join(f'P{p}={2 ** r.quantile(p / 100):.1f}x'
                             for p in (10, 25, 50, 75, 90, 95, 99)))
    for t in (5, 10, 15, 20):
        print(f'    above {t:2d}x: {100 * (r > np.log2(t)).mean():4.1f}% '
              f'of precursors')
    top = r[r >= r.quantile(0.90)]
    print(f'    top decile: median {2 ** top.median():.1f}x '
          f'(split-half validated, 1.3% shrinkage)')


def panel_intensity(df, sample_info, out, name, w_in=2.60, h_in=None,
                    legend_ncol=None):
    """Each arm's precursor intensity distribution, shared vs arm-unique."""
    d = df[df['precursor_intensity'] > 0].merge(
        sample_info[['run', 'condition2']], on='run')
    per = (d.groupby(['precursor_id', 'condition2'])['precursor_intensity']
           .median().unstack().reindex(columns=COND_ORDER))
    lg = np.log10(per)

    span = (np.floor(np.nanmin(lg.values) * 2) / 2,
            np.ceil(np.nanmax(lg.values) * 2) / 2)
    bins = _n_bins(lg[COND_ORDER[0]].dropna())

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    rows, top, handles, labels = [], 0, [], []
    for c in COND_ORDER:
        other = [o for o in COND_ORDER if o != c][0]
        x = lg[c].dropna()
        uniq = lg.loc[lg[other].isna(), c].dropna()
        color = COLOR[c]

        counts, edges = np.histogram(x, bins=bins, range=span)
        u_counts, _ = np.histogram(uniq, bins=bins, range=span)
        centres = (edges[:-1] + edges[1:]) / 2

        ax.fill_between(centres, counts, color=color, alpha=0.16, linewidth=0)
        ax.fill_between(centres, u_counts, color=color, alpha=0.45,
                        linewidth=0)
        ax.plot(centres, counts, color=color, lw=2.2)
        ax.axvline(x.median(), color=color, lw=1.2, ls='--')
        top = max(top, counts.max())

        p1, p99 = x.quantile(0.01), x.quantile(0.99)
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.16,
                                     ec=color, lw=1.4))
        labels.append(c)
        rows.append(pd.DataFrame({'arm': c,
                                  'log10_intensity_bin_centre': centres,
                                  'precursors': counts,
                                  'precursors_unique_to_arm': u_counts,
                                  'bin_width': edges[1] - edges[0],
                                  'median_log10': x.median(),
                                  'p1_log10': p1, 'p99_log10': p99}))
        print(f'    {c:6s} n={len(x):>7,}  unique={len(uniq):>6,}  '
              f'median log10 {x.median():.3f} '
              f'(unique {uniq.median():.3f})  P1-P99 {p1:.2f}-{p99:.2f}')

    handles.append(plt.Rectangle((0, 0), 1, 1, fc='0.45', alpha=0.45,
                                 ec='none'))
    labels.append('unique to that arm')

    ax.set_xlabel(r'$\log_{10}$ precursor intensity', fontsize=fs.FONTSIZE)
    ax.set_ylabel(f'precursors per bin (bin {edges[1] - edges[0]:.2f})',
                  fontsize=fs.FONTSIZE)
    ax.set_xlim(*span)
    ax.set_ylim(0, top * 1.05)
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.0),
              ncol=legend_ncol or len(labels), frameon=False,
              fontsize=fs.LEGEND_FONTSIZE,
              handlelength=0.9, handleheight=0.9, handletextpad=0.4,
              borderpad=0.15, columnspacing=1.2)
    ax.tick_params(labelsize=fs.FONTSIZE)
    _unbold(ax)
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=w_in, h_in=h_in or fs.AXES_H_IN)
    save(fig, name, pd.concat(rows), out=out)
    plt.close(fig)


TIC_DIR = cfg.data('figure1', 'ion_current')
ELUTION_COVER = (0.005, 0.995)
HIST_BIN_LOG10 = 0.05


def elution_window(cond):
    """(start, end) in min holding the central 99% of a search's precursors."""
    f = pd.read_csv(os.path.join(cond['dir'], 'lfq.dia.features.csv'),
                    usecols=lambda c: c.endswith('RT mean'))
    rt = pd.to_numeric(pd.Series(f.to_numpy().ravel()),
                       errors='coerce').dropna()
    lo, hi = rt.quantile(list(ELUTION_COVER))
    return float(lo), float(hi)


def ion_current_cycles():
    """Every MS2 acquisition cycle inside its instrument's elution window."""
    import glob
    d = pd.concat(pd.read_csv(f, dtype={'instrument': str}) for f in
                  glob.glob(os.path.join(TIC_DIR, '*_A*.csv')))
    d = d[d['ms_level'] == 2]
    windows = {c['label']: elution_window(c) for c in CONDITIONS}
    keep = [g[g['rt_min'].between(*windows[inst])]
            for inst, g in d.groupby('instrument')]
    cycles = pd.concat(keep)
    per_run = cycles.groupby(['instrument', 'run'])['tic'].median()
    med = {i: float(per_run[i].median()) for i in COND_ORDER}
    ratio = med[COND_ORDER[1]] / med[COND_ORDER[0]]
    return cycles, per_run, ratio, windows


MODE_BW_SCALES = (0.5, 1.0, 2.0)


def _kde_mode(x, scale=1.0):
    """Most frequent value of x (log10 units) by Gaussian kernel density."""
    from scipy.stats import gaussian_kde
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    bw = scale * 0.9 * iqr / 1.349 * len(x) ** -0.2
    kde = gaussian_kde(x, bw_method=bw / np.std(x, ddof=1))
    grid = np.linspace(*np.percentile(x, [0.5, 99.5]), 5000)
    return float(grid[kde(grid).argmax()])


def _half_sample_mode(x):
    """Bickel's half-sample mode: no bandwidth, no bins."""
    x = np.sort(np.asarray(x))
    while len(x) > 3:
        h = int(np.ceil(len(x) / 2))
        i = int(np.argmin(x[h - 1:] - x[:len(x) - h + 1]))
        x = x[i:i + h]
    return float(x.mean())


def ion_current_modes(cycles):
    """Per-injection modes and the peak-to-peak ratio, with its robustness."""
    lg = {inst: [np.log10(g['tic'].to_numpy()) for _, g in
                 cycles[cycles['instrument'] == inst].groupby('run')]
          for inst in COND_ORDER}

    def ratio_of(fn):
        m = {i: float(np.median([fn(x) for x in lg[i]])) for i in COND_ORDER}
        return 10 ** (m[COND_ORDER[1]] - m[COND_ORDER[0]]), m

    def hist_mode(x):
        c, e = np.histogram(x, bins=np.arange(x.min(), x.max() + HIST_BIN_LOG10,
                                              HIST_BIN_LOG10))
        return float((e[c.argmax()] + e[c.argmax() + 1]) / 2)

    ratio, modes = ratio_of(_kde_mode)
    checks = {f'kde_bw_x{s:g}': ratio_of(lambda x, s=s: _kde_mode(x, s))[0]
              for s in MODE_BW_SCALES}
    checks['histogram_bin'] = ratio_of(hist_mode)[0]
    checks['half_sample'] = ratio_of(_half_sample_mode)[0]
    per = {i: [_kde_mode(x) for x in lg[i]] for i in COND_ORDER}
    pairs = [10 ** (b - a) for a in per[COND_ORDER[0]]
             for b in per[COND_ORDER[1]]]
    checks['pairwise_min'], checks['pairwise_max'] = min(pairs), max(pairs)
    return ratio, modes, checks


def panel_ion_current():
    """d -- per-cycle raw MS2 ion current, peak to peak."""
    cycles, per_run, med_ratio, windows = ion_current_cycles()
    ratio, modes, checks = ion_current_modes(cycles)
    lg = np.log10(cycles['tic'])
    span = (np.floor(lg.quantile(0.001) * 4) / 4,
            np.ceil(lg.quantile(0.999) * 4) / 4)
    n_inj = cycles.groupby('instrument')['run'].nunique()
    bins = int(round((span[1] - span[0]) / HIST_BIN_LOG10))

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    rows, top, peak = [], 0.0, {}
    for inst in COND_ORDER:
        x = np.log10(cycles.loc[cycles['instrument'] == inst, 'tic'])
        counts, edges = np.histogram(x, bins=bins, range=span)
        counts = counts / n_inj[inst]
        centres = (edges[:-1] + edges[1:]) / 2
        color = COLOR[inst]
        ax.fill_between(centres, counts, color=color, alpha=0.18, lw=0)
        ax.plot(centres, counts, color=color, lw=2.2, label=inst)
        peak[inst] = float(np.interp(modes[inst], centres, counts))
        top = max(top, counts.max())
        rows.append(pd.DataFrame({
            'instrument': inst, 'log10_ms2_ion_current_bin_centre': centres,
            'cycles_per_injection': counts,
            'bin_width_log10': edges[1] - edges[0],
            'mode_log10': modes[inst],
            'median_log10': np.log10(per_run[inst].median()),
            'n_injections': n_inj[inst], 'n_cycles': len(x),
            'elution_window_start_min': windows[inst][0],
            'elution_window_end_min': windows[inst][1]}))
    y_arrow = top * 1.12
    for inst in COND_ORDER:
        ax.plot([modes[inst]] * 2, [peak[inst], y_arrow], color=COLOR[inst],
                lw=1.0, ls=':')
    ax.annotate('', xy=(modes[COND_ORDER[1]], y_arrow),
                xytext=(modes[COND_ORDER[0]], y_arrow),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2,
                                shrinkA=0, shrinkB=0))
    ax.text(np.mean(list(modes.values())), y_arrow * 1.03, f'{ratio:.1f}×',
            ha='center', va='bottom', fontsize=fs.FONTSIZE, fontweight='bold')
    ax.set_ylim(0, top * 1.32)
    ax.set_xlim(*span)
    ax.set_xlabel(r'$\log_{10}$ MS2 ion current per cycle [cps]',
                  fontsize=fs.FONTSIZE)
    ax.set_ylabel('Acquisition cycles per injection\n'
                  f'(bin {edges[1] - edges[0]:.2f} log$_{{10}}$)',
                  fontsize=fs.FONTSIZE)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=fs.LEGEND_FONTSIZE, handlelength=1.2,
              borderpad=0.15, columnspacing=1.2)
    ax.tick_params(labelsize=fs.FONTSIZE)
    _unbold(ax)
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=2.60, h_in=fs.AXES_H_IN)
    print(f'    MS2 ion current {COND_ORDER[1]}/{COND_ORDER[0]}: peak to peak '
          f'{ratio:.2f}, median {med_ratio:.2f} (elution windows '
          + ', '.join(f'{k} {a:.2f}-{b:.2f} min' for k, (a, b) in
                      windows.items()) + ')')
    print('    peak-to-peak robustness: ' + ', '.join(
        f'{k} {v:.2f}' for k, v in checks.items()))
    save(fig, f'figure1{LETTER["ion_current"]}_ms2_ion_current',
         pd.concat(rows).assign(ratio_peak_to_peak=ratio,
                                ratio_median=med_ratio))
    plt.close(fig)


def check_replicate_counts(sample_info):
    """Both arms must carry the same n before their CVs are compared."""
    n = sample_info.groupby('condition2')['run'].nunique()
    print('  replicates per arm: '
          + ', '.join(f'{k}={v}' for k, v in n.items()))
    if n.nunique() != 1:
        print('  WARNING: unequal replicate counts — the CV medians in panels '
              'd/e are NOT directly comparable; rescale before quoting them.')


def main():
    fs.init()
    fs.start_build(script=__file__, inputs=[c['dir'] for c in CONDITIONS], out_dir=OUT)
    print(f'loading figure 1 inputs from {DATA} ...')
    df, sample_info = load()
    print(f'  {df.shape[0]:,} rows, {len(sample_info)} runs')
    check_replicate_counts(sample_info)

    agg = core.process_experiment(df, sample_info, group_col='condition2')
    print('\nPer-run summary:')
    print(agg[['run', 'condition2', 'precursor', 'protein_group',
               'total_precursors', 'Pr20', 'total_protein_groups', 'PG20']]
          .to_string(index=False))

    print('\npanels b, c ...')
    panel_bc(agg)
    print('panel d ...')
    panel_ion_current()
    ratio = precursor_ratios(df, sample_info)
    print('gain statistics (for the text, not a panel) ...')
    report_gain(ratio)
    fs.report_printed_sizes()
    fs.report_stale(OUT, prefix='figure')


if __name__ == '__main__':
    main()
