"""Supplementary Figure 5: effect of the cycle time."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(cfg.DATA_ROOT, 'H032_E350')
OUT = cfg.panels(__file__)

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

LETTER = {'peak_width': 'a', 'sampling': 'b', 'undersampled': None,
          'ids_pg': 'c', 'ids_prec': 'd'}

METHODS = [
    {'dir': '500SPD_3p3Th_0p4s', 'cycle_s': 0.402, 'width_th': 3.3},
    {'dir': '500SPD_2p6Th_0p5s', 'cycle_s': 0.502, 'width_th': 2.6},
    {'dir': '500SPD_2p1Th_0p6s', 'cycle_s': 0.602, 'width_th': 2.1},
    {'dir': '500SPD_1p8Th_0p7s', 'cycle_s': 0.702, 'width_th': 1.8},
]
for m in METHODS:
    m['label'] = f'{m["cycle_s"]:.1f} s'

ORDER = [m['label'] for m in METHODS]
BY_DIR = {m['dir']: m for m in METHODS}
COLOR = {m['label']: core.PALETTE_SINGLE[i] for i, m in enumerate(METHODS)}

NYQUIST = 2.0

THOUSANDS = FuncFormatter(lambda v, _: f'{v:,.0f}')


def load_searches():
    experiments = []
    for m in METHODS:
        folder = os.path.join(DATA, m['dir'])
        features = os.path.join(folder, 'lfq.dia.features.csv')
        runs = [c[:-len(' m/z')] for c in pd.read_csv(features, nrows=0).columns
                if c.endswith(' m/z') and c != 'm/z']
        if len(runs) != 4:
            raise ValueError(f'{m["dir"]}: {len(runs)} runs, expected 4')
        experiments.append({
            'path': features,
            'protein_path': os.path.join(folder, 'lfq.dia.proteins.csv'),
            'file_tags': runs,
            'condition1': 'HeLa, 500 SPD, ZT Scan',
            'condition2': m['label']})
    df, sample_info = core.load_experiments(experiments)
    sample_info['replicate'] = sample_info.groupby('condition2').cumcount() + 1
    return df, sample_info


def load_widths():
    """Per-precursor FWHM, relabelled from the search folder to the cycle time."""
    w = pd.read_csv(os.path.join(DATA, 'lc_peak_width_sourcedata.csv'))
    w = w[w['condition'].isin(BY_DIR)].copy()
    w['label'] = w['condition'].map(lambda d: BY_DIR[d]['label'])
    w['cycle_s'] = w['condition'].map(lambda d: BY_DIR[d]['cycle_s'])
    w = w[np.isfinite(w['fwhm_s'])]
    return w


def panel_peak_width(widths):
    """a -- the FWHM distribution of each method, as a cumulative distribution."""
    groups = [widths.loc[widths['label'] == lab, 'fwhm_s'].to_numpy()
              for lab in ORDER]
    n_runs = {lab: widths.loc[widths['label'] == lab, 'run'].nunique()
              for lab in ORDER}
    lo, hi = np.percentile(np.concatenate(groups), [0.5, 99.0])

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 3.4))
    ax.axhline(0.5, color='0.55', linewidth=0.8, linestyle=':', zorder=1)
    rows = []
    for lab, vals in zip(ORDER, groups):
        x = np.sort(vals)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=COLOR[lab], linewidth=2.0, zorder=3)
        q = np.linspace(0, 1, 401)
        rows.append(pd.DataFrame({'cycle': lab, 'cumulative_fraction': q,
                                  'fwhm_s': np.quantile(vals, q),
                                  'median_fwhm_s': float(np.median(vals)),
                                  'n_precursors': len(vals),
                                  'n_runs': n_runs[lab]}))

    ax.set_xlabel('Peak FWHM [s]', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Cumulative fraction of precursors', fontsize=fs.FONTSIZE)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)

    handles = [plt.Line2D([], [], color=COLOR[lab], linewidth=2.0, label=lab)
               for lab in ORDER]
    ax.legend(handles=handles, loc='lower right', frameon=False,
              fontsize=fs.LEGEND_FONTSIZE, handlelength=1.4,
              labelspacing=0.3, borderpad=0.2,
              title='cycle time',
              title_fontproperties={'size': fs.LEGEND_FONTSIZE})

    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=fs.PANEL_W_IN * 0.78,
                            h_in=fs.AXES_H_IN)
    fs.save_panel(fig, OUT,
                  f'supplementary_figure5{LETTER["peak_width"]}_peak_width',
                  pd.concat(rows, ignore_index=True), dpi=300)
    plt.close(fig)

    print(f'    ECDF over {lo:.2f}-{hi:.2f} s')
    for lab, v in zip(ORDER, groups):
        print(f'      {lab}  median FWHM {np.median(v):.3f} s  '
              f'n = {len(v):,} over {n_runs[lab]} runs')


def sampling_table(widths):
    """Points across the FWHM per method, from each method's measured width."""
    med = widths.groupby('label')['fwhm_s'].median()
    rows = []
    for m in METHODS:
        lab = m['label']
        rows.append({
            'cycle': lab, 'cycle_s': m['cycle_s'], 'width_th': m['width_th'],
            'median_fwhm_s': float(med[lab]),
            'points_per_fwhm_measured': float(med[lab]) / m['cycle_s']})
    return pd.DataFrame(rows)


def panel_sampling(sampling):
    """b -- points across the peak against cycle time (measured widths)."""
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 3.4))

    ax.axhline(NYQUIST, color=fs.LEGEND_DARK, linewidth=1.0, linestyle=':',
               zorder=1)
    ax.annotate('Nyquist limit', xy=(sampling['cycle_s'].max(), NYQUIST),
                xytext=(0, 4), textcoords='offset points', ha='right',
                va='bottom', fontsize=fs.LEGEND_FONTSIZE,
                color=fs.LEGEND_DARK)

    col = 'points_per_fwhm_measured'
    ax.plot(sampling['cycle_s'], sampling[col], color=fs.LEGEND_DARK,
            linewidth=1.4, linestyle='-', zorder=2)
    for _, r in sampling.iterrows():
        ax.plot(r['cycle_s'], r[col], marker='o', markersize=7,
                markerfacecolor=COLOR[r['cycle']],
                markeredgecolor=COLOR[r['cycle']], markeredgewidth=1.6,
                linestyle='none', zorder=3)

    ax.set_xlabel('Cycle time [s]', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Points across the FWHM', fontsize=fs.FONTSIZE)
    ax.set_xticks(sampling['cycle_s'])
    ax.xaxis.set_major_locator(FixedLocator(sampling['cycle_s'].tolist()))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.1f}'))
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)

    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=fs.PANEL_W_IN * 0.78,
                            h_in=fs.AXES_H_IN)
    fs.save_panel(fig, OUT,
                  f'supplementary_figure5{LETTER["sampling"]}_points_per_fwhm',
                  sampling, dpi=300)
    plt.close(fig)

    print(sampling.round(3).to_string(index=False))
    for col, name in [('points_per_fwhm_measured', 'measured')]:
        ok = sampling[sampling[col] >= NYQUIST]
        last = ok['cycle'].iloc[-1] if not ok.empty else 'none'
        print(f'    slowest cycle still at or above {NYQUIST:g} points '
              f'({name}): {last}')


def undersampled_table(widths):
    """Share of precursors measured at fewer than `NYQUIST` points per FWHM."""
    rows = []
    for m in METHODS:
        v = widths.loc[widths['label'] == m['label'], 'fwhm_s'].to_numpy()
        ratio = v / m['cycle_s']
        rows.append({'cycle': m['label'], 'cycle_s': m['cycle_s'],
                     'width_th': m['width_th'], 'n_precursors': len(v),
                     'median_points_per_fwhm': float(np.median(ratio)),
                     'below_nyquist_pct': float(np.mean(ratio < NYQUIST) * 100)})
    return pd.DataFrame(rows)


def panel_ids(agg):
    """c, d -- identifications against cycle time, with the CV < 20% subset."""
    for key, total_col, cv_col, ylabel, stem in [
        ('ids_pg', 'total_protein_groups', 'PG20', '# protein groups',
         'protein_group_ids'),
        ('ids_prec', 'total_precursors', 'Pr20', '# precursors',
         'precursor_ids'),
    ]:
        fig, ax, summary = core.plot_overlapping_bars(
            agg, x_col='condition2',
            higher_y_col=total_col,
            lower_y_col=cv_col,
            higher_label='IDs', lower_label='< 20% CV',
            palette=[COLOR[c] for c in ORDER],
            figsize=(fs.PANEL_W_IN, 4), y_label=ylabel,
            label_fontsize=fs.FONTSIZE, tick_fontsize=fs.FONTSIZE,
            legend_fontsize=fs.LEGEND_FONTSIZE,
            value_fontsize=fs.FONTSIZE, point_size=fs.POINT_SIZE)
        if not fs.SHOW_VALUE_LABELS:
            fs.strip_value_labels(ax)
        ax.set_xlabel('Cycle time', fontsize=fs.FONTSIZE)
        ax.yaxis.set_major_formatter(THOUSANDS)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        fs.tidy(ax)
        ax.tick_params(labelsize=fs.FONTSIZE)
        ax.xaxis.label.set_fontsize(fs.FONTSIZE)
        ax.yaxis.label.set_fontsize(fs.FONTSIZE)

        tot = (agg.groupby('condition2')[total_col].first().reindex(ORDER))
        q20 = agg.groupby('condition2')[cv_col].first().reindex(ORDER)
        print(f'    {ylabel}')
        for c in ORDER:
            print(f'      {c:6s} {tot[c]:>9,.0f}   <20% CV {q20[c]:>9,.0f}')
        print(f'      totals peak at {tot.idxmax()!r}, quantifiable peak at '
              f'{q20.idxmax()!r}')

        fig.tight_layout()
        fs.set_bar_geometry_inches(fig, ax)
        fs.save_panel(fig, OUT,
                      f'supplementary_figure5{LETTER[key]}_{stem}',
                      summary, dpi=300)
        plt.close(fig)


def main():
    fs.init()
    fs.start_build(script=__file__, inputs=[DATA], out_dir=OUT)
    print(f'loading widths from {DATA} ...')
    widths = load_widths()
    print(f'  {len(widths):,} measured peaks over '
          f'{widths["label"].nunique()} methods')

    print(f'\npanel {LETTER["peak_width"]} ...')
    panel_peak_width(widths)

    sampling = sampling_table(widths)
    print(f'\npanel {LETTER["sampling"]} ...')
    panel_sampling(sampling)

    under = undersampled_table(widths)
    print(under.to_string(index=False))

    print(f'\nloading searches from {DATA} ...')
    df, sample_info = load_searches()
    print(f'  {df.shape[0]:,} rows, {len(sample_info)} runs')
    agg = core.process_experiment(df, sample_info, group_col='condition2')

    print(f'\npanels {LETTER["ids_pg"]}, {LETTER["ids_prec"]} ...')
    panel_ids(agg)
    fs.report_printed_sizes()
    fs.report_stale(OUT, prefix='supplementary_figure')


if __name__ == '__main__':
    main()
