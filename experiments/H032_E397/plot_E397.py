"""PFF+ vs PFF- bar panels for the aSyn pS129 assay (H032_E397)."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd

import zenotof8600_analytics as core

OUT_DIR = pathlib.Path(cfg.cache('H032_E397', 'analysis'))
SOURCE = OUT_DIR / 'e397_lib_measurements.csv'

EXCLUDE = ()

ORDER = ('PFF-', 'PFF+')
COLOR = [core.PALETTE_SINGLE[2], core.PALETTE_SINGLE[0]]

TEST = 'welch'

PANELS = (
    ('lh_ps129', 'light / heavy pS129', 'pS129, heavy-normalised', 'float'),
    ('occupancy', 'pS129 occupancy [%]', 'pS129 occupancy', 'float'),
    ('light_ps129', 'light pS129 [peak area]', 'pS129, raw signal',
     'float'),
    ('lh_unmod', 'light / heavy unmodified', 'total aSyn peptide', 'float'),
)


def table():
    """One row per run, summed over the three Met-oxidation states."""
    d = pd.read_csv(SOURCE)
    q = d[~d['run'].isin(EXCLUDE)]
    s = (q.groupby(['run', 'panel', 'form', 'base'])['area'].sum()
          .unstack(['form', 'base']).reset_index())
    t = pd.DataFrame({'run': s['run'], 'condition': s['panel']})
    t['light_ps129'] = s[('l', 'pS129')].to_numpy()
    t['lh_ps129'] = (s[('l', 'pS129')] / s[('h', 'pS129')]).to_numpy()
    t['lh_unmod'] = (s[('l', 'unmod')] / s[('h', 'unmod')]).to_numpy()
    t['occupancy'] = 100 * t['lh_ps129'] / (t['lh_ps129'] + t['lh_unmod'])
    t['condition'] = pd.Categorical(t['condition'], ORDER, ordered=True)
    return t.sort_values(['condition', 'run'])


def main():
    core.init_plotting()
    OUT_DIR.mkdir(exist_ok=True)
    t = table()
    def draw(ax, col, ylabel, title, fmt):
        core.plot_bar(
            t, x_col='condition', y_col=col, ax=ax,
            palette=COLOR, y_label=ylabel, x_label='', title=title,
            show_points=True, value_format=fmt, value_fontsize=8,
            label_fontsize=10, tick_fontsize=10,
            significance_test=TEST, significance_show_ns=True,
            significance_fontsize=11)
        ax.xaxis.label.set_fontweight('normal')
        ax.yaxis.label.set_fontweight('normal')
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    for col, ylabel, title, fmt in PANELS:
        fig, ax = plt.subplots(figsize=(4, 4))
        draw(ax, col, ylabel, title, fmt)
        fig.tight_layout()
        core.fix_bar_geometry(fig, ax, h_in=2.50)
        core.scale_replicate_points(ax, ax.collections)
        name = f'E397_{col}'
        for ext in ('pdf', 'png'):
            fig.savefig(OUT_DIR / f'{name}.{ext}', dpi=300,
                        bbox_inches='tight')
        src = t[['run', 'condition', col]].rename(columns={col: 'value'})
        src.insert(0, 'readout', col)
        src['bar_height_mean'] = src['condition'].map(
            t.groupby('condition', observed=True)[col].mean())
        src.to_csv(OUT_DIR / f'{name}_sourcedata.csv', index=False)
        plt.close(fig)
        print(f'  wrote {name}.pdf|.png|_sourcedata.csv')

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for a, (col, ylabel, title, fmt) in zip(axes.ravel(), PANELS):
        draw(a, col, ylabel, title, fmt)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(OUT_DIR / f'E397_panels.{ext}', dpi=300,
                    bbox_inches='tight')
    t.melt(id_vars=['run', 'condition'],
           value_vars=[c for c, *_ in PANELS],
           var_name='readout', value_name='value').to_csv(
        OUT_DIR / 'E397_panels_sourcedata.csv', index=False)
    plt.close(fig)
    print('  wrote E397_panels.pdf|.png|_sourcedata.csv')

    from scipy import stats as st
    n_per_arm = int(t.groupby('condition', observed=True).size().min())
    print(f'\n=== means and {TEST} p-values (n={n_per_arm} per arm) ===')
    for col, _, title, _f in PANELS:
        a = t.loc[t['condition'] == 'PFF+', col].to_numpy()
        b = t.loc[t['condition'] == 'PFF-', col].to_numpy()
        p = st.ttest_ind(a, b, equal_var=False).pvalue
        print(f'  {title:26s} PFF- {b.mean():9.4f}   PFF+ {a.mean():9.4f}   '
              f'{a.mean() / b.mean():6.2f}x   p = {p:.2g}')


if __name__ == '__main__':
    main()
