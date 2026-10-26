"""Supplementary Figure 4f-g: processed blanks against single cells."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = cfg.panels(__file__)
FEATURES_WITHOUT = os.path.join(cfg.DATA_ROOT, 'H032_E388', 'SC+blank',
                                'lfq.dia.features.csv')

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'figure4'))
import figure_style as fs
import box_panel as bp
import single_cell as sc

SEARCHES = ['with reference', 'without reference']
SAMPLE_TYPES = [sc.CONDITIONS[0]['label'], sc.SINGLE_CELL]
COLOR = {t: sc.COLOR[t] for t in SAMPLE_TYPES}
LEVELS = {'f': ('protein_groups', '# protein groups', 'protein_groups'),
          'g': ('precursors', '# precursors', 'precursors')}
UNIT_IN = 1.30


def _condition_of(run):
    for key, label in sc.RELABEL.items():
        if key in run:
            return label
    for c in sc.CONDITIONS:
        if c['match'] in run:
            return c['label']
    return None


def load():
    """Per-run counts for blanks and cells in both searches, same runs."""
    ref, cutoff = sc.apply_exclusions(sc.load())
    ref = ref[ref['condition2'].isin(SAMPLE_TYPES)].copy()
    ref['search'] = SEARCHES[0]
    keep = dict(zip(ref['run'], zip(ref['included'],
                                    ref['exclusion_reason'])))

    runs = bp.run_names(FEATURES_WITHOUT)
    experiments = []
    for t in SAMPLE_TYPES:
        tags = [r for r in runs if _condition_of(r) == t]
        experiments.append({'path': FEATURES_WITHOUT, 'file_tags': tags,
                            'condition1': 'H032_E388', 'condition2': t})
    new = bp.identification_counts(experiments, SAMPLE_TYPES,
                                   strip=r'.*H032_E388_|.*_TiHe_')
    missing = set(new['run']) ^ set(ref['run'])
    if missing:
        raise ValueError(f'the two searches hold different runs: {missing}')
    new['included'] = new['run'].map(lambda r: keep[r][0])
    new['exclusion_reason'] = new['run'].map(lambda r: keep[r][1])
    new['search'] = SEARCHES[1]

    ids = pd.concat([ref, new], ignore_index=True)
    ids['condition2'] = pd.Categorical(ids['condition2'].astype(str),
                                       categories=SAMPLE_TYPES, ordered=True)
    ids['search'] = pd.Categorical(ids['search'], categories=SEARCHES,
                                   ordered=True)
    return ids.sort_values(['search', 'condition2', 'run'])


def draw(ids, letter):
    col, ylabel, stem = LEVELS[letter]
    shown = ids[ids['included']]
    n_per = shown.groupby(['search', 'condition2'], observed=True)['run'].count()
    small = {k for k, n in n_per.items() if n < bp.MIN_BOX_N}
    boxed = shown.copy()
    boxed[col] = [v if (s, t) not in small else float('nan')
                  for s, t, v in zip(boxed['search'], boxed['condition2'],
                                     boxed[col])]
    fig, ax = core.plot_boxplot_with_points(
        boxed, x_col='search', y_col=col, hue_col='condition2',
        palette=COLOR, figsize=(fs.PANEL_W_IN, 4), y_label=ylabel,
        x_label='', show_points=True,
        point_size=bp.POINT_DIAM, point_alpha=0.85,
        label_fontsize=fs.FONTSIZE, tick_fontsize=fs.FONTSIZE,
        legend_fontsize=fs.LEGEND_FONTSIZE, xtick_rotation=0, ymin=0)
    for (s, t) in sorted(small, key=str):
        x = SEARCHES.index(s) + (-0.15 if t == SAMPLE_TYPES[0] else 0.15)
        bp.points_with_median(ax, x, shown.loc[(shown['search'] == s) &
                              (shown['condition2'] == t), col], COLOR[t],
                              half_width=0.12, seed=SEARCHES.index(s))
    fs.tidy(ax)
    ax.tick_params(labelsize=fs.FONTSIZE)
    ax.xaxis.label.set_fontsize(fs.FONTSIZE)
    ax.yaxis.label.set_fontsize(fs.FONTSIZE)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f'{v / 1000:,.0f}k' if v >= 1000 else f'{v:,.0f}'))
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[t],
                             edgecolor='black', linewidth=0.8)
               for t in SAMPLE_TYPES]
    ax.legend(handles, SAMPLE_TYPES, frameon=False,
              fontsize=fs.LEGEND_FONTSIZE, loc='upper right',
              handlelength=0.9, handleheight=0.9, handletextpad=0.4)

    stats = (shown.groupby(['search', 'condition2'], observed=True)[col]
             .agg(['median', 'min', 'max', 'count']))
    print(f'    {ylabel}')
    for (s, t), r in stats.iterrows():
        print(f'      {s:18s} {t:7s} n={int(r["count"]):>2}  median '
              f'{r["median"]:>8,.0f}  range {r["min"]:,.0f}-{r["max"]:,.0f}')

    source = ids[['search', 'run', 'well', 'condition2', col, 'included',
                  'exclusion_reason']].rename(
        columns={'condition2': 'sample_type', col: 'value'})
    source['quantity'] = col
    fig.tight_layout()
    fs.set_categorical_width_inches(fig, ax, unit_in=UNIT_IN)
    name = f'supplementary_figure4{letter}_blank_control_{stem}'
    w, _ = fs.save_panel(fig, OUT, name, source, dpi=300)
    print(f'    {name}: {w:.2f} in wide')
    plt.close(fig)


def main(letters=('f', 'g')):
    ids = load()
    for letter in letters:
        draw(ids, letter)


if __name__ == '__main__':
    fs.init()
    fs.start_build(script=__file__, inputs=[FEATURES_WITHOUT, sc.FEATURES],
                   out_dir=OUT)
    main()
    fs.report_printed_sizes()
