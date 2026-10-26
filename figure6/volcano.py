"""Figure 6, panel e -- PFF+ against PFF-, ZT Scan, with alpha-synuclein named."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(cfg.DATA_ROOT, 'H032_E397', 'PFF_repeat')
OUT = cfg.panels(__file__)

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

FAILED_RUNS = {}

CONDITIONS = [('PFF-', 'A'), ('PFF+', 'B')]
COND_ORDER = [c for c, _ in CONDITIONS]

UP_COLOR = core.PALETTE_SINGLE[0]
DOWN_COLOR = core.PALETTE_SINGLE[2]
HIGHLIGHT_COLOR = '#D62728'

HIGHLIGHT_GENES = ['SNCA']
FC_THRESHOLD = 0.585
PADJ_THRESHOLD = 0.05
MIN_VALID = 3
METHOD = 'moderated'


def _well(run):
    """The plate well of a run name, ignoring any suffix after it."""
    for token in reversed(run.split('_')):
        if re.fullmatch(r'[A-Za-z]\d+', token):
            return token.upper()
    raise ValueError(f'no plate well in run name {run!r}')


def load():
    """The eight-run search; `FAILED_RUNS` is empty for the repeat."""
    features = os.path.join(DATA, 'lfq.dia.features.csv')
    runs = [c[:-len(' m/z')] for c in pd.read_csv(features, nrows=0).columns
            if c.endswith(' m/z') and c != 'm/z']
    keep = [r for r in runs if _well(r) not in FAILED_RUNS]
    dropped = sorted(set(runs) - set(keep))
    for r in dropped:
        print(f'  excluding {_well(r)}: {FAILED_RUNS[_well(r)]}')
    if len(keep) != len(runs) - len(FAILED_RUNS):
        raise ValueError(f'expected to drop {len(FAILED_RUNS)} runs, '
                         f'dropped {len(runs) - len(keep)}')

    df, sample_info = core.load_experiments([{
        'path': features,
        'protein_path': os.path.join(DATA, 'lfq.dia.proteins.csv'),
        'file_tags': keep,
        'condition1': 'HEK + aSyn PFF, 200 SPD ZT Scan',
        'condition2': 'all'}])

    def condition_of(run):
        well = _well(run)
        for label, prefix in CONDITIONS:
            if well.startswith(prefix):
                return label
        raise ValueError(f'cannot assign a condition to {run!r}')

    sample_info['condition2'] = sample_info['run'].map(condition_of)
    sample_info['replicate'] = (
        sample_info.groupby('condition2').cumcount() + 1)
    counts = sample_info['condition2'].value_counts()
    print('  ' + ', '.join(f'{c} n={counts[c]}' for c in COND_ORDER))
    return df, sample_info


def panel(df, sample_info, letter):
    """e -- PFF+ over PFF-, protein level, SNCA named."""
    res = core.plot_volcano(
        df, sample_info, level='protein', group_col='condition2',
        condition_a='PFF+', condition_b='PFF-',
        fc_threshold=FC_THRESHOLD, padj_threshold=PADJ_THRESHOLD,
        min_valid_per_condition=MIN_VALID, method=METHOD,
        palette=[UP_COLOR, DOWN_COLOR],
        highlight_genes=HIGHLIGHT_GENES, highlight_color=HIGHLIGHT_COLOR,
        highlight_size=90, label_highlighted=True, label_top_n=0,
        figsize=(fs.PANEL_W_IN, 3.6),
        label_fontsize=fs.FONTSIZE, tick_fontsize=fs.FONTSIZE,
        legend_fontsize=fs.LEGEND_FONTSIZE, legend_loc='upper left')
    fig, ax, table = _unpack(res)

    ax.set_xlabel('log$_2$ fold change, PFF+ / PFF$-$', fontsize=fs.FONTSIZE)
    ax.set_ylabel('$-$log$_{10}$ adjusted p', fontsize=fs.FONTSIZE)
    leg = ax.get_legend()
    if leg is not None:
        handles = list(leg.legend_handles)
        labels = [t.get_text() for t in leg.get_texts()]
        leg.remove()
        ax.legend(handles, labels, loc='lower center',
                  bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
                  fontsize=fs.LEGEND_FONTSIZE, handlelength=1.0,
                  handletextpad=0.4, borderpad=0.0, columnspacing=0.8,
                  labelspacing=0.25)

    fs.tidy(ax)
    ax.tick_params(labelsize=fs.FONTSIZE)
    ax.margins(x=0.08, y=0.10)

    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=2.35, h_in=2.05)
    fs.save_panel(fig, OUT, f'figure6{letter}_volcano_pff', table, dpi=300)
    plt.close(fig)
    _report(table)
    return table


def _unpack(res):
    """`plot_volcano` returns (fig, ax, table) or a dict, depending on version."""
    if isinstance(res, dict):
        return res['fig'], res['ax'], res.get('results', res.get('table'))
    return res


def _report(table):
    cols = {c.lower(): c for c in table.columns}
    gene = cols.get('genes', cols.get('gene'))
    fc = cols.get('log2_fc', cols.get('log2fc'))
    padj = cols.get('padj', cols.get('p_adj', cols.get('adj_p')))
    sig = table[(table[padj] < PADJ_THRESHOLD) &
                (table[fc].abs() > FC_THRESHOLD)]
    print(f'    {len(table):,} protein groups tested, {len(sig):,} significant '
          f'(|log2 FC| > {FC_THRESHOLD}, adj p < {PADJ_THRESHOLD})')
    print(f'      up in PFF+ {int((sig[fc] > 0).sum()):,}   '
          f'down {int((sig[fc] < 0).sum()):,}')
    for g in HIGHLIGHT_GENES + ['SNCB']:
        hit = table[table[gene].astype(str).str.split(';').apply(
            lambda xs: g in [x.strip() for x in xs])]
        for _, r in hit.iterrows():
            mark = '<- highlighted' if g in HIGHLIGHT_GENES else ''
            print(f'      {g:6s} log2 FC {r[fc]:+.2f}  adj p {r[padj]:.2e} '
                  f'{mark}')


def main(letter='e'):
    fs.init()
    fs.start_build(script=__file__, inputs=[DATA], out_dir=OUT)
    print(f'loading {DATA} ...')
    df, sample_info = load()
    print(f'  {df.shape[0]:,} rows, {len(sample_info)} runs')
    print(f'\npanel {letter} ...')
    panel(df, sample_info, letter)
    fs.report_stale(OUT, prefix='figure')


if __name__ == '__main__':
    main()
