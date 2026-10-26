"""Supplementary figure 3 — ZT Scan against Zeno SWATH on the HYE design."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import logging
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'figure3'))
import figure3 as f3
import make_supplementary_composites as sc

for _noisy in ('directlfq', 'fontTools', 'matplotlib'):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

METHODS = list(f3.METHODS)
SPECIES_ORDER = f3.SPECIES_ORDER
SPECIES_COLORS = f3.SPECIES_COLORS

MARKER = {'ZT Scan': 'o', 'Zeno SWATH': '^'}
FILLED = {'ZT Scan': True, 'Zeno SWATH': True}


def save(fig, name, source_df):
    return fs.save_panel(fig, OUT, name, source_df, dpi=300)


def measure(quality_min=None):
    """Ratio tables (and the per-pair frames) for both methods."""
    out = {}
    for m in METHODS:
        print(f'\n=== {m}'
              + (f', Quality >= {quality_min}' if quality_min else '') + ' ===')
        df, si = f3.load(m)
        pq = f3.pair_quant(df, si, quality_min=quality_min, label=f'{m} ')
        t = f3.ratio_tables(pq, si)
        out[m] = dict(t, pairs=pq, sample_info=si)
    return out


TWO_UP_W_IN = f3.ROW_C_W_IN
SCATTER_H_IN = 2.6
TWO_UP_AXES_H_IN = f3.ROW_C_AXES_H_IN


def panel_accuracy_both(acc, letter, label):
    """Figure 3f's panel with the Zeno SWATH series switched on."""
    zt, dia = acc['ZT Scan'], acc['Zeno SWATH']
    f3.panel_accuracy(zt['accuracy'], zt['fit'],
                      dia=(dia['accuracy'], dia['fit']), out=OUT,
                      name=f'supplementary_figure3{letter}_ratio_accuracy_'
                           f'{label}',
                      total_w_in=TWO_UP_W_IN, h_in=TWO_UP_AXES_H_IN)


def panel_scatter(res, method, letter):
    a, b = f3.SCATTER_PAIR
    tag = method.lower().replace(' ', '_')
    f3.panel_species_ratio(
        res[method]['pairs'][f3.SCATTER_PAIR], res[method]['sample_info'],
        out=OUT, size=(f3.ROW_B_W_IN, SCATTER_H_IN),
        target_w=sc.full_row_width('supplementary_figure3', letter),
        name=f'supplementary_figure3{letter}_species_ratio_{a.lower()}_vs_'
             f'{b.lower()}_{tag}')


def panel_b(res, letter='b'):
    """b — spread between proteins of one species, both methods."""
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    rows = []
    for m in METHODS:
        b = res[m]['between']
        for sp in SPECIES_ORDER:
            s = b[b['species'] == sp]
            ax.scatter(s['expected'], s['error_pct'],
                       s=40, marker=MARKER[m],
                       facecolor=SPECIES_COLORS[sp] if FILLED[m] else 'none',
                       edgecolor=SPECIES_COLORS[sp] if not FILLED[m]
                       else 'black',
                       linewidth=1.2 if not FILLED[m] else 0.4, zorder=3)
            for _, r in s.iterrows():
                rows.append({'method': m, **r.to_dict()})
    ax.axvline(0, color='0.6', lw=0.8)
    ax.set_xlabel('Expected $\\log_2$ ratio', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Precision [%]', fontsize=fs.FONTSIZE)
    ax.set_ylim(0, None)
    f3.legend_two_rows(ax, *f3.species_method_handles(methods=METHODS))
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    fig.tight_layout()
    fs.fit_panel_width(fig, ax, total_w_in=TWO_UP_W_IN,
                       h_in=TWO_UP_AXES_H_IN)
    save(fig, f'supplementary_figure3{letter}_between_protein_error',
         pd.DataFrame(rows))
    plt.close(fig)


def panel_c(res, letter='c'):
    """c — replicate CV against the species' share of the mixture."""
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    rows = []
    for m in METHODS:
        c = res[m]['cv'].sort_values('load_pct')
        for sp in SPECIES_ORDER:
            s = c[c['species'] == sp]
            ax.plot(s['load_pct'], s['protein_cv_pct'],
                    marker=MARKER[m], ms=5.5, lw=1.8,
                    ls=f3.METHOD_LINESTYLE[m],
                    color=SPECIES_COLORS[sp],
                    markerfacecolor=SPECIES_COLORS[sp] if FILLED[m] else 'white')
            for _, r in s.iterrows():
                rows.append({'method': m, **r.to_dict()})
    ax.set_xlabel('Species share of mixture [%]', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Protein quant CV [%]', fontsize=fs.FONTSIZE)
    ax.set_ylim(0, None)
    handles = [plt.Line2D([], [], marker=MARKER[m], ms=6, color='0.35',
                          ls=f3.METHOD_LINESTYLE[m],
                          markerfacecolor='0.35' if FILLED[m] else 'white',
                          label=m) for m in METHODS]
    ax.legend(handles=handles, loc='upper right', frameon=False,
              fontsize=fs.LEGEND_FONTSIZE, handlelength=1.8,
              handletextpad=0.5, borderpad=0.2, labelspacing=0.3)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    fig.tight_layout()
    fs.fit_panel_width(fig, ax, total_w_in=TWO_UP_W_IN,
                       h_in=TWO_UP_AXES_H_IN)
    save(fig, f'supplementary_figure3{letter}_protein_cv', pd.DataFrame(rows))
    plt.close(fig)


LETTERS = {'scatter_dia': 'a', 'accuracy': 'b', 'between_protein': 'c',
           'protein_cv': 'd'}


def main(*letters):
    wanted = set(letters) or set(LETTERS.values())
    unknown = wanted - set(LETTERS.values())
    if unknown:
        raise SystemExit(f'unknown panel(s) {sorted(unknown)}')
    fs.init()
    fs.start_build(script=__file__, inputs=[f3.DATA], out_dir=OUT)
    L = {k: v in wanted for k, v in LETTERS.items()}

    if L['accuracy'] or L['between_protein'] or L['protein_cv'] \
            or L['scatter_dia']:
        res = measure()
        print('slopes, complete cases, no filter: ' + '  '.join(
            f'{m} {res[m]["fit"]["slope"]:.3f}' for m in METHODS))
        if L['accuracy']:
            panel_accuracy_both(res, LETTERS['accuracy'], 'complete')
        if L['between_protein']:
            panel_b(res, LETTERS['between_protein'])
        if L['protein_cv']:
            panel_c(res, LETTERS['protein_cv'])
        if L['scatter_dia']:
            panel_scatter(res, 'Zeno SWATH', LETTERS['scatter_dia'])

    fs.report_printed_sizes()
    fs.report_stale(OUT, prefix='supplementary_figure')


if __name__ == '__main__':
    main(*sys.argv[1:])
