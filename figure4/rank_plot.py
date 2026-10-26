"""Figure 4d/4g -- rank-abundance curves with named proteins."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import zenotof8600_analytics as core

import box_panel as bp
import dvp_tissue as dvp
import single_cell as sc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = cfg.panels(__file__)
fs = bp.fs

MIN_RUNS_FRAC = 0.5

MAX_LABELS = 10

CHAR_W_EM = 0.58
LINE_H_EM = 1.05
LABEL_PAD_PT = 9

AXES_W_IN = 2.24

TISSUE_AREA = '200k_um2'


def _rank(features, runs):
    df, _ = core.load_experiments([{'path': features, 'file_tags': runs,
                                    'condition1': 'x', 'condition2': 'x'}])
    d = df[df['precursor_intensity'] > 0].dropna(subset=['genes'])
    per_run = (d.groupby(['run', 'protein_group'])
               .agg(intensity=('precursor_intensity', 'sum'),
                    genes=('genes', 'first')).reset_index())
    per_run['log10'] = np.log10(per_run['intensity'])
    prot = (per_run.groupby('protein_group')
            .agg(log10=('log10', 'median'), genes=('genes', 'first'),
                 n_runs=('run', 'nunique')).reset_index()
            .sort_values('log10', ascending=False).reset_index(drop=True))
    prot['rank'] = prot.index + 1
    upper = prot['genes'].astype(str).str.upper().str.replace(',', ';')
    prot['syms'] = [frozenset(x.strip() for x in v if x.strip())
                    for v in upper.str.split(';')]
    prot['gene'] = upper.str.split(';').str[0].str.strip()
    return prot


def single_cell_table():
    """Protein groups of the 19 kept single cells, ranked."""
    ids, _ = sc.apply_exclusions(sc.load())
    runs = list(ids.loc[(ids['condition2'] == sc.SINGLE_CELL)
                        & ids['included'], 'run'])
    return _rank(sc.FEATURES, runs), len(runs)


def tissue_table():
    """Protein groups of the smallest DVP dissection, ranked."""
    path = os.path.join(dvp.DATA, TISSUE_AREA, 'lfq.dia.features.csv')
    runs = bp.run_names(path)
    return _rank(path, runs), len(runs)


def select_labels(prot, candidates, n_runs, xlim, ylim):
    """Greedy two-sided placement: preference order in, placed labels out."""
    x_per_in = (xlim[1] - xlim[0]) / AXES_W_IN
    y_per_in = (ylim[1] - ylim[0]) / fs.AXES_H_IN
    em = fs.LEGEND_FONTSIZE / 72.0
    pad_x = LABEL_PAD_PT / 72.0 * x_per_in
    h = LINE_H_EM * em * y_per_in
    placed, boxes = [], []

    for gene, klass in candidates:
        hit = prot[prot['syms'].map(lambda v: gene in v)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        if r['n_runs'] < MIN_RUNS_FRAC * n_runs:
            continue
        w = CHAR_W_EM * em * len(gene) * x_per_in
        for side, dy in (('right', 0.0), ('left', 0.0),
                         ('right', 1.15 * h), ('right', -1.15 * h),
                         ('left', 1.15 * h), ('left', -1.15 * h)):
            cy = r['log10'] + dy
            y0, y1 = cy - h / 2, cy + h / 2
            if y0 < ylim[0] or y1 > ylim[1]:
                continue
            if side == 'right':
                x0, x1 = r['rank'] + pad_x, r['rank'] + pad_x + w
            else:
                x0, x1 = r['rank'] - pad_x - w, r['rank'] - pad_x
            if x0 < xlim[0] or x1 > xlim[1]:
                continue
            if any(not (x1 < b[0] or x0 > b[1] or y1 < b[2] or y0 > b[3])
                   for b in boxes):
                continue
            boxes.append((x0, x1, y0, y1))
            placed.append((gene, klass, side, dy, r))
            break
        if len(placed) >= MAX_LABELS:
            break
    return placed


def draw(panel):
    prot, n_runs = panel['loader']()
    span = prot['log10'].max() - prot['log10'].min()
    print(f'\n{panel["name"]}: {len(prot):,} protein groups over {n_runs} '
          f'runs, {span:.1f} orders '
          f'(log10 {prot["log10"].min():.2f}-{prot["log10"].max():.2f})')

    xlim = (0, len(prot) * 1.02)
    pad = span * 0.04
    ylim = (prot['log10'].min() - pad, prot['log10'].max() + pad)
    placed = select_labels(prot, panel['candidates'], n_runs, xlim, ylim)

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    ax.plot(prot['rank'], prot['log10'], color='#B8B8B8', lw=1.6, zorder=1)
    shown = []
    for gene, klass, side, dy, r in placed:
        ax.scatter([r['rank']], [r['log10']], s=26, zorder=4,
                   color=panel['classes'][klass], edgecolor='black',
                   linewidth=0.5)
        dy_pt = dy / ((ylim[1] - ylim[0]) / fs.AXES_H_IN) * 72.0
        ax.annotate(panel.get('display', str)(gene),
                    xy=(r['rank'], r['log10']),
                    xytext=(LABEL_PAD_PT if side == 'right' else -LABEL_PAD_PT,
                            dy_pt), textcoords='offset points',
                    fontsize=fs.LEGEND_FONTSIZE, va='center', zorder=5,
                    ha='left' if side == 'right' else 'right',
                    arrowprops=dict(arrowstyle='-', lw=0.4, color='0.55',
                                    shrinkA=0, shrinkB=2))
        shown.append(gene)
        print(f'    {klass[:16]:17s} rank {int(r["rank"]):5,}  '
              f'log10 {r["log10"]:5.2f}  {int(r["n_runs"])}/{n_runs}  {gene}')
    print(f'    placed {len(shown)} of {len(panel["candidates"])} candidates')

    ax.set_xlabel(panel['xlabel'], fontsize=fs.FONTSIZE)
    ax.set_ylabel('log$_{10}$ intensity', fontsize=fs.FONTSIZE)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=fs.FONTSIZE)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f'{v / 1000:,.0f}k' if v >= 1000 else f'{v:,.0f}'))
    fs.tidy(ax)

    handles = [plt.Line2D([], [], marker='o', ms=5, ls='none', color=c,
                          mec='black', mew=0.5, label=k)
               for k, c in panel['classes'].items()]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=2 if len(handles) <= 2 else 3, frameon=False,
              fontsize=fs.LEGEND_FONTSIZE, handlelength=1.0,
              handletextpad=0.4, borderpad=0.0, columnspacing=0.8)

    src = prot[['rank', 'protein_group', 'gene', 'log10', 'n_runs']].rename(
        columns={'log10': 'log10_intensity'})
    src['labelled'] = prot['syms'].map(lambda v: bool(v & set(shown)))
    src['label_drawn'] = prot['syms'].map(
        lambda v: next((g for g in shown if g in v), ''))
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=AXES_W_IN, h_in=fs.AXES_H_IN)
    fs.save_panel(fig, OUT, panel['name'], src, dpi=300)
    plt.close(fig)


SC_CLASSES = {'Housekeeping': core.PALETTE_SINGLE[2],
              'Transcription factor': core.PALETTE_SINGLE[0]}
SC_CANDIDATES = [
    ('ACTB', 'Housekeeping'), ('STAT3', 'Transcription factor'),
    ('LMNB1', 'Housekeeping'), ('SP1', 'Transcription factor'),
    ('PCNA', 'Housekeeping'), ('RELA', 'Transcription factor'),
    ('EEF1A1', 'Housekeeping'), ('CREB1', 'Transcription factor'),
    ('NFKB1', 'Transcription factor'), ('TFAM', 'Transcription factor'),
    ('SMARCA4', 'Transcription factor'), ('GTF2I', 'Transcription factor'),
    ('HSP90AB1', 'Housekeeping'), ('STAT1', 'Transcription factor'),
    ('VIM', 'Housekeeping'), ('ILF3', 'Transcription factor'),
]

BRAIN_CLASSES = {'Neuron / synapse': core.PALETTE_SINGLE[0],
                 'Glia / myelin': core.PALETTE_SINGLE[3]}
BRAIN_CANDIDATES = [
    ('MBP', 'Glia / myelin'), ('SNAP25', 'Neuron / synapse'),
    ('GRIN1', 'Neuron / synapse'), ('TMEM119', 'Glia / myelin'),
    ('GFAP', 'Glia / myelin'), ('GAD1', 'Neuron / synapse'),
    ('P2RY12', 'Glia / myelin'), ('PVALB', 'Neuron / synapse'),
    ('DLG4', 'Neuron / synapse'), ('AQP4', 'Glia / myelin'),
    ('SLC32A1', 'Neuron / synapse'), ('S100B', 'Glia / myelin'),
    ('AIF1', 'Glia / myelin'), ('SLC1A2', 'Glia / myelin'),
    ('CAMK2A', 'Neuron / synapse'), ('MOG', 'Glia / myelin'),
    ('PLP1', 'Glia / myelin'), ('TH', 'Neuron / synapse'),
    ('SYP', 'Neuron / synapse'), ('CNP', 'Glia / myelin'),
    ('NEFH', 'Neuron / synapse'), ('SYN1', 'Neuron / synapse'),
    ('MAPT', 'Neuron / synapse'), ('CALB1', 'Neuron / synapse'),
]

PANELS = [
    dict(name='figure4d_single_cell_rank', classes=SC_CLASSES,
         candidates=SC_CANDIDATES, loader=single_cell_table,
         xlabel='Protein group rank'),
    dict(name='figure4g_tissue_rank', classes=BRAIN_CLASSES,
         candidates=BRAIN_CANDIDATES, loader=tissue_table,
         xlabel='Protein group rank',
         display=lambda g: g[:1] + g[1:].lower()),
]


def main():
    fs.init()
    fs.start_build(script=__file__, inputs=[os.path.join(cfg.DATA_ROOT, 'H032_E388'),
                               os.path.join(cfg.DATA_ROOT, 'H032_E389')], out_dir=OUT)
    for panel in PANELS:
        draw(panel)
    fs.report_stale(OUT, prefix='figure')


if __name__ == '__main__':
    main()
