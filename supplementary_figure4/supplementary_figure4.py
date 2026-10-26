"""Supplementary Figure 4: Q1 width screen at single-cell-equivalent load."""

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
DATA = os.path.join(cfg.DATA_ROOT, 'H032_E382')
OUT = cfg.panels(__file__)

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

CONDITIONS = [
    {'label': '100 vw\nDIA', 'dir': '40SPD_100vw12ms', 'width_th': None},
    {'label': '5.7', 'dir': '40SPD_5p7Th_1p48s', 'width_th': 5.7},
    {'label': '7.2', 'dir': '40SPD_7p2Th_1p48s', 'width_th': 7.2},
    {'label': '10.1', 'dir': '40SPD_10p1Th_1p48s', 'width_th': 10.1},
    {'label': '14.5', 'dir': '40SPD_14p5Th_1p48s', 'width_th': 14.5},
]
COND_ORDER = [c['label'] for c in CONDITIONS]

COLOR = {c: core.PALETTE_SINGLE[i % len(core.PALETTE_SINGLE)]
         for i, c in enumerate(COND_ORDER)}

LETTER = {'ids_pg': 'a', 'ids_prec': 'b',
          'disco_ratio': 'c', 'disco_precursors': 'd', 'disco_protein_groups': 'e',
          'blank_pg': 'f', 'blank_prec': 'g'}

def load():
    experiments = []
    for c in CONDITIONS:
        d = os.path.join(DATA, c['dir'])
        f = os.path.join(d, 'lfq.dia.features.csv')
        runs = [x[:-len(' m/z')] for x in pd.read_csv(f, nrows=0).columns
                if x.endswith(' m/z') and x != 'm/z']
        if len(runs) != 4:
            raise ValueError(f'{c["dir"]}: {len(runs)} runs, expected 4')
        experiments.append({
            'path': f, 'protein_path': os.path.join(d, 'lfq.dia.proteins.csv'),
            'file_tags': runs,
            'condition1': 'single-cell-equivalent HEK, 40 SPD, 1.48 s',
            'condition2': c['label']})
    df, sample_info = core.load_experiments(experiments)
    sample_info['replicate'] = sample_info.groupby('condition2').cumcount() + 1
    return df, sample_info


def panel_ids(agg):
    """a, b — identifications against Q1 width, with the CV<20% subset."""
    for key, total_col, cv_col, ylabel in [
        ('ids_pg', 'total_protein_groups', 'PG20', '# protein groups'),
        ('ids_prec', 'total_precursors', 'Pr20', '# precursors'),
    ]:
        name = f'supplementary_figure4{LETTER[key]}_' + (
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
        ax.set_xlabel('Q1 isolation width [Th]', fontsize=fs.FONTSIZE)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        fs.tidy(ax)
        ax.tick_params(labelsize=fs.FONTSIZE)
        ax.xaxis.label.set_fontsize(fs.FONTSIZE)
        ax.yaxis.label.set_fontsize(fs.FONTSIZE)

        tot = (agg.groupby('condition2')[total_col].first().reindex(COND_ORDER))
        q20 = agg.groupby('condition2')[cv_col].first().reindex(COND_ORDER)
        print(f'    {ylabel}')
        for c in COND_ORDER:
            print(f'      {c.replace(chr(10), " "):10s} {tot[c]:>9,.0f}   '
                  f'<20% CV {q20[c]:>9,.0f}')
        print(f'      totals peak at '
              f'{tot.idxmax().replace(chr(10), " ")!r}, quantifiable peak at '
              f'{q20.idxmax().replace(chr(10), " ")!r}')

        fig.tight_layout()
        fs.set_bar_geometry_inches(fig, ax)
        fs.save_panel(fig, OUT, name, summary, dpi=300)
        plt.close(fig)


def main():
    fs.init()
    fs.start_build(script=__file__, inputs=[DATA], out_dir=OUT)
    print(f'loading from {DATA} ...')
    df, sample_info = load()
    print(f'  {df.shape[0]:,} rows, {len(sample_info)} runs')
    agg = core.process_experiment(df, sample_info, group_col='condition2')
    print('\npanels a, b ...')
    panel_ids(agg)

    import disco_panels
    print('\npanels c, d, e -- discovery mode, H032_E387 ...')
    disco_panels.main((LETTER['disco_ratio'], LETTER['disco_precursors'],
                       LETTER['disco_protein_groups']))

    import blank_control
    print('\npanels f, g -- processed blanks, with and without references ...')
    blank_control.main((LETTER['blank_pg'], LETTER['blank_prec']))

    fs.report_printed_sizes()
    fs.report_stale(OUT, prefix='supplementary_figure')


if __name__ == '__main__':
    main()
