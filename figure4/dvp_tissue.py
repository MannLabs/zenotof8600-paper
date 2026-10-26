"""Figure 4: deep visual proteomics tissue depth against dissected area."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os

import zenotof8600_analytics as core

import box_panel as bp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = cfg.panels(__file__)
DATA = os.path.join(cfg.DATA_ROOT, 'H032_E389')

CONDITIONS = [
    {'label': '10k', 'dir': '10k_um2', 'wells': 'A3-D3'},
    {'label': '100k', 'dir': '100k_um2', 'wells': 'A4-D4'},
    {'label': '200k', 'dir': '200k_um2', 'wells': 'A5-D5'},
]

AREA_UNIT = r'$\mu$m$^2$'

LETTER = 'f'
MAIN_LEVEL = 'protein_groups'


def _check_distinct_runs(experiments):
    """Refuse two conditions that hold the same run."""
    seen = {}
    for exp in experiments:
        for run in exp['file_tags']:
            if run in seen:
                raise ValueError(
                    f'conditions {seen[run]!r} and {exp["condition2"]!r} both '
                    f'contain run {run.split("E389_")[-1]!r} -- one of the two '
                    f'search folders is a copy of the other, or was exported '
                    f'from the wrong runs. Expected wells: '
                    + ', '.join(f'{c["label"]}={c["wells"]}'
                                for c in CONDITIONS))
            seen[run] = exp['condition2']


def load():
    """Identification counts per run, one search folder per condition."""
    experiments, order = [], []
    for c in CONDITIONS:
        path = os.path.join(DATA, c['dir'], 'lfq.dia.features.csv')
        if not os.path.exists(path):
            print(f'  SKIPPING {c["label"]}: no search at '
                  f'{os.path.join(c["dir"], "lfq.dia.features.csv")}')
            continue
        experiments.append({'path': path, 'file_tags': bp.run_names(path),
                            'condition1': 'H032_E389',
                            'condition2': c['label']})
        order.append(c['label'])
    if not experiments:
        raise ValueError(f'no search folder found under {DATA}')
    _check_distinct_runs(experiments)

    ids = bp.identification_counts(experiments, order,
                                   strip=r'.*H032_E389_|.*_TiHe_')
    ids['included'] = True
    ids['exclusion_reason'] = ''
    return ids, order


COLOR = {c['label']: core.PALETTE_SINGLE[i] for i, c in enumerate(CONDITIONS)}


def main():
    bp.fs.start_build(script=__file__, inputs=[DATA], out_dir=OUT)
    ids, order = load()

    print(f'\n{len(ids)} runs over {len(order)} areas\n')
    print(ids[['well', 'condition2', 'precursors', 'peptides',
               'protein_groups']].to_string(index=False))

    print('\nmedian per area (drawn runs only)')
    suffix, col, ylabel = next(l for l in bp.LEVELS if l[0] == MAIN_LEVEL)
    bp.draw(ids, name=f'figure4{LETTER}_dvp_tissue_{suffix}', col=col,
            ylabel=ylabel, cond_order=order, color=COLOR, out_dir=OUT,
            xlabel=f'Dissected area ({AREA_UNIT})')

    bp.report_small_groups(ids, order)
    pg = ids.groupby('condition2', observed=True)['protein_groups'].median()
    if '10k' in pg and '200k' in pg:
        print(f'\n200k over 10k: {pg["200k"] / pg["10k"]:.2f}x protein groups '
              f'for 20x the area')
    bp.fs.report_stale(OUT, prefix='figure')


if __name__ == '__main__':
    main()
