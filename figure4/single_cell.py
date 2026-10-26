"""Figure 4 -- single-cell HeLa depth, H032_E388."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os

import numpy as np

import zenotof8600_analytics as core

import box_panel as bp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = cfg.panels(__file__)
FEATURES = os.path.join(cfg.DATA_ROOT, 'H032_E388', 'all_cells',
                        'lfq.dia.features.csv')

CONDITIONS = [
    {'label': 'Blank', 'match': '_processed_blank_'},
    {'label': '1 cell', 'match': '_single_cell_'},
    {'label': '20 cells', 'match': '_10_cells_'},
    {'label': '5 ng', 'match': '_HeLa_5ng_'},
]
UNIT_IN = 0.52
LETTER = 'b'
MAIN_LEVEL = 'protein_groups'

COND_ORDER = [c['label'] for c in CONDITIONS]

SINGLE_CELL = CONDITIONS[1]['label']
COLOR = {c['label']: core.PALETTE_SINGLE[i] for i, c in enumerate(CONDITIONS)}

EXCLUDE_RUNS = {}

RELABEL = {
    '_single_cell_S3_B6': 'Blank',
    '_processed_blank_S3_H9': '1 cell',
}

OUTLIER_K = 3.0


def load():
    """Identification counts per run, conditions split by run-name substring."""
    runs = bp.run_names(FEATURES)

    def condition_of(r):
        for key, label in RELABEL.items():
            if key in r:
                return label
        for c in CONDITIONS:
            if c['match'] in r:
                return c['label']
        return None

    experiments = []
    for c in CONDITIONS:
        tags = [r for r in runs if condition_of(r) == c['label']]
        if not tags:
            raise ValueError(f'no run matches {c["match"]!r} in {FEATURES}')
        experiments.append({'path': FEATURES, 'file_tags': tags,
                            'condition1': 'H032_E388',
                            'condition2': c['label']})
    return bp.identification_counts(experiments, COND_ORDER,
                                    strip=r'.*H032_E388_|.*_TiHe_')


def apply_exclusions(ids):
    """Flag named runs and failed single cells; return (ids, cutoff)."""
    ids = ids.copy()
    ids['included'] = True
    ids['exclusion_reason'] = ''

    named = ids['well'].isin(EXCLUDE_RUNS)
    ids.loc[named, 'included'] = False
    ids.loc[named, 'exclusion_reason'] = ids.loc[named, 'well'].map(
        EXCLUDE_RUNS)

    sc = ids['condition2'] == SINGLE_CELL
    cells = ids.loc[sc & ids['included'], 'run'].tolist()
    df, si = core.load_experiments([{'path': FEATURES, 'file_tags': cells,
                                     'condition1': 'H032_E388',
                                     'condition2': SINGLE_CELL}])
    _, kept, qc = core.filter_outlier_runs(df, si, cohort_col=None,
                                           plot=False)
    cutoff = float(qc['threshold'].iloc[0])
    failed = sc & ids['included'] & ~ids['run'].isin(kept['run'])
    ids.loc[failed, 'included'] = False
    ids.loc[failed, 'exclusion_reason'] = (
        f'failed cell -- below {cutoff:,.0f} precursors (2-component GMM on '
        f'log10 precursors per run, equal-posterior boundary)')
    return ids, cutoff


def main():
    bp.fs.start_build(script=__file__, inputs=[os.path.dirname(FEATURES)], out_dir=OUT)
    ids, cutoff = apply_exclusions(load())

    print(f'\n{len(ids)} runs, {(~ids["included"]).sum()} excluded')
    for r in ids[~ids['included']].itertuples():
        print(f'  - {r.well:24s} {r.protein_groups:>6,} PG   '
              f'{r.exclusion_reason}')
    print(f'  failed-cell cutoff: {cutoff:,.0f} precursors (GMM)')

    print('\nmedian per sample type (drawn runs only)')
    suffix, col, ylabel = next(l for l in bp.LEVELS if l[0] == MAIN_LEVEL)
    bp.draw(ids, name=f'figure4{LETTER}_single_cell_{suffix}', col=col,
            ylabel=ylabel, cond_order=COND_ORDER, color=COLOR,
            out_dir=OUT, unit_in=UNIT_IN, show_excluded=True)

    sc = ids.loc[(ids['condition2'] == SINGLE_CELL) & ids['included'],
                 'protein_groups']
    print(f'\nsingle cells kept: n={len(sc)}  median={sc.median():,.0f}  '
          f'IQR {sc.quantile(.25):,.0f}-{sc.quantile(.75):,.0f}  '
          f'range {sc.min():,.0f}-{sc.max():,.0f}')
    bp.report_small_groups(ids, COND_ORDER)
    bp.fs.report_stale(OUT, prefix='figure')


if __name__ == '__main__':
    main()
