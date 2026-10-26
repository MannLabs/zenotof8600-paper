"""aSyn pS129 in E397, measured through the library."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg
import sys

import numpy as np
import pandas as pd

import zenotof8600_analytics as core

sys.path.insert(0, cfg.data('H032_E395'))
sys.path.insert(0, cfg.data('H032_E391'))
import analyze_E395 as e395
from check_run_sanity import RT_SCHEDULED, WINDOW_S

RAW_DIR = e395.pathlib.Path(cfg.raw('H032_E397', 'raw'))
PREFIX = '20260921_Zeno2_Eno0_6p4min_TiHe_SA_H032_E397_'
CACHE_DIR = cfg.raw('ms_data_cache')
OUT_DIR = e395.pathlib.Path(cfg.cache('H032_E397', 'analysis'))

GROUPS = (('PFF-', ('A1', 'A2', 'A3', 'A4_repeat')),
          ('PFF+', ('B1', 'B2', 'B3', 'B4_repeat')))

Q1_TOL = 0.004

SPECIES = ('unmod', 'unmod_ox1', 'unmod_ox2',
           'pS129', 'pS129_ox1', 'pS129_ox2')


def load(run):
    return core.raw.load_ms_data(
        str(RAW_DIR / f'{PREFIX}{run}.wiff'), cache_dir=CACHE_DIR,
        reader_options={'keep_k_peaks_per_spec': 10 ** 7})


def main():
    core.raw.register_sil_modifications()
    sp, frags = e395._species()
    keep = frags[
        frags.apply(lambda r: (r['ion_type'], int(r['ion_number']))
                    in e395.QUANT_SET and r['fragment_charge'] == 1, axis=1)]

    runs = [r for _, rr in GROUPS for r in rr]
    ms_files = {r: load(r) for r in runs}
    core.raw.assert_targets_resolved(ms_files[runs[0]], sp['q1_mz'],
                                     tol=Q1_TOL, names=sp['compound_id'])
    print(f'Q1 tolerance {Q1_TOL} Th isolates all {len(sp)} channels')

    out = []
    for species in SPECIES:
        ids = [f'aSyn_h_{species}', f'aSyn_l_{species}']
        idx = sp.index[sp['compound_id'].isin(ids)]
        sub = keep[keep['target_idx'].isin(idx)]
        centre = RT_SCHEDULED[ids[0]]
        half = WINDOW_S / 60.0
        heavy_idx = int(sp.index[sp['compound_id'] == ids[0]][0])
        q = core.raw.quantify_targets(
            ms_files, sub, reference_by='sequence',
            reference_selector=heavy_idx, fragment_filter=None,
            match='precursor', match_tol=Q1_TOL,
            rt_range=(centre - half, centre + half), verbose=False)
        q['species'] = species
        q['compound_id'] = q['target_idx'].map(sp['compound_id'])
        q['form'] = q['compound_id'].str.split('_').str[1]
        q['base'] = 'pS129' if 'pS129' in species else 'unmod'
        out.append(q)
    d = pd.concat(out, ignore_index=True)
    d['panel'] = d['run'].map({r: g for g, rr in GROUPS for r in rr})
    OUT_DIR.mkdir(exist_ok=True)
    d.to_csv(OUT_DIR / 'e397_lib_measurements.csv', index=False)

    pd.set_option('display.width', 240)
    s = d.groupby(['run', 'panel', 'form', 'base'])['area'].sum().reset_index()
    print('\n=== area, summed over the three oxidation states ===')
    print(s.pivot_table(index=['form', 'base'], columns='run',
                        values='area').round(0).to_string())

    piv = s.pivot_table(index=['panel', 'run'], columns=['form', 'base'],
                        values='area')
    r = pd.DataFrame(index=piv.index)
    r['L/H unmod'] = piv[('l', 'unmod')] / piv[('h', 'unmod')]
    r['L/H pS129'] = piv[('l', 'pS129')] / piv[('h', 'pS129')]
    r['occupancy %'] = 100 * r['L/H pS129'] / (r['L/H pS129'] + r['L/H unmod'])
    print('\n=== ratios ===')
    print(r.round(4).to_string())

    from scipy import stats as st
    print('\n=== PFF+ vs PFF- ===')
    for col in ('L/H pS129', 'occupancy %', 'L/H unmod'):
        a_ = r.xs('PFF+', level='panel')[col].to_numpy()
        b_ = r.xs('PFF-', level='panel')[col].to_numpy()
        print(f'  {col:14s} PFF- {b_.mean():9.4f}  PFF+ {a_.mean():9.4f}  '
              f'{a_.mean() / b_.mean():6.2f}x  '
              f'p = {st.ttest_ind(a_, b_, equal_var=False).pvalue:.2g}')

    print('\n=== light pS129: shape against its heavy partner (trace_r) ===')
    lp = d[(d['form'] == 'l') & (d['base'] == 'pS129')]
    print(lp.pivot_table(index='run', columns='species',
                         values='trace_r').round(3).to_string())
    print(f'\nwrote {OUT_DIR / "e397_lib_measurements.csv"}')


if __name__ == '__main__':
    main()
