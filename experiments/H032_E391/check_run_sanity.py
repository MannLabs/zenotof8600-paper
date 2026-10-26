"""One-command sanity check on a single run, before committing a queue."""

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

from annotate_spectra import MAX_FRAGMENT_CHARGE
from make_final_method import all_species
from plot_quant_xic import XIC_TOL_PPM
from score_fragment_coverage import FRAG_TYPES, RAW_DIR
from score_quant_feasibility import COUNTS_PER_ION

CACHE_DIR = cfg.raw('ms_data_cache')
PREFIX_D = '20260920_Zeno2_Eno0_6p4min_TiHe_SA_H032_E391_box2_'

RT_SCHEDULED = {
    'aSyn_h_unmod': 5.31, 'aSyn_h_unmod_ox1': 5.10, 'aSyn_h_unmod_ox2': 4.49,
    'aSyn_h_pS129': 5.40, 'aSyn_h_pS129_ox1': 5.26, 'aSyn_h_pS129_ox2': 4.62,
    'aSyn_l_unmod': 5.31, 'aSyn_l_unmod_ox1': 5.10, 'aSyn_l_unmod_ox2': 4.49,
    'aSyn_l_pS129': 5.40, 'aSyn_l_pS129_ox1': 5.26, 'aSyn_l_pS129_ox2': 4.62,
}

WINDOW_S = 8.0

QUANT_SET = (('y', 3), ('z', 4), ('y', 4), ('z', 5), ('z', 6))
DIAGNOSTIC = ('y', 13)

MIN_IONS_FOR_RT = 5.0


def measure(run, prefix=PREFIX_D):
    core.raw.register_sil_modifications()
    sp = core.precursor_q1_mz(all_species(), isotope='most_abundant')
    frags = core.raw.target_fragments(sp, frag_types=FRAG_TYPES['ead'],
                                      min_mz=150.0, max_mz=3000.0)
    ms = core.raw.load_ms_data(
        str(RAW_DIR / f'{prefix}{run}.wiff'), cache_dir=CACHE_DIR,
        reader_options={'keep_k_peaks_per_spec': 10 ** 7})
    rows = []
    for i, t in sp.iterrows():
        cid = t['compound_id']
        q1, centre = float(t['q1_mz']), RT_SCHEDULED[cid]
        sub = frags[(frags['target_idx'] == i)
                    & (frags['fragment_charge'] <= MAX_FRAGMENT_CHARGE)]
        half = WINDOW_S / 60.0
        traces, quant = {}, 0.0
        for ion, num in QUANT_SET + (DIAGNOSTIC,):
            r = sub[(sub['ion_type'] == ion) & (sub['ion_number'] == num)
                    & (sub['fragment_charge'] == 1)]
            if r.empty:
                continue
            x = core.raw.targeted_xic(ms, q1, float(r['product_mz'].iloc[0]),
                                      mz_tol=XIC_TOL_PPM)
            m = (x['rt'] >= centre - half) & (x['rt'] <= centre + half)
            y = x.loc[m, 'intensity'].to_numpy() / COUNTS_PER_ION
            traces[f'{ion}{num}'] = (x.loc[m, 'rt'].to_numpy(), y)
            if (ion, num) != DIAGNOSTIC:
                quant += y.sum()
        if not traces:
            rows.append({'compound_id': cid, 'ions': 0.0, 'found': False})
            continue
        rt, tot = traces['y3'][0], sum(v for _, v in traces.values())
        apex = float(rt[int(np.argmax(tot))]) if tot.max() > 0 else np.nan
        rows.append({
            'compound_id': cid, 'centre': centre, 'apex': round(apex, 3),
            'dev_s': round((apex - centre) * 60, 1) if apex == apex else np.nan,
            'ions': round(quant, 1),
            'y13_ions': round(float(traces.get('y13', (0, [0]))[1].sum()), 1),
            'found': quant > 0})
    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run = sys.argv[1]
    d = measure(run)
    pd.set_option('display.width', 220)
    d['margin_s'] = (WINDOW_S - d['dev_s'].abs()).round(1)
    d.loc[d['ions'] < MIN_IONS_FOR_RT, ['dev_s', 'margin_s']] = np.nan

    print(f'=== {run} ===')
    print(d[['compound_id', 'centre', 'apex', 'dev_s', 'margin_s',
             'ions', 'y13_ions']].to_string(index=False))

    heavy = d[d['compound_id'].str.contains('_h_')]
    print(f'\n  heavy quant ions, summed over oxidation states: '
          f"{heavy['ions'].sum():.0f}")
    missing = d.loc[~d['found'], 'compound_id'].tolist()
    print(f'  channels found: {int(d["found"].sum())}/12'
          + (f'  MISSING {missing}' if missing else ''))
    worst = d['dev_s'].abs().max()
    if worst == worst:
        early = d.loc[d['dev_s'].idxmin()]
        print(f'  worst deviation {worst:.1f} s '
              f'(min margin {WINDOW_S - worst:.1f} s); earliest channel is '
              f"{early['compound_id']} at {early['dev_s']:+.1f} s")
        if worst > WINDOW_S * 0.6:
            print('  ** over 60% of the window used - re-centre before '
                  'running the queue **')


if __name__ == '__main__':
    main()
