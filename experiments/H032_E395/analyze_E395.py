"""alpha-synuclein pS129 in the real sample matrix - E395."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg
import pathlib
import sys

import numpy as np
import pandas as pd

import zenotof8600_analytics as core

sys.path.insert(0, cfg.data('H032_E391'))
from annotate_spectra import MAX_FRAGMENT_CHARGE
from check_run_sanity import RT_SCHEDULED, WINDOW_S
from make_final_method import all_species
from score_fragment_coverage import FRAG_TYPES
from score_quant_feasibility import COUNTS_PER_ION

XIC_TOL_PPM = 'auto'

RAW_DIR = pathlib.Path(cfg.raw('H032_E395', 'raw'))
CACHE_DIR = cfg.raw('ms_data_cache')
OUT_DIR = pathlib.Path(cfg.cache('H032_E395', 'analysis'))
PREFIX = '20260920_Zeno2_Eno0_6p4min_TiHe_SA_H032_E395_'

GROUPS = (('cond B', ('B1', 'B2', 'B3')),
          ('cond D', ('D1', 'D2', 'D3', 'D4')))

QUANT_SET = (('y', 3), ('z', 4), ('y', 4), ('z', 5), ('z', 6))

DIAGNOSTIC = ('y', 13)

INTEG_HALF_MIN = 3.5 / 60.0

BASELINE_EXCLUDE_MIN = 5.0 / 60.0

COELUTION_TOL_MIN = 0.04
MIN_COELUTING = 3

MIN_IONS_FOR_RT = 5.0


def _species():
    core.raw.register_sil_modifications()
    sp = core.precursor_q1_mz(all_species(), isotope='most_abundant')
    frags = core.raw.target_fragments(sp, frag_types=FRAG_TYPES['ead'],
                                      min_mz=150.0, max_mz=3000.0)
    return sp, frags


def _traces(ms, q1, sub, centre):
    """Per-transition XIC over the scheduled window, in ions."""
    half = WINDOW_S / 60.0
    out = {}
    for ion, num in QUANT_SET + (DIAGNOSTIC,):
        r = sub[(sub['ion_type'] == ion) & (sub['ion_number'] == num)
                & (sub['fragment_charge'] == 1)]
        if r.empty:
            continue
        x = core.raw.targeted_xic(ms, q1, float(r['product_mz'].iloc[0]),
                                  mz_tol=XIC_TOL_PPM)
        m = (x['rt'] >= centre - half) & (x['rt'] <= centre + half)
        out[f'{ion}{num}'] = (x.loc[m, 'rt'].to_numpy(),
                              x.loc[m, 'intensity'].to_numpy() / COUNTS_PER_ION)
    return out


def _integrate(rt, y, apex):
    """Background-subtracted ions over the peak, the baseline, and the uncertainty an empty channel would still show."""
    off = np.abs(rt - apex) > BASELINE_EXCLUDE_MIN
    base = float(np.mean(y[off])) if off.sum() >= 5 else 0.0
    sd = float(np.std(y[off])) if off.sum() >= 5 else np.nan
    peak = np.abs(rt - apex) <= INTEG_HALF_MIN
    n = float(peak.sum())
    return float(np.sum(y[peak] - base)), base, n, sd * np.sqrt(n)


def _summed_apex(traces):
    """Apex of the summed quant transitions of one channel."""
    if not traces:
        return np.nan
    rt = traces['y3'][0]
    tot = sum(y for k, (_, y) in traces.items() if k != 'y13')
    return float(rt[int(np.argmax(tot))]) if tot.max() > 0 else np.nan


def measure(ms, sp, frags, run):
    """One row per (channel, transition), light pinned to its heavy apex."""
    per_channel = {}
    for i, t in sp.iterrows():
        cid = t['compound_id']
        sub = frags[(frags['target_idx'] == i)
                    & (frags['fragment_charge'] <= MAX_FRAGMENT_CHARGE)]
        per_channel[cid] = _traces(ms, float(t['q1_mz']), sub,
                                   RT_SCHEDULED[cid])

    rows = []
    for cid, tr in per_channel.items():
        if not tr:
            rows.append({'run': run, 'compound_id': cid, 'ion': None,
                         'kind': 'quant', 'ions': 0.0, 'acquired': False})
            continue
        form, base_name = cid.split('_')[1], cid.split('_')[2]
        own_apex = _summed_apex(tr)
        ref_apex = own_apex
        heavy_cid = cid.replace('_l_', '_h_')
        if form == 'l' and per_channel.get(heavy_cid):
            h = _summed_apex(per_channel[heavy_cid])
            if h == h:
                ref_apex = h
        for ion, (irt, iy) in tr.items():
            ions, baseline, n, sigma = _integrate(irt, iy, ref_apex)
            ion_apex = float(irt[int(np.argmax(iy))]) if iy.max() > 0 else np.nan
            rows.append({
                'run': run, 'compound_id': cid, 'form': form,
                'base': base_name,
                'ox': cid.split('_ox')[1] if '_ox' in cid else '0',
                'ion': ion, 'kind': 'diagnostic' if ion == 'y13' else 'quant',
                'centre': RT_SCHEDULED[cid], 'own_apex': round(own_apex, 4),
                'ref_apex': round(ref_apex, 4), 'ion_apex': round(ion_apex, 4),
                'ions': ions, 'gross_ions': float(iy.sum()),
                'baseline_ions': baseline, 'sigma_ions': sigma,
                'n_scans_integrated': n, 'acquired': True})
    return rows


def _piv(d, idx, col, val, agg='sum'):
    return d.pivot_table(index=idx, columns=col, values=val, aggfunc=agg)


def main(runs=None):
    sp, frags = _species()
    wanted = list(runs) if runs else [r for _, rr in GROUPS for r in rr]
    panel = {r: g for g, rr in GROUPS for r in rr}

    rows = []
    for run in wanted:
        ms = core.raw.load_ms_data(
            str(RAW_DIR / f'{PREFIX}{run}.wiff'), cache_dir=CACHE_DIR,
            reader_options={'keep_k_peaks_per_spec': 10 ** 7})
        for r in measure(ms, sp, frags, run):
            r['panel'] = panel.get(run, '?')
            rows.append(r)
    d = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    d.to_csv(OUT_DIR / 'e395_quant_sourcedata.csv', index=False)
    pd.set_option('display.width', 240)

    q = d[d['kind'] == 'quant']

    print('=== window fit: deviation of the apex from the scheduled centre, '
          'seconds (+-8 s window) ===')
    w = (q.groupby(['run', 'compound_id'])
          .agg(centre=('centre', 'first'), apex=('own_apex', 'first'),
               ions=('ions', 'sum')).reset_index())
    w['dev_s'] = ((w['apex'] - w['centre']) * 60).round(1)
    w.loc[w['ions'] < MIN_IONS_FOR_RT, 'dev_s'] = np.nan
    print(_piv(w, 'compound_id', 'run', 'dev_s', 'first').round(1).to_string())
    worst = w['dev_s'].abs().max()
    n_acq = int(d.groupby(['run', 'compound_id'])['acquired'].first().sum())
    print(f'  worst |deviation| {worst:.1f} s (margin {WINDOW_S - worst:.1f} s)'
          f'   channels acquired {n_acq}/{12 * len(wanted)}')

    print('\n=== quant ions per channel (background-subtracted, summed over '
          'the 5 transitions) ===')
    print(_piv(q, 'compound_id', 'run', 'ions').round(0).to_string())

    print('\n=== summed over the three oxidation states ===')
    s = q.groupby(['run', 'panel', 'form', 'base'])['ions'].sum().reset_index()
    print(_piv(s, ['form', 'base'], 'run', 'ions').round(0).to_string())

    print('\n=== heavy / light per form (H:L), with the detection limit ===')
    piv = s.pivot_table(index=['panel', 'run'], columns=['form', 'base'],
                        values='ions')
    lod = (q[q['form'] == 'l'].assign(v=lambda x: x['sigma_ions'] ** 2)
             .groupby(['panel', 'run', 'base'])['v'].sum()
             .pow(0.5).mul(3).unstack('base'))
    out = pd.DataFrame(index=piv.index)
    out['H:L unmod'] = piv[('h', 'unmod')] / piv[('l', 'unmod')]
    out['H:L pS129'] = piv[('h', 'pS129')] / piv[('l', 'pS129')]
    out['light pS129'] = piv[('l', 'pS129')]
    out['3sd limit'] = lod['pS129']
    out['detected'] = np.where(out['light pS129'] > out['3sd limit'],
                               'yes', 'NO')
    out['H:L pS129 floor'] = piv[('h', 'pS129')] / lod['pS129']
    out['heavy p:u'] = piv[('h', 'pS129')] / piv[('h', 'unmod')]
    print(out.round(2).to_string())
    print('  H:L pS129 is only a measurement where `detected` is yes; '
          'elsewhere read\n  `H:L pS129 floor`, the lowest light pS129 this '
          'run could have shown.')

    print('\n=== pS129 occupancy ===')
    occ = pd.DataFrame(index=piv.index)
    lh_u = piv[('l', 'unmod')] / piv[('h', 'unmod')]
    lh_p = piv[('l', 'pS129')] / piv[('h', 'pS129')]
    lh_lim = lod['pS129'] / piv[('h', 'pS129')]
    occ['measured'] = lh_p / (lh_p + lh_u)
    occ['upper bound'] = lh_lim / (lh_lim + lh_u)
    print(occ.round(4).to_string())
    print('  Both assume an equimolar heavy spike of pS129 and unmod '
          '(measured p:u above);\n  divide by the spike true pS129:unmod to '
          'correct.  Where nothing is detected\n  only `upper bound` is a '
          'result.')

    print('\n=== light pS129: does it co-elute with its heavy partner? ===')
    lp = q[(q['form'] == 'l') & (q['base'] == 'pS129')].copy()
    lp['d_s'] = ((lp['ion_apex'] - lp['ref_apex']) * 60).round(1)
    lp['coelutes'] = (lp['ion_apex'] - lp['ref_apex']).abs() <= COELUTION_TOL_MIN
    for run in wanted:
        sub = lp[lp['run'] == run]
        for cid in sorted(sub['compound_id'].unique()):
            c = sub[sub['compound_id'] == cid]
            n = int(c['coelutes'].sum())
            verdict = 'REAL' if n >= MIN_COELUTING else 'not resolved'
            print(f'  {run:4s} {cid:22s} {c["ions"].sum():8.1f} ions  '
                  f'{n}/{len(c)} co-elute  -> {verdict}')
            print('         ' + '  '.join(
                f'{r.ion}:{r.ions:7.1f}@{r.d_s:+.1f}s' for r in c.itertuples()))

    print('\n=== y13 confirmation (phospho-specific, not summed into quant) ===')
    y = d[d['kind'] == 'diagnostic']
    print(_piv(y, ['form', 'base'], 'run', 'ions').round(1).to_string())

    print(f'\nwrote {OUT_DIR / "e395_quant_sourcedata.csv"}')


if __name__ == '__main__':
    main(sys.argv[1:] or None)
