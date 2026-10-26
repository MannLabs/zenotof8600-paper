"""pS129 occupancy from the ZT Scan DIA runs of H032_E397."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys
import time

import h5py
import hdf5plugin
import numpy as np
import pandas as pd

import zenotof8600_analytics as core
from zenotof8600_analytics.ztscan.patch import read_q1_window

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(cfg.DATA_ROOT, 'H032_E397')
HDF_DIR = cfg.raw('ztscan_full')
PREFIX = '20260921_Zeno2_Eno0_6p4min_TiHe_SA_H032_E397_'
GROUPS = (('PFF-', ('A5_repeat', 'A6_repeat', 'A7_repeat', 'A8_repeat')),
          ('PFF+', ('B5_repeat', 'B6_repeat', 'B7_repeat', 'B8_repeat')))
REFERENCE_RUN = 'B5_repeat'

for p in (cfg.data('H032_E395'), cfg.data('H032_E391')):
    if p not in sys.path:
        sys.path.insert(0, p)

N_TRANSITIONS = 6
FRAG_PPM = 25.0
ISOBARIC_PPM = 30.0
RT_SEARCH_MIN = 0.6
INTEGRATE_FWHM = 1.5


def hdf(run):
    return os.path.join(HDF_DIR, f'{PREFIX}{run}.hdf')


def species_and_fragments():
    import analyze_E395 as e395
    from check_run_sanity import RT_SCHEDULED
    sp, _ = e395._species()
    fr = core.raw.target_fragments(sp, frag_types=['y_z1', 'y_z2'],
                                   min_mz=250.0, max_mz=2000.0)
    fr = fr[fr['ion_number'] >= 3]
    sp = sp.copy()
    sp['rt_sched'] = sp['compound_id'].map(RT_SCHEDULED)
    sp['form'] = sp['compound_id'].str.split('_').str[1]
    sp['species'] = sp['compound_id'].str.split('_', n=2).str[2]
    sp['base'] = np.where(sp['species'].str.startswith('pS129'),
                          'pS129', 'unmod')
    return sp, fr


def drop_isobaric(sp, fr):
    """Remove ions within ISOBARIC_PPM of an ion of ANOTHER channel whose precursor shares a ZT window (|dq1| < 2.5 Th): they would be one peak."""
    keep = np.ones(len(fr), bool)
    q1 = sp['q1_mz']
    mz = fr['product_mz'].to_numpy()
    t = fr['target_idx'].to_numpy()
    for i in range(len(fr)):
        near = (np.abs(q1.loc[t] .to_numpy() - q1.loc[t[i]]) < 2.5) & (t != t[i])
        if np.any(np.abs(mz[near] - mz[i]) / mz[i] * 1e6 < ISOBARIC_PPM):
            keep[i] = False
    return fr[keep]


def channel_traces(path, sp, fr, half_iso):
    """{target_idx: (rt per cycle, summed fragment intensity per ion)}."""
    lo, hi = sp['q1_mz'].min() - 2 * half_iso, sp['q1_mz'].max() + 2 * half_iso
    rt_lo = sp['rt_sched'].min() - RT_SEARCH_MIN
    rt_hi = sp['rt_sched'].max() + RT_SEARCH_MIN
    win = read_q1_window(path, float(lo), float(hi), float(rt_lo),
                         float(rt_hi), n_cycles=0, mz_range=(200.0, 2100.0))
    rt1 = win['rt1'][(win['rt1'] >= rt_lo) & (win['rt1'] <= rt_hi)]
    edges = np.append(rt1, rt1[-1] + np.median(np.diff(rt1)))
    cyc = np.searchsorted(edges, win['s_rt'], side='right') - 1
    n_spec = len(win['s_rt'])
    spec = np.repeat(np.arange(n_spec), np.diff(win['off']))
    key = spec * 4096.0 + win['mz'].astype(np.float64)
    o = np.argsort(key, kind='stable')
    key = key[o]
    cum = np.concatenate(([0.0], np.cumsum(win['inten'][o], dtype=np.float64)))
    out = {}
    for t, g in fr.groupby('target_idx'):
        q1 = float(sp.loc[t, 'q1_mz'])
        js = np.flatnonzero((np.abs(win['s_q1'] - q1) <= half_iso)
                            & (cyc >= 0) & (cyc < len(rt1)))
        f = g['product_mz'].to_numpy()
        ions = g['ion_type'].str.cat(g['ion_number'].astype(str)) + \
            '+' + g['fragment_charge'].astype(str)
        jj, ff = np.meshgrid(js, f, indexing='ij')
        base = jj.ravel() * 4096.0
        fv = ff.ravel()
        tol = fv * FRAG_PPM * 1e-6
        v = (cum[np.searchsorted(key, base + fv + tol, side='right')]
             - cum[np.searchsorted(key, base + fv - tol, side='left')])
        v = v.reshape(len(js), len(f))
        per_cycle = np.zeros((len(rt1), len(f)))
        np.add.at(per_cycle, cyc[js], v)
        out[t] = (rt1, per_cycle, list(ions))
    return out


def half_isolation(path):
    with h5py.File(path, 'r') as h:
        sg = h['ms_data']['spectrum_df']
        m = sg['ms_level'][:] == 2
        w = sg['isolation_upper_mz'][:][m] - sg['isolation_lower_mz'][:][m]
    return float(np.median(w)) / 2


def choose_transitions(sp, fr, traces):
    """Per channel: the N strongest ions of its HEAVY partner, by summed signal in the reference run; light and heavy then use the same ion names, so their ratio compares like with like."""
    chosen = {}
    for species, g in sp.groupby('species'):
        h = int(g.index[g['form'] == 'h'][0])
        rt, pc, ions = traces[h]
        top = np.argsort(pc.sum(axis=0))[::-1][:N_TRANSITIONS]
        names = [ions[k] for k in top]
        for t in g.index:
            chosen[int(t)] = names
    return chosen


def measure_run(run, sp, fr, chosen):
    path = hdf(run)
    half = half_isolation(path)
    tr = channel_traces(path, sp, fr, half)
    rows = []
    for species, g in sp.groupby('species'):
        h = int(g.index[g['form'] == 'h'][0])
        l = int(g.index[g['form'] == 'l'][0])

        def summed(t):
            rt, pc, ions = tr[t]
            cols = [ions.index(n) for n in chosen[t] if n in ions]
            return rt, pc[:, cols].sum(axis=1)

        rt_h, y_h = summed(h)
        pk = core.raw.pick_peak(rt_h, y_h, baseline="offpeak",
                            interpolate_points=500)
        apex, fwhm = float(pk['apex_rt']), float(pk['fwhm'])
        if not (np.isfinite(apex) and np.isfinite(fwhm) and fwhm > 0):
            apex = float(g['rt_sched'].iloc[0])
            fwhm = 0.04
        w = INTEGRATE_FWHM * fwhm
        for t, form in ((h, 'h'), (l, 'l')):
            rt, y = summed(t)
            a = core.raw.integrate_peak(rt, y, apex - w, apex + w,
                                        baseline='offpeak')
            m = np.abs(rt - apex) <= w
            r = (np.corrcoef(y[m], y_h[m])[0, 1]
                 if m.sum() > 3 and np.std(y[m]) > 0 and np.std(y_h[m]) > 0
                 else np.nan)
            rows.append(dict(run=run, species=species, form=form,
                             base=g['base'].iloc[0],
                             compound_id=sp.loc[t, 'compound_id'],
                             q1_mz=float(sp.loc[t, 'q1_mz']),
                             anchor_rt=apex, fwhm=fwhm, area=a['area'],
                             n_points=int(m.sum()), trace_r=r,
                             transitions=';'.join(chosen[t]),
                             half_isolation_th=half))
    return pd.DataFrame(rows)


def occupancy(d):
    s = d.groupby(['run', 'form', 'base'])['area'].sum().unstack(
        ['form', 'base'])
    r = pd.DataFrame(index=s.index)
    r['lh_unmod'] = s[('l', 'unmod')] / s[('h', 'unmod')]
    r['lh_ps129'] = s[('l', 'pS129')] / s[('h', 'pS129')]
    r['occupancy'] = 100 * r['lh_ps129'] / (r['lh_ps129'] + r['lh_unmod'])
    r['group'] = r.index.map({x: g for g, rr in GROUPS for x in rr})
    return r.reset_index()


def main(*runs):
    os.makedirs(OUT, exist_ok=True)
    core.raw.register_sil_modifications()
    sp, fr = species_and_fragments()
    n0 = len(fr)
    fr = drop_isobaric(sp, fr)
    print(f'{len(sp)} channels, {len(fr)} of {n0} y ions kept '
          f'(isobaric within {ISOBARIC_PPM:g} ppm dropped)')
    ref = hdf(REFERENCE_RUN)
    t0 = time.time()
    chosen = choose_transitions(sp, fr, channel_traces(ref, sp, fr,
                                                       half_isolation(ref)))
    for sp_name, g in sp.groupby('species'):
        print(f'  {sp_name:10s} {chosen[int(g.index[0])]}')
    print(f'  transitions fixed on {REFERENCE_RUN} in {time.time() - t0:.0f} s')
    want = runs or [r for _, rr in GROUPS for r in rr]
    parts = []
    for run in want:
        if not os.path.exists(hdf(run)):
            print(f'  {run}: no HDF yet -- skipped')
            continue
        t0 = time.time()
        parts.append(measure_run(run, sp, fr, chosen))
        print(f'  {run}: {time.time() - t0:.0f} s')
    d = pd.concat(parts, ignore_index=True)
    d.to_csv(os.path.join(OUT, 'ztscan_ps129_measurements.csv'), index=False)
    occ = occupancy(d)
    occ.to_csv(os.path.join(OUT, 'ztscan_ps129_occupancy.csv'), index=False)
    pd.set_option('display.width', 200)
    print(occ.round(4).to_string(index=False))
    print(occ.groupby('group')['occupancy'].agg(['mean', 'std', 'count'])
          .round(3).to_string())
    print('light pS129 trace_r vs heavy:')
    print(d[(d.form == 'l') & (d.base == 'pS129')]
          .pivot_table(index='run', columns='species', values='trace_r')
          .round(2).to_string())


if __name__ == '__main__':
    main(*sys.argv[1:])
