"""Measure the Q1 locating precision at EVERY isolation width."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import argparse
import os
import sys

import numpy as np
import pandas as pd

ZT = _os.path.join(cfg.REPO_ROOT, 'experiments', 'H032_E376')
sys.path.insert(0, _os.path.join(cfg.REPO_ROOT, 'workflows'))
sys.path.insert(0, ZT)

import e376_targets as cp
import figure_q1_resolves_window as W
from q1_estimator_compare import parab_centroids
from ztscan_common import by_ions, load_axis, q1_correction
from ztscan_patch import extract_patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = cfg.data('H032_E376')
RUN_PREFIX = '20260905_Zeno2_Eno0_32p8min_TiHe_SA_H032_E376_'
OUT = os.path.join(cfg.DATA_ROOT, 'H032_E376', 'q1_precision_by_width.csv')

RUNS = [
    ('A5', 1.0, '40SPD_1Th_1p48s'),
    ('A9', 2.8, '40SPD_2p8Th_1p48s'),
    ('A10', 2.8, '40SPD_2p8Th_1p48s'),
    ('B1', 4.3, '40SPD_4p3Th_1p48s'),
    ('B5', 5.7, '40SPD_5p7Th_1p48s'),
    ('B9', 7.2, '40SPD_7p2Th_1p48s'),
]

N_PATCHES = 8
TOP_FRAC = 0.30
N_CYCLES = 11


def sigma_iqr(x):
    x = pd.Series(x).dropna()
    if len(x) < 4:
        return np.nan
    return float((x.quantile(0.75) - x.quantile(0.25)) / 1.349)


def detrend(err, against, n_bins=8):
    """Subtract the binned median of `err` against `against`, shape-free."""
    err = np.asarray(err, float)
    against = np.asarray(against, float)
    ok = np.isfinite(err) & np.isfinite(against)
    if ok.sum() < 3 * n_bins:
        return err - np.nanmedian(err)
    edges = np.quantile(against[ok], np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    idx = np.clip(np.searchsorted(edges, against, side='left') - 1,
                  0, n_bins - 1)
    out = err.copy()
    for b in range(n_bins):
        m = ok & (idx == b)
        if m.sum() >= 3:
            out[m] = err[m] - np.median(err[m])
    return out


def measure(tag, width, search):
    """One run: place abundant precursors by the parabolic Q1 vertex."""
    hdf = cfg.raw('ztscan_full', f'{RUN_PREFIX}{tag}.hdf')
    if not os.path.exists(hdf):
        print(f'  {tag}: no HDF at {hdf} -- skipped')
        return pd.DataFrame()

    cp.FEAT = os.path.join(ROOT, search, 'lfq.dia.features.csv')
    cp.TAG, cp.RUN = tag, RUN_PREFIX
    cp.N_PATCHES = N_PATCHES

    d = cp.identified()
    best = d.groupby(['seq', 'z']).area.max()
    cut = best.quantile(1 - TOP_FRAC)
    pct = best.rank(pct=True) * 100
    axis = load_axis(tag)
    print(f'  {tag} ({width} Th): {len(d):,} identified, '
          f'calibration {"found" if axis is not None else "none -- detrended"}')

    rows, tried = [], []
    for _, rt, q0 in cp.choose_patches(d):
        patch = extract_patch(hdf, rt, q0, q0 + cp.Q1_WIDTH,
                              n_cycles=N_CYCLES)
        if patch is None or not len(patch['mz']):
            continue
        corr = (q1_correction(axis, patch['q1_axis'], rt)
                if axis is not None else 0.0)
        qs = patch['q1_axis'] - corr

        sel = d[(np.abs(d.rt - rt) <= cp.RT_TOL) & (d.mz >= q0)
                & (d.mz <= q0 + cp.Q1_WIDTH)]
        sel = (sel.groupby(['seq', 'z'])
               .agg(mz=('mz', 'median'), area=('area', 'max')).reset_index())
        sel = sel[sel.area >= cut]
        tried += [dict(run=tag, width_th=width, patch=f'rt{rt:.0f}_q{q0:.0f}',
                       seq=r.seq, z=int(r.z), mz=float(r.mz),
                       area_pct=float(pct.get((r.seq, r.z), np.nan)))
                  for r in sel.itertuples()]
        if len(sel) < 2:
            continue

        ions = [np.asarray(by_ions([r.seq], [int(r.z)])[0], float).ravel()
                for r in sel.itertuples()]
        for k, r in enumerate(sel.itertuples()):
            others = np.concatenate([ions[j] for j in range(len(ions))
                                     if j != k])
            u = ions[k][np.min(np.abs(ions[k][:, None] - others[None, :]),
                               axis=1) > W.SHARED_TOL]
            fr = W.frag_rows(patch, u)
            if len(fr) < W.MIN_FRAG:
                continue
            H = patch['q1_prof'][[x[1] for x in fr]]
            c_gate = W.centroids(qs, H)[0]
            sn = np.isfinite(c_gate)
            ok = sn & (W.consensus_cos(H, sn) >= W.MIN_COS)
            if ok.sum() < W.MIN_FRAG:
                continue
            cc = parab_centroids(qs, H[ok])
            cc = cc[np.isfinite(cc)]
            if len(cc) < W.MIN_FRAG:
                continue
            rows.append(dict(run=tag, width_th=width,
                             patch=f'rt{rt:.0f}_q{q0:.0f}', seq=r.seq,
                             z=int(r.z), mz=float(r.mz), area=float(r.area),
                             area_pct=float(pct.get((r.seq, r.z), np.nan)),
                             n_frag=int(len(cc)),
                             centroid=float(np.median(cc)),
                             err_raw=float(np.median(cc) - r.mz),
                             within=sigma_iqr(cc)))
        del patch

    out = pd.DataFrame(rows)
    TRIED.append(pd.DataFrame(tried))
    if len(out):
        e = detrend(out['err_raw'].to_numpy(), out['mz'].to_numpy())
        out['err'] = detrend(e, out['patch'].astype('category').cat.codes
                             .to_numpy())
        print(f'    {len(out)} precursors placed over '
              f'{out.patch.nunique()} patches   '
              f'sigma raw {sigma_iqr(out.err_raw):.3f} -> '
              f'detrended {sigma_iqr(out.err):.3f} Th   '
              f'within {out.within.median():.3f} Th')
    return out


TRIED = []

GATES = {'strict': dict(MIN_SNR=5.0, MIN_COS=0.95, MIN_FRAG=4),
         'relaxed': dict(MIN_SNR=3.0, MIN_COS=0.80, MIN_FRAG=3)}


def main():
    global TOP_FRAC, N_PATCHES, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--top-frac', type=float, default=TOP_FRAC)
    ap.add_argument('--patches', type=int, default=N_PATCHES)
    ap.add_argument('--runs', default='', help='comma list of tags, e.g. A9')
    ap.add_argument('--gates', choices=list(GATES), default='strict')
    a = ap.parse_args()
    for k, v in GATES[a.gates].items():
        setattr(W, k, v)
    TOP_FRAC, N_PATCHES = a.top_frac, a.patches
    if TOP_FRAC != 0.30:
        OUT = OUT.replace('.csv', '_all.csv' if TOP_FRAC >= 1.0
                          else f'_top{round(100 * TOP_FRAC)}.csv')
    runs = [r for r in RUNS if not a.runs or r[0] in a.runs.split(',')]
    if N_PATCHES != 8:
        OUT = OUT.replace('.csv', f'_p{N_PATCHES}.csv')
    if a.gates != 'strict':
        OUT = OUT.replace('.csv', f'_{a.gates}.csv')
    if a.runs:
        OUT = OUT.replace('.csv', '_test_' + '_'.join(r[0] for r in runs)
                          + '.csv')
    print(f'gates {a.gates} {GATES[a.gates]}')
    print(f'top {100 * TOP_FRAC:.0f}% by abundance, {N_PATCHES} patches/run '
          f'-> {OUT}')
    parts = [measure(tag, w, s) for tag, w, s in runs]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    if TOP_FRAC != 0.30 or N_PATCHES != 8:
        pd.concat(TRIED, ignore_index=True).to_csv(
            OUT.replace('.csv', '_candidates.csv'), index=False)

    print('\nQ1 locating precision by isolation width (parabolic vertex)')
    print('  width   n   sigma_raw  sigma_detrended  within')
    for w, g in df.groupby('width_th'):
        print(f'  {w:>5} Th {len(g):>4}   {sigma_iqr(g.err_raw):>8.3f}   '
              f'{sigma_iqr(g.err):>13.3f}   {g.within.median():>6.3f}')
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
