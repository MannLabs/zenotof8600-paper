"""Four ways to read a Q1 position off a transmission profile, measured."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, _os.path.join(cfg.REPO_ROOT, 'workflows'))
sys.path.insert(0, _os.path.join(cfg.REPO_ROOT, 'experiments', 'H032_E376'))
import e376_targets as cp
from e376_targets import HDF, ROOT, TAG
from e376_targets import RT_TOL, Q1_WIDTH, identified
from figure_q1_resolves_window import (CENT_HALF, MIN_COS,
                                       MIN_FRAG, MIN_SNR, SHARED_TOL,
                                       centroids_interp, centroids_grid,
                                       consensus_cos, frag_rows)
from zenotof8600_analytics.ztscan.profile import peak_index
from ztscan_common import by_ions, load_axis, q1_correction
from ztscan_patch import extract_patch

N_PATCHES = 8
N_CYCLES = 11
TOP_FRAC = 0.30
OUT = os.path.join(ROOT, 'ztscan_api', 'figures', 'paper')


def gate(qs, H):
    """The S/N gate, shared by every estimator so only the rule differs."""
    ok = np.zeros(len(H), bool)
    base = np.zeros(len(H))
    for k, prof in enumerate(H):
        if prof.max() <= 0:
            continue
        j = int(np.argmax(prof))
        off = np.abs(qs - qs[j]) > CENT_HALF
        b = np.median(prof[off]) if off.sum() >= 5 else 0.0
        base[k] = b
        ok[k] = (prof.max() / b >= MIN_SNR) if b > 0 else True
    return ok, base


def parab_centroids(qs, H):
    """The whole-file Q1 measurement's rule: parabolic vertex at the argmax."""
    idx = peak_index(np.asarray(H, float))
    return np.interp(idx, np.arange(len(qs)), qs)


def wmean_centroids(qs, H, base):
    """Plain intensity-weighted mean of the whole profile, baseline removed."""
    out = np.full(len(H), np.nan)
    for k, prof in enumerate(H):
        w = np.clip(prof - base[k], 0, None)
        if w.sum() > 0:
            out[k] = float((qs * w).sum() / w.sum())
    return out


def sig(x):
    x = pd.Series(x).dropna()
    if len(x) < 4:
        return np.nan
    return float((x.quantile(0.75) - x.quantile(0.25)) / 1.349)


def main():
    d = identified()
    cut = d.groupby(['seq', 'z']).area.max().quantile(1 - TOP_FRAC)
    axis = load_axis(TAG)
    cp.N_PATCHES = N_PATCHES
    rows = []
    for n, rt, q0 in cp.choose_patches(d):
        patch = extract_patch(HDF, rt, q0, q0 + Q1_WIDTH, n_cycles=N_CYCLES)
        if patch is None or not len(patch['mz']):
            continue
        qs = patch['q1_axis'] - q1_correction(axis, patch['q1_axis'], rt)
        sel = d[(np.abs(d.rt - rt) <= RT_TOL) & (d.mz >= q0)
                & (d.mz <= q0 + Q1_WIDTH)]
        sel = (sel.groupby(['seq', 'z'])
               .agg(mz=('mz', 'median'), area=('area', 'max')).reset_index())
        sel = sel[sel.area >= cut]
        if len(sel) < 2:
            continue
        ions = [np.asarray(by_ions([r.seq], [int(r.z)])[0], float).ravel()
                for r in sel.itertuples()]
        for k, r in enumerate(sel.itertuples()):
            others = np.concatenate([ions[j] for j in range(len(ions))
                                     if j != k])
            u = ions[k][np.min(np.abs(ions[k][:, None] - others[None, :]),
                               axis=1) > SHARED_TOL]
            fr = frag_rows(patch, u)
            if len(fr) < MIN_FRAG:
                continue
            H = patch['q1_prof'][[x[1] for x in fr]]
            ok0, base = gate(qs, H)
            est = {
                'interp': centroids_interp(qs, H)[0],
                'parab': np.where(ok0, parab_centroids(qs, H), np.nan),
                'wmean': np.where(ok0, wmean_centroids(qs, H, base), np.nan),
                'grid': centroids_grid(qs, H)[0],
            }
            sn = np.isfinite(est['grid'])
            ok = sn & (consensus_cos(H, sn) >= MIN_COS)
            if ok.sum() < MIN_FRAG:
                continue
            row = dict(patch=f'rt{rt:.0f}_q{q0:.0f}', seq=r.seq, z=int(r.z),
                       mz=float(r.mz), n_frag=int(ok.sum()))
            for name, v in est.items():
                w = v[ok]
                w = w[np.isfinite(w)]
                row[f'err_{name}'] = (float(np.median(w) - r.mz) if len(w)
                                      else np.nan)
                row[f'within_{name}'] = sig(w)
                row[f'n_{name}'] = len(w)
            rows.append(row)
        print(f'  rt{rt:.0f}_q{q0:.0f}: {len(rows)} precursors', flush=True)
        del patch
    r = pd.DataFrame(rows)
    if not len(r):
        print('nothing placed')
        return
    r.to_csv(os.path.join(OUT, 'q1_estimator_compare.csv'), index=False)
    print(f'\n{len(r)} abundant precursors, {r.patch.nunique()} patches, '
          f'scored against known precursor m/z\n')
    print(f'{"estimator":>10s} {"used by":>34s} {"median":>8s} '
          f'{"ACROSS":>8s} {"within":>8s} {"|e|<=0.25":>10s} {"frags":>6s}')
    who = {'interp': 'Q1 calibration + paper figure',
           'parab': 'whole-file Q1 measurement',
           'wmean': 'intensity-weighted mean',
           'grid': 'retired - was figure only'}
    best = None
    for name in ('interp', 'parab', 'wmean', 'grid'):
        e = r[f'err_{name}'].dropna()
        a = sig(e)
        print(f'{name:>10s} {who[name]:>34s} {e.median():+8.3f} {a:8.3f} '
              f'{r[f"within_{name}"].median():8.3f} '
              f'{(e.abs() <= 0.25).mean():10.1%} {int(r[f"n_{name}"].sum()):6d}')
        if best is None or a < best[1]:
            best = (name, a)
    print(f'\n  best ACROSS-precursor: {best[0]} at {best[1]:.3f} Th')
    print('\n  "within" is the fragment-to-fragment scatter inside one '
          'peptide. A rule can\n  make it small by putting every fragment in '
          'the same wrong place, so it is\n  not the figure of merit - '
          '"ACROSS" is.')

    e = r['err_interp'].dropna()
    se = sig(e) / np.sqrt(len(e))
    print(f'\n  figure vs calibration residual: median {e.median():+.3f} Th '
          f'+- {se:.3f} (s.e.)')
    print(f'  -> {"still a real offset, so the constant is needed" if abs(e.median()) > 2 * se else "consistent with zero; the constant could be dropped"}')


if __name__ == '__main__':
    main()
