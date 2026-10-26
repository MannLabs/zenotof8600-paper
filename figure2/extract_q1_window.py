"""Cache everything figure 2's Q1-window panel needs, so it never reads the HDF."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import argparse
import json
import os
import sys

import numpy as np

ZT = _os.path.join(cfg.REPO_ROOT, 'experiments', 'H032_E376')
sys.path.insert(0, _os.path.join(cfg.REPO_ROOT, 'workflows'))
sys.path.insert(0, ZT)

import figure_q1_resolves_window as W
from ztscan_common import by_ions, load_axis, q1_correction

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(cfg.DATA_ROOT, 'H032_E376', 'q1_window_A9.npz')


def main(which='old'):
    global OUT
    if which == 'wholefile':
        sys.path.insert(0, HERE)
        import figure2 as f2
        with open(f2.Q1_AXIS_WHOLEFILE) as fh:
            axis = json.load(fh)
        bias = 0.0
        OUT = OUT.replace('.npz', '_wholefile.npz')
        if os.path.exists(OUT):
            raise SystemExit(f'{OUT} exists -- tag it deleteme_ first')
    else:
        axis, bias = load_axis(W.TAG), W.ESTIMATOR_BIAS
    info = W.true_mz()
    order = np.argsort([m for m, _ in info])
    peps = [W.PEPTIDES[i] for i in order]
    mzs = np.array([info[i][0] for i in order])
    areas = np.array([info[i][1] for i in order])
    q_cen = 0.5 * (mzs.min() + mzs.max())
    win_lo, win_hi = q_cen - W.DIA_WIN / 2, q_cen + W.DIA_WIN / 2
    q_lo, q_hi = win_lo - W.DRAW_PAD, win_hi + W.DRAW_PAD

    ions, izs = [], []
    for s, z in peps:
        m, _, zc = by_ions([s], [z], with_charge=True)
        ions.append(np.asarray(m[0], float).ravel())
        izs.append(np.asarray(zc[0], int).ravel())
    uniq = []
    for k in range(len(peps)):
        others = np.concatenate([ions[j] for j in range(len(peps)) if j != k])
        m = np.min(np.abs(ions[k][:, None] - others[None, :]),
                   axis=1) > W.SHARED_TOL
        uniq.append(ions[k][m])

    patch = W.load_patch(mzs.min() - W.PATCH_PAD, mzs.max() + W.PATCH_PAD)
    qs = (patch['q1_axis'] - q1_correction(axis, patch['q1_axis'], W.RT_APEX)
          - bias)

    out = {'axis': np.array(which), 'qs': qs, 'mzs': mzs, 'areas': areas,
           'win_lo': win_lo, 'win_hi': win_hi, 'q_lo': q_lo, 'q_hi': q_hi,
           'q_cen': q_cen, 'dia_win': W.DIA_WIN, 'rt_apex': W.RT_APEX,
           'sigma_a9': W.SIGMA_A9,
           'labels': np.array([f'{s} {z}+' for s, z in peps])}

    FM, H, C, OK = [], [], [], []
    for k in range(len(peps)):
        rows = W.frag_rows(patch, uniq[k])
        fm = np.array([r[0] for r in rows])
        h = (patch['q1_prof'][[r[1] for r in rows]] if rows
             else np.zeros((0, len(qs))))
        c = W.centroids(qs, h)[0] if len(h) else np.zeros(0)
        sn = np.isfinite(c)
        co = W.consensus_cos(h, sn) if len(h) else np.zeros(0)
        ok = sn & (co >= W.MIN_COS)
        FM.append(fm)
        H.append(h)
        C.append(c)
        OK.append(ok)
        print(f'  {peps[k][0]:>18s} {peps[k][1]}+ {mzs[k]:8.3f}  '
              f'{len(fm):3d} in patch, {int(ok.sum()):3d} pass S/N + cosine')

    keep = [k for k in range(len(peps)) if OK[k].sum() >= W.MIN_FRAG]
    med = np.array([np.median(C[k][OK[k]]) for k in keep])
    ordk = [keep[i] for i in np.argsort(med)]
    out['kept'] = np.array(ordk)

    for k in ordk:
        out[f'p{k}_profiles'] = H[k][OK[k]]
        out[f'p{k}_centroids'] = C[k][OK[k]]
        out[f'p{k}_frag_mz'] = FM[k][OK[k]]

    ms1 = W.ms1_at_apex(q_lo, q_hi)
    if ms1 is not None:
        pm, pv, cm = ms1
        out['ms1_mz'], out['ms1_int'] = pm, pv
        out['ms1_centroids'] = cm[(cm >= q_lo) & (cm <= q_hi)]

    inwin = (qs >= win_lo) & (qs <= win_hi)
    tot_int = patch['q1_prof'][:, inwin].sum(axis=1)
    shown = ((patch['mz'] >= W.FRAG_MZ_LO) & (patch['mz'] <= W.FRAG_MZ_HI)
             & (tot_int > 0))
    out['spec_mz'] = patch['mz'][shown]
    out['spec_int'] = tot_int[shown]
    out['frag_mz_lo'], out['frag_mz_hi'] = W.FRAG_MZ_LO, W.FRAG_MZ_HI
    for k in ordk:
        rows = [r for r in W.frag_rows(patch, FM[k][OK[k]])
                if W.FRAG_MZ_LO <= r[0] <= W.FRAG_MZ_HI and tot_int[r[1]] > 0]
        out[f'p{k}_spec_mz'] = np.array([patch['mz'][r[1]] for r in rows])
        out[f'p{k}_spec_int'] = np.array([tot_int[r[1]] for r in rows])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f'\n{len(ordk)} peptides placed; wrote {OUT} '
          f'({os.path.getsize(OUT) / 1e6:.1f} MB)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--axis', choices=('old', 'wholefile'), default='old')
    main(ap.parse_args().axis)
