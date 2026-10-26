"""Q1 locating precision of EVERY identified precursor, over the WHOLE file."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_q1_precision as M
from ztscan_patch import read_q1_window, patch_from_window

W, cp = M.W, M.cp
OUT_DIR = os.path.join(cfg.DATA_ROOT, 'H032_E376', 'q1_wholefile')
TILE_STEP = 5
Q1_BLOCK = 20.0


def arm_ions(d):
    """{(seq, z): b/y ion m/z array} for every precursor, one by_ions call."""
    keys = list(d.groupby(['seq', 'z']).size().index)
    ions = M.by_ions([k[0] for k in keys], [int(k[1]) for k in keys])[0]
    return {k: np.asarray(v, float).ravel() for k, v in zip(keys, ions)}


def unique_mask(ion_lists):
    """Per precursor: which of its ions no OTHER precursor has within SHARED_TOL."""
    owner = np.concatenate([np.full(len(v), i) for i, v in
                            enumerate(ion_lists)]) if ion_lists else []
    allm = np.concatenate(ion_lists) if ion_lists else np.array([])
    o = np.argsort(allm, kind='stable')
    ms, ow = allm[o], np.asarray(owner)[o]
    lo = np.searchsorted(ms, ms - W.SHARED_TOL, side='left')
    hi = np.searchsorted(ms, ms + W.SHARED_TOL, side='right')
    shared = np.zeros(len(ms), bool)
    for k in np.flatnonzero(hi - lo > 1):
        sl = slice(lo[k], hi[k])
        near = np.abs(ms[sl] - ms[k]) < W.SHARED_TOL
        if np.any(ow[sl][near] != ow[k]):
            shared[k] = True
    back = np.empty(len(ms), bool)
    back[o] = ~shared
    out, a = [], 0
    for v in ion_lists:
        out.append(back[a:a + len(v)])
        a += len(v)
    return out


def match_rows(patch_sorted, fmz):
    """W.frag_rows, vectorised: (fragment m/z, patch row) per fragment found."""
    ms, order, inten = patch_sorted
    lo = np.searchsorted(ms, fmz - W.FRAG_TOL, side='left')
    hi = np.searchsorted(ms, fmz + W.FRAG_TOL, side='right')
    out = []
    for t, a, b in zip(fmz, lo, hi):
        if b > a:
            j = a + int(np.argmax(inten[a:b]))
            out.append((float(t), int(order[j])))
    return out


def place(qs, H):
    """Gates + parabolic vertex for one precursor's fragment Q1 profiles, under whatever W.MIN_* are set."""
    c_gate = W.centroids(qs, H)[0]
    sn = np.isfinite(c_gate)
    ok = sn & (W.consensus_cos(H, sn) >= W.MIN_COS)
    if ok.sum() < W.MIN_FRAG:
        return None
    cc = M.parab_centroids(qs, H[ok])
    cc = cc[np.isfinite(cc)]
    return cc if len(cc) >= W.MIN_FRAG else None


def set_gates(name):
    for k, v in M.GATES[name].items():
        setattr(W, k, v)


def tile_rows(patch, qs, cand, assigned, ions, pct, tag, width, rt_c, q0,
              verify=False):
    """Place every ASSIGNED precursor of one tile under both gate sets."""
    keys = list(zip(cand['seq'], cand['z']))
    ilists = [ions[k] for k in keys]
    umask = unique_mask(ilists)
    o = np.argsort(patch['mz'], kind='stable')
    ps = (patch['mz'][o], o, patch['intensity'][o])
    rows = []
    for k_i, r in enumerate(cand.itertuples()):
        if not assigned[k_i]:
            continue
        u = ilists[k_i][umask[k_i]]
        fr = match_rows(ps, u)
        if verify:
            others = np.concatenate([ilists[j] for j in range(len(ilists))
                                     if j != k_i]) if len(ilists) > 1 \
                else np.array([])
            u0 = (ilists[k_i][np.min(np.abs(ilists[k_i][:, None]
                                            - others[None, :]), axis=1)
                              > W.SHARED_TOL] if len(others) else ilists[k_i])
            assert np.array_equal(np.sort(u0), np.sort(u)), 'uniqueness'
            assert W.frag_rows(patch, u) == fr, 'frag_rows'
        row = dict(run=tag, width_th=width, tile=f'rt{rt_c:.2f}_q{q0:.0f}',
                   seq=r.seq, z=int(r.z), mz=float(r.mz), rt=float(r.rt),
                   area=float(r.area),
                   area_pct=float(pct.get((r.seq, r.z), np.nan)),
                   n_unique=int(len(u)), n_found=int(len(fr)))
        H = patch['q1_prof'][[x[1] for x in fr]] if fr else None
        for g in ('strict', 'relaxed'):
            set_gates(g)
            cc = place(qs, H) if (H is not None
                                  and len(fr) >= W.MIN_FRAG) else None
            row[f'n_frag_{g}'] = 0 if cc is None else int(len(cc))
            row[f'err_raw_{g}'] = (np.nan if cc is None
                                   else float(np.median(cc) - r.mz))
            row[f'within_{g}'] = (np.nan if cc is None
                                  else M.sigma_iqr(cc))
        rows.append(row)
    return rows


def measure_arm(tag, width, search, *, max_windows=None, max_peaks=None,
                verify=False):
    hdf = cfg.raw('ztscan_full', f'{M.RUN_PREFIX}{tag}.hdf')
    cp.FEAT = os.path.join(M.ROOT, search, 'lfq.dia.features.csv')
    cp.TAG, cp.RUN = tag, M.RUN_PREFIX
    d = cp.identified()
    d = (d.groupby(['seq', 'z'])
         .agg(mz=('mz', 'median'), rt=('rt', 'median'), area=('area', 'max'))
         .reset_index())
    pct = d.set_index(['seq', 'z'])['area'].rank(pct=True) * 100
    t0 = time.time()
    ions = arm_ions(d)
    print(f'  {tag} ({width} Th): {len(d):,} precursors, ions in '
          f'{time.time() - t0:.0f} s', flush=True)
    axis = M.load_axis(tag)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'{tag}.csv')
    starts = np.arange(np.floor(d.mz.min() / Q1_BLOCK) * Q1_BLOCK,
                       d.mz.max() + Q1_BLOCK, Q1_BLOCK)
    if max_windows:
        dens = [((d.mz >= q) & (d.mz < q + Q1_BLOCK)).sum() for q in starts]
        starts = starts[np.argsort(dens)[::-1][:max_windows]]
        out_path = out_path.replace('.csv', f'_test{max_windows}.csv')
    if os.path.exists(out_path):
        raise SystemExit(f'{out_path} exists -- tag it deleteme_ first; '
                         f'this script never overwrites a measurement')
    total = 0
    for wi, q0 in enumerate(starts):
        sub = d[(d.mz >= q0) & (d.mz < q0 + Q1_BLOCK)]
        if not len(sub):
            continue
        t_r = time.time()
        win = read_q1_window(hdf, float(q0), float(q0 + Q1_BLOCK),
                             float(sub.rt.min() - cp.RT_TOL),
                             float(sub.rt.max() + cp.RT_TOL),
                             n_cycles=M.N_CYCLES)
        t_r = time.time() - t_r
        if win is None:
            continue
        rt1 = win['rt1']
        centres = rt1[M.N_CYCLES // 2::TILE_STEP]
        near = np.abs(sub.rt.to_numpy()[:, None] - centres[None, :])
        home = np.argmin(near, axis=1)
        rows, t_b, t_p = [], 0.0, 0.0
        for ti in np.unique(home):
            rt_c = float(centres[ti])
            t1 = time.time()
            kw = {} if max_peaks is None else {'max_peaks': max_peaks}
            patch = patch_from_window(win, rt_c, n_cycles=M.N_CYCLES, **kw)
            t_b += time.time() - t1
            if patch is None or not len(patch['mz']):
                continue
            t1 = time.time()
            corr = (M.q1_correction(axis, patch['q1_axis'], rt_c)
                    if axis is not None else 0.0)
            qs = patch['q1_axis'] - corr
            cand = sub[np.abs(sub.rt - rt_c) <= cp.RT_TOL]
            assigned = (home[np.abs(sub.rt - rt_c).to_numpy() <= cp.RT_TOL]
                        == ti) if len(cand) else []
            if len(cand):
                rows += tile_rows(patch, qs, cand.reset_index(drop=True),
                                  np.asarray(assigned), ions, pct, tag, width,
                                  rt_c, q0, verify=verify)
            t_p += time.time() - t1
        part = pd.DataFrame(rows)
        part.to_csv(out_path, mode='a', header=not os.path.exists(out_path),
                    index=False)
        total += len(part)
        n_s = int((part.get('n_frag_strict', pd.Series(dtype=int)) > 0).sum())
        print(f'    window {wi + 1}/{len(starts)} {q0:.0f} Th: '
              f'{len(sub):,} precursors, {len(np.unique(home))} tiles | '
              f'read {t_r:.0f} s, build {t_b:.0f} s, place {t_p:.0f} s | '
              f'{n_s} strict-placed', flush=True)
        del win
    print(f'  {tag}: {total:,} rows -> {out_path}')
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='A9')
    ap.add_argument('--windows', type=int, default=0,
                    help='test: only the N densest Q1 windows')
    ap.add_argument('--max-peaks', type=int, default=None)
    ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()
    for tag, width, search in M.RUNS:
        if tag in a.runs.split(','):
            measure_arm(tag, width, search, max_windows=a.windows or None,
                        max_peaks=a.max_peaks, verify=a.verify)


if __name__ == '__main__':
    main()
