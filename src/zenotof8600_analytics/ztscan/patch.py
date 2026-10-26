"""Extract a Q1 x RT x m/z patch of ZT Scan data, with a profile per peak."""

from __future__ import annotations

import h5py
import hdf5plugin
import numpy as np
import time as _time

from ..raw import centroid_profile, merge_peaks

MAX_PEAKS = 20000

RDCC_NBYTES = 512 * 1024 ** 2
RDCC_NSLOTS = 100003

MZ_TOL = 0.02
MIN_PEAK_INTENSITY = 0.0


def extract_patch(hdf_path, rt_centre, q1_lo, q1_hi, *, n_cycles=11,
                  mz_range=(200.0, 1800.0), max_peaks=MAX_PEAKS):
    """All peaks in one Q1 x RT patch, each with a Q1 and an RT profile.

    Parameters
      rt_centre:  minutes; the patch spans ``n_cycles`` cycles around it.
      q1_lo/hi:   the Q1 range to take, in Th.
      max_peaks:  keep only this many peaks, strongest first. A patch holds
                  thousands of lattice samples; the cut is on intensity alone.
    """
    win = read_q1_window(hdf_path, q1_lo, q1_hi, rt_centre, rt_centre,
                         n_cycles=n_cycles, mz_range=mz_range)
    if win is None:
        return None
    return patch_from_window(win, rt_centre, n_cycles=n_cycles,
                             max_peaks=max_peaks)


def read_q1_window(hdf_path, q1_lo, q1_hi, rt_lo, rt_hi, *, n_cycles=11,
                   mz_range=(200.0, 1800.0), progress=False):
    """Every MS2 spectrum of one Q1 window over an RT range, read ONCE."""
    with h5py.File(hdf_path, 'r', rdcc_nbytes=RDCC_NBYTES,
                   rdcc_nslots=RDCC_NSLOTS) as h:
        g = h['ms_data']
        sg = g['spectrum_df']
        ml = sg['ms_level'][:]
        q1 = sg['precursor_mz'][:]
        rt = sg['rt'][:]
        st = sg['peak_start_idx'][:].astype(np.int64)
        sp = sg['peak_stop_idx'][:].astype(np.int64)
        mz_ds, in_ds = g['peak_df']['mz'], g['peak_df']['Intensity']

        rt1 = np.sort(rt[ml == 1])
        pad = n_cycles * float(np.median(np.diff(rt1))) if len(rt1) > 1 else 0.0
        sel = np.where((ml == 2) & (rt >= rt_lo - pad) & (rt <= rt_hi + pad)
                       & (q1 >= q1_lo) & (q1 <= q1_hi))[0]
        if sel.size == 0:
            return None
        sel = sel[np.argsort(st[sel], kind='stable')]
        mzs, ins, keep_spec = [], [], []
        step = max(len(sel) // 20, 1)
        t0 = _time.time()
        for n, i in enumerate(sel):
            if progress and n and n % step == 0:
                el = _time.time() - t0
                print(f'      read {n:,}/{len(sel):,} spectra '
                      f'({n / len(sel):.0%}, {el:.0f} s, '
                      f'~{el / n * (len(sel) - n):.0f} s left)', flush=True)
            a, b = int(st[i]), int(sp[i])
            if b <= a:
                continue
            m, v = mz_ds[a:b], in_ds[a:b]
            k = (m >= mz_range[0]) & (m <= mz_range[1])
            if not k.any():
                continue
            mzs.append(m[k])
            ins.append(v[k])
            keep_spec.append(i)
    if not mzs:
        return None
    keep_spec = np.asarray(keep_spec)
    rt_order = np.argsort(rt[keep_spec], kind='stable')
    keep_spec = keep_spec[rt_order]
    counts = np.array([len(mzs[i]) for i in rt_order], dtype=np.int64)
    return dict(mz=np.concatenate([mzs[i] for i in rt_order]),
                inten=np.concatenate([ins[i] for i in rt_order]),
                off=np.concatenate(([0], np.cumsum(counts))),
                s_rt=rt[keep_spec], s_q1=q1[keep_spec], rt1=rt1)


def patch_from_window(win, rt_centre, *, n_cycles=11, max_peaks=MAX_PEAKS):
    """One patch, built from an already-read Q1 window."""
    rt1 = win['rt1']
    k = int(np.argmin(np.abs(rt1 - rt_centre)))
    lo_c = max(0, k - n_cycles // 2)
    hi_c = min(len(rt1), lo_c + n_cycles)
    rt_axis = rt1[lo_c:hi_c]
    if len(rt_axis) < 2:
        return None
    edges = np.append(rt_axis, rt_axis[-1] + (rt_axis[-1] - rt_axis[-2]))

    s_rt = win['s_rt']
    lo = int(np.searchsorted(s_rt, edges[0], side='left'))
    hi = int(np.searchsorted(s_rt, edges[-1], side='left'))
    if hi <= lo:
        return None
    q1_axis = np.unique(win['s_q1'][lo:hi])
    cyc = np.clip(np.searchsorted(edges, s_rt[lo:hi], side='right') - 1,
                  0, len(rt_axis) - 1)
    qi = np.searchsorted(q1_axis, win['s_q1'][lo:hi])

    a, b = int(win['off'][lo]), int(win['off'][hi])
    if b <= a:
        return None
    mz_all = win['mz'][a:b]
    in_all = win['inten'][a:b].astype(np.float64)
    spec_all = np.repeat(np.arange(hi - lo, dtype=np.int32),
                         np.diff(win['off'][lo:hi + 1]))

    u_mz, u_in = merge_peaks(mz_all, in_all, tol=1.0, tol_unit='ppm',
                             position='median', scan_id=spec_all)
    pk_mz, pk_int = centroid_profile(u_mz, u_in)[:2]
    if len(pk_mz) == 0:
        return None
    order = np.argsort(pk_int)[::-1][:max_peaks]
    pk_mz, pk_int = pk_mz[np.sort(order)], pk_int[np.sort(order)]

    idx = np.searchsorted(pk_mz, mz_all).clip(1, len(pk_mz) - 1)
    left = np.abs(pk_mz[idx - 1] - mz_all)
    right = np.abs(pk_mz[idx] - mz_all)
    nearest = np.where(left < right, idx - 1, idx)
    ok = np.minimum(left, right) <= MZ_TOL
    nearest, w, s = nearest[ok], in_all[ok], spec_all[ok]
    nq, nc, npk = len(q1_axis), len(rt_axis), len(pk_mz)
    q1_prof = np.bincount(nearest * nq + qi[s], weights=w,
                          minlength=npk * nq).reshape(npk, nq)
    rt_prof = np.bincount(nearest * nc + cyc[s], weights=w,
                          minlength=npk * nc).reshape(npk, nc)
    return dict(mz=pk_mz, intensity=pk_int, q1_prof=q1_prof, rt_prof=rt_prof,
                q1_axis=q1_axis, rt_axis=rt_axis, n_spectra=int(hi - lo))
