"""Figure 2 input: retention time x Q1 ion-distribution grid of a ZT Scan run."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import sys
import time

import h5py
import hdf5plugin
import numpy as np

PREFIX = '20260905_Zeno2_Eno0_32p8min_TiHe_SA_H032_E376_'
PEAKS_PER_BLOCK = 50_000_000


def _spectrum_blocks(start, stop, n_spec):
    """Yield [b0, b1) spectrum ranges holding <= PEAKS_PER_BLOCK peaks."""
    b0 = 0
    while b0 < n_spec:
        b1 = int(np.searchsorted(start, start[b0] + PEAKS_PER_BLOCK))
        b1 = min(max(b1, b0 + 1), n_spec)
        yield b0, b1
        b0 = b1


def build(hdf_path):
    """(tic, tic_hi, npk, seen, rt1, q1_centres) on the cycle x Q1 grid."""
    t0 = time.time()
    with h5py.File(hdf_path, 'r') as h:
        g = h['ms_data']
        sg = g['spectrum_df']
        ms_level = sg['ms_level'][:]
        q1 = sg['precursor_mz'][:]
        rt = sg['rt'][:]
        start = sg['peak_start_idx'][:].astype(np.int64)
        stop = sg['peak_stop_idx'][:].astype(np.int64)
        mz_ds, int_ds = g['peak_df']['mz'], g['peak_df']['Intensity']
        n_spec = len(ms_level)
        is2 = ms_level == 2
        rt1 = np.sort(rt[ms_level == 1])
        q1_centres = np.unique(q1[is2])
        n_cyc, n_q1 = len(rt1), len(q1_centres)
        cyc_all = np.clip(np.searchsorted(rt1, rt, side='right') - 1,
                          0, n_cyc - 1)
        q1_idx_all = np.searchsorted(q1_centres, q1)
        n_cell = n_cyc * n_q1
        tic = np.zeros(n_cell)
        tic_hi = np.zeros(n_cell)
        npk = np.zeros(n_cell)
        seen = np.zeros(n_cell, dtype=bool)
        cnt_all = (stop - start).astype(np.int64)
        f2 = cyc_all[is2] * n_q1 + q1_idx_all[is2]
        npk += np.bincount(f2, weights=cnt_all[is2].astype(float),
                           minlength=n_cell)
        seen[f2] = True
        for n, (b0, b1) in enumerate(_spectrum_blocks(start, stop, n_spec)):
            sel = is2[b0:b1]
            if not sel.any():
                continue
            p0, p1 = int(start[b0]), int(stop[b1 - 1])
            inten = int_ds[p0:p1].astype(np.float64)
            cnt = cnt_all[b0:b1]
            off = (start[b0:b1] - p0)
            ok = cnt > 0
            s_all = np.zeros(b1 - b0)
            s_hi = np.zeros(b1 - b0)
            if ok.any():
                s_all[ok] = np.add.reduceat(inten, off[ok])
                mz = mz_ds[p0:p1]
                above = mz > np.repeat(q1[b0:b1], cnt).astype(np.float32)
                s_hi[ok] = np.add.reduceat(inten * above, off[ok])
            flat = cyc_all[b0:b1] * n_q1 + q1_idx_all[b0:b1]
            tic += np.bincount(flat[sel], weights=s_all[sel], minlength=n_cell)
            tic_hi += np.bincount(flat[sel], weights=s_hi[sel],
                                  minlength=n_cell)
            if n % 5 == 0:
                print(f'  block {n}: spectra {b1:,}/{n_spec:,}  '
                      f'{time.time() - t0:.0f}s', flush=True)
    shape = (n_cyc, n_q1)
    return (tic.reshape(shape), tic_hi.reshape(shape), npk.reshape(shape),
            seen.reshape(shape), rt1, q1_centres)


def main(tag='A9'):
    tic, tic_hi, npk, seen, rt1, q1c = build(
        cfg.raw('ztscan_full', f'{PREFIX}{tag}.hdf'))
    out = cfg.data('H032_E376', f'ztscan_rt_q1_map_{tag}_grid.npz')
    np.savez_compressed(out, tic=tic, tic_hi=tic_hi, npk=npk.astype(np.int32),
                        seen=seen, rt=rt1, q1=q1c)
    print(f'wrote {out}')


if __name__ == '__main__':
    main(*sys.argv[1:2])
