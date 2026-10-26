"""Dump the fragment peaks around identified precursors, per RT slice."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, _os.path.join(cfg.REPO_ROOT, 'workflows'))
import zenotof8600_analytics as core

import precursor_mass_q1_vs_ms1 as pm
from precursor_mass_q1_vs_ms1 import select_window, stream_spectra

from ztscan_common import OUT_DIR, by_ions, load_axis, q1_correction


Q1_PATCH_FWHM = 2.5
CYC_HALF = 3
SEED_N = 20
APEX_CYC = 1
CAL_AXIS = None
MIN_PEAK = 1000.0
MIN_Q1_POINTS = 4
MZ_TOL = 0.02

LATTICE_AT_500 = 4.36e-3
GAP_FACTOR = 2.0

PICK_TOL, PICK_COS = 0.28, 0.93
SLICE_WIDTH = 1.2
DUMP_N = 200
DUMP_SUBDIR = 'q1_patch_dumps'


def lattice_step(mz):
    """Native TOF sample spacing at m/z."""
    return LATTICE_AT_500 * np.sqrt(np.asarray(mz, float) / 500.0)


def centroid_profile(mz, inten, qw=None, extra=()):
    """Collapse native PROFILE samples into centroided peaks."""
    if not len(mz):
        e = np.array([])
        base = (e, e, e, e)
        return base + ([np.array([]) for _ in extra],) if extra else base
    brk = np.where(np.diff(mz) > GAP_FACTOR * lattice_step(mz[:-1]))[0] + 1
    starts = np.concatenate(([0], brk)).astype(int)
    tot = np.add.reduceat(inten, starts)
    good = tot > 0
    npts = np.diff(np.concatenate((starts, [len(mz)])))
    hm = inten >= 0.5 * np.repeat(np.maximum.reduceat(inten, starts), npts)
    wsum = np.add.reduceat(inten * hm, starts)
    use = good & (wsum > 0)
    mzc = np.where(use, np.add.reduceat(inten * mz * hm, starts),
                   np.add.reduceat(inten * mz, starts))
    den = np.where(use, wsum, tot)
    mzc = mzc[good] / den[good]
    npts = npts[good]
    q1c = (np.add.reduceat(qw, starts)[good] / tot[good]
           if qw is not None else np.full(int(good.sum()), np.nan))
    if not extra:
        return mzc, tot[good], q1c, npts
    ex = [np.add.reduceat(a, starts, axis=0)[good] for a in extra]
    return mzc, tot[good], q1c, npts, ex


def collect(hdf_path, w, ids, fwhm0):
    """Per precursor: every peak in its (Q1 patch x elution cycles) region."""
    q1_axis = np.unique(w['q1'])
    step = float(np.median(np.diff(q1_axis)))
    n_cyc = int(w['cycle'].max()) + 1
    n_prof = 2 * CYC_HALF + 1
    rt_axis = np.array([w['rt'][w['cycle'] == c].mean() for c in range(n_cyc)])
    qh = int(np.ceil(Q1_PATCH_FWHM * fwhm0 / step))

    theo = ids.theo_mz.to_numpy(float)
    qi0 = np.searchsorted(q1_axis, theo).clip(0, len(q1_axis) - 1)
    ci0 = np.abs(rt_axis[None, :]
                 - ids.rt.to_numpy(float)[:, None]).argmin(axis=1)

    bucket = [[] for _ in range(len(ids))]
    qw = max(1, int(round(0.5 * fwhm0 / step)))
    win = [[] for _ in range(len(ids))]
    t0, done, kept = time.time(), 0, 0
    for k, m, v in stream_spectra(hdf_path, w):
        done += 1
        qi = int(np.searchsorted(q1_axis, w['q1'][k]))
        if qi >= len(q1_axis) or q1_axis[qi] != w['q1'][k] or not len(m):
            continue
        ci = int(w['cycle'][k])
        keep = v >= MIN_PEAK
        if not keep.any():
            continue
        mk, vk = m[keep].astype(np.float32), v[keep].astype(np.float32)
        lo = np.searchsorted(qi0, qi - qh)
        hi = np.searchsorted(qi0, qi + qh, side='right')
        for p in range(lo, hi):
            coff = ci - int(ci0[p])
            if abs(coff) > CYC_HALF:
                continue
            near = abs(qi - qi0[p]) <= qw
            bucket[p].append((mk, vk, float(q1_axis[qi]), coff, near))
            kept += len(mk)
            if near:
                win[p].append((mk, vk))
        if done % 25000 == 0:
            print(f'    {done:,}/{len(w["idx"]):,} spectra  {kept:,} peak '
                  f'copies  {time.time() - t0:.0f}s', flush=True)

    rows = []
    for p in range(len(ids)):
        if not bucket[p]:
            rows.append(None)
            continue
        lens = [len(b[0]) for b in bucket[p]]
        mm = np.concatenate([b[0] for b in bucket[p]])
        vv = np.concatenate([b[1] for b in bucket[p]]).astype(np.float64)
        qq = np.repeat(np.array([b[2] for b in bucket[p]]), lens)
        cc = np.repeat(np.array([b[3] for b in bucket[p]], np.int64), lens)
        nn = np.repeat(np.array([b[4] for b in bucket[p]], bool), lens)
        bucket[p] = None
        order = np.argsort(mm, kind='stable')
        mm, vv, qq, cc, nn = mm[order], vv[order], qq[order], cc[order], nn[order]
        u, inv = np.unique(mm, return_inverse=True)
        nu = len(u)

        ap = np.abs(cc) <= APEX_CYC
        isum = np.bincount(inv[ap], weights=vv[ap], minlength=nu)
        iq = np.bincount(inv[ap], weights=(vv * qq)[ap], minlength=nu)
        sel = nn
        rt = np.bincount(inv[sel] * n_prof + (cc[sel] + CYC_HALF),
                         weights=vv[sel],
                         minlength=nu * n_prof).reshape(nu, n_prof)
        both = ap & nn
        ipk = np.bincount(inv[both], weights=vv[both], minlength=nu)
        npts = np.bincount(inv[ap], minlength=nu)

        ok = (npts >= MIN_Q1_POINTS) & (isum > 0)
        mzc, tot, q1c, nlat, rtc = centroid_profile(
            u[ok].astype(float), isum[ok], iq[ok],
            extra=(ipk[ok], rt[ok]))
        rows.append({'tab': pd.DataFrame({'mz': mzc, 'intensity': rtc[0],
                                          'q1_centroid': q1c,
                                          'patch_intensity': tot,
                                          'n_lattice': nlat}),
                     'rt': rtc[1]})

    base = []
    for p in range(len(ids)):
        if not win[p]:
            base.append(None)
            continue
        mm = np.concatenate([b[0] for b in win[p]])
        vv = np.concatenate([b[1] for b in win[p]]).astype(np.float64)
        win[p] = None
        o = np.argsort(mm, kind='stable')
        u, inv = np.unique(mm[o], return_inverse=True)
        isum = np.bincount(inv, weights=vv[o])
        mzc, tot, _, _ = centroid_profile(u.astype(float), isum)
        base.append(pd.DataFrame({'mz': mzc, 'intensity': tot}))
    return rows, base


def is_theoretical(mz, wanted, tol=MZ_TOL):
    """Boolean per observed peak: does it match any wanted m/z within tol."""
    if not len(wanted) or not len(mz):
        return np.zeros(len(mz), bool)
    j = np.searchsorted(wanted, mz)
    out = np.zeros(len(mz), bool)
    for d in (-1, 0):
        k = np.clip(j + d, 0, len(wanted) - 1)
        out |= np.abs(wanted[k] - mz) <= tol
    return out


def annotate(rows, ids, targets):
    """Per precursor: peak m/z, intensity, raw Q1 deviation, b/y flag."""
    out = []
    for p in range(len(ids)):
        r = rows[p]
        if r is None or not len(r['tab']):
            out.append(None)
            continue
        mzp = float(ids.theo_mz.iloc[p])
        t = r['tab'].copy()
        t['dev'] = t.q1_centroid.to_numpy() - mzp
        t['is_by'] = is_theoretical(t.mz.to_numpy(), targets[p])
        t['prec_mz'] = mzp
        t['prec_rt'] = float(ids.rt.iloc[p])
        v = t.intensity.to_numpy(float)
        t['rel'] = v / v.max() if v.max() > 0 else 0.0
        out.append(t)
    return out


def add_rt_score(rows, ann, axis, q1_tol):
    """Co-elution score: cosine of each peak's elution profile against the precursor's, in the SAME (Q1 x RT) pass."""
    for p, t in enumerate(ann):
        if t is None:
            continue
        prof = rows[p]['rt']
        dev = t.dev_cal.to_numpy(float)
        seed = np.where(np.abs(dev) <= 0.5 * q1_tol)[0]
        if len(seed) < 3:
            t['rt_cos'] = np.nan
            continue
        seed = seed[np.argsort(t.intensity.to_numpy()[seed])[::-1][:SEED_N]]
        P = prof[seed]
        nrm = np.linalg.norm(P, axis=1)
        P = P[nrm > 0] / nrm[nrm > 0, None]
        if len(P) < 3:
            t['rt_cos'] = np.nan
            continue
        ref = np.median(P, axis=0)
        n = np.linalg.norm(ref)
        if n <= 0:
            t['rt_cos'] = np.nan
            continue
        ref = ref / n
        norms = np.linalg.norm(prof, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            t['rt_cos'] = np.where(norms > 0, prof @ ref / norms, np.nan)


def apply_axis(ann, axis):
    """Stamp the calibrated Q1 deviation onto every peak, once."""
    for t in ann:
        if t is None or not len(t):
            continue
        t['dev_cal'] = t.dev - q1_correction(
            axis, t.prec_mz.to_numpy(float), t.prec_rt.to_numpy(float))
    return ann


def fit_axis(ann):
    """Affine Q1 mass axis from the peaks that ARE known b/y ions."""
    d = pd.concat([t[t.is_by] for t in ann if t is not None])
    d = d[np.abs(d.dev) < 3.0]
    b = core.binned_stats(d.prec_mz.to_numpy(float), d.dev.to_numpy(float),
                          bins=12)
    x, y = b['x'].to_numpy(float), b['median'].to_numpy(float)
    g = np.isfinite(x) & np.isfinite(y)
    slope, icept = np.polyfit(x[g], y[g], 1)
    return float(icept), float(slope)


def dump_slice(hdf_path, features_csv, run, rt_centre, tag, fwhm0, out_dir):
    """Save one RT slice's per-precursor peaks, profiles and identities."""
    os.makedirs(out_dir, exist_ok=True)
    pm.N_PRECURSORS = 10 ** 9
    w = select_window(hdf_path, rt_centre, SLICE_WIDTH)
    ids = pm.load_ids(features_csv, run, float(w['rt'].min()),
                      float(w['rt'].max()))
    ids = ids.nlargest(DUMP_N, 'area').sort_values('theo_mz').reset_index(
        drop=True)
    targets, sites = by_ions(ids.sequence.to_numpy(),
                             ids.z.astype(int).to_numpy())
    rows, _ = collect(hdf_path, w, ids, fwhm0)
    ann = annotate(rows, ids, targets)
    axis = CAL_AXIS or fit_axis(ann)
    apply_axis(ann, axis)
    add_rt_score(rows, ann, axis, PICK_TOL)

    tabs, profs, owner = [], [], []
    for p, t in enumerate(ann):
        if t is None or not len(t):
            continue
        t = t.copy()
        t['owner'] = p
        tabs.append(t)
        profs.append(rows[p]['rt'])
        owner.append(np.full(len(t), p))
    tab = pd.concat(tabs, ignore_index=True)
    prof = np.concatenate(profs, axis=0).astype(np.float32)
    base = os.path.join(out_dir, f'patchdump_{tag}_rt{rt_centre:.1f}')
    tab.drop(columns=[c for c in ('n_lattice',) if c in tab]).to_parquet(
        base + '_peaks.parquet', index=False)
    np.savez_compressed(base + '_profiles.npz', prof=prof,
                        owner=np.concatenate(owner))
    ids.to_csv(base + '_precursors.csv', index=False)
    with open(base + '_meta.txt', 'w') as fh:
        fh.write('axis_kind table\n' if isinstance(axis, dict) else
                 f'axis_intercept {axis[0]}\naxis_slope {axis[1]}\n')
        fh.write(f'fwhm0 {fwhm0}\ncyc_half {CYC_HALF}\n'
                 f'pick_tol {PICK_TOL}\npick_cos {PICK_COS}\n')
    print(f'  dumped {len(tab):,} peaks over {len(ids)} patches -> '
          f'{os.path.basename(base)}_*')


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    hdf_path, features_csv, run = sys.argv[1:4]
    tag, dump_rt = 'cond', []
    for a in sys.argv[4:]:
        if a.startswith('--tag='):
            tag = a.split('=', 1)[1]
        elif a.startswith('--dump='):
            dump_rt = [float(x) for x in a.split('=', 1)[1].split(',')]
    if not dump_rt:
        sys.exit(__doc__)

    global CAL_AXIS
    CAL_AXIS = load_axis(tag)
    if CAL_AXIS is None:
        print('  no Q1 calibration file - each slice will fit its own axis '
              '(run q1_axis_stability.py to measure one)', flush=True)

    npz = os.path.join(os.path.dirname(OUT_DIR),
                       f'precursor_mass_q1_vs_ms1_{tag}_q1profiles.npz')
    fwhm0 = float(np.load(npz)['fwhm0']) if os.path.exists(npz) else 1.7
    for c in dump_rt:
        print(f'dump slice RT {c}', flush=True)
        dump_slice(hdf_path, features_csv, run, c, tag, fwhm0,
                   os.path.join(OUT_DIR, DUMP_SUBDIR))


if __name__ == '__main__':
    main()
