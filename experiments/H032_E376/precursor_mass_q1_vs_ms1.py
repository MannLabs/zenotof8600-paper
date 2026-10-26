"""Precursor m/z from the Q1 dimension vs from the TOFMS (MS1) scan."""

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
import matplotlib.pyplot as plt

sys.path.insert(0, _os.path.join(cfg.REPO_ROOT, 'workflows'))
import zenotof8600_analytics as core
from zenotof8600_analytics.raw.extraction import (
    peak_shape, window_indices)
from alphabase.peptide.fragment import (
    create_fragment_mz_dataframe, get_charged_frag_types)

from q1_resolution_scan import select_window, stream_spectra
from spectrum_merge import merge_profile

OUT_DIR = cfg.cache('H032_E376')

RT_WIDTH = 1.0
MZ_BINS = 5
ABUNDANCE_BINS = 5
MZ_RANGE = (360.0, 960.0)
TARGET_LIST = None
N_PRECURSORS = 5000
N_CANDIDATES = 16
N_FRAG = 8
FRAG_MZ_RANGE = (250.0, 1800.0)
FRAG_TOL = 0.03
MIN_FRAG_FOR_Q1 = 3

Q1_HALF_TH = 8.0
CYC_HALF = 5
SEARCH_FWHM = 1.5
SEARCH_MAX_TH = 2.0

MS1_HALF = 0.12
MS1_SEARCH = 0.04
MS1_SCANS = 3


def load_ids(features_csv, run, rt_lo, rt_hi):
    """Unmodified identified precursors in the RT window, most abundant first."""
    cols = ['Peptide', 'm/z', 'z', f'{run} m/z', f'{run} RT mean',
            f'{run} Normalized Area']
    d = pd.read_csv(features_csv, usecols=cols, low_memory=False)
    d = d.rename(columns={'m/z': 'theo_mz', 'Peptide': 'sequence',
                          f'{run} m/z': 'peaks_mz',
                          f'{run} RT mean': 'rt',
                          f'{run} Normalized Area': 'area'})
    for c in ('theo_mz', 'rt', 'area', 'z', 'peaks_mz'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['theo_mz', 'rt', 'area', 'z']).query('area > 0')
    d = d[d.sequence.astype(str).str.fullmatch(r'[A-Z]+').fillna(False)]
    d = d[(d.rt >= rt_lo) & (d.rt <= rt_hi)]
    if TARGET_LIST:
        want = pd.read_csv(TARGET_LIST)
        keys = set(zip(want.sequence.astype(str), want.z.astype(int)))
        d = d[[ (str(a), int(b)) in keys
                for a, b in zip(d.sequence, d.z) ]]
        print(f'  {len(d)} of the shared target list fall in this RT window',
              flush=True)
        return (d.sort_values('area', ascending=False)
                 .drop_duplicates(['sequence', 'z'])
                 .sort_values('theo_mz').reset_index(drop=True))
    n_before = len(d)
    d = d[(d.theo_mz >= MZ_RANGE[0]) & (d.theo_mz <= MZ_RANGE[1])]
    if n_before != len(d):
        print(f'  {n_before - len(d)} precursors outside the '
              f'{MZ_RANGE[0]:.0f}-{MZ_RANGE[1]:.0f} Th method range dropped '
              f'(sweep edges, clipped transmission)', flush=True)
    return stratified_sample(d)


def stratified_sample(d, n=None, mz_bins=MZ_BINS, ab_bins=ABUNDANCE_BINS):
    """Even coverage of the m/z x abundance plane, not the top N by area."""
    n = N_PRECURSORS if n is None else n
    d = d.sort_values('area', ascending=False).drop_duplicates(
        ['sequence', 'z']).reset_index(drop=True)
    if len(d) <= n:
        return d.sort_values('theo_mz').reset_index(drop=True)
    mzb = pd.qcut(d.theo_mz, min(mz_bins, d.theo_mz.nunique()),
                  labels=False, duplicates='drop')
    abb = pd.qcut(np.log10(d.area), min(ab_bins, d.area.nunique()),
                  labels=False, duplicates='drop')
    cells = list(d.groupby([mzb, abb]).indices.values())
    rng = np.random.default_rng(0)
    quota, chosen = n // max(len(cells), 1), []
    short = 0
    for idx in cells:
        take = min(quota, len(idx))
        chosen.append(rng.choice(idx, take, replace=False))
        short += quota - take
    chosen = np.concatenate(chosen) if chosen else np.array([], int)
    if short > 0:
        rest = np.setdiff1d(np.arange(len(d)), chosen)
        if len(rest):
            chosen = np.concatenate(
                [chosen, rng.choice(rest, min(short, len(rest)),
                                    replace=False)])
    out = d.iloc[np.sort(chosen)]
    print(f'stratified {len(out)} of {len(d)} precursors over '
          f'{len(cells)} m/z x abundance cells '
          f'(m/z {out.theo_mz.min():.0f}-{out.theo_mz.max():.0f}, '
          f'area {np.log10(out.area.min()):.1f}-'
          f'{np.log10(out.area.max()):.1f} log10)', flush=True)
    return out.sort_values('theo_mz').reset_index(drop=True)


def _unused_top_by_area(d):
    return (d.sort_values('area', ascending=False)
             .drop_duplicates(['sequence', 'z'])
             .head(N_PRECURSORS)
             .sort_values('theo_mz')
             .reset_index(drop=True))


def theoretical_fragments(ids):
    """b/y singly-charged ions per precursor."""
    prec = pd.DataFrame({'sequence': ids.sequence.to_numpy(),
                         'mods': '', 'mod_sites': '',
                         'charge': ids.z.astype(int).to_numpy()})
    prec['nAA'] = prec.sequence.str.len()
    fdf = create_fragment_mz_dataframe(prec,
                                       get_charged_frag_types(['b', 'y'], 1))
    mz_out, owner, kind, num = [], [], [], []
    for pi, (a, b, n_aa) in enumerate(zip(prec.frag_start_idx,
                                          prec.frag_stop_idx, prec.nAA)):
        blk = fdf.iloc[int(a):int(b)]
        r = np.arange(len(blk))
        v = np.concatenate([blk['b_z1'].to_numpy(), blk['y_z1'].to_numpy()])
        t = np.concatenate([np.zeros(len(blk), np.int64),
                            np.ones(len(blk), np.int64)])
        j = np.concatenate([r + 1, int(n_aa) - (r + 1)])
        keep = (v >= FRAG_MZ_RANGE[0]) & (v <= FRAG_MZ_RANGE[1])
        v, t, j = v[keep], t[keep], j[keep]
        o = np.argsort(v)[::-1]
        v, t, j = v[o], t[o], j[o]
        sel = [i for i in range(len(v))
               if i == 0 or abs(v[i] - v[i - 1]) > 1e-3][:N_CANDIDATES]
        mz_out.append(v[sel])
        kind.append(t[sel])
        num.append(j[sel])
        owner.append(np.full(len(sel), pi, dtype=np.int64))
    return (np.concatenate(mz_out), np.concatenate(owner),
            np.concatenate(kind), np.concatenate(num))


def q1_profiles(hdf_path, w, theo, owner_rt, frag_mz, frag_owner):
    """prof[fragment, cycle offset, q1 offset], in ONE streaming pass."""
    q1_axis = np.unique(w['q1'])
    step = float(np.median(np.diff(q1_axis)))
    n_cyc = int(w['cycle'].max()) + 1
    rt_axis = np.array([w['rt'][w['cycle'] == c].mean() for c in range(n_cyc)])
    qh = int(np.ceil(Q1_HALF_TH / step))

    owner_mz = theo[frag_owner]
    qi0 = np.searchsorted(q1_axis, owner_mz).clip(0, len(q1_axis) - 1)
    ci0 = np.abs(rt_axis[None, :]
                 - owner_rt[frag_owner][:, None]).argmin(axis=1)

    n_f, n_q = len(frag_mz), 2 * qh + 1
    prof = np.zeros((n_f, 2 * CYC_HALF + 1, n_q))
    print(f'  {n_f:,} theoretical fragments · Q1 patch {n_q} x '
          f'{2 * CYC_HALF + 1} = {prof.nbytes / 1e6:.0f} MB', flush=True)

    lo_b = np.asarray(frag_mz - FRAG_TOL, dtype=np.float32)
    hi_b = np.asarray(frag_mz + FRAG_TOL, dtype=np.float32)
    qi_of = {float(q): i for i, q in enumerate(q1_axis)}
    t0, done = time.time(), 0
    for k, m, v in stream_spectra(hdf_path, w):
        done += 1
        qi = qi_of.get(float(w['q1'][k]))
        if qi is None or not len(m):
            continue
        ci = int(w['cycle'][k])
        a = np.searchsorted(owner_mz, q1_axis[max(0, qi - qh)])
        b = np.searchsorted(owner_mz,
                            q1_axis[min(len(q1_axis) - 1, qi + qh)],
                            side='right')
        if b <= a:
            continue
        dq = qi - qi0[a:b]
        dc = ci - ci0[a:b]
        ok = (np.abs(dq) <= qh) & (np.abs(dc) <= CYC_HALF)
        if not ok.any():
            continue
        idx = np.arange(a, b)[ok]
        cs = np.concatenate(([0.0], np.cumsum(v, dtype=np.float64)))
        left, right = window_indices(m, lo_b[idx], hi_b[idx])
        np.add.at(prof, (idx, dc[ok] + CYC_HALF, dq[ok] + qh),
                  cs[right] - cs[left])
        if done % 20000 == 0:
            print(f'    {done:,}/{len(w["idx"]):,} spectra  '
                  f'{time.time() - t0:.0f}s', flush=True)
    return prof, q1_axis, qi0, qh, step


def frag_curve(summed, q1_axis, qi0, qh, fi):
    """(q1 positions, summed intensity) for one fragment, edges handled."""
    lo = max(0, qi0[fi] - qh)
    hi = min(len(q1_axis) - 1, qi0[fi] + qh)
    j0, j1 = lo - qi0[fi] + qh, hi - qi0[fi] + qh
    return q1_axis[lo:hi + 1], summed[fi][j0:j1 + 1]


def q1_estimate(theo, frag_mz, frag_owner, summed, q1_axis, qi0, qh, fwhm0):
    """Per-fragment Q1 centroid, then the per-precursor median."""
    heights = summed.max(axis=1)
    cen = np.full(len(frag_mz), np.nan)
    for fi in range(len(frag_mz)):
        x, y = frag_curve(summed, q1_axis, qi0, qh, fi)
        if len(x) < 5 or y.max() <= 0:
            continue
        half = SEARCH_FWHM * fwhm0
        if SEARCH_MAX_TH is not None:
            half = min(half, SEARCH_MAX_TH)
        sh = peak_shape(x, y, centre=float(theo[frag_owner[fi]]), search=half)
        if np.isfinite(sh['width']):
            cen[fi] = 0.5 * (float(sh['left_x']) + float(sh['right_x']))

    out, used = {}, np.zeros(len(frag_mz), bool)
    for pi in range(len(theo)):
        f = np.where((frag_owner == pi) & np.isfinite(cen))[0]
        if len(f) < MIN_FRAG_FOR_Q1:
            continue
        f = f[np.argsort(heights[f])[::-1][:N_FRAG]]
        used[f] = True
        out[pi] = (float(np.median(cen[f])), len(f),
                   float(np.std(cen[f], ddof=1)), float(heights[f].max()))
    return out, cen, heights, used


def ms1_estimate(hdf_path, ids, rt_lo, rt_hi):
    """Half-maximum centroid of the precursor peak in the TOFMS survey."""
    with h5py.File(hdf_path, 'r') as h:
        sg = h['ms_data']['spectrum_df']
        ml, rt = sg['ms_level'][:], sg['rt'][:]
        start = sg['peak_start_idx'][:].astype(np.int64)
        stop = sg['peak_stop_idx'][:].astype(np.int64)
        m1 = np.where((ml == 1) & (rt >= rt_lo - 0.1) & (rt <= rt_hi + 0.1))[0]
        mz_ds = h['ms_data']['peak_df']['mz']
        int_ds = h['ms_data']['peak_df']['Intensity']
        scans = []
        for k in m1:
            s_, e_ = int(start[k]), int(stop[k])
            mm, vv = mz_ds[s_:e_], int_ds[s_:e_]
            o = np.argsort(mm, kind='stable')
            scans.append((mm[o], vv[o]))
    rt1 = rt[m1]
    print(f'  MS1: {len(m1)} scans, {sum(len(s[0]) for s in scans):,} peaks',
          flush=True)

    n = len(ids)
    mz_out = np.full(n, np.nan)
    fwhm_out = np.full(n, np.nan)
    apex_out = np.full(n, np.nan)
    step_out = np.full(n, np.nan)
    for i, (t, r) in enumerate(zip(ids.theo_mz.to_numpy(), ids.rt.to_numpy())):
        mm, vv = [], []
        for s in np.argsort(np.abs(rt1 - r))[:MS1_SCANS]:
            sm, sv = scans[s]
            a, b = window_indices(sm,
                                  np.asarray([t - MS1_HALF], dtype=sm.dtype),
                                  np.asarray([t + MS1_HALF], dtype=sm.dtype))
            mm.append(sm[a[0]:b[0]])
            vv.append(sv[a[0]:b[0]])
        m = np.concatenate(mm)
        v = np.concatenate(vv)
        if len(m) < 5:
            continue
        u, y = merge_profile(m, v)
        if len(u) < 5 or y.max() <= 0:
            continue
        if len(u) > 4:
            step_out[i] = float(np.percentile(np.diff(u), 10))
        sh = peak_shape(u, y, centre=t, search=MS1_SEARCH)
        if not np.isfinite(sh['width']):
            continue
        sel = (u >= float(sh['left_x'])) & (u <= float(sh['right_x']))
        if sel.sum() >= 3 and y[sel].sum() > 0:
            mz_out[i] = float(np.average(u[sel], weights=y[sel]))
            fwhm_out[i] = float(sh['width'])
            apex_out[i] = float(sh['height'])
    return mz_out, fwhm_out, apex_out, step_out


def sigma_iqr(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 4:
        return np.nan
    return float((np.quantile(v, .75) - np.quantile(v, .25)) / 1.349)


CAL_BINS = 12


def q1_calibration(mz, err):
    """Affine Q1 mass-axis fit: err ~ a + b x m/z, on BIN MEDIANS."""
    mz, err = np.asarray(mz, float), np.asarray(err, float)
    ok = np.isfinite(mz) & np.isfinite(err)
    b = core.binned_stats(mz[ok], err[ok], bins=CAL_BINS)
    x, y = b['x'].to_numpy(float), b['median'].to_numpy(float)
    good = np.isfinite(x) & np.isfinite(y)
    slope, intercept = np.polyfit(x[good], y[good], 1)

    lin = np.full(len(mz), np.nan)
    lin[ok] = err[ok] - (intercept + slope * mz[ok])
    free = np.full(len(mz), np.nan)
    edges = np.quantile(mz[ok], np.linspace(0, 1, CAL_BINS + 1))
    which = np.clip(np.digitize(mz[ok], edges[1:-1]), 0, CAL_BINS - 1)
    e = err[ok].copy()
    for k in range(CAL_BINS):
        sel = which == k
        if sel.sum() >= 5:
            e[sel] -= np.median(err[ok][sel])
    free[ok] = e
    return intercept, slope, lin, free


def figure(d, label, fwhm0, cal, out_base):
    c_ms1, c_q1 = core.PALETTE_SINGLE[0], core.PALETTE_SINGLE[1]
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.5))

    for a, col, hue, name, unit in (
            (ax[0][0], 'ms1_err_ppm', c_ms1, 'MS1 (TOFMS)', 'ppm'),
            (ax[0][1], 'q1_err_th', c_q1, 'Q1 dimension', 'Th')):
        v = d[col].dropna()
        med, s = float(v.median()), sigma_iqr(v)
        a.hist(v, bins=50, color=hue, alpha=0.8, linewidth=0)
        a.axvline(med, color='black', ls='--', lw=1.2)
        a.set_xlabel(f'measured - theoretical [{unit}]', fontsize=10)
        a.set_ylabel('precursors', fontsize=10)
        a.set_title(f'{name}   (n = {len(v):,})\n'
                    f'offset {med:+.4g} {unit}  ·  '
                    fr'$\sigma_{{IQR}}$ {s:.3g} {unit}',
                    fontsize=11, fontweight='bold')

    a = ax[0][2]
    v = d['q1_cal_th'].dropna()
    s_lin, s_free = sigma_iqr(v), sigma_iqr(d['q1_cal_free_th'])
    mz0 = float(d.theo_mz.median())
    a.hist(v, bins=50, color=c_q1, alpha=0.8, linewidth=0)
    a.axvline(0, color='black', lw=1.2, ls='--')
    a.set_xlabel('residual after affine Q1 calibration [Th]', fontsize=10)
    a.set_ylabel('precursors', fontsize=10)
    a.set_title(f'Q1 dimension, calibrated   (n = {len(v):,})\n'
                fr'$\sigma_{{IQR}}$ {s_lin:.4g} Th = {s_lin / mz0 * 1e6:.0f} '
                f'ppm  ·  shape-free {s_free:.4g} Th',
                fontsize=11, fontweight='bold')

    for a, col, hue, ylab in ((ax[1][0], 'ms1_err_ppm', c_ms1,
                               'MS1 error [ppm]'),
                              (ax[1][1], 'q1_err_th', c_q1,
                               'Q1 error [Th]')):
        s = d.dropna(subset=[col])
        a.scatter(s.theo_mz, s[col], s=9, color=hue, alpha=0.4, linewidth=0)
        if len(s) > 30:
            b = core.binned_stats(s.theo_mz.to_numpy(float),
                                  s[col].to_numpy(float), bins=8)
            a.plot(b['x'].to_numpy(float), b['median'].to_numpy(float),
                   color=hue, lw=2.0, marker='o', ms=4.5,
                   markeredgecolor='white', markeredgewidth=0.6)
        a.axhline(0, color='black', lw=0.8, ls=':')
        a.set_xlabel('Precursor m/z (Q1 dimension)', fontsize=10)
        a.set_ylabel(ylab, fontsize=10)
        a.set_title('Error across the sweep', fontsize=11, fontweight='bold')

    a = ax[1][1]
    xs = np.linspace(d.theo_mz.min(), d.theo_mz.max(), 100)
    a.plot(xs, cal[0] + cal[1] * xs, color='black', lw=1.3, ls='--',
           zorder=6)
    a.set_title('Error across the sweep — an affine ramp\n'
                f'{cal[0]:+.3f} Th {cal[1]:+.2e} $\\times$ m/z  '
                f'(zero at m/z {-cal[0] / cal[1]:.0f})',
                fontsize=11, fontweight='bold')

    a = ax[1][2]
    s = d.dropna(subset=['q1_frag_sd', 'area'])
    a.scatter(np.log10(s.area), s.q1_frag_sd, s=9, color=c_q1, alpha=0.4,
              linewidth=0)
    if len(s) > 30:
        b = core.binned_stats(np.log10(s.area.to_numpy(float)),
                              s.q1_frag_sd.to_numpy(float), bins=8)
        a.plot(b['x'].to_numpy(float), b['median'].to_numpy(float),
               color=c_q1, lw=2.0, marker='o', ms=4.5,
               markeredgecolor='white', markeredgewidth=0.6)
    a.axhline(fwhm0 / 2, color='black', lw=0.9, ls='--')
    a.text(0.98, fwhm0 / 2, 'half the Q1 FWHM ',
           transform=a.get_yaxis_transform(), ha='right', va='bottom',
           fontsize=8, color='black')
    a.set_xlabel(r'$\log_{10}$ precursor area (PEAKS)', fontsize=10)
    a.set_ylabel('SD between a precursor' + "'" + 's fragments [Th]',
                 fontsize=10)
    a.set_title('Scatter within one precursor', fontsize=11,
                fontweight='bold')

    for a in ax.ravel():
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    fig.suptitle(
        f'Precursor m/z from the Q1 transmission centroid vs from the TOFMS '
        f'survey  ·  {label}\n'
        f'both scored against the search-reported (theoretical) m/z  ·  '
        f'Q1 FWHM {fwhm0:.2f} Th  ·  up to {N_FRAG} theoretical b/y ions '
        f'per precursor, median',
        fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    hdf_path, features_csv, run = sys.argv[1:4]
    rt_centre, label, stem = None, run, 'precursor_mass_q1_vs_ms1'
    for a in sys.argv[4:]:
        if a.startswith('--rt='):
            rt_centre = float(a.split('=', 1)[1])
        elif a.startswith('--label='):
            label = a.split('=', 1)[1]
        elif a.startswith('--stem='):
            stem = a.split('=', 1)[1]
        elif a.startswith('--targets='):
            globals()['TARGET_LIST'] = a.split('=', 1)[1]
        elif a.startswith('--search-max='):
            globals()['SEARCH_MAX_TH'] = float(a.split('=', 1)[1])
        elif a.startswith('--n='):
            globals()['N_PRECURSORS'] = int(a.split('=', 1)[1])
    core.init_plotting()

    w = select_window(hdf_path, rt_centre, RT_WIDTH)
    rt_lo, rt_hi = float(w['rt'].min()), float(w['rt'].max())
    ids = load_ids(features_csv, run, rt_lo, rt_hi)
    print(f'{len(ids)} identified unmodified precursors · '
          f'm/z {ids.theo_mz.min():.0f}-{ids.theo_mz.max():.0f} · '
          f'RT {rt_lo:.2f}-{rt_hi:.2f} min', flush=True)
    theo = ids.theo_mz.to_numpy(float)
    owner_rt = ids.rt.to_numpy(float)

    frag_mz, frag_owner, frag_kind, frag_num = theoretical_fragments(ids)
    prof, q1_axis, qi0, qh, step = q1_profiles(hdf_path, w, theo, owner_rt,
                                               frag_mz, frag_owner)
    summed = prof.sum(axis=1)

    ws = []
    n_strong = max(20, len(frag_mz) // 20)
    for fi in np.argsort(summed.max(axis=1))[::-1][:n_strong]:
        x, y = frag_curve(summed, q1_axis, qi0, qh, fi)
        sh = peak_shape(x, y, centre=float(theo[frag_owner[fi]]),
                        search=Q1_HALF_TH)
        if np.isfinite(sh['width']):
            ws.append(float(sh['width']))
    fwhm0 = float(np.median(ws)) if ws else 2.0
    print(f'  Q1 FWHM {fwhm0:.3f} Th  (step {step:.4f} Th, '
          f'{fwhm0 / step:.1f} samples/FWHM)', flush=True)

    q1, cen, heights, used = q1_estimate(theo, frag_mz, frag_owner, summed,
                                         q1_axis, qi0, qh, fwhm0)
    nan4 = (np.nan,) * 4
    ids['q1_mz'] = [q1.get(i, nan4)[0] for i in range(len(ids))]
    ids['q1_n_frag'] = [q1.get(i, nan4)[1] for i in range(len(ids))]
    ids['q1_frag_sd'] = [q1.get(i, nan4)[2] for i in range(len(ids))]
    ids['q1_height'] = [q1.get(i, nan4)[3] for i in range(len(ids))]

    (ids['ms1_mz'], ids['ms1_fwhm'], ids['ms1_height'],
     ids['ms1_step']) = ms1_estimate(hdf_path, ids, rt_lo, rt_hi)

    ids['ms1_err_ppm'] = (ids.ms1_mz - ids.theo_mz) / ids.theo_mz * 1e6
    ids['peaks_err_ppm'] = (ids.peaks_mz - ids.theo_mz) / ids.theo_mz * 1e6
    ids['q1_err_th'] = ids.q1_mz - ids.theo_mz
    ids['q1_err_ppm'] = ids.q1_err_th / ids.theo_mz * 1e6
    ids['q1_fwhm'] = fwhm0
    ids['ms1_res'] = ids.theo_mz / ids.ms1_fwhm

    a0, b0, lin, free = q1_calibration(ids.theo_mz.to_numpy(float),
                                       ids.q1_err_th.to_numpy(float))
    ids['q1_cal_th'] = lin
    ids['q1_cal_free_th'] = free

    base = os.path.join(OUT_DIR, stem)
    d = ids[ids.ms1_err_ppm.notna() | ids.q1_err_th.notna()]
    figure(d, label, fwhm0, (a0, b0), base)
    ids.to_csv(base + '_sourcedata.csv', index=False)

    pd.DataFrame({
        'precursor': frag_owner,
        'sequence': ids.sequence.to_numpy()[frag_owner],
        'z': ids.z.to_numpy()[frag_owner],
        'prec_theo_mz': theo[frag_owner],
        'prec_rt': owner_rt[frag_owner],
        'prec_area': ids.area.to_numpy()[frag_owner],
        'frag_kind': np.where(frag_kind == 0, 'b', 'y'),
        'frag_num': frag_num,
        'frag_mz': frag_mz,
        'q1_centroid': cen,
        'q1_height': heights,
        'used_in_median': used,
        'q1_fwhm': fwhm0,
        'q1_step': step,
    }).to_csv(base + '_fragments.csv', index=False)

    np.savez_compressed(base + '_q1profiles.npz',
                        prof=summed.astype(np.float32),
                        q1_axis=q1_axis.astype(np.float64),
                        qi0=qi0.astype(np.int32), qh=np.int32(qh),
                        frag_mz=frag_mz, frag_owner=frag_owner,
                        fwhm0=np.float64(fwhm0))

    mz0 = float(ids.theo_mz.median())
    print(f'\n{label}   n = {len(ids)} precursors, median m/z {mz0:.1f}')
    print(f'  {"estimate":18s} {"n":>4s}  {"offset":>15s}  '
          f'{"sigma_IQR":>14s}   equivalent')
    for name, col, unit in (
            ('MS1 (TOFMS)', 'ms1_err_ppm', 'ppm'),
            ('PEAKS reported', 'peaks_err_ppm', 'ppm'),
            ('Q1 raw', 'q1_err_th', 'Th'),
            ('Q1 raw', 'q1_err_ppm', 'ppm'),
            ('Q1 calibrated', 'q1_cal_th', 'Th'),
            ('Q1 cal shape-free', 'q1_cal_free_th', 'Th')):
        v = ids[col].dropna()
        if not len(v):
            continue
        med, s = float(v.median()), sigma_iqr(v)
        eq = (f'{med / 1e6 * mz0 * 1000:+.1f} / {s / 1e6 * mz0 * 1000:.1f} mTh'
              if unit == 'ppm' else
              f'{med / mz0 * 1e6:+.0f} / {s / mz0 * 1e6:.0f} ppm')
        print(f'  {name:18s} {len(v):4d}  {med:+11.5g} {unit:3s}  '
              f'{s:10.5g} {unit:3s}   {eq}')
    ok = ids.ms1_fwhm.notna()
    if ok.any():
        pts = float((ids.ms1_fwhm[ok] / ids.ms1_step[ok]).median())
        print(f'  MS1 resolution {ids.ms1_res[ok].median():,.0f} '
              f'(FWHM {ids.ms1_fwhm[ok].median() * 1000:.1f} mTh, '
              f'grid {ids.ms1_step[ok].median() * 1000:.2f} mTh, '
              f'{pts:.1f} points/FWHM)')
        print(f'  Q1  resolution {mz0 / fwhm0:,.0f} '
              f'(FWHM {fwhm0 * 1000:.0f} mTh, '
              f'grid {step * 1000:.0f} mTh, {fwhm0 / step:.1f} points/FWHM)')
    print(f'  Q1 mass-axis fit: {a0:+.4f} Th {b0:+.3e} x m/z '
          f'(zero crossing m/z {-a0 / b0:.0f}, '
          f'span error {b0 * 1e6:+.0f} ppm)')
    sub = ids.dropna(subset=['ms1_err_ppm', 'q1_cal_th'])
    if len(sub) > 3:
        print(f'  MS1 error vs calibrated Q1 error: Pearson r = '
              f'{np.corrcoef(sub.ms1_err_ppm, sub.q1_cal_th)[0, 1]:+.3f} '
              f'(n = {len(sub)}) - independent measurements')
    print(f'  fragments per precursor: median '
          f'{ids.q1_n_frag.median():.0f} of {N_CANDIDATES} candidates')
    print(f'wrote {base}.pdf / .png / _sourcedata.csv')


if __name__ == '__main__':
    main()
