"""Q1 placement of the co-eluting precursors in one 2.8 Th isolation window."""

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
import matplotlib.pyplot as plt

sys.path.insert(0, _os.path.join(cfg.REPO_ROOT, 'workflows'))
sys.path.insert(0, _os.path.join(cfg.REPO_ROOT, 'experiments', 'H032_E376'))
import zenotof8600_analytics as core
from e376_targets import HDF, ROOT, TAG
from zenotof8600_analytics.raw import (centroid_profile,
                                      peak_shape)
from zenotof8600_analytics.ztscan.profile import peak_index
from q1_isotope_offset import offset_from_mz
from ztscan_common import (by_ions, load_axis, load_ms1,
                           q1_correction)
from ztscan_patch import extract_patch

OUT = os.path.join(ROOT, 'ztscan_api', 'figures', 'paper')

RT_APEX = 25.41
PEPTIDES = [('AFAVGVQQVLLK', 2), ('IGASTLLSDIER', 2), ('LDPSIFESLQK', 2),
            ('GSYGDLGGPIITTQVTIPK', 3)]

DIA_WIN = 2.8
DRAW_PAD = 1.0
PATCH_PAD = 8.0
N_CYCLES = 11
MS1_HALF_RT = 0.05
FRAG_TOL = 0.02
SHARED_TOL = 0.05
CENT_HALF = 1.3
MIN_SNR = 5.0
MIN_COS = 0.95
MIN_FRAG = 4
MIN_PEPTIDES = 3
ESTIMATOR_BIAS = -0.125
SIGMA_A9 = 0.127
MS1_MARK_FRAC = 0.01
FRAG_MZ_LO, FRAG_MZ_HI = 200.0, 1250.0
NEUTRON = 1.00335


def zero_fill(mz, inten, factor=3.0):
    """Drop a zero either side of every real gap, for plotting."""
    if len(mz) < 2:
        return mz, inten
    d = np.diff(mz)
    step = np.median(d[d > 0]) if np.any(d > 0) else 0.0
    if step <= 0:
        return mz, inten
    out_mz, out_in = [], []
    for i in range(len(mz)):
        if i and d[i - 1] > factor * step:
            out_mz += [mz[i - 1] + step * 0.5, mz[i] - step * 0.5]
            out_in += [0.0, 0.0]
        out_mz.append(mz[i])
        out_in.append(inten[i])
    return np.asarray(out_mz), np.asarray(out_in)


def true_mz():
    """Measured precursor m/z and abundance, from the identifications."""
    from e376_targets import identified
    d = identified()
    pre = d.groupby(['seq', 'z']).agg(mz=('mz', 'median'),
                                      area=('area', 'max'))
    return [(float(pre.loc[(s, z), 'mz']), float(pre.loc[(s, z), 'area']))
            for s, z in PEPTIDES]


def load_patch(lo, hi):
    """The patch the Q1 measurement itself works on, not a parallel readout."""
    p = extract_patch(HDF, RT_APEX, lo, hi, n_cycles=N_CYCLES)
    if p is None or not len(p['mz']):
        raise SystemExit('no patch at this position')
    print(f"patch: {len(p['mz']):,} peaks, {len(p['q1_axis'])} Q1 positions "
          f"({np.median(np.diff(p['q1_axis'])):.3f} Th step), "
          f"{len(p['rt_axis'])} cycles")
    return p


def frag_rows(patch, fmz):
    """(fragment m/z, patch row) for each fragment present in the patch."""
    out = []
    for t in fmz:
        j = np.flatnonzero(np.abs(patch['mz'] - t) <= FRAG_TOL)
        if j.size:
            out.append((float(t), int(j[np.argmax(patch['intensity'][j])])))
    return out


def consensus_cos(H, ok):
    """Cosine of each fragment's Q1 profile against its peptide's consensus."""
    cos = np.zeros(H.shape[0])
    if ok.sum() < 3:
        return cos
    N = H[ok] / np.maximum(H[ok].max(axis=1)[:, None], 1e-12)
    c = np.median(N, axis=0)
    n = np.linalg.norm(c)
    if n <= 0:
        return cos
    c = c / n
    A = H / np.maximum(H.max(axis=1)[:, None], 1e-12)
    return (A @ c) / np.maximum(np.linalg.norm(A, axis=1), 1e-12)


def centroids_interp(qs, H):
    """Q1 position per fragment: the INTERPOLATED half-maximum midpoint."""
    out, snr = np.full(H.shape[0], np.nan), np.zeros(H.shape[0])
    for k, prof in enumerate(H):
        if prof.max() <= 0:
            continue
        j = int(np.argmax(prof))
        near = np.abs(qs - qs[j]) <= CENT_HALF
        off = ~near
        base = np.median(prof[off]) if off.sum() >= 5 else 0.0
        snr[k] = prof.max() / base if base > 0 else np.inf
        if snr[k] < MIN_SNR:
            continue
        try:
            sh = peak_shape(qs, prof - base, centre=float(qs[j]),
                            search=CENT_HALF)
        except Exception:
            continue
        lx, rx = sh.get('left_x'), sh.get('right_x')
        if lx is None or rx is None or not np.isfinite([lx, rx]).all():
            continue
        out[k] = 0.5 * (float(lx) + float(rx))
    return out, snr


def centroids(qs, H):
    """Q1 position per fragment: the PARABOLIC VERTEX at the profile maximum."""
    out, snr = np.full(H.shape[0], np.nan), np.zeros(H.shape[0])
    idx = peak_index(np.asarray(H, float))
    pos = np.interp(idx, np.arange(len(qs)), qs)
    for k, prof in enumerate(H):
        if prof.max() <= 0:
            continue
        j = int(np.argmax(prof))
        off = np.abs(qs - qs[j]) > CENT_HALF
        base = np.median(prof[off]) if off.sum() >= 5 else 0.0
        snr[k] = prof.max() / base if base > 0 else np.inf
        if snr[k] < MIN_SNR:
            continue
        out[k] = pos[k]
    return out, snr


def centroids_grid(qs, H):
    """The lattice-bound Q1 centroid rule, compared in q1_estimator_compare."""
    out, snr = np.full(H.shape[0], np.nan), np.zeros(H.shape[0])
    for k, prof in enumerate(H):
        if prof.max() <= 0:
            continue
        j = int(np.argmax(prof))
        near = np.abs(qs - qs[j]) <= CENT_HALF
        off = ~near
        base = np.median(prof[off]) if off.sum() >= 5 else 0.0
        snr[k] = prof.max() / base if base > 0 else np.inf
        if snr[k] < MIN_SNR:
            continue
        w = prof - base
        hm = near & (w >= 0.5 * w.max())
        if w[hm].sum() <= 0:
            continue
        out[k] = float((qs[hm] * w[hm]).sum() / w[hm].sum())
    return out, snr


def ms1_at_apex(lo, hi):
    """The single MS1 scan nearest the apex, as PROFILE plus its centroids."""
    scans, rts = load_ms1(HDF, RT_APEX - MS1_HALF_RT, RT_APEX + MS1_HALF_RT)
    if not len(scans):
        return None
    k = int(np.argmin(np.abs(np.asarray(rts, float) - RT_APEX)))
    pm, pv = scans[k]
    w = (pm >= lo) & (pm <= hi)
    if not w.any():
        return None
    pm, pv = pm[w], pv[w].astype(float)
    cm, _ = centroid_profile(pm, pv)[:2]
    return pm, pv, cm


def prepare():
    """Select the peptides, their unique fragments and their Q1 profiles."""
    axis = load_axis(TAG)

    info = true_mz()
    order = np.argsort([m for m, _ in info])
    peps = [PEPTIDES[i] for i in order]
    mzs = np.array([info[i][0] for i in order])
    areas = np.array([info[i][1] for i in order])
    q_cen = 0.5 * (mzs.min() + mzs.max())
    win_lo, win_hi = q_cen - DIA_WIN / 2, q_cen + DIA_WIN / 2
    q_lo, q_hi = win_lo - DRAW_PAD, win_hi + DRAW_PAD
    print(f'window {mzs.min():.3f}-{mzs.max():.3f} Th (span '
          f'{mzs.max() - mzs.min():.2f} Th, closest pair '
          f'{np.diff(np.sort(mzs)).min():.3f} Th), abundance spread '
          f'{areas.max() / areas.min():.1f}x')

    ions, izs = [], []
    for s, z in peps:
        m, _, zc = by_ions([s], [z], with_charge=True)
        ions.append(np.asarray(m[0], float).ravel())
        izs.append(np.asarray(zc[0], int).ravel())
    uniq, uniq_z = [], []
    for k in range(len(peps)):
        others = np.concatenate([ions[j] for j in range(len(peps)) if j != k])
        m = np.min(np.abs(ions[k][:, None] - others[None, :]),
                   axis=1) > SHARED_TOL
        uniq.append(ions[k][m])
        uniq_z.append(izs[k][m])

    patch = load_patch(mzs.min() - PATCH_PAD, mzs.max() + PATCH_PAD)
    qs = (patch['q1_axis'] - q1_correction(axis, patch['q1_axis'], RT_APEX)
          - ESTIMATOR_BIAS)

    FM, H, C, OK, ISO = [], [], [], [], []
    zof = [dict(zip(u, uz)) for u, uz in zip(uniq, uniq_z)]
    for k in range(len(peps)):
        rows = frag_rows(patch, uniq[k])
        fm = np.array([r[0] for r in rows])
        ISO.append(np.array([offset_from_mz(t, zof[k][t], mzs[k], peps[k][1])
                             for t in fm]))
        h = (patch['q1_prof'][[r[1] for r in rows]] if rows
             else np.zeros((0, len(qs))))
        c = centroids(qs, h)[0] if len(h) else np.zeros(0)
        sn = np.isfinite(c)
        co = consensus_cos(h, sn) if len(h) else np.zeros(0)
        ok = sn & (co >= MIN_COS)
        FM.append(fm)
        H.append(h)
        C.append(c)
        OK.append(ok)
        print(f'  {peps[k][0]:>18s} {peps[k][1]}+ {mzs[k]:8.3f}  '
              f'{len(ions[k]):3d} b/y, {len(uniq[k]):3d} unique, '
              f'{len(fm):3d} in patch, {int(sn.sum()):3d} pass S/N, '
              f'{int(ok.sum()):3d} also pass cosine')

    keep = [k for k in range(len(peps)) if OK[k].sum() >= MIN_FRAG]
    print(f'\n{len(keep)} of {len(peps)} peptides placed with >= {MIN_FRAG} '
          f'unique fragments')
    if len(keep) < MIN_PEPTIDES:
        raise SystemExit('too few peptides placed - not a figure')
    med = np.array([np.median(C[k][OK[k]]) for k in keep])
    sig = np.array([(np.percentile(C[k][OK[k]], 75)
                     - np.percentile(C[k][OK[k]], 25)) / 1.349 for k in keep])
    o = np.argsort(med)
    ordk = [keep[i] for i in o]
    cols = {k: core.PALETTE_SINGLE[i % len(core.PALETTE_SINGLE)]
            for i, k in enumerate(ordk)}
    gap = np.diff(med[o])
    err = med - mzs[keep]
    across = float((np.percentile(err, 75) - np.percentile(err, 25)) / 1.349)
    scale = max(across, SIGMA_A9)
    print(f'closest pair {gap.min():.3f} Th = {gap.min() / scale:.1f} sigma '
          f'against the {scale:.3f} Th across-precursor error')
    print(f'  within one peptide, fragment to fragment: '
          f'{np.median(sig):.3f} Th sigma_IQR')
    print(f'  centroid - true m/z: median {np.median(err):+.3f} Th, '
          f'sigma_IQR {across:.3f} Th')
    print(f'  measured positions in true m/z order: '
          f'{bool(np.all(np.argsort(med) == np.argsort(mzs[keep])))}')
    return dict(peps=peps, mzs=mzs, areas=areas, qs=qs, patch=patch,
                FM=FM, H=H, C=C, OK=OK, keep=keep, ordk=ordk, cols=cols,
                win_lo=win_lo, win_hi=win_hi, q_lo=q_lo, q_hi=q_hi,
                q_cen=q_cen)


def main():
    os.makedirs(OUT, exist_ok=True)
    core.init_plotting()
    P = prepare()
    peps, mzs, qs, patch = P['peps'], P['mzs'], P['qs'], P['patch']
    FM, H, C, OK = P['FM'], P['H'], P['C'], P['OK']
    keep, ordk, cols = P['keep'], P['ordk'], P['cols']
    win_lo, win_hi = P['win_lo'], P['win_hi']
    q_lo, q_hi, q_cen = P['q_lo'], P['q_hi'], P['q_cen']

    fig, axes = plt.subplots(4, 1, figsize=(8, 12),
                             gridspec_kw=dict(height_ratios=[1, 1.1, 1.1, 1]))

    ax = axes[0]
    ax.axvspan(win_lo, win_hi, color='0.55', alpha=0.16, lw=0, zorder=0)
    ms1 = ms1_at_apex(q_lo, q_hi)
    if ms1 is not None:
        pm, pv, cm = ms1
        zx, zy = zero_fill(pm, pv)
        ax.plot(zx, zy, color='0.35', lw=0.9, zorder=2)
        ax.fill_between(zx, 0, zy, color='0.75', alpha=0.35, zorder=1)
        cm = cm[(cm >= q_lo) & (cm <= q_hi)]
        ch = np.interp(cm, pm, pv)
        vis = ch >= MS1_MARK_FRAC * pv.max()
        ax.plot(cm[vis], ch[vis], ls='none', marker='v', ms=4.0,
                color='0.15', zorder=4)
    for k in keep:
        ax.axvline(mzs[k], color=cols[k], lw=1.2, ls='--', alpha=0.9, zorder=3)
    ax.set_ylabel('MS1 intensity')
    ax.set_xlim(q_lo, q_hi)
    ax.set_title(f'A   MS1 at RT {RT_APEX} min', fontsize=10,
                 fontweight='bold')

    ax = axes[1]
    ax.axvspan(win_lo, win_hi, color='0.55', alpha=0.10, lw=0, zorder=0)
    for k in ordk:
        sub = H[k][OK[k]]
        for row in sub:
            ax.plot(qs, row / row.max(), color=cols[k], lw=0.6, alpha=0.28)
        prof = (sub / sub.max(axis=1)[:, None]).mean(axis=0)
        ax.plot(qs, prof / prof.max(), color=cols[k], lw=2.2,
                label=f'{peps[k][0]} {peps[k][1]}+  ({int(OK[k].sum())} b/y)')
        ax.axvline(mzs[k], color=cols[k], lw=1.0, ls='--', alpha=0.7)
    ax.set_ylabel('fragment intensity\n(each normalised)')
    ax.set_xlim(q_lo, q_hi)
    ax.set_ylim(0, 1.06)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    ax.set_title('B   Fragment Q1 profiles', fontsize=10, fontweight='bold')

    ax = axes[2]
    rng = np.random.default_rng(0)
    ax.axvspan(win_lo, win_hi, color='0.55', alpha=0.10, lw=0, zorder=0)
    for i, k in enumerate(ordk):
        y = len(keep) - 1 - i
        cc = C[k][OK[k]]
        ax.axhline(y, color='0.9', lw=0.6, zorder=0)
        ax.scatter(cc, y + rng.uniform(-0.22, 0.22, cc.size), s=16,
                   color=cols[k], alpha=0.85, linewidth=0.3,
                   edgecolor='white', zorder=3)
        ax.plot([np.median(cc)], [y], marker='|', ms=18, color=cols[k],
                mew=2.4, zorder=4)
        ax.axvline(mzs[k], color=cols[k], lw=1.0, ls='--', alpha=0.7, zorder=1)
        ax.annotate(f'{peps[k][0]} {peps[k][1]}+', (0.008, y + 0.33),
                    xycoords=('axes fraction', 'data'), fontsize=8,
                    color=cols[k], va='center')
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(keep) + 0.15)
    ax.set_xlim(q_lo, q_hi)
    ax.set_xlabel('Q1 dimension (m/z)   =   precursor m/z')
    ax.set_title('C   Per-fragment Q1 centroids', fontsize=10,
                 fontweight='bold')
    h = [plt.Line2D([], [], ls='--', color='0.35', lw=1.0,
                    label='precursor m/z')]
    ax.legend(handles=h, frameon=False, fontsize=7.5, loc='upper right',
              handletextpad=0.6)

    ax = axes[3]
    inwin = (qs >= win_lo) & (qs <= win_hi)
    tot_int = patch['q1_prof'][:, inwin].sum(axis=1)
    shown = ((patch['mz'] >= FRAG_MZ_LO) & (patch['mz'] <= FRAG_MZ_HI)
             & (tot_int > 0))
    ax.vlines(patch['mz'][shown], 0, tot_int[shown], color='0.72', lw=0.6,
              zorder=1)
    tot = tot_int[shown].sum()
    claimed, nhit = 0.0, 0
    for k in ordk:
        rows = [r for r in frag_rows(patch, FM[k][OK[k]])
                if FRAG_MZ_LO <= r[0] <= FRAG_MZ_HI and tot_int[r[1]] > 0]
        ax.vlines([patch['mz'][r[1]] for r in rows], 0,
                  [tot_int[r[1]] for r in rows], color=cols[k], lw=1.5,
                  zorder=3, label=f'{peps[k][0]} ({len(rows)})')
        claimed += sum(tot_int[r[1]] for r in rows)
        nhit += len(rows)
    ax.set_xlabel('Fragment m/z')
    ax.set_ylabel('summed MS2 intensity')
    ax.set_xlim(FRAG_MZ_LO, FRAG_MZ_HI)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    ax.set_title(f'D   One {DIA_WIN:g} Th DIA window at {q_cen:.1f} Th',
                 fontsize=10, fontweight='bold')
    print(f'chimeric {DIA_WIN:g} Th window: {int(shown.sum()):,} peaks, '
          f'{nhit} assigned b/y ions from {len(keep)} peptides, '
          f'{claimed / tot:.1%} of its intensity')

    fig.tight_layout()
    base = os.path.join(OUT, 'q1_resolves_window')
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight')
    print(f'wrote {base}.pdf / .png')

    rows = []
    for k in ordk:
        for f in np.where(OK[k])[0]:
            for j, q in enumerate(qs):
                rows.append(dict(peptide=f'{peps[k][0]} {peps[k][1]}+',
                                 precursor_mz=mzs[k], frag_mz=FM[k][f],
                                 q1=q, intensity=H[k][f, j],
                                 norm=H[k][f, j] / H[k][f].max(),
                                 centroid=C[k][f]))
    pd.DataFrame(rows).to_csv(base + '_sourcedata.csv', index=False)
    print(f'wrote {base}_sourcedata.csv')


if __name__ == '__main__':
    main()
