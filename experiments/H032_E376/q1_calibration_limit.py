"""How accurate could the Q1 dimension be made?"""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import math
import os
import sys
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, _os.path.join(cfg.REPO_ROOT, 'workflows'))
import zenotof8600_analytics as core
from zenotof8600_analytics.raw.extraction import peak_shape
from alphabase.constants.aa import AA_Composition
from alphabase.constants.isotope import formula_dist
from alphabase.constants.atom import parse_formula

IN_DIR = cfg.cache('H032_E376')
OUT_DIR = os.path.join(IN_DIR, 'q1_calibration')

CONDITIONS = [('40SPD 1 Th (A5)', 'precursor_mass_q1_vs_ms1_A5'),
              ('40SPD 2.8 Th (A9)', 'precursor_mass_q1_vs_ms1_A9')]

NEUTRON = 1.0033548
TARGET_PPM = 5.0
MS1_PPM = 2.9
CAL_BINS = 12
MIN_FRAG = 3
CENTROID_FWHM = 1.2
N_ISO = 6
ESTIMATORS = ['halfmax', 'weighted', 'model']


@lru_cache(maxsize=400_000)
def complement_dist(residues, add_water):
    """Isotope distribution of a peptide sub-sequence (the complement)."""
    comp = {'C': 0, 'H': 0, 'N': 0, 'O': 0, 'S': 0}
    for r in residues:
        c = AA_Composition.get(r)
        if c is None:
            return None
        for e, n in c.items():
            comp[e] = comp.get(e, 0) + n
    if add_water:
        comp['H'] += 2
        comp['O'] += 1
    if comp['C'] <= 0:
        return None
    f = ''.join(f'{e}({n})' for e, n in comp.items() if n > 0)
    d = np.asarray(formula_dist(parse_formula(f))[0], float)
    d = d[:N_ISO]
    s = d.sum()
    return tuple(d / s) if s > 0 else None


def complement_of(seq, kind, j):
    """b_j -> residues j+1..n (plus water); y_j -> residues 1..n-j."""
    seq = str(seq)
    return (seq[int(j):], True) if kind == 'b' else (seq[:len(seq) - int(j)],
                                                     False)


def add_isotope_prediction(df):
    w, mean_k = [], np.full(len(df), np.nan)
    for i, (seq, kind, j) in enumerate(zip(df.sequence, df.frag_kind,
                                           df.frag_num)):
        res, water = complement_of(seq, kind, j)
        d = complement_dist(res, water) if res else None
        w.append(d)
        if d is not None:
            d = np.asarray(d)
            mean_k[i] = float((d * np.arange(len(d))).sum())
    df = df.copy()
    df['iso_w'] = w
    df['mean_k'] = mean_k
    df['pred_off'] = mean_k * NEUTRON / df.z.to_numpy(float)
    return df.dropna(subset=['pred_off'])


def frag_curve(prof, q1_axis, qi0, qh, fi):
    lo = max(0, int(qi0[fi]) - qh)
    hi = min(len(q1_axis) - 1, int(qi0[fi]) + qh)
    j0, j1 = lo - int(qi0[fi]) + qh, hi - int(qi0[fi]) + qh
    return q1_axis[lo:hi + 1], prof[fi][j0:j1 + 1]


def baseline_sub(y):
    """Remove a flat baseline taken as the 10th percentile of the window."""
    b = np.percentile(y, 10)
    return np.clip(y - b, 0, None)


def empirical_lineshape(prof, q1_axis, qi0, qh, df, fwhm0):
    """The measured Q1 transmission lineshape."""
    ok = df.index.to_numpy()
    score = df.pred_off.to_numpy()
    strong = df.q1_height.to_numpy()
    pick = np.where((score <= np.quantile(score, 0.25))
                    & (strong >= np.quantile(strong, 0.75)))[0]
    grid = np.linspace(-2.5, 2.5, 201)
    acc, n = np.zeros_like(grid), 0
    for i in pick:
        fi = int(ok[i])
        x, y = frag_curve(prof, q1_axis, qi0, qh, fi)
        y = baseline_sub(y)
        if len(x) < 7 or y.max() <= 0:
            continue
        sh = peak_shape(x, y, centre=float(df.prec_theo_mz.iloc[i]),
                        search=1.5 * fwhm0)
        if not np.isfinite(sh['width']):
            continue
        c = 0.5 * (float(sh['left_x']) + float(sh['right_x']))
        acc += np.interp(grid, (x - c) / fwhm0, y / y.max(), left=0, right=0)
        n += 1
    if n == 0:
        return grid, np.exp(-4 * np.log(2) * grid ** 2), 0
    k = acc / n
    return grid, k / k.max(), n


def estimate_all(prof, q1_axis, qi0, qh, df, fwhm0, kern):
    """halfmax / weighted / model estimate of the precursor Q1 position."""
    kx, ky = kern
    n = len(df)
    out = {e: np.full(n, np.nan) for e in ESTIMATORS}
    idx = df.index.to_numpy()
    theo = df.prec_theo_mz.to_numpy(float)
    zz = df.z.to_numpy(float)
    ws = df.iso_w.to_numpy()
    mu_grid = np.linspace(-1.2 * fwhm0, 1.2 * fwhm0, 97)
    t0 = time.time()
    for i in range(n):
        fi = int(idx[i])
        x, y = frag_curve(prof, q1_axis, qi0, qh, fi)
        if len(x) < 7:
            continue
        y = baseline_sub(y)
        if y.max() <= 0:
            continue
        sh = peak_shape(x, y, centre=theo[i], search=1.5 * fwhm0)
        if not np.isfinite(sh['width']):
            continue
        hm = 0.5 * (float(sh['left_x']) + float(sh['right_x']))
        out['halfmax'][i] = hm

        sel = np.abs(x - hm) <= CENTROID_FWHM * fwhm0
        if sel.sum() >= 3 and y[sel].sum() > 0:
            out['weighted'][i] = float(np.average(x[sel], weights=y[sel]))

        w = ws[i]
        if w is None:
            continue
        w = np.asarray(w, float)
        delta = NEUTRON / zz[i]
        mus = hm + mu_grid
        u = (x[None, :] - mus[:, None]) / fwhm0
        m = np.zeros_like(u)
        for k, wk in enumerate(w):
            if wk > 1e-4:
                m += wk * np.interp(u - k * delta / fwhm0, kx, ky,
                                    left=0, right=0)
        den = (m * m).sum(axis=1)
        good = den > 0
        if not good.any():
            continue
        amp = np.where(good, (m * y[None, :]).sum(axis=1) / np.where(den, den, 1), 0)
        resid = ((amp[:, None] * m - y[None, :]) ** 2).sum(axis=1)
        resid[~good] = np.inf
        out['model'][i] = float(mus[int(np.argmin(resid))])
        if i and i % 2000 == 0:
            print(f'    {i:,}/{n:,}  {time.time() - t0:.0f}s', flush=True)
    return out


def sigma_iqr(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 4:
        return np.nan
    return float((np.quantile(v, .75) - np.quantile(v, .25)) / 1.349)


def affine_detrend(mz, err, bins=CAL_BINS):
    """Remove a + b x m/z, fitted on BIN MEDIANS (tails would drag a fit)."""
    mz, err = np.asarray(mz, float), np.asarray(err, float)
    ok = np.isfinite(mz) & np.isfinite(err)
    out = np.full(len(mz), np.nan)
    if ok.sum() < 12:
        if ok.any():
            out[ok] = err[ok] - np.median(err[ok])
        return out, np.nan, 0.0
    bins = int(np.clip(ok.sum() // 20, 3, bins))
    b = core.binned_stats(mz[ok], err[ok], bins=bins)
    x, y = b['x'].to_numpy(float), b['median'].to_numpy(float)
    g = np.isfinite(x) & np.isfinite(y)
    if g.sum() < 3:
        out[ok] = err[ok] - np.median(err[ok])
        return out, float(np.median(err[ok])), 0.0
    slope, icept = np.polyfit(x[g], y[g], 1)
    out[ok] = err[ok] - (icept + slope * mz[ok])
    return out - np.nanmedian(out), icept, slope


def per_precursor(df, col):
    u = df[df.used_in_median & np.isfinite(df[col])]
    g = u.groupby('precursor')
    out = g.agg(est=(col, 'median'), n=(col, 'size'), sd=(col, 'std'),
                mz=('prec_theo_mz', 'first'), z=('z', 'first'),
                area=('prec_area', 'first'))
    return out[out.n >= MIN_FRAG]


def ladder(df):
    """sigma_IQR of the per-precursor estimate, correction by correction."""
    rows, series = [], {}
    for est in ESTIMATORS:
        df[f'err_{est}'] = df[est] - df.prec_theo_mz
        df[f'iso_{est}'] = df[f'err_{est}'] - df.pred_off

    base = per_precursor(df, 'err_halfmax')
    v = base.est.to_numpy()
    rows.append(('raw (half-max)', sigma_iqr(v), len(v)))
    series['raw (half-max)'] = v
    v1, _, _ = affine_detrend(base.mz.to_numpy(), v)
    rows.append(('+ affine mass axis', sigma_iqr(v1), len(v1)))
    series['+ affine mass axis'] = v1

    wc = per_precursor(df, 'err_weighted')
    v2, _, _ = affine_detrend(wc.mz.to_numpy(), wc.est.to_numpy())
    rows.append(('weighted centroid', sigma_iqr(v2), len(v2)))
    series['weighted centroid'] = v2

    wi = per_precursor(df, 'iso_weighted')
    v3, _, _ = affine_detrend(wi.mz.to_numpy(), wi.est.to_numpy())
    rows.append(('+ isotope model', sigma_iqr(v3), len(v3)))
    series['+ isotope model'] = v3

    mo = per_precursor(df, 'err_model')
    v4, _, _ = affine_detrend(mo.mz.to_numpy(), mo.est.to_numpy())
    rows.append(('lineshape forward fit', sigma_iqr(v4), len(v4)))
    series['lineshape forward fit'] = v4

    v5 = np.full(len(mo), np.nan)
    zs = mo.z.to_numpy()
    for q in np.unique(zs):
        m = zs == q
        v5[m], _, _ = affine_detrend(mo.mz.to_numpy()[m],
                                     mo.est.to_numpy()[m])
    rows.append(('+ per-charge axis', sigma_iqr(v5), int(np.isfinite(v5).sum())))
    series['+ per-charge axis'] = v5

    top = (mo.area >= mo.area.quantile(0.75)).to_numpy()
    rows.append(('(top-quartile subset)', sigma_iqr(v5[top]), int(top.sum())))
    series['(top-quartile subset)'] = v5[top]
    return pd.DataFrame(rows, columns=['stage', 'sigma_th', 'n']), series, mo


def within_precursor(df, est):
    """Fixed-effects test - centring on each precursor removes the mass axis exactly, so only the isotope term can survive."""
    u = df[df.used_in_median & np.isfinite(df[est])].copy()
    u['n_in'] = u.groupby('precursor')[est].transform('size')
    u = u[u.n_in >= MIN_FRAG].copy()
    u['err'] = u[est] - u.prec_theo_mz
    u['err_c'] = u.err - u.groupby('precursor').err.transform('mean')
    u['pred_c'] = u.pred_off - u.groupby('precursor').pred_off.transform('mean')
    x, y = u.pred_c.to_numpy(float), u.err_c.to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 40 or np.ptp(x[ok]) == 0:
        return u, np.nan, np.nan, np.nan
    b = core.binned_stats(x[ok], y[ok], bins=9)
    bx, by = b['x'].to_numpy(float), b['median'].to_numpy(float)
    g = np.isfinite(bx) & np.isfinite(by)
    if g.sum() < 3:
        return u, np.nan, np.nan, np.nan
    slope, icept = np.polyfit(bx[g], by[g], 1)
    return u, float(slope), float(icept), float(np.corrcoef(x[ok], y[ok])[0, 1])


def save(fig, base):
    fig.savefig(base + '.pdf', bbox_inches='tight', dpi=600)
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight')
    print(f'  wrote {os.path.basename(base)}.pdf / .png')


def fig_lineshape(data, out_base):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    for ci, D in enumerate(data):
        hue = core.PALETTE_SINGLE[ci]
        kx, ky = D['kern']
        ax[0].plot(kx, ky, color=hue, lw=2.0,
                   label=f"{D['label']}  (n = {D['n_kern']})")
    g = np.linspace(-2.5, 2.5, 201)
    ax[0].plot(g, np.exp(-4 * np.log(2) * g ** 2), color='black', lw=1.1,
               ls='--', label='Gaussian of the same FWHM')
    ax[0].axhline(0.5, color='gray', lw=0.7, ls=':')
    ax[0].set_xlabel('Q1 offset from apex [FWHM]', fontsize=10)
    ax[0].set_ylabel('normalised transmission', fontsize=10)
    ax[0].set_title('The measured Q1 lineshape\n(fragments with least isotope '
                    'smear)', fontsize=11, fontweight='bold')
    ax[0].legend(fontsize=8.5, frameon=False)

    a = ax[1]
    D = data[0]
    kx, ky = D['kern']
    fwhm0 = D['fwhm0']
    u = np.linspace(-2.5, 3.5, 400)
    a.plot(u, np.interp(u, kx, ky), color='black', lw=1.6,
           label='monoisotopic only')
    for j, (mk, zz, lab) in enumerate([(0.25, 2, 'small complement'),
                                       (0.75, 2, 'large complement'),
                                       (0.75, 3, 'large complement, 3+')]):
        d = np.exp(-mk) * mk ** np.arange(N_ISO) / \
            np.array([math.factorial(k) for k in range(N_ISO)])
        d /= d.sum()
        m = np.zeros_like(u)
        for k, wk in enumerate(d):
            m += wk * np.interp(u - k * (NEUTRON / zz) / fwhm0, kx, ky,
                                left=0, right=0)
        cen = float(np.average(u, weights=np.clip(m, 0, None)))
        a.plot(u, m / m.max(), color=core.PALETTE_SINGLE[j + 1], lw=1.8,
               label=fr'{lab}, $\langle k\rangle$={mk}: centroid '
                     f'{cen * fwhm0:+.3f} Th')
    a.axvline(0, color='gray', lw=0.7, ls=':')
    a.set_xlabel('Q1 offset from the TRUE precursor position [FWHM]',
                 fontsize=10)
    a.set_ylabel('normalised transmission', fontsize=10)
    a.set_title('The blend the Q1 window actually sees\n'
                '(isotopologues are unresolved)', fontsize=11,
                fontweight='bold')
    a.legend(fontsize=8, frameon=False)

    a = ax[2]
    xs = np.arange(len(ESTIMATORS))
    wid = 0.38
    for ci, D in enumerate(data):
        hue = core.PALETTE_SINGLE[ci]
        sl = [D['wp'][e][1] for e in ESTIMATORS]
        a.bar(xs + (ci - 0.5) * wid, sl, wid, color=hue, alpha=0.85,
              edgecolor='black', linewidth=0.5, label=D['label'])
        for x, v in zip(xs + (ci - 0.5) * wid, sl):
            a.text(x, v + 0.02, f'{v:.2f}', ha='center', va='bottom',
                   fontsize=8)
    a.axhline(1.0, color='black', lw=1.3, ls='--')
    a.text(1.45, 1.03, 'full response — target for a raw centroid',
           ha='center', va='bottom', fontsize=8)
    a.axhline(0, color='black', lw=1.3, ls=':')
    a.text(2.45, 0.03, 'removed — target for the forward fit', ha='right',
           va='bottom', fontsize=8)
    a.set_xticks(xs)
    a.set_xticklabels(['half-max\nmidpoint', 'weighted\ncentroid',
                       'lineshape\nforward fit'], fontsize=9)
    a.set_ylabel('within-precursor response to the isotope term', fontsize=10)
    a.set_title('Does the estimator see the isotope shift?\n'
                'raw centroids should read 1.0; the fit should read 0',
                fontsize=11, fontweight='bold')
    a.legend(fontsize=8.5, frameon=False, loc='lower left')

    for a in ax.ravel():
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    fig.suptitle('The Q1 lineshape, and why the centroid estimator is the '
                 'whole question', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out_base)


def fig_mechanism(data, out_base):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for ci, D in enumerate(data):
        hue = core.PALETTE_SINGLE[ci]
        df = D['df']
        s = df[df.used_in_median]

        a = ax[0][0]
        a.scatter(s.prec_theo_mz, s.pred_off, s=6, color=hue, alpha=0.20,
                  linewidth=0, rasterized=True)
        b = core.binned_stats(s.prec_theo_mz.to_numpy(float),
                              s.pred_off.to_numpy(float), bins=10)
        a.plot(b['x'], b['median'], color=hue, lw=2.0, marker='o', ms=4,
               markeredgecolor='white', markeredgewidth=0.6,
               label=D['label'])

        u, slope, icept, r = D['wp']['weighted']
        a = ax[0][1]
        a.scatter(u.pred_c, u.err_c, s=6, color=hue, alpha=0.18, linewidth=0,
                  rasterized=True)
        b = core.binned_stats(u.pred_c.to_numpy(float),
                              u.err_c.to_numpy(float), bins=9)
        a.plot(b['x'], b['median'], color=hue, lw=2.2, marker='o', ms=5,
               markeredgecolor='white', markeredgewidth=0.6,
               label=f"{D['label']}: slope {slope:.2f}, r = {r:+.2f}")

        a = ax[1][0]
        g = u.groupby('precursor')
        sd0 = g.err.std().dropna()
        iso = u.err - u.pred_off
        sd1 = iso.groupby(u.precursor).std().dropna()
        hi = float(np.nanpercentile(sd0, 98))
        bins = np.linspace(0, hi, 45)
        a.hist(sd0, bins=bins, color=hue, alpha=0.28, linewidth=0,
               label=f"{D['label']}: raw, median {sd0.median():.3f} Th")
        a.hist(sd1, bins=bins, histtype='step', color=hue, lw=2.0,
               label=f'   isotope-corrected, {sd1.median():.3f} Th')

        a = ax[1][1]
        best = D['series']['+ per-charge axis']
        mo = D['mo']
        ok = np.isfinite(best)
        b = core.binned_stats(np.log10(mo.area.to_numpy(float)[ok]),
                              np.abs(best[ok]), bins=8)
        a.plot(b['x'], (b['q_hi'] - b['q_lo']) / 1.349, color=hue, lw=2.0,
               marker='o', ms=4.5, markeredgecolor='white',
               markeredgewidth=0.6, label=D['label'])

    ax[0][0].set_xlabel('Precursor m/z', fontsize=10)
    ax[0][0].set_ylabel('Predicted Q1 offset [Th]', fontsize=10)
    ax[0][0].set_title('The complement carries the heavy atoms\n'
                       r'offset $= \langle k\rangle_{compl}\times$1.0034$/z$',
                       fontsize=11, fontweight='bold')
    ax[0][0].legend(fontsize=8.5, frameon=False, loc='upper left')

    a = ax[0][1]
    lim = float(np.nanpercentile(
        np.abs(np.concatenate([D['wp']['weighted'][0].pred_c.to_numpy(float)
                               for D in data])), 98))
    a.plot([-lim, lim], [-lim, lim], color='black', lw=1.1, ls='--',
           label='1:1 (model exact)')
    a.set_xlim(-lim, lim)
    a.set_xlabel('Predicted offset, centred on its precursor [Th]',
                 fontsize=10)
    a.set_ylabel('Measured Q1 error, centred [Th]', fontsize=10)
    a.set_title('Within-precursor test, weighted centroid\n'
                '(no mass-axis term can survive this)', fontsize=11,
                fontweight='bold')
    a.legend(fontsize=8.5, frameon=False, loc='upper left')

    a = ax[1][0]
    a.set_xlabel("SD between one precursor's fragment centroids [Th]",
                 fontsize=10)
    a.set_ylabel('precursors', fontsize=10)
    a.set_title("Does the model tighten a precursor's own fragments?",
                fontsize=11, fontweight='bold')
    a.legend(fontsize=8, frameon=False)

    a = ax[1][1]
    a.set_xlabel(r'$\log_{10}$ precursor area', fontsize=10)
    a.set_ylabel(r'$\sigma_{IQR}$ of the best estimate [Th]', fontsize=10)
    a.set_title('Is what remains ion statistics?', fontsize=11,
                fontweight='bold')
    a.legend(fontsize=8.5, frameon=False)
    a.set_ylim(bottom=0)

    for a in ax.ravel():
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    fig.suptitle('Why a Q1 centroid is not a precursor mass: the fragment is '
                 'transmitted at the isotopologues of its COMPLEMENT',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out_base)


def fig_ladder(data, out_base):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    stages = list(data[0]['lad'].stage)
    xs = np.arange(len(stages))
    wid = 0.38
    a = ax[0]
    for ci, D in enumerate(data):
        hue = core.PALETTE_SINGLE[ci]
        ppm = D['lad'].sigma_th / D['mz0'] * 1e6
        a.bar(xs + (ci - 0.5) * wid, ppm, wid, color=hue, alpha=0.85,
              edgecolor='black', linewidth=0.5, label=D['label'])
        for x, v in zip(xs + (ci - 0.5) * wid, ppm):
            a.text(x, v * 1.05, f'{v:.0f}', ha='center', va='bottom',
                   fontsize=8)
    a.axhline(TARGET_PPM, color='black', lw=1.4, ls='--')
    a.text(-0.45, TARGET_PPM * 1.22,
           f'{TARGET_PPM:.0f} ppm — the single-quadrupole claim', ha='left',
           va='bottom', fontsize=8.5)
    a.axhline(MS1_PPM, color=core.PALETTE_SINGLE[3], lw=1.4, ls=':')
    a.text(-0.45, MS1_PPM * 0.78,
           f'TOFMS on this same data ({MS1_PPM} ppm)', ha='left', va='top',
           fontsize=8.5, color=core.PALETTE_SINGLE[3])
    a.set_yscale('log')
    a.set_ylim(1.8, 1600)
    a.set_xticks(xs)
    a.set_xticklabels(stages, rotation=32, ha='right', fontsize=8.5)
    a.set_ylabel(r'$\sigma_{IQR}$ of precursor m/z [ppm]', fontsize=10)
    a.set_title('Every correction, stacked', fontsize=11, fontweight='bold')
    a.legend(fontsize=9, frameon=False, loc='upper center', ncol=2)

    a = ax[1]
    D = data[0]
    for i, k in enumerate(['+ affine mass axis', 'weighted centroid',
                           '+ isotope model', '+ per-charge axis']):
        v = np.asarray(D['series'][k], float)
        v = v[np.isfinite(v)] / D['mz0'] * 1e6
        a.hist(v, bins=70, range=(-800, 800), histtype='step', lw=2.0,
               color=core.PALETTE_SINGLE[i],
               label=f'{k} ({sigma_iqr(v):.0f} ppm)')
    a.axvline(0, color='black', lw=0.8, ls=':')
    a.set_xlabel('precursor m/z error [ppm]', fontsize=10)
    a.set_ylabel('precursors', fontsize=10)
    a.set_title(f"{D['label']} — the distribution at each stage", fontsize=11,
                fontweight='bold')
    a.legend(fontsize=8.5, frameon=False)

    for a in ax.ravel():
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    fig.suptitle('How far calibration can take the Q1 dimension',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out_base)


def fig_demand(data, out_base):
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.8))
    src = {'demand': [], 'averaging': [], 'specificity': []}
    a = ax[0]
    rows = []
    for ci, D in enumerate(data):
        best = float(D['lad'].sigma_th.iloc[-2])
        rows.append((f"Q1 · {D['label'].split('(')[0].strip()}", D['fwhm0'],
                     best, D['mz0'], core.PALETTE_SINGLE[ci]))
    rows.append(('single quadrupole\n(unit res., TrueCal — claimed)', 0.60,
                 TARGET_PPM * 1e-6 * 500.0, 500.0, core.PALETTE_SINGLE[4]))
    rows.append(('TOFMS\n(this run, measured)', 0.0146,
                 MS1_PPM * 1e-6 * 480.0, 480.0, core.PALETTE_SINGLE[3]))
    y = np.arange(len(rows))
    have = np.array([r[1] / r[2] for r in rows])
    need = np.array([r[1] / (TARGET_PPM * 1e-6 * r[3]) for r in rows])
    a.barh(y - 0.19, have, 0.36, color=[r[4] for r in rows], alpha=0.95,
           edgecolor='black', linewidth=0.5, label='achieved')
    a.barh(y + 0.19, need, 0.36, color=[r[4] for r in rows], alpha=0.32,
           edgecolor='black', linewidth=0.5,
           label=f'needed for {TARGET_PPM:.0f} ppm')
    for yy, h, n in zip(y, have, need):
        a.text(h * 1.12, yy - 0.19, f'{h:,.0f}', va='center', fontsize=8)
        a.text(n * 1.12, yy + 0.19, f'{n:,.0f}', va='center', fontsize=8,
               color='dimgray')
    src['demand'] = pd.DataFrame(
        {'analyser': [r[0].replace('\n', ' ') for r in rows],
         'fwhm_th': [r[1] for r in rows],
         'centroid_precision_th': [r[2] for r in rows],
         'mz': [r[3] for r in rows],
         'sharpening_achieved': have, 'sharpening_needed_5ppm': need})
    a.set_yticks(y)
    a.set_yticklabels([r[0] for r in rows], fontsize=9)
    a.set_xscale('log')
    a.set_xlim(3, 3e4)
    a.set_xlabel('FWHM / centroid precision  (sharpening factor)',
                 fontsize=10)
    a.set_title(f'What {TARGET_PPM:.0f} ppm asks of each analyser',
                fontsize=11, fontweight='bold')
    a.legend(fontsize=8.5, frameon=False, loc='lower right')
    a.invert_yaxis()

    a = ax[1]
    for ci, D in enumerate(data):
        hue = core.PALETTE_SINGLE[ci]
        best = np.asarray(D['series']['+ per-charge axis'], float)
        nn = D['mo'].n.to_numpy()
        ok = np.isfinite(best)
        pts = [(int(k), sigma_iqr(best[ok & (nn == k)]))
               for k in np.unique(nn[ok]) if (ok & (nn == k)).sum() >= 25]
        if len(pts) < 2:
            continue
        xs_, ys_ = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        a.plot(xs_, ys_, color=hue, lw=2.0, marker='o', ms=5,
               markeredgecolor='white', markeredgewidth=0.6,
               label=f"{D['label']} (n$\\geq$25 per point)")
        a.plot(xs_, ys_[0] * np.sqrt(xs_[0] / xs_), color=hue, lw=1.1,
               ls='--')
        src['averaging'].append(pd.DataFrame(
            {'condition': D['label'], 'n_fragments': xs_, 'sigma_th': ys_,
             'ideal_sqrt_n_th': ys_[0] * np.sqrt(xs_[0] / xs_)}))
    a.set_xlabel('fragments averaged per precursor', fontsize=10)
    a.set_ylabel(r'$\sigma_{IQR}$ of the calibrated estimate [Th]',
                 fontsize=10)
    a.set_title('Does averaging more fragments help?\n'
                r'dashed = ideal $1/\sqrt{n}$ from the first point',
                fontsize=11, fontweight='bold')
    a.legend(fontsize=8.5, frameon=False)
    a.set_ylim(bottom=0)

    a = ax[2]
    for ci, D in enumerate(data):
        hue = core.PALETTE_SINGLE[ci]
        u = D['wp']['weighted'][0]
        res = np.abs((u.err_c - u.pred_c).to_numpy(float))
        b = core.binned_stats(u.frag_mz.to_numpy(float), res, bins=9)
        a.plot(b['x'], b['median'], color=hue, lw=2.0, marker='o', ms=5,
               markeredgecolor='white', markeredgewidth=0.6,
               label=D['label'])
        a.fill_between(b['x'], b['q_lo'], b['q_hi'], color=hue, alpha=0.15,
                       linewidth=0)
        t = b.copy()
        t['condition'] = D['label']
        src['specificity'].append(t)
    a.set_xlabel('Fragment m/z  (higher = more specific)', fontsize=10)
    a.set_ylabel('|within-precursor residual| [Th]', fontsize=10)
    a.set_title('Is the floor co-isolation?\n'
                'residual after the isotope term, vs fragment specificity',
                fontsize=11, fontweight='bold')
    a.legend(fontsize=8.5, frameon=False)
    a.set_ylim(bottom=0)

    for a in ax.ravel():
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    fig.suptitle('What 5 ppm would require, and why neither averaging nor '
                 'abundance gets there', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out_base)
    out = [src['demand'].assign(panel='demand')]
    for k in ('averaging', 'specificity'):
        if src[k]:
            out.append(pd.concat(src[k]).assign(panel=k))
    pd.concat(out, ignore_index=True).to_csv(out_base + '_sourcedata.csv',
                                             index=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    core.init_plotting()

    data = []
    for label, stem in CONDITIONS:
        fp = os.path.join(IN_DIR, stem + '_fragments.csv')
        np_ = os.path.join(IN_DIR, stem + '_q1profiles.npz')
        if not (os.path.exists(fp) and os.path.exists(np_)):
            print(f'skipping {label}: missing inputs')
            continue
        print(f'\n{label}')
        df = pd.read_csv(fp)
        z = np.load(np_)
        prof, q1_axis = z['prof'], z['q1_axis']
        qi0, qh, fwhm0 = z['qi0'], int(z['qh']), float(z['fwhm0'])
        df = add_isotope_prediction(df).reset_index(drop=False)
        df = df.rename(columns={'index': 'frag_idx'}).set_index('frag_idx')

        kx, ky, n_kern = empirical_lineshape(prof, q1_axis, qi0, qh, df, fwhm0)
        print(f'  lineshape from {n_kern} fragments; FWHM {fwhm0:.3f} Th')
        est = estimate_all(prof, q1_axis, qi0, qh, df, fwhm0, (kx, ky))
        for e in ESTIMATORS:
            df[e] = est[e]
        lad, series, mo = ladder(df)
        wp = {e: within_precursor(df, e) for e in ESTIMATORS}
        mz0 = float(df.prec_theo_mz.median())
        data.append(dict(label=label, df=df, lad=lad, series=series, mo=mo,
                         wp=wp, kern=(kx, ky), n_kern=n_kern, fwhm0=fwhm0,
                         mz0=mz0))

        print(f'  predicted isotope offset: median {df.pred_off.median():.4f} '
              f'Th ({df.pred_off.quantile(.02):.3f}-'
              f'{df.pred_off.quantile(.98):.3f})')
        for e in ESTIMATORS:
            print(f'  within-precursor slope, {e:9s}: {wp[e][1]:+.3f} '
                  f'(1.0 = exact)  r = {wp[e][3]:+.3f}')
        print(f'  {"stage":24s} {"sigma_Th":>10s} {"ppm":>7s}   n')
        for _, r in lad.iterrows():
            print(f'  {r.stage:24s} {r.sigma_th:10.4f} '
                  f'{r.sigma_th / mz0 * 1e6:7.0f}   {int(r.n)}')

    if not data:
        sys.exit('no inputs - run precursor_mass_q1_vs_ms1.py first')

    fig_lineshape(data, os.path.join(OUT_DIR, 'q1_lineshape_estimators'))
    fig_mechanism(data, os.path.join(OUT_DIR, 'q1_isotope_mechanism'))
    fig_ladder(data, os.path.join(OUT_DIR, 'q1_calibration_ladder'))
    fig_demand(data, os.path.join(OUT_DIR, 'q1_accuracy_demand'))

    src = []
    for D in data:
        u = D['wp']['weighted'][0].copy()
        u['condition'] = D['label']
        src.append(u.drop(columns=['iso_w']))
    pd.concat(src).to_csv(os.path.join(
        OUT_DIR, 'q1_isotope_mechanism_sourcedata.csv'), index=False)

    lads = []
    for D in data:
        t = D['lad'].copy()
        t['condition'] = D['label']
        t['median_mz'] = D['mz0']
        t['sigma_ppm'] = t.sigma_th / D['mz0'] * 1e6
        t['q1_fwhm_th'] = D['fwhm0']
        lads.append(t)
    pd.concat(lads).to_csv(os.path.join(
        OUT_DIR, 'q1_calibration_ladder_sourcedata.csv'), index=False)

    ks = []
    for D in data:
        kx, ky = D['kern']
        ks.append(pd.DataFrame({'condition': D['label'], 'offset_fwhm': kx,
                                'transmission': ky}))
    pd.concat(ks).to_csv(os.path.join(
        OUT_DIR, 'q1_lineshape_estimators_sourcedata.csv'), index=False)
    print(f'\noutputs in {OUT_DIR}')


if __name__ == '__main__':
    main()
