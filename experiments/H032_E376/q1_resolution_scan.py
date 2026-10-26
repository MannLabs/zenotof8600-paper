"""ZT Scan performance against abundance: Q1 and m/z resolution and precision."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys

import h5py

from alpharaw.ms_data_base import MSData_Base
import hdf5plugin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, _os.path.join(cfg.REPO_ROOT, 'workflows'))
import zenotof8600_analytics as core
from zenotof8600_analytics.raw.extraction import (
    peak_shape, window_indices)

OUT_DIR = cfg.cache('H032_E376')

N_TARGETS = 600
N_STRATA = 6
FRAG_TOL = 0.03
MZ_HALF = 0.06
MIN_FRAG_MZ = 150.0
MIN_CYCLES = 3
ROUGH_SEARCH = 3.0

SEARCH_FWHM = 2.0
CENTROID_FWHM = 1.0
EDGE_FWHM = 1.5
MIN_CYCLES_FOR_RT = 6
BUFFER_ROWS = 20_000_000
RT_WIDTH = 1.0


def select_window(hdf_path, rt_centre=None, rt_width=1.0):
    """Spectrum METADATA for an RT window - no peak data read."""
    with h5py.File(hdf_path, 'r') as h:
        sg = h['ms_data']['spectrum_df']
        ms_level = sg['ms_level'][:]
        rt = sg['rt'][:]
        q1 = sg['precursor_mz'][:]
        start = sg['peak_start_idx'][:].astype(np.int64)
        stop = sg['peak_stop_idx'][:].astype(np.int64)
        n_peaks = h['ms_data']['peak_df']['mz'].shape[0]
    if rt_centre is None:
        rt_centre = 0.5 * (rt.min() + rt.max())
    lo, hi = rt_centre - rt_width / 2, rt_centre + rt_width / 2
    sel = np.where((ms_level == 2) & (rt >= lo) & (rt <= hi))[0]
    if not sel.size:
        raise SystemExit(f'no MS2 spectra in RT {lo:.2f}-{hi:.2f} min')

    order = np.lexsort((rt[sel], q1[sel]))
    sel = sel[order]
    cyc = np.zeros(len(sel), dtype=np.int64)
    _, first = np.unique(q1[sel], return_index=True)
    for a_, b_ in zip(first, np.append(first[1:], len(sel))):
        cyc[a_:b_] = np.arange(b_ - a_)
    print(f'RT {lo:.2f}-{hi:.2f} min: {len(sel):,} MS2 spectra', flush=True)
    return {'idx': sel, 'q1': q1[sel], 'cycle': cyc, 'rt': rt[sel],
            'start': start[sel], 'stop': stop[sel], 'n_peaks': n_peaks}


def stream_spectra(hdf_path, w, which=None, buffer_rows=BUFFER_ROWS):
    """Yield (position in `w`, mz, intensity) through a SLIDING BUFFER."""
    pos = np.arange(len(w['idx'])) if which is None else np.asarray(which)
    pos = pos[np.argsort(w['start'][pos])]
    with h5py.File(hdf_path, 'r') as h:
        mz_ds = h['ms_data']['peak_df']['mz']
        int_ds = h['ms_data']['peak_df']['Intensity']
        buf_lo = buf_hi = -1
        mz_buf = int_buf = None
        for k in pos:
            s_, e_ = int(w['start'][k]), int(w['stop'][k])
            if e_ <= s_:
                continue
            if mz_buf is None or s_ < buf_lo or e_ > buf_hi:
                buf_lo = s_
                buf_hi = min(w['n_peaks'], max(e_, s_ + buffer_rows))
                mz_buf = mz_ds[buf_lo:buf_hi]
                int_buf = int_ds[buf_lo:buf_hi]
            m = mz_buf[s_ - buf_lo:e_ - buf_lo]
            v = int_buf[s_ - buf_lo:e_ - buf_lo]
            o = np.argsort(m, kind='stable')
            yield int(k), m[o], v[o]


def pick_targets_streamed(hdf_path, w, n, n_strata, every=20):
    """Distinct fragment m/z, stratified over abundance."""
    sub = np.arange(0, len(w['idx']), every)
    mzs_, its_ = [], []
    for _k, m, v in stream_spectra(hdf_path, w, sub):
        keep = m >= MIN_FRAG_MZ
        mzs_.append(m[keep])
        its_.append(v[keep])
    mz = np.concatenate(mzs_)
    it = np.concatenate(its_)
    order = np.argsort(it)[::-1]
    mz, it = mz[order], it[order]
    seen, keep_i = [], []
    for i in range(len(mz)):
        if seen and np.min(np.abs(np.asarray(seen) - mz[i])) <= 5 * FRAG_TOL:
            continue
        seen.append(mz[i])
        keep_i.append(i)
        if len(seen) >= n * 4:
            break
    mzs, ints = np.asarray(seen), it[keep_i]
    edges = np.quantile(np.log10(ints), np.linspace(0, 1, n_strata + 1))
    rng, chosen = np.random.default_rng(0), []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = np.where((np.log10(ints) >= a) & (np.log10(ints) <= b))[0]
        if sel.size:
            chosen.extend(rng.choice(sel, size=min(max(1, n // n_strata),
                                                   sel.size), replace=False))
    return np.sort(mzs[np.asarray(sorted(set(chosen)))])


def q1_profiles_streamed(hdf_path, w, q1_axis, n_cyc, targets):
    """prof[cycle, q1, target], filled in ONE streaming pass."""
    prof = np.zeros((n_cyc, len(q1_axis), len(targets)))
    qi_of = {float(q): i for i, q in enumerate(q1_axis)}
    for k, m, v in stream_spectra(hdf_path, w):
        qi = qi_of.get(float(w['q1'][k]))
        if qi is None:
            continue
        lo = np.asarray(targets - FRAG_TOL, dtype=m.dtype)
        hi = np.asarray(targets + FRAG_TOL, dtype=m.dtype)
        left, right = window_indices(m, lo, hi)
        prof[int(w['cycle'][k]), qi] += [v[a:b].sum()
                                         for a, b in zip(left, right)]
    return prof


def rough_q1_fwhm(prof, q1_axis):
    """Median Q1 FWHM of the strongest targets, to size every later window."""
    summed = prof.sum(axis=0)
    n_strong = max(5, summed.shape[1] // 20)
    w = []
    for ti in np.argsort(summed.max(axis=0))[::-1][:n_strong]:
        sh = peak_shape(q1_axis, summed[:, ti], search=ROUGH_SEARCH)
        if np.isfinite(sh['width']):
            w.append(float(sh['width']))
    return float(np.median(w)) if w else np.nan


def weighted_centroid(x, y, centre, half):
    """Intensity-weighted mean inside a fixed window around `centre`."""
    m = np.abs(x - centre) <= half
    if m.sum() < 3:
        return np.nan
    tot = y[m].sum()
    return float(np.average(x[m], weights=y[m])) if tot > 0 else np.nan


def mz_samples(store, q1, ci, frag):
    """Raw (m/z, intensity) samples near `frag`, from the pass-C store."""
    got = store.get((float(q1), int(ci)))
    if got is None:
        return None, None
    mz_s, int_s = got
    a, b = window_indices(mz_s,
                          np.asarray([frag - MZ_HALF], dtype=mz_s.dtype),
                          np.asarray([frag + MZ_HALF], dtype=mz_s.dtype))
    return mz_s[a[0]:b[0]], int_s[a[0]:b[0]]


def prefetch_for_mz(hdf_path, w, needed):
    """PASS C: read only the (q1, cycle) spectra the m/z measurements need."""
    want = {(float(q), int(c)) for q, c in needed}
    pos = [k for k in range(len(w['idx']))
           if (float(w['q1'][k]), int(w['cycle'][k])) in want]
    store = {}
    for k, m, v in stream_spectra(hdf_path, w, pos):
        store[(float(w['q1'][k]), int(w['cycle'][k]))] = (m, v)
    print(f'  pass C: {len(store):,} spectra re-read for the m/z panels')
    return store


def measure(store, q1_axis, n_cyc, targets, prof, fwhm0, rt_axis):
    """Per fragment: Q1 width/precision, m/z width/precision, RT width."""
    search, cen_win = SEARCH_FWHM * fwhm0, CENTROID_FWHM * fwhm0
    lo_edge = q1_axis.min() + EDGE_FWHM * fwhm0
    hi_edge = q1_axis.max() - EDGE_FWHM * fwhm0
    rows = []
    for ti, frag in enumerate(targets):
        elution = prof[:, :, ti].sum(axis=1)
        if not np.isfinite(elution).any() or elution.max() <= 0:
            continue
        rt_fwhm = np.nan
        if n_cyc >= MIN_CYCLES_FOR_RT:
            rs = peak_shape(rt_axis, elution)
            apex_rt = float(rs['apex_x'])
            if np.isfinite(rs['width']):
                rt_fwhm = float(rs['width'])
                keep_c = np.abs(rt_axis - apex_rt) <= rs['width']
            else:
                keep_c = np.abs(rt_axis - apex_rt) <= 2 * np.median(
                    np.diff(rt_axis))
        else:
            keep_c = np.ones(n_cyc, dtype=bool)
        if keep_c.sum() < MIN_CYCLES:
            keep_c = np.zeros(n_cyc, dtype=bool)
            keep_c[np.argsort(elution)[::-1][:MIN_CYCLES]] = True

        summed = prof[keep_c, :, ti].sum(axis=0)
        sh = peak_shape(q1_axis, summed, search=search)
        if not np.isfinite(sh['width']):
            continue
        apex = float(sh['apex_x'])
        if not (lo_edge <= apex <= hi_edge):
            continue
        cen = [weighted_centroid(q1_axis, prof[ci, :, ti], apex, cen_win)
               for ci in np.where(keep_c)[0]]
        cen = [c for c in cen if np.isfinite(c)]
        if len(cen) < MIN_CYCLES:
            continue

        qa = q1_axis[np.argmin(np.abs(q1_axis - apex))]
        pooled_mz, pooled_it, mz_cen = [], [], []
        for ci in np.where(keep_c)[0]:
            m, i = mz_samples(store, qa, int(ci), frag)
            if m is None or len(m) < 3:
                continue
            pooled_mz.append(m)
            pooled_it.append(i)
            if i.sum() > 0:
                mz_cen.append(float(np.average(m, weights=i)))
        mz_fwhm = np.nan
        if pooled_mz:
            am = np.concatenate(pooled_mz)
            ai = np.concatenate(pooled_it)
            o = np.argsort(am)
            uniq, inv = np.unique(am[o], return_inverse=True)
            summed_mz = np.bincount(inv, weights=ai[o])
            if len(uniq) >= 4:
                s2 = peak_shape(uniq, summed_mz)
                if np.isfinite(s2['width']):
                    mz_fwhm = float(s2['width'])

        q1_cen = 0.5 * (float(sh['left_x']) + float(sh['right_x']))
        rows.append({
            'fragment_mz': frag,
            'apex_intensity': float(sh['height']),
            'q1_fwhm': float(sh['width']),
            'q1_centroid': q1_cen,
            'q1_resolution': q1_cen / float(sh['width']),
            'q1_precision': float(np.std(cen, ddof=1)),
            'mz_fwhm': mz_fwhm,
            'mz_resolution': (frag / mz_fwhm if np.isfinite(mz_fwhm)
                              and mz_fwhm > 0 else np.nan),
            'mz_precision': (float(np.std(mz_cen, ddof=1))
                             if len(mz_cen) >= MIN_CYCLES else np.nan),
            'rt_fwhm_s': rt_fwhm * 60 if np.isfinite(rt_fwhm) else np.nan,
            'n_cycles': len(cen),
        })
    return pd.DataFrame(rows)


def vs_panels(datasets, params, out_base, subtitle):
    """One panel per (parameter x axis), one hue per dataset."""
    XLAB = {'log10_intensity': 'log$_{10}$ apex intensity',
            'fragment_mz': 'Fragment m/z',
            'q1_centroid': 'Q1 m/z'}
    fig, axes = plt.subplots(len(params), 2,
                             figsize=(10.4, 3.3 * len(params)), squeeze=False)
    for r, (col, ylabel, logy, second) in enumerate(params):
        for c, xk in enumerate(('log10_intensity', second)):
            xlabel = XLAB[xk]
            ax = axes[r][c]
            drawn = False
            for di, (label, d) in enumerate(datasets):
                hue = core.PALETTE_SINGLE[di % len(core.PALETTE_SINGLE)]
                sub = d[np.isfinite(d[col])]
                if len(sub) < 10:
                    continue
                drawn = True
                ax.scatter(sub[xk], sub[col], s=7, color=hue, alpha=0.22,
                           linewidth=0, rasterized=True)
                bs = core.binned_stats(sub[xk].to_numpy(float),
                                       sub[col].to_numpy(float), bins=10)
                x = bs['x'].to_numpy(float)
                lab = label
                if logy and xk == 'log10_intensity':
                    sl = np.polyfit(sub[xk].to_numpy(float),
                                    np.log10(sub[col].to_numpy(float)), 1)[0]
                    lab = f'{label}  (slope {sl:+.2f})'
                ax.plot(x, bs['median'].to_numpy(float), color=hue, lw=1.9,
                        marker='o', ms=4, label=lab)
                ax.fill_between(x, bs['q_lo'].to_numpy(float),
                                bs['q_hi'].to_numpy(float), color=hue,
                                alpha=0.16, lw=0)
            if not drawn:
                ax.text(0.5, 0.5, f'{col}: too few measurable', ha='center',
                        va='center', transform=ax.transAxes, fontsize=9)
                ax.set_axis_off()
                continue
            if logy:
                ax.set_yscale('log')
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if len(datasets) > 1 or (logy and xk == 'log10_intensity'):
                ax.legend(frameon=False, fontsize=8, loc='best')
    fig.suptitle(subtitle, fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
    print(f'wrote {out_base}.pdf / .png')


def plan_mz_reads(q1_axis, n_cyc, targets, prof, fwhm0, rt_axis):
    """Which (q1, cycle) spectra the m/z panels will need."""
    need = []
    for ti in range(len(targets)):
        elution = prof[:, :, ti].sum(axis=1)
        if elution.max() <= 0:
            continue
        if n_cyc >= MIN_CYCLES_FOR_RT:
            rs = peak_shape(rt_axis, elution)
            apex_rt = float(rs['apex_x'])
            keep_c = (np.abs(rt_axis - apex_rt) <= rs['width']
                      if np.isfinite(rs['width'])
                      else np.abs(rt_axis - apex_rt) <=
                      2 * np.median(np.diff(rt_axis)))
        else:
            keep_c = np.ones(n_cyc, dtype=bool)
        if keep_c.sum() < MIN_CYCLES:
            keep_c = np.zeros(n_cyc, dtype=bool)
            keep_c[np.argsort(elution)[::-1][:MIN_CYCLES]] = True
        summed = prof[keep_c, :, ti].sum(axis=0)
        sh = peak_shape(q1_axis, summed, search=SEARCH_FWHM * fwhm0)
        if not np.isfinite(sh['width']):
            continue
        qa = q1_axis[np.argmin(np.abs(q1_axis - float(sh['apex_x'])))]
        need += [(qa, ci) for ci in np.where(keep_c)[0]]
    return need


def scan_one(hdf_path, rt_centre):
    """Measure one file in three streaming passes; nothing large is held."""
    w = select_window(hdf_path, rt_centre, RT_WIDTH)
    q1_axis = np.unique(w['q1'])
    n_cyc = int(w['cycle'].max()) + 1
    q1_step = float(np.median(np.diff(q1_axis)))
    print(f'  {len(q1_axis)} Q1 positions (step {q1_step:.3f} Th) x '
          f'{n_cyc} cycles')

    targets = pick_targets_streamed(hdf_path, w, N_TARGETS, N_STRATA)
    prof = q1_profiles_streamed(hdf_path, w, q1_axis, n_cyc, targets)
    fwhm0 = rough_q1_fwhm(prof, q1_axis)
    if not np.isfinite(fwhm0):
        raise SystemExit('could not measure a rough Q1 FWHM')
    print(f'  rough Q1 FWHM {fwhm0:.3f} Th -> search +-{SEARCH_FWHM * fwhm0:.2f} Th'
          f'  ({fwhm0 / q1_step:.1f} samples across the FWHM)')

    rt_axis = np.array([w['rt'][w['cycle'] == c].mean() for c in range(n_cyc)])
    need = plan_mz_reads(q1_axis, n_cyc, targets, prof, fwhm0, rt_axis)
    store = prefetch_for_mz(hdf_path, w, need)

    d = measure(store, q1_axis, n_cyc, targets, prof, fwhm0, rt_axis)
    d['log10_intensity'] = np.log10(d['apex_intensity'])
    d['q1_step'] = q1_step
    return d, fwhm0, n_cyc


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    specs, rt_centre = [], None
    for arg in sys.argv[1:]:
        if arg.startswith('--rt='):
            rt_centre = float(arg.split('=', 1)[1])
            continue
        path, _, lab = arg.partition('=')
        specs.append((path, lab or os.path.basename(path)[:26]))
    core.init_plotting()

    datasets, widths = [], []
    for path, lab in specs:
        print(f'=== {lab}')
        d, fwhm0, n_cyc = scan_one(path, rt_centre)
        print(f'  {len(d)} fragments measured; Q1 span '
              f'{d.q1_centroid.min():.0f}-{d.q1_centroid.max():.0f} Th')
        d['condition'] = lab
        datasets.append((lab, d))
        widths.append(fwhm0)

    params = [('q1_fwhm', 'Q1 FWHM [Th]', False, 'q1_centroid'),
              ('q1_resolution', r'Q1 resolving power (m/$\Delta$m)', False,
               'q1_centroid'),
              ('q1_precision', 'Q1 centroid SD [Th]', True, 'q1_centroid'),
              ('mz_fwhm', 'm/z FWHM [Th]', True, 'fragment_mz'),
              ('mz_precision', 'm/z centroid SD [Th]', True, 'fragment_mz'),
              ('rt_fwhm_s', 'Chromatographic FWHM [s]', False, 'fragment_mz')]
    stem = 'q1_resolution_scan' if len(datasets) == 1 else         'q1_resolution_scan_compare'
    base = os.path.join(OUT_DIR, stem)
    vs_panels(datasets, params, base,
              'ZT Scan performance vs abundance and m/z · ' +
              ' vs '.join(f'{l} ({len(d)} frag)' for l, d in datasets))
    pd.concat([d for _, d in datasets], ignore_index=True).to_csv(
        base + '_sourcedata.csv', index=False)

    print()
    hdr = f'{"condition":26s} {"n":>5s} {"Q1 FWHM":>9s} {"Q1 R":>7s} '           f'{"Q1 prec":>9s} {"m/z FWHM":>9s} {"RT FWHM":>8s}'
    print(hdr)
    print('-' * len(hdr))
    for lab, d in datasets:
        print(f'{lab:26s} {len(d):5d} {d.q1_fwhm.median():9.3f} '
              f'{d.q1_resolution.median():7.0f} {d.q1_precision.median():9.4f} '
              f'{d.mz_fwhm.median():9.4f} {d.rt_fwhm_s.median():8.2f}')


if __name__ == '__main__':
    main()
