"""Peak-level operations: merging, matching, picking, integrating."""

from __future__ import annotations

import numpy as np

from .loading import _mz_window


def merge_peaks(mz, intensity, *, tol=1.0, tol_unit='ppm',
                position='median', scan_id=None):
    """Merge peaks from several spectra onto one ascending m/z axis.

    Parameters
      mz, intensity:  concatenated samples from any number of spectra, any
                      order.
      tol/tol_unit:   samples closer than this are the same peak. 'ppm'
                      (default) or 'Da'.
      position:       how a merged peak's m/z is decided: 'median' (default)
                      is the median of the distinct input positions, ignoring
                      intensity, for profile samples; 'weighted' is the
                      intensity-weighted mean, for centroids.
      scan_id:        optional, one label per sample naming the spectrum it
                      came from. With it, a merged point holds at most one
                      sample per spectrum: a group that would take two samples
                      of the same spectrum is cut at its largest internal gap
                      until none does.

    Returns
      (mz, intensity) arrays, ascending, one entry per merged peak. Intensity
      is always the sum.
    """
    mz = np.asarray(mz, dtype=np.float64).ravel()
    intensity = np.asarray(intensity, dtype=np.float64).ravel()
    if mz.size != intensity.size:
        raise ValueError(f'mz and intensity differ in length: '
                         f'{mz.size} vs {intensity.size}')
    if position not in ('median', 'weighted'):
        raise ValueError(f"position must be 'median' or 'weighted', "
                         f"got {position!r}")
    if mz.size == 0:
        return np.empty(0), np.empty(0)
    if not np.all(np.isfinite(mz)):
        raise ValueError('mz contains non-finite values')

    if scan_id is not None:
        scan_id = np.asarray(scan_id).ravel()
        if scan_id.size != mz.size:
            raise ValueError(f'scan_id and mz differ in length: '
                             f'{scan_id.size} vs {mz.size}')

    order = np.argsort(mz, kind='stable')
    mz, intensity = mz[order], intensity[order]
    if mz.size == 1:
        return mz.copy(), intensity.copy()

    d = np.diff(mz)
    if tol_unit == 'ppm':
        tol_abs = mz[:-1] * tol * 1e-6
    elif tol_unit in ('da', 'Da', 'th', 'Th'):
        tol_abs = np.full(mz.size - 1, float(tol))
    else:
        raise ValueError(f"tol_unit must be 'ppm' or 'Da', got {tol_unit!r}")

    new_grp = d > tol_abs
    if scan_id is not None:
        new_grp = _split_repeated_scans(new_grp, d, scan_id[order])
    grp = np.concatenate(([0], np.cumsum(new_grp)))
    inten_out = np.bincount(grp, weights=intensity)

    if position == 'weighted':
        tot = np.where(inten_out > 0, inten_out, 1.0)
        wmz = np.bincount(grp, weights=intensity * mz) / tot
        starts = np.concatenate(([True], new_grp))
        return np.where(inten_out > 0, wmz, mz[starts]), inten_out

    if not np.any((d > 0) & ~new_grp):
        return mz[np.concatenate(([True], new_grp))], inten_out

    new_val = np.concatenate(([True], (d > 0) | new_grp))
    vals = mz[new_val]
    cnt = np.bincount(grp[new_val], minlength=inten_out.size)
    start = np.concatenate(([0], np.cumsum(cnt)[:-1]))
    lo = start + (cnt - 1) // 2
    hi = start + cnt // 2
    return 0.5 * (vals[lo] + vals[hi]), inten_out

def _split_repeated_scans(new_grp, d, scan):
    """Add group boundaries until no group holds two samples of one scan."""
    n = scan.size
    grp = np.concatenate(([0], np.cumsum(new_grp)))
    _, inv = np.unique(scan, return_inverse=True)
    m = int(inv.max()) + 1
    key = np.sort(grp.astype(np.int64) * m + inv)
    dup_keys = key[1:][key[1:] == key[:-1]]
    if dup_keys.size == 0:
        return new_grp
    starts = np.concatenate(([0], np.flatnonzero(new_grp) + 1))
    stops = np.concatenate((starts[1:], [n]))
    new_grp = new_grp.copy()
    for g in np.unique(dup_keys // m):
        a, b = int(starts[g]), int(stops[g])
        for c in _segment_one_per_scan(inv[a:b], d[a:b - 1]):
            new_grp[a + c - 1] = True
    return new_grp


def _segment_one_per_scan(scan, gaps):
    """Cut one sorted run into contiguous segments with no scan repeated."""
    L = scan.size
    big = float(np.sum(gaps)) + 1.0
    cost = np.full(L + 1, np.inf)
    back = np.zeros(L + 1, dtype=np.int64)
    cost[0] = 0.0
    last = {}
    lo = 0
    for j in range(1, L + 1):
        s = int(scan[j - 1])
        if s in last and last[s] >= lo:
            lo = last[s] + 1
        last[s] = j - 1
        for i in range(lo, j):
            c = cost[i] + (big - gaps[i - 1] if i > 0 else 0.0)
            if c < cost[j]:
                cost[j], back[j] = c, i
    cuts, j = [], L
    while j > 0:
        i = int(back[j])
        if i > 0:
            cuts.append(i)
        j = i
    return cuts


def profile_tolerance_ok(mz, tol, tol_unit='ppm'):
    """True when ``tol`` is small enough not to chain profile samples."""
    mz = np.asarray(mz, dtype=np.float64).ravel()
    if mz.size < 3:
        return True
    d = np.diff(np.sort(mz))
    d = d[d > 0]
    if d.size == 0:
        return True
    step = float(np.percentile(d, 10))
    centre = float(np.median(mz))
    tol_abs = centre * tol * 1e-6 if tol_unit == 'ppm' else float(tol)
    return tol_abs < step

def window_indices(sorted_mz, lo, hi):
    """`searchsorted` bounds for one or many [lo, hi] m/z windows."""
    mz = np.asarray(sorted_mz)
    scalar = np.isscalar(lo) and np.isscalar(hi)
    lo_a = np.asarray(lo, dtype=mz.dtype)
    hi_a = np.asarray(hi, dtype=mz.dtype)
    a = np.searchsorted(mz, lo_a, side='left')
    b = np.searchsorted(mz, hi_a, side='right')
    if scalar:
        return int(a), int(b)
    return a, b

def match_peaks(spec_mz, spec_intensity, target_mz, *, tol=20.0, tol_unit='ppm',
                reduce='max'):
    """Match a spectrum's peaks to target m/z values within a tolerance.

    Parameters
      spec_mz, spec_intensity:  the spectrum (any order).
      target_mz:                scalar or array of m/z to look up.
      tol / tol_unit:           match window (default 20 ppm).
      reduce:                   'max' or 'sum' within each window.
    """
    smz = np.asarray(spec_mz, dtype=float)
    sint = np.asarray(spec_intensity, dtype=float)
    targets = np.atleast_1d(np.asarray(target_mz, dtype=float))
    order = np.argsort(smz)
    smz, sint = smz[order], sint[order]

    out_int = np.zeros(targets.size, dtype=float)
    out_mz = np.full(targets.size, np.nan, dtype=float)
    for i, t in enumerate(targets):
        lo, hi = _mz_window(float(t), tol, tol_unit)
        a = np.searchsorted(smz, lo, side='left')
        b = np.searchsorted(smz, hi, side='right')
        if b <= a:
            continue
        seg_int, seg_mz = sint[a:b], smz[a:b]
        if reduce == 'max':
            j = int(np.argmax(seg_int))
            out_int[i], out_mz[i] = seg_int[j], seg_mz[j]
        elif reduce == 'sum':
            s = float(seg_int.sum())
            out_int[i] = s
            out_mz[i] = float((seg_mz * seg_int).sum() / s) if s > 0 else np.nan
        else:
            raise ValueError(f"reduce must be 'max' or 'sum', got {reduce!r}")
    return out_int, out_mz


OFFPEAK_EXCLUDE_FWHM = 3.0
OFFPEAK_MIN_POINTS = 5

def _baseline_value(intensity, baseline):
    """Resolve a baseline spec against a trace."""
    if baseline is None:
        return 0.0
    if isinstance(baseline, str):
        if baseline == 'median':
            return float(np.median(intensity))
        if baseline.startswith('p') and baseline != 'offpeak':
            return float(np.percentile(intensity, float(baseline[1:])))
        raise ValueError(
            "baseline must be 'median', 'offpeak', 'p<percentile>', None or a "
            f'number, got {baseline!r}')
    return float(baseline)

def pick_peak(rt, intensity, *, rt_range=None, height_fraction=0.5,
              baseline='median', interpolate_points=None, min_sn=None):
    """Pick the dominant chromatographic peak by half-maximum over a baseline.

    Parameters
      rt, intensity:    equal-length arrays (e.g. from ``extract_xic``).
      rt_range:         optional (min, max) to restrict picking.
      height_fraction:  fraction of apex height above baseline (0.5 = FWHM).
      baseline:         'offpeak' measures the peak, excludes
                        ``OFFPEAK_EXCLUDE_FWHM`` half-widths either side of the
                        apex, takes the median of the rest and re-measures,
                        falling back to 'p5' when fewer than
                        ``OFFPEAK_MIN_POINTS`` remain; 'median' (default) is
                        the median of the window; 'p<k>' the k-th percentile;
                        None/0 for zero, or a float.
      interpolate_points:
                        resample onto this many evenly spaced points before
                        measuring.
      min_sn:           return the empty result unless ``apex / baseline``
                        reaches this. Only meaningful with a non-zero baseline.
    """
    rt = np.asarray(rt, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if rt_range is not None:
        lo, hi = rt_range
        m = (rt >= lo) & (rt <= hi)
        rt, intensity = rt[m], intensity[m]

    empty = dict(apex_rt=np.nan, apex_intensity=np.nan, left_rt=np.nan,
                 right_rt=np.nan, fwhm=0.0, area=0.0, baseline=np.nan,
                 sn=np.nan, n_points=0)
    if intensity.size == 0 or not np.any(intensity > 0):
        return empty

    if interpolate_points and rt.size >= 2:
        fine_rt = np.linspace(rt.min(), rt.max(), int(interpolate_points))
        intensity = np.interp(fine_rt, rt, intensity)
        rt = fine_rt

    if isinstance(baseline, str) and baseline == 'offpeak':
        rough = pick_peak(rt, intensity, height_fraction=height_fraction,
                          baseline='p5')
        off = None
        if np.isfinite(rough['apex_rt']) and rough['fwhm'] > 0:
            keep = (np.abs(rt - rough['apex_rt'])
                    > OFFPEAK_EXCLUDE_FWHM * rough['fwhm'] / 2.0)
            if keep.sum() >= OFFPEAK_MIN_POINTS:
                off = float(np.median(intensity[keep]))
        baseline = off if off is not None else 'p5'

    base = _baseline_value(intensity, baseline)
    apex = int(np.argmax(intensity))
    sn = float(intensity[apex] / base) if base > 0 else np.inf
    if min_sn is not None and not (sn >= min_sn):
        return empty
    thr = base + height_fraction * (intensity[apex] - base)

    lo = apex
    while lo > 0 and intensity[lo] > thr:
        lo -= 1
    hi = apex
    while hi < len(intensity) - 1 and intensity[hi] > thr:
        hi += 1

    _trapz = getattr(np, 'trapezoid', np.trapz)
    area = float(_trapz(intensity[lo:hi + 1], rt[lo:hi + 1]))
    return dict(
        apex_rt=float(rt[apex]), apex_intensity=float(intensity[apex]),
        left_rt=float(rt[lo]), right_rt=float(rt[hi]),
        fwhm=float(rt[hi] - rt[lo]), area=area, baseline=base, sn=sn,
        n_points=int(hi - lo + 1),
    )

def points_in_window(rt, center, half_width):
    """Number of acquired points within ``center +/- half_width``."""
    rt = np.asarray(rt, dtype=float)
    return int(np.sum((rt >= center - half_width) & (rt <= center + half_width)))

def integrate_peak(rt, intensity, rt_start, rt_end, *, baseline='min_boundary'):
    """Trapezoidal area between two retention times, over a baseline."""
    rt = np.asarray(rt, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    m = (rt >= rt_start) & (rt <= rt_end)
    rt_w, int_w = rt[m], intensity[m]
    if rt_w.size < 2:
        return dict(area=0.0, baseline=np.nan, n_points=int(rt_w.size),
                    rt_start=float(rt_start), rt_end=float(rt_end))

    rt = np.asarray(rt, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if baseline == 'min_boundary':
        base = float(min(int_w[0], int_w[-1]))
    elif baseline == 'none':
        base = 0.0
    elif baseline == 'offpeak':
        half = (rt_end - rt_start) / 2.0
        centre = (rt_start + rt_end) / 2.0
        off = np.abs(rt - centre) > OFFPEAK_EXCLUDE_FWHM * half
        base = (float(np.mean(intensity[off]))
                if off.sum() >= OFFPEAK_MIN_POINTS
                else _baseline_value(int_w, 'p5'))
    else:
        base = _baseline_value(int_w, baseline)

    _trapz = getattr(np, 'trapezoid', np.trapz)
    area = float(_trapz(np.clip(int_w - base, 0, None), rt_w))
    return dict(area=area, baseline=base, n_points=int(rt_w.size),
                rt_start=float(rt_start), rt_end=float(rt_end))

def peak_shape(x, y, *, centre=None, search=None, height_fraction=0.5):
    """Height, width and area of the peak in ``y`` nearest ``centre``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nan = dict(apex_x=np.nan, height=np.nan, width=np.nan, area=np.nan,
               left_x=np.nan, right_x=np.nan, n_points=0)
    if x.size != y.size or x.size < 3 or not np.isfinite(y).any():
        return nan

    seg_x, seg_y = x, y
    if centre is not None and search is not None:
        lo_i, hi_i = window_indices(x, centre - search, centre + search)
        if hi_i - lo_i < 1:
            return nan
        k = int(lo_i) + int(np.argmax(y[lo_i:hi_i]))
    elif centre is not None:
        k = int(np.argmin(np.abs(x - centre)))
    else:
        k = int(np.argmax(y))
    apex = seg_y[k]
    if not np.isfinite(apex) or apex <= 0:
        return nan

    i = k
    while i > 0 and seg_y[i - 1] < seg_y[i]:
        i -= 1
    j = k
    while j < seg_y.size - 1 and seg_y[j + 1] < seg_y[j]:
        j += 1
    area = float(np.trapezoid(seg_y[i:j + 1], seg_x[i:j + 1])) if j > i else 0.0

    level = apex * height_fraction

    def crossing(step):
        p = k
        while i <= p + step <= j and seg_y[p + step] > level:
            p += step
        q = p + step
        if not (i <= q <= j) or seg_y[p] == seg_y[q]:
            return np.nan
        f = (seg_y[p] - level) / (seg_y[p] - seg_y[q])
        return float(seg_x[p] + f * (seg_x[q] - seg_x[p]))

    left, right = crossing(-1), crossing(+1)
    width = abs(right - left) if np.isfinite(left) and np.isfinite(right) else np.nan
    return dict(apex_x=float(seg_x[k]), height=float(apex), width=width,
                area=area, left_x=left, right_x=right, n_points=int(j - i + 1))
