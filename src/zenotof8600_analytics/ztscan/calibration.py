"""Q1 axis calibration from residual precursors - no identifications needed."""

from __future__ import annotations

import numpy as np
import pandas as pd

NEUTRON = 1.00335


def binned_median(x, y, n_bins, *, min_n=150):
    """Quantile-binned median with the standard error of the median."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    q = pd.qcut(pd.Series(x).rank(method='first'), n_bins, labels=False)
    c, m, e, nn = [], [], [], []
    for k in range(n_bins):
        s = (q == k).to_numpy()
        if s.sum() < min_n:
            continue
        v = y[s]
        sig = np.subtract(*np.percentile(v, [75, 25])) / 1.349
        c.append(float(np.median(x[s])))
        m.append(float(np.median(v)))
        e.append(float(1.253 * sig / np.sqrt(s.sum())))
        nn.append(int(s.sum()))
    return (np.asarray(c), np.asarray(m), np.asarray(e), np.asarray(nn))


def _loess(x, y, at, frac=0.30):
    """LOESS through (x, y), evaluated at `at`, constant outside the range."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    f = lowess(y, x, frac=frac, it=3, return_sorted=True)
    return np.interp(at, f[:, 0], f[:, 1], left=f[0, 1], right=f[-1, 1])


def fit_q1_axis(q1_pos, precursor_mz, rt=None, *, n_mz_bins=60,
                min_per_bin=200, loess_frac=0.30, n_knots=24):
    """Fit a Q1 correction table from residual-precursor measurements."""
    q1_pos = np.asarray(q1_pos, float)
    precursor_mz = np.asarray(precursor_mz, float)
    dev = precursor_mz - q1_pos
    ok = np.isfinite(dev) & np.isfinite(q1_pos)
    if rt is not None:
        rt = np.asarray(rt, float)
        ok &= np.isfinite(rt)
    q1_pos, dev = q1_pos[ok], dev[ok]
    rt = rt[ok] if rt is not None else None
    if q1_pos.size < max(min_per_bin, 2):
        raise ValueError(f'too few measurements to fit: {q1_pos.size} '
                         f'(need at least min_per_bin = {min_per_bin})')
    n_mz_bins = int(max(2, min(n_mz_bins, q1_pos.size // min_per_bin)))

    slope = 0.0
    if rt is not None and np.ptp(rt) > 0:
        c, m, _, _ = binned_median(q1_pos, dev, n_mz_bins, min_n=min_per_bin)
        resid = dev - np.interp(q1_pos, c, m)
        rc, rm, _, rn = binned_median(
            rt, resid, max(2, min(20, rt.size // min_per_bin)),
            min_n=min_per_bin)
        if rc.size >= 2:
            slope = float(np.polyfit(rc, rm, 1, w=np.sqrt(rn))[0])

    c, m, _, _ = binned_median(q1_pos, dev - (slope * rt if rt is not None
                                              else 0.0),
                               n_mz_bins, min_n=min_per_bin)
    if c.size < 2:
        raise ValueError(
            f'no m/z bin reached min_per_bin={min_per_bin} from '
            f'{q1_pos.size} measurements; lower min_per_bin or supply more')
    knots = np.linspace(q1_pos.min(), q1_pos.max(), n_knots)
    corr = _loess(c, m, knots, frac=loess_frac)
    return dict(kind='table',
                mz_knots=[float(v) for v in knots],
                mz_corr=[float(-v) for v in corr],
                slope_rt=float(-slope),
                n_precursors=int(q1_pos.size),
                source='residual_precursor')


