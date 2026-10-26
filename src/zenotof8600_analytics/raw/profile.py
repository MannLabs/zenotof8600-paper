"""Turning native PROFILE samples into centroided peaks."""

from __future__ import annotations

import numpy as np

LATTICE_AT_500 = 0.00436
GAP_FACTOR = 2.0


def lattice_step(mz):
    """Native TOF sample spacing at m/z."""
    return LATTICE_AT_500 * np.sqrt(np.asarray(mz, float) / 500.0)


def _valley_breaks(inten, starts, valley_frac):
    """Break points where two peaks OVERLAP without a gap between them."""
    ends = np.append(starts[1:], len(inten))
    out = []
    for a, b in zip(starts, ends):
        if b - a < 5:
            continue
        y = inten[a:b]
        mins = np.flatnonzero((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:])) + 1
        for i in mins:
            left, right = y[:i].max(), y[i + 1:].max()
            if left > 0 and right > 0 and y[i] < valley_frac * min(left, right):
                out.append(a + i + 1)
    return np.asarray(sorted(out), dtype=int)


def centroid_profile(mz, inten, qw=None, extra=(), position_thresh=0.7,
                     valley_frac=0.8):
    """Collapse native PROFILE samples into centroided peaks."""
    if not len(mz):
        e = np.array([])
        base = (e, e, e, e)
        return base + ([np.array([]) for _ in extra],) if extra else base
    brk = np.where(np.diff(mz) > GAP_FACTOR * lattice_step(mz[:-1]))[0] + 1
    starts = np.concatenate(([0], brk)).astype(int)
    if valley_frac is not None:
        extra_brk = _valley_breaks(inten, starts, valley_frac)
        if extra_brk.size:
            starts = np.unique(np.concatenate((starts, extra_brk)))
    tot = np.add.reduceat(inten, starts)
    good = tot > 0
    npts = np.diff(np.concatenate((starts, [len(mz)])))
    hm = inten >= position_thresh * np.repeat(
        np.maximum.reduceat(inten, starts), npts)
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
