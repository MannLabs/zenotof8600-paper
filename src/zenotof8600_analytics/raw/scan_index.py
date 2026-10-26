"""Numpy view of one run's spectra, built once, for fast scan selection."""
from __future__ import annotations

import numpy as np

try:
    import numba as _nb
except ImportError:
    _nb = None

_ATTR = '_pa_scan_index'

WINDOW_CHOICES = ('central', 'union')

RECENTRE_SCAN_CHOICES = ('own', 'all')


def _check(value, choices, name):
    if value not in choices:
        raise ValueError(f'{name} must be one of {choices}, got {value!r}')


class ScanIndex:
    """Sorted positions of one ms_data's spectra, per level and per window."""

    def __init__(self, ms_data):
        s = ms_data.spectrum_df
        self.n_spec = len(s)
        self.rt = s['rt'].to_numpy(float)
        self.level = (s['ms_level'].to_numpy() if 'ms_level' in s.columns
                      else np.zeros(self.n_spec, dtype=int))
        self.start = s['peak_start_idx'].to_numpy(np.int64)
        self.stop = s['peak_stop_idx'].to_numpy(np.int64)
        self.pmz = ms_data.peak_df['mz'].to_numpy()
        self.pint = ms_data.peak_df['intensity'].to_numpy()
        self._peak_df_id = id(ms_data.peak_df)

        self._by_level = {}
        for lev in [None, *np.unique(self.level).tolist()]:
            pos = (np.arange(self.n_spec) if lev is None
                   else np.nonzero(self.level == lev)[0])
            o = np.argsort(self.rt[pos], kind='stable')
            self._by_level[lev] = (self.rt[pos[o]], pos[o])

        self.has_isolation = ('isolation_lower_mz' in s.columns
                              and 'isolation_upper_mz' in s.columns)
        self.win_level = np.array([], dtype=self.level.dtype)
        self.win_lo = self.win_hi = np.array([], dtype=float)
        self._win_scans = []
        if self.has_isolation:
            lo = s['isolation_lower_mz'].to_numpy(float)
            hi = s['isolation_upper_mz'].to_numpy(float)
            ok = np.isfinite(lo) & np.isfinite(hi)
            pos = np.nonzero(ok)[0]
            if pos.size:
                key = np.lexsort((self.rt[pos], hi[pos], lo[pos],
                                  self.level[pos]))
                pos = pos[key]
                kl, klo, khi = self.level[pos], lo[pos], hi[pos]
                brk = np.nonzero((np.diff(kl) != 0) | (np.diff(klo) != 0)
                                 | (np.diff(khi) != 0))[0] + 1
                edges = np.concatenate(([0], brk, [pos.size]))
                wl, wlo, whi = [], [], []
                for a, b in zip(edges[:-1], edges[1:]):
                    p = pos[a:b]
                    p = p[np.argsort(self.rt[p], kind='stable')]
                    wl.append(kl[a])
                    wlo.append(klo[a])
                    whi.append(khi[a])
                    self._win_scans.append((self.rt[p], p))
                self.win_level = np.asarray(wl)
                self.win_lo = np.asarray(wlo, float)
                self.win_hi = np.asarray(whi, float)

    def valid_for(self, ms_data):
        return (len(ms_data.spectrum_df) == self.n_spec
                and id(ms_data.peak_df) == self._peak_df_id)

    @staticmethod
    def _slice(sorted_rt, pos, rt_range):
        if rt_range is None:
            return pos
        a = np.searchsorted(sorted_rt, rt_range[0], 'left')
        b = np.searchsorted(sorted_rt, rt_range[1], 'right')
        return pos[a:b]

    def windows_for(self, precursor_mz, ms_level=None, window='central'):
        """Indices of the isolation window(s) an XIC of this precursor reads."""
        _check(window, WINDOW_CHOICES, 'window')
        hit = (self.win_lo <= precursor_mz) & (self.win_hi >= precursor_mz)
        if ms_level is not None:
            hit &= self.win_level == ms_level
        ws = np.nonzero(hit)[0]
        if window == 'central' and ws.size > 1:
            keep = []
            for lev in np.unique(self.win_level[ws]):
                w = ws[self.win_level[ws] == lev]
                margin = np.minimum(precursor_mz - self.win_lo[w],
                                    self.win_hi[w] - precursor_mz)
                keep.append(w[int(np.argmax(margin))])
            ws = np.asarray(sorted(keep))
        return ws

    def scans(self, *, ms_level=None, rt_range=None, precursor_mz=None,
              window='central'):
        """Positions of the matching spectra, in index (file) order."""
        if precursor_mz is not None and self.has_isolation:
            parts = [self._slice(*self._win_scans[w], rt_range)
                     for w in self.windows_for(precursor_mz, ms_level, window)]
            if not parts:
                return np.array([], dtype=np.int64)
            return np.unique(np.concatenate(parts))
        if ms_level not in self._by_level:
            return np.array([], dtype=np.int64)
        return np.sort(self._slice(*self._by_level[ms_level], rt_range))


def scan_index(ms_data):
    """The cached ScanIndex of `ms_data`, built on first use."""
    ix = getattr(ms_data, _ATTR, None)
    if ix is None or not ix.valid_for(ms_data):
        ix = ScanIndex(ms_data)
        try:
            setattr(ms_data, _ATTR, ix)
        except AttributeError:
            pass
    return ix


BACKEND = 'numba' if _nb is not None else 'numpy'


if _nb is not None:
    @_nb.njit(cache=True)
    def _nb_collect(pmz, pint, starts, stops, pos, lows, highs):
        nt = lows.size
        counts = np.zeros(nt, np.int64)
        for i in pos:
            smz = pmz[starts[i]:stops[i]]
            for t in range(nt):
                counts[t] += (np.searchsorted(smz, highs[t])
                              - np.searchsorted(smz, lows[t]))
        off = np.zeros(nt + 1, np.int64)
        for t in range(nt):
            off[t + 1] = off[t] + max(counts[t], 0)
        x = np.empty(off[nt], pmz.dtype)
        y = np.empty(off[nt], pint.dtype)
        fill = off[:nt].copy()
        for i in pos:
            s0 = starts[i]
            smz = pmz[s0:stops[i]]
            for t in range(nt):
                lo = np.searchsorted(smz, lows[t])
                hi = np.searchsorted(smz, highs[t])
                for k in range(lo, hi):
                    x[fill[t]] = smz[k]
                    y[fill[t]] = pint[s0 + k]
                    fill[t] += 1
        return x, y, off


def collect_profiles(ix, positions, lows, highs):
    """Per target t, the (m/z, intensity) points in [lows[t], highs[t]) over `positions`, concatenated in scan order then m/z order - exactly what `locate_peak_mz` accumulated one target at a time."""
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)
    positions = np.asarray(positions, dtype=np.int64)
    if BACKEND == 'numba' and _nb is not None:
        return _nb_collect(ix.pmz, ix.pint, ix.start, ix.stop, positions,
                           lows, highs)
    xs = [[] for _ in lows]
    ys = [[] for _ in lows]
    for i in positions:
        s0, s1 = ix.start[i], ix.stop[i]
        smz = ix.pmz[s0:s1]
        lo = np.searchsorted(smz, lows)
        hi = np.searchsorted(smz, highs)
        for t in np.nonzero(hi > lo)[0]:
            xs[t].append(smz[lo[t]:hi[t]])
            ys[t].append(ix.pint[s0:s1][lo[t]:hi[t]])
    off = np.zeros(len(lows) + 1, np.int64)
    for t in range(len(lows)):
        off[t + 1] = off[t] + sum(len(a) for a in xs[t])
    x = (np.concatenate([a for t in xs for a in t]) if off[-1]
         else np.array([], dtype=ix.pmz.dtype))
    y = (np.concatenate([a for t in ys for a in t]) if off[-1]
         else np.array([], dtype=ix.pint.dtype))
    return x, y, off
