"""Manual-mode raw-file extraction on top of alpharaw."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

_READER_MODULE = {
    'thermo': 'alpharaw.thermo', 'thermo_raw': 'alpharaw.thermo',
    'sciex': 'alpharaw.sciex', 'sciex_wiff': 'alpharaw.sciex',
    'sciex_raw': 'alpharaw.sciex', 'mzml': 'alpharaw.mzml',
}
_EXT_TO_READER = {
    '.raw': 'thermo', '.wiff': 'sciex', '.mzml': 'mzml', '.mzml.gz': 'mzml',
}


from .loading import (
    _mz_window, load_ms_data, estimate_parse_memory, _preflight_memory,
    _cache_is_readable, _reader_opts, _fmt_limit, _stamp_peak_limit,
    _cached_peak_limit, DEFAULT_READER_OPTIONS, KEEP_ALL_PEAKS, PARSE_BYTES_PER_PEAK, PARSE_MIN_FREE_GB, PARSE_PEAKS_PER_GB, PARSE_SAFETY, _EXT_TO_READER, _PEAK_LIMIT_ATTR, _READER_MODULE)
from .peaks import (
    merge_peaks, window_indices, match_peaks, pick_peak, points_in_window, integrate_peak, peak_shape, _baseline_value, OFFPEAK_EXCLUDE_FWHM, OFFPEAK_MIN_POINTS,
)
from .scan_index import (
    scan_index, collect_profiles, ScanIndex, WINDOW_CHOICES,
    RECENTRE_SCAN_CHOICES, _check)


def extract_xic(ms_data, mz, *, mz_tol='auto', mz_tol_unit='ppm', rt_range=None,
                ms_level=1, precursor_mz=None, recentre=True,
                recentre_rt_range=None, mz_shift_ppm=None, window='central',
                recentre_scans='own'):
    """Extracted-ion chromatogram for one or more m/z targets.

    Parameters
      ms_data:      object with ``.spectrum_df`` and ``.peak_df`` (an alpharaw
                    reader or anything with the same columns).
      mz:           a single m/z, or a sequence of m/z summed into one trace
                    (e.g. several isotopes of the same precursor).
      mz_tol:       half-window tolerance (default 30). On profile data the
                    window spans the peak; use a narrower one on centroided data.
      mz_shift_ppm: apply this ppm shift to every target m/z instead of
                    recentring (see measure_mz_shift_ppm).
      recentre_rt_range: retention range over which the m/z window is
                    recentred, when that differs from the range being
                    extracted.
      mz_tol_unit:  'ppm' (default) or 'Da'.
      rt_range:     (min, max) retention-time window in minutes; None = whole run.
      ms_level:     MS level to draw from (default 1); None = any level.
      precursor_mz: for MS2 (DIA) fragment XICs, keep only spectra whose
                    isolation window (``isolation_lower_mz`` /
                    ``isolation_upper_mz``) contains this m/z. Ignored when the
                    spectra carry no isolation columns.
      window:       'central' (default) reads only the isolation window the
                    precursor sits most centrally in; 'union' reads every
                    window containing it (see scan_index.WINDOW_CHOICES).
      recentre_scans: 'own' (default) recentres the m/z window on the scans
                    the XIC is read from; 'all' on every scan of the level in
                    the range (see scan_index.RECENTRE_SCAN_CHOICES).
    """
    _check(window, WINDOW_CHOICES, 'window')
    _check(recentre_scans, RECENTRE_SCAN_CHOICES, 'recentre_scans')
    ix = scan_index(ms_data)
    idxs = ix.scans(ms_level=ms_level, rt_range=rt_range,
                    precursor_mz=precursor_mz, window=window)

    mz_tol = resolve_mz_tol(mz_tol, mz_tol_unit, ms_data, ms_level=ms_level)
    mzs = np.atleast_1d(np.asarray(mz, dtype=float))
    if mz_shift_ppm is not None and np.isfinite(mz_shift_ppm):
        mzs = mzs * (1.0 + float(mz_shift_ppm) / 1e6)
    elif recentre:
        fwhm = (mz_tol / XIC_FWHM_FRACTION if mz_tol_unit == 'ppm'
                else None)
        loc_range = (recentre_rt_range if recentre_rt_range is not None
                     else rt_range)
        loc_pos = ix.scans(ms_level=ms_level, rt_range=loc_range,
                           precursor_mz=(precursor_mz
                                         if recentre_scans == 'own' else None),
                           window=window)
        mzs = np.array(_locate_many(ms_data, ix, [float(m) for m in mzs],
                                    loc_pos, fwhm_ppm=fwhm,
                                    ms_level=ms_level))
    windows = [_mz_window(float(m), mz_tol, mz_tol_unit) for m in mzs]

    peak_mz, peak_int = ix.pmz, ix.pint
    starts, stops, rt = ix.start, ix.stop, ix.rt

    out_rt = np.empty(len(idxs), dtype=float)
    out_int = np.zeros(len(idxs), dtype=float)
    for k, i in enumerate(idxs):
        out_rt[k] = rt[i]
        smz = peak_mz[starts[i]:stops[i]]
        sint = peak_int[starts[i]:stops[i]]
        total = 0.0
        for lo_mz, hi_mz in windows:
            sel = (smz >= lo_mz) & (smz <= hi_mz)
            if sel.any():
                total += float(sint[sel].sum())
        out_int[k] = total

    order = np.argsort(out_rt)
    return pd.DataFrame({'rt': out_rt[order], 'intensity': out_int[order]})

def sum_spectra(ms_data, *, rt_range=None, ms_level=2, precursor_mz=None,
                bin_tol=20.0, bin_tol_unit='ppm', position='median',
                window='central', one_per_scan=False):
    """Sum spectra over a retention-time window into one merged spectrum.

    Parameters
      ms_data:        object with ``.spectrum_df`` and ``.peak_df``.
      rt_range:       (min, max) minutes; None = whole run.
      ms_level:       spectrum level to sum (default 2); None = any.
      precursor_mz:   if given and the spectra carry ``isolation_lower_mz`` /
                      ``isolation_upper_mz``, keep only windows covering it.
      bin_tol:        peak-merge tolerance (default 20).
      bin_tol_unit:   'ppm' (default) or 'Da'.
      position:       merged-peak m/z rule, passed to ``merge_peaks``:
                      'median' (default, for profile input) or 'weighted'
                      (for centroids).
      window:         'central' (default) or 'union', as in extract_xic.
      one_per_scan:   if True, a merged peak takes at most one sample per
                      spectrum (``merge_peaks(scan_id=)``).
    """
    _check(window, WINDOW_CHOICES, 'window')
    ix = scan_index(ms_data)
    keep_pos = ix.scans(ms_level=ms_level, rt_range=rt_range,
                        precursor_mz=precursor_mz, window=window)
    starts, stops = ix.start, ix.stop
    peak_mz, peak_int = ix.pmz, ix.pint

    chunks_mz, chunks_int, chunks_scan = [], [], []
    for i in keep_pos:
        chunks_mz.append(peak_mz[starts[i]:stops[i]])
        chunks_int.append(peak_int[starts[i]:stops[i]])
        chunks_scan.append(np.full(stops[i] - starts[i], i))
    if not chunks_mz:
        return pd.DataFrame({'mz': [], 'intensity': []})
    mz = np.concatenate(chunks_mz)
    inten = np.concatenate(chunks_int)
    if mz.size == 0:
        return pd.DataFrame({'mz': [], 'intensity': []})

    out_mz, out_int = merge_peaks(
        mz, inten, tol=bin_tol, tol_unit=bin_tol_unit, position=position,
        scan_id=np.concatenate(chunks_scan) if one_per_scan else None)
    return pd.DataFrame({'mz': out_mz, 'intensity': out_int})

class _SpectrumView:
    """A subset of an ms_data's spectra, sharing its peak table."""

    def __init__(self, spectrum_df, peak_df):
        self.spectrum_df = spectrum_df
        self.peak_df = peak_df

def filter_spectra(ms_data, *, ms_level=None, nce=None, rt_range=None,
                   query=None):
    """Restrict an ms_data to a subset of its spectra, without copying peaks."""
    spec = ms_data.spectrum_df
    mask = np.ones(len(spec), dtype=bool)
    if ms_level is not None:
        mask &= spec['ms_level'].to_numpy() == ms_level
    if nce is not None:
        values = [nce] if np.isscalar(nce) else list(nce)
        nce_col = spec['nce'].to_numpy()
        hit = np.zeros(len(spec), dtype=bool)
        for v in values:
            hit |= np.abs(nce_col - float(v)) < 0.5
        mask &= hit
    if rt_range is not None:
        rt = spec['rt'].to_numpy()
        mask &= (rt >= rt_range[0]) & (rt <= rt_range[1])
    out = spec[mask]
    if query:
        out = out.query(query)
    return _SpectrumView(out, ms_data.peak_df)

def peak_from_fragments(ms_data, precursor_mz, fragment_mz, *, rt_range=None,
                        mz_tol='auto', mz_tol_unit='ppm', top_n=6,
                        height_fraction=0.5, baseline='median'):
    """Locate a precursor's LC peak from the co-elution of its fragment ions."""
    fragment_mz = np.atleast_1d(np.asarray(fragment_mz, dtype=float))
    traces, totals = [], []
    for fmz in fragment_mz:
        x = extract_xic(ms_data, float(fmz), mz_tol=mz_tol, mz_tol_unit=mz_tol_unit,
                        rt_range=rt_range, ms_level=2, precursor_mz=precursor_mz)
        traces.append(x)
        totals.append(float(x['intensity'].sum()))

    empty = pd.DataFrame({'rt': [], 'intensity': []})
    if not traces or all(len(t) == 0 for t in traces):
        return pick_peak(np.array([]), np.array([])), empty, np.array([])

    order = np.argsort(totals)[::-1]
    chosen = [i for i in order if totals[i] > 0][:top_n]
    rt = traces[order[0]]['rt'].to_numpy()
    if not chosen:
        return (pick_peak(rt, np.zeros(len(rt))),
                pd.DataFrame({'rt': rt, 'intensity': np.zeros(len(rt))}),
                np.array([]))
    summed = np.zeros(len(rt), dtype=float)
    for i in chosen:
        summed += traces[i]['intensity'].to_numpy()
    peak = pick_peak(rt, summed, height_fraction=height_fraction, baseline=baseline)
    return peak, pd.DataFrame({'rt': rt, 'intensity': summed}), fragment_mz[chosen]


XIC_FWHM_FRACTION = 0.5

XIC_FALLBACK_PPM = 10.0

RESOLUTION_N_SPECTRA = 24
RESOLUTION_N_PEAKS = 12
RESOLUTION_SEARCH_PPM = 120.0
RESOLUTION_MIN_POINTS = 5

RESOLUTION_MIN_REL_INTENSITY = 0.02

RESOLUTION_MIN_TIC_FRAC = 0.10
_RESOLUTION_ATTR = '_pa_resolution_ppm'


def measure_resolution_ppm(ms_data, *, ms_level=2, n_spectra=None,
                           n_peaks=None, search_ppm=None, min_points=None):
    """Median FWHM of the profile peaks in this file, in ppm."""
    n_spectra = RESOLUTION_N_SPECTRA if n_spectra is None else n_spectra
    n_peaks = RESOLUTION_N_PEAKS if n_peaks is None else n_peaks
    search_ppm = RESOLUTION_SEARCH_PPM if search_ppm is None else search_ppm
    min_points = RESOLUTION_MIN_POINTS if min_points is None else min_points

    spec = ms_data.spectrum_df
    if ms_level is not None and 'ms_level' in spec.columns:
        spec = spec[spec['ms_level'] == ms_level]
    if not len(spec):
        return float('nan')
    peak_int_all = ms_data.peak_df['intensity'].to_numpy()
    csum = np.concatenate(([0.0], np.cumsum(peak_int_all, dtype=float)))
    starts_all = spec['peak_start_idx'].to_numpy()
    stops_all = spec['peak_stop_idx'].to_numpy()
    tic = csum[stops_all] - csum[starts_all]
    if not np.isfinite(tic).any() or tic.max() <= 0:
        return float('nan')
    has_signal = np.nonzero(tic >= tic.max() * RESOLUTION_MIN_TIC_FRAC)[0]
    order = has_signal[np.argsort(tic[has_signal])[::-1]][:n_spectra]
    take = spec.iloc[np.sort(order)]

    peak_mz = ms_data.peak_df['mz'].to_numpy()
    peak_int = ms_data.peak_df['intensity'].to_numpy()
    widths = []
    for _, row in take.iterrows():
        lo, hi = int(row['peak_start_idx']), int(row['peak_stop_idx'])
        mz, it = peak_mz[lo:hi], peak_int[lo:hi]
        if mz.size < min_points:
            continue
        inner = np.arange(1, mz.size - 1)
        if not inner.size:
            continue
        loc = inner[(it[inner] > it[inner - 1]) & (it[inner] >= it[inner + 1])]
        if not loc.size:
            continue
        floor = float(it.max()) * RESOLUTION_MIN_REL_INTENSITY
        for k in loc[np.argsort(it[loc])[::-1][:n_peaks]]:
            if it[k] < floor:
                break
            centre = float(mz[k])
            half = centre * search_ppm / 1e6
            sel = (mz >= centre - half) & (mz <= centre + half)
            if sel.sum() < min_points:
                continue
            shape = peak_shape(mz[sel], it[sel], centre=centre)
            w = shape.get('width', float('nan'))
            if np.isfinite(w) and w > 0 and shape['n_points'] >= min_points:
                widths.append(w / centre * 1e6)
    return float(np.median(widths)) if widths else float('nan')


def xic_tolerance_ppm(ms_data, *, fraction=None, fallback=None,
                      ms_level=2, cache=True):
    """The XIC half-window this file's own resolution implies, in ppm."""
    fraction = XIC_FWHM_FRACTION if fraction is None else fraction
    fallback = XIC_FALLBACK_PPM if fallback is None else fallback
    res = getattr(ms_data, _RESOLUTION_ATTR, None) if cache else None
    if res is None:
        res = measure_resolution_ppm(ms_data, ms_level=ms_level)
        if cache:
            try:
                setattr(ms_data, _RESOLUTION_ATTR, res)
            except AttributeError:
                pass
    return fallback if not np.isfinite(res) else float(res) * fraction


RECENTRE_SEARCH_FWHM = 0.6

RECENTRE_MIN_CONTRAST = 3.0


def locate_peak_mz(ms_data, mz, *, fwhm_ppm=None, rt_range=None, ms_level=2,
                   search_fwhm=None, min_contrast=None, precursor_mz=None,
                   window='central'):
    """The MEASURED m/z of the peak near `mz`, or `mz` itself if there is none."""
    fwhm_ppm = (xic_tolerance_ppm(ms_data, ms_level=ms_level) * 2
                if fwhm_ppm is None else fwhm_ppm)
    ix = scan_index(ms_data)
    pos = ix.scans(ms_level=ms_level, rt_range=rt_range,
                   precursor_mz=precursor_mz, window=window)
    return _locate_many(ms_data, ix, [float(mz)], pos, fwhm_ppm=fwhm_ppm,
                        ms_level=ms_level, search_fwhm=search_fwhm,
                        min_contrast=min_contrast)[0]


def _locate_many(ms_data, ix, mzs, positions, *, fwhm_ppm, ms_level=2,
                 search_fwhm=None, min_contrast=None):
    """locate_peak_mz for several m/z over one set of scans, in one pass."""
    fwhm_ppm = (xic_tolerance_ppm(ms_data, ms_level=ms_level) * 2
                if fwhm_ppm is None else fwhm_ppm)
    search_fwhm = RECENTRE_SEARCH_FWHM if search_fwhm is None else search_fwhm
    min_contrast = (RECENTRE_MIN_CONTRAST if min_contrast is None
                    else min_contrast)
    mzs = [float(m) for m in mzs]
    if not len(positions):
        return mzs
    halfs = [m * fwhm_ppm * search_fwhm / 1e6 for m in mzs]
    x, y, off = collect_profiles(ix, positions,
                                 [m - h for m, h in zip(mzs, halfs)],
                                 [m + h for m, h in zip(mzs, halfs)])
    return [_centroid_from_points(m, x[off[t]:off[t + 1]],
                                  y[off[t]:off[t + 1]], fwhm_ppm=fwhm_ppm,
                                  half=halfs[t], min_contrast=min_contrast)
            for t, m in enumerate(mzs)]


def _centroid_from_points(mz, x, y, *, fwhm_ppm, half, min_contrast):
    """The position estimate of locate_peak_mz, from its collected points."""
    if x.size < 3 or y.max() <= 0:
        return float(mz)

    step = mz * fwhm_ppm / 1e6 / 32.0
    edges = np.arange(mz - half, mz + half + step, step)
    prof, _ = np.histogram(x, bins=edges, weights=y)
    if not prof.size or prof.max() <= 0:
        return float(mz)
    base = float(np.percentile(prof, 10))
    if base > 0 and prof.max() < base * min_contrast:
        return float(mz)

    centres = (edges[:-1] + edges[1:]) / 2.0
    k = int(np.argmax(prof))
    if k <= 0 or k >= prof.size - 1:
        return float(mz)
    apex = centres[k]
    core_half = mz * fwhm_ppm / 1e6 * 0.5
    sel = np.abs(x - apex) <= core_half
    if sel.sum() < 3 or y[sel].sum() <= 0:
        return float(mz)
    yb = y[sel] - np.median(y[~sel]) if (~sel).sum() >= 5 else y[sel]
    yb = np.clip(yb, 0, None)
    if yb.sum() <= 0:
        return float(mz)
    return float((x[sel] * yb).sum() / yb.sum())


def measure_mz_shift_ppm(ms_data, mz, *, fwhm_ppm=None, rt_range=None,
                         ms_level=2):
    """The median ppm offset between theoretical m/z and the measured peaks."""
    mzs = np.atleast_1d(np.asarray(mz, dtype=float))
    shifts = []
    for m in mzs:
        found = locate_peak_mz(ms_data, float(m), fwhm_ppm=fwhm_ppm,
                               rt_range=rt_range, ms_level=ms_level)
        if np.isfinite(found) and found > 0:
            shifts.append((found - m) / m * 1e6)
    return float(np.median(shifts)) if shifts else float('nan')


def resolve_mz_tol(mz_tol, mz_tol_unit, ms_data, *, ms_level=2):
    """Turn ``mz_tol='auto'`` into a number; pass anything else through."""
    if isinstance(mz_tol, str):
        if mz_tol != 'auto':
            raise ValueError(f"mz_tol must be a number or 'auto', "
                             f"got {mz_tol!r}")
        if mz_tol_unit != 'ppm':
            raise ValueError("mz_tol='auto' is only defined in ppm")
        return xic_tolerance_ppm(ms_data, ms_level=ms_level)
    return mz_tol
