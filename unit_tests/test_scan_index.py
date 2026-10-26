"""The scan index behind extract_xic / sum_spectra / locate_peak_mz."""
import numpy as np
import pandas as pd
import pytest

from zenotof8600_analytics.raw import scan_index as si
from zenotof8600_analytics.raw.extraction import (
    extract_xic, sum_spectra, locate_peak_mz, filter_spectra, merge_peaks,
    resolve_mz_tol, xic_tolerance_ppm, _mz_window, XIC_FWHM_FRACTION,
    RECENTRE_SEARCH_FWHM, RECENTRE_MIN_CONTRAST)


def _ref_extract_xic(ms_data, mz, *, mz_tol='auto', mz_tol_unit='ppm', rt_range=None, ms_level=1, precursor_mz=None, recentre=True, recentre_rt_range=None, mz_shift_ppm=None):
    spec = ms_data.spectrum_df
    keep = np.ones(len(spec), dtype=bool)
    if ms_level is not None and 'ms_level' in spec.columns:
        keep &= spec['ms_level'].to_numpy() == ms_level
    rt = spec['rt'].to_numpy()
    if rt_range is not None:
        lo, hi = rt_range
        keep &= (rt >= lo) & (rt <= hi)
    if precursor_mz is not None and 'isolation_lower_mz' in spec.columns:
        ilo = spec['isolation_lower_mz'].to_numpy()
        ihi = spec['isolation_upper_mz'].to_numpy()
        keep &= (ilo <= precursor_mz) & (ihi >= precursor_mz)
    mz_tol = resolve_mz_tol(mz_tol, mz_tol_unit, ms_data, ms_level=ms_level)
    mzs = np.atleast_1d(np.asarray(mz, dtype=float))
    if mz_shift_ppm is not None and np.isfinite(mz_shift_ppm):
        mzs = mzs * (1.0 + float(mz_shift_ppm) / 1000000.0)
    elif recentre:
        fwhm = mz_tol / XIC_FWHM_FRACTION if mz_tol_unit == 'ppm' else None
        loc_range = recentre_rt_range if recentre_rt_range is not None else rt_range
        mzs = np.array([_ref_locate_peak_mz(ms_data, float(m), fwhm_ppm=fwhm, rt_range=loc_range, ms_level=ms_level) for m in mzs])
    windows = [_mz_window(float(m), mz_tol, mz_tol_unit) for m in mzs]
    peak_mz = ms_data.peak_df['mz'].to_numpy()
    peak_int = ms_data.peak_df['intensity'].to_numpy()
    starts = spec['peak_start_idx'].to_numpy()
    stops = spec['peak_stop_idx'].to_numpy()
    idxs = np.nonzero(keep)[0]
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


def _ref_sum_spectra(ms_data, *, rt_range=None, ms_level=2, precursor_mz=None, bin_tol=20.0, bin_tol_unit='ppm', position='median'):
    spec = ms_data.spectrum_df
    keep = np.ones(len(spec), dtype=bool)
    if ms_level is not None and 'ms_level' in spec.columns:
        keep &= spec['ms_level'].to_numpy() == ms_level
    if rt_range is not None:
        rt = spec['rt'].to_numpy()
        lo, hi = rt_range
        keep &= (rt >= lo) & (rt <= hi)
    if precursor_mz is not None and 'isolation_lower_mz' in spec.columns:
        ilo = spec['isolation_lower_mz'].to_numpy()
        ihi = spec['isolation_upper_mz'].to_numpy()
        keep &= (ilo <= precursor_mz) & (ihi >= precursor_mz)
    starts = spec['peak_start_idx'].to_numpy()
    stops = spec['peak_stop_idx'].to_numpy()
    peak_mz = ms_data.peak_df['mz'].to_numpy()
    peak_int = ms_data.peak_df['intensity'].to_numpy()
    chunks_mz, chunks_int = ([], [])
    for i in np.nonzero(keep)[0]:
        chunks_mz.append(peak_mz[starts[i]:stops[i]])
        chunks_int.append(peak_int[starts[i]:stops[i]])
    if not chunks_mz:
        return pd.DataFrame({'mz': [], 'intensity': []})
    mz = np.concatenate(chunks_mz)
    inten = np.concatenate(chunks_int)
    if mz.size == 0:
        return pd.DataFrame({'mz': [], 'intensity': []})
    out_mz, out_int = merge_peaks(mz, inten, tol=bin_tol, tol_unit=bin_tol_unit, position=position)
    return pd.DataFrame({'mz': out_mz, 'intensity': out_int})


def _ref_locate_peak_mz(ms_data, mz, *, fwhm_ppm=None, rt_range=None, ms_level=2, search_fwhm=None, min_contrast=None):
    fwhm_ppm = xic_tolerance_ppm(ms_data, ms_level=ms_level) * 2 if fwhm_ppm is None else fwhm_ppm
    search_fwhm = RECENTRE_SEARCH_FWHM if search_fwhm is None else search_fwhm
    min_contrast = RECENTRE_MIN_CONTRAST if min_contrast is None else min_contrast
    half = mz * fwhm_ppm * search_fwhm / 1000000.0
    spec = ms_data.spectrum_df
    keep = np.ones(len(spec), dtype=bool)
    if ms_level is not None and 'ms_level' in spec.columns:
        keep &= spec['ms_level'].to_numpy() == ms_level
    if rt_range is not None:
        rt = spec['rt'].to_numpy()
        keep &= (rt >= rt_range[0]) & (rt <= rt_range[1])
    if not keep.any():
        return float(mz)
    pmz = ms_data.peak_df['mz'].to_numpy()
    pint = ms_data.peak_df['intensity'].to_numpy()
    starts = spec['peak_start_idx'].to_numpy()
    stops = spec['peak_stop_idx'].to_numpy()
    xs, ys = ([], [])
    for i in np.nonzero(keep)[0]:
        smz = pmz[starts[i]:stops[i]]
        lo = np.searchsorted(smz, mz - half)
        hi = np.searchsorted(smz, mz + half)
        if hi > lo:
            xs.append(smz[lo:hi])
            ys.append(pint[starts[i]:stops[i]][lo:hi])
    if not xs:
        return float(mz)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if x.size < 3 or y.max() <= 0:
        return float(mz)
    step = mz * fwhm_ppm / 1000000.0 / 32.0
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
    core_half = mz * fwhm_ppm / 1000000.0 * 0.5
    sel = np.abs(x - apex) <= core_half
    if sel.sum() < 3 or y[sel].sum() <= 0:
        return float(mz)
    yb = y[sel] - np.median(y[~sel]) if (~sel).sum() >= 5 else y[sel]
    yb = np.clip(yb, 0, None)
    if yb.sum() <= 0:
        return float(mz)
    return float((x[sel] * yb).sum() / yb.sum())


WINDOWS = [(400.0, 413.0), (412.0, 425.0), (424.0, 437.0)]


class _Run:
    def __init__(self, n_cycles=40, seed=0, fragments=(), bg_peaks=300):
        rng = np.random.default_rng(seed)
        rows, mzs, ints, start = [], [], [], 0
        cycle_s = 0.5 / 60
        for c in range(n_cycles):
            for w in [None, *range(len(WINDOWS))]:
                rt = c * cycle_s + (0 if w is None else (w + 1) * cycle_s / 8)
                mz = np.sort(rng.uniform(150, 1500, bg_peaks))
                it = rng.uniform(1, 200, bg_peaks)
                for fw, fmz, amp, apex, ppm in fragments:
                    if w != fw:
                        continue
                    a = amp * np.exp(-0.5 * ((c - apex) / 2.0) ** 2)
                    centre = fmz * (1 + ppm / 1e6)
                    grid = centre * (1 + np.linspace(-60, 60, 25) / 1e6)
                    prof = a * np.exp(-4 * np.log(2)
                                      * ((grid - centre) / (centre * 30e-6)) ** 2)
                    mz = np.concatenate([mz, grid])
                    it = np.concatenate([it, prof])
                o = np.argsort(mz, kind='stable')
                mz, it = mz[o].astype(np.float32), it[o].astype(np.float32)
                lo, hi = (np.nan, np.nan) if w is None else WINDOWS[w]
                rows.append({'rt': rt, 'ms_level': 1 if w is None else 2,
                             'isolation_lower_mz': lo,
                             'isolation_upper_mz': hi,
                             'peak_start_idx': start,
                             'peak_stop_idx': start + mz.size})
                start += mz.size
                mzs.append(mz)
                ints.append(it)
        self.spectrum_df = pd.DataFrame(rows)
        self.peak_df = pd.DataFrame({'mz': np.concatenate(mzs),
                                     'intensity': np.concatenate(ints)})


def _random_run(seed):
    rng = np.random.default_rng(seed)
    frags = [(int(rng.integers(0, 3)), float(rng.uniform(200, 1400)),
              float(rng.uniform(1e3, 1e5)), int(rng.integers(5, 35)),
              float(rng.uniform(-10, 10))) for _ in range(12)]
    return _Run(seed=seed, fragments=frags), frags


def _same(a, b):
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)
    for c in a.columns:
        np.testing.assert_array_equal(a[c].to_numpy(), b[c].to_numpy())


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_union_settings_match_the_reference_extract_xic(seed):
    ms, frags = _random_run(seed)
    rng = np.random.default_rng(100 + seed)
    for _ in range(15):
        prec = float(rng.uniform(401, 436))
        mzs = [f[1] for f in frags[:int(rng.integers(1, 6))]]
        rr = tuple(sorted(rng.uniform(0, 40 * 0.5 / 60, 2)))
        for kw in ({}, {'recentre': False}, {'mz_shift_ppm': 3.0},
                   {'recentre_rt_range': rr}):
            old = _ref_extract_xic(ms, mzs, mz_tol=20.0, ms_level=2,
                                   precursor_mz=prec, rt_range=rr, **kw)
            new = extract_xic(ms, mzs, mz_tol=20.0, ms_level=2,
                              precursor_mz=prec, rt_range=rr, window='union',
                              recentre_scans='all', **kw)
            _same(old, new)


def test_ms1_and_no_precursor_paths_match_the_reference():
    ms, frags = _random_run(3)
    for mzs in ([frags[0][1]], [f[1] for f in frags[:4]]):
        for level in (1, 2, None):
            _same(_ref_extract_xic(ms, mzs, mz_tol=20.0, ms_level=level),
                  extract_xic(ms, mzs, mz_tol=20.0, ms_level=level,
                              window='union', recentre_scans='all'))


def test_locate_peak_mz_without_precursor_is_unchanged():
    ms, frags = _random_run(4)
    for _, fmz, *_ in frags:
        for rr in (None, (0.05, 0.25)):
            assert (locate_peak_mz(ms, fmz, fwhm_ppm=40.0, rt_range=rr)
                    == _ref_locate_peak_mz(ms, fmz, fwhm_ppm=40.0,
                                           rt_range=rr))


def test_sum_spectra_union_matches_the_reference():
    ms, _ = _random_run(5)
    for prec in (405.0, 412.5, 430.0):
        _same(_ref_sum_spectra(ms, precursor_mz=prec, rt_range=(0.05, 0.2)),
              sum_spectra(ms, precursor_mz=prec, rt_range=(0.05, 0.2),
                          window='union'))


def test_numba_and_numpy_backends_agree_exactly():
    ms, frags = _random_run(6)
    mzs = [f[1] for f in frags]
    out = {}
    for backend in ('numpy', 'numba'):
        if backend == 'numba' and si._nb is None:
            pytest.skip('numba not installed')
        old = si.BACKEND
        si.BACKEND = backend
        try:
            out[backend] = extract_xic(ms, mzs, mz_tol=20.0, ms_level=2,
                                       precursor_mz=412.5, window='union',
                                       recentre_scans='all')
        finally:
            si.BACKEND = old
    _same(out['numpy'], out['numba'])


def test_union_interleaves_two_windows_central_reads_one():
    ms = _Run(fragments=[(1, 600.0, 1e5, 20, 0.0)])
    union = extract_xic(ms, 600.0, mz_tol=20.0, ms_level=2,
                        precursor_mz=412.8, window='union', recentre=False)
    central = extract_xic(ms, 600.0, mz_tol=20.0, ms_level=2,
                          precursor_mz=412.8, recentre=False)
    assert len(union) == 2 * len(central) == 80
    inner = union['intensity'].to_numpy()[30:50]
    assert (inner[::2] < inner[1::2]).all() or (inner[1::2] < inner[::2]).all()
    assert central['intensity'].max() > 1e4


def test_central_prefers_the_window_with_the_wider_margin():
    ix = si.scan_index(_Run(n_cycles=2))
    assert list(ix.windows_for(412.2, 2, 'central')) == [0]
    assert list(ix.windows_for(412.8, 2, 'central')) == [1]
    assert list(ix.windows_for(412.5, 2, 'union')) == [0, 1]
    assert list(ix.windows_for(418.0, 2, 'central')) == [1]


def test_sum_spectra_central_reads_one_window():
    ms = _Run(n_cycles=10)
    u = sum_spectra(ms, precursor_mz=412.5, window='union', bin_tol=1e-3)
    c = sum_spectra(ms, precursor_mz=412.5, bin_tol=1e-3)
    assert len(c) < len(u)


def test_own_window_recentring_is_not_pulled_onto_another_windows_ion():
    theo = 700.0
    ms = _Run(fragments=[(2, theo, 5e4, 20, 3.0), (0, theo, 1e6, 20, 12.0)])
    rr = (15 * 0.5 / 60, 25 * 0.5 / 60)
    kw = dict(fwhm_ppm=40.0, rt_range=rr, ms_level=2)
    own = locate_peak_mz(ms, theo, precursor_mz=430.0, **kw)
    pooled = locate_peak_mz(ms, theo, **kw)
    assert abs((own / theo - 1) * 1e6 - 3.0) < 1.5
    assert (pooled / theo - 1) * 1e6 > 8.0
    xo = extract_xic(ms, theo, mz_tol=20.0, ms_level=2, precursor_mz=430.0,
                     rt_range=rr)
    xa = extract_xic(ms, theo, mz_tol=20.0, ms_level=2, precursor_mz=430.0,
                     rt_range=rr, recentre_scans='all')
    assert xo['intensity'].max() > xa['intensity'].max()


def test_index_is_cached_and_rebuilt_for_a_view():
    ms = _Run(n_cycles=5)
    assert si.scan_index(ms) is si.scan_index(ms)
    view = filter_spectra(ms, ms_level=2)
    assert si.scan_index(view) is not si.scan_index(ms)
    assert si.scan_index(view).n_spec == len(view.spectrum_df)


def test_unknown_choices_are_rejected():
    ms = _Run(n_cycles=2)
    with pytest.raises(ValueError, match='window must be'):
        extract_xic(ms, 600.0, ms_level=2, precursor_mz=410.0,
                    window='nearest')
    with pytest.raises(ValueError, match='recentre_scans must be'):
        extract_xic(ms, 600.0, ms_level=2, precursor_mz=410.0,
                    recentre_scans='some')
