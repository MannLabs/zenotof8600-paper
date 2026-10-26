"""Recentring must be told where the peak is."""
import numpy as np
import pandas as pd

from zenotof8600_analytics.raw import targeted
from zenotof8600_analytics.raw.extraction import extract_xic


class _MS:

    def __init__(self, n=80, mz=500.0, offset_ppm=8.0, fwhm_ppm=30.0,
                 apex_scan=40, rt_sigma=1.5, bg=600.0):
        rng = np.random.default_rng(0)
        true_mz = mz * (1 + offset_ppm / 1e6)
        bg_amp = bg
        rows, peaks, start = [], [], 0
        grid = mz * (1 + np.linspace(-6, 6, 41) * fwhm_ppm / 1e6)
        for i in range(n):
            amp = 1e5 * np.exp(-0.5 * ((i - apex_scan) / rt_sigma) ** 2)
            sig = amp * np.exp(-4 * np.log(2)
                               * ((grid - true_mz)
                                  / (mz * fwhm_ppm / 1e6)) ** 2)
            bg = bg_amp * np.exp(-4 * np.log(2)
                               * ((grid - mz * (1 - 40 / 1e6))
                                  / (mz * fwhm_ppm / 1e6)) ** 2)
            inten = sig + bg + rng.uniform(0, 2, grid.size)
            peaks.append(pd.DataFrame({'mz': grid, 'intensity': inten}))
            rows.append({'rt': i * 0.005, 'ms_level': 2, 'precursor_mz': 700.0,
                         'isolation_lower_mz': 699.0,
                         'isolation_upper_mz': 701.0,
                         'peak_start_idx': start,
                         'peak_stop_idx': start + grid.size})
            start += grid.size
        self.spectrum_df = pd.DataFrame(rows)
        self.peak_df = pd.concat(peaks, ignore_index=True)
        self.true_mz, self.theory_mz = true_mz, mz
        self.apex_rt = apex_scan * 0.005


def test_recentring_over_the_peak_lands_on_the_analyte():
    from zenotof8600_analytics.raw.extraction import locate_peak_mz
    ms = _MS()
    half = 0.008
    blind = locate_peak_mz(ms, ms.theory_mz, fwhm_ppm=30.0, ms_level=2)
    onpeak = locate_peak_mz(ms, ms.theory_mz, fwhm_ppm=30.0, ms_level=2,
                            rt_range=(ms.apex_rt - half, ms.apex_rt + half))
    err = lambda v: abs(v - ms.true_mz) / ms.true_mz * 1e6
    assert err(onpeak) < err(blind), (err(onpeak), err(blind))
    assert err(onpeak) < 5.0, err(onpeak)


def test_recentre_rt_range_does_not_truncate_the_trace():
    ms = _MS()
    full = extract_xic(ms, ms.theory_mz, mz_tol=15.0, ms_level=2,
                       recentre=False)
    steered = extract_xic(ms, ms.theory_mz, mz_tol=15.0, ms_level=2,
                          recentre=True,
                          recentre_rt_range=(ms.apex_rt - 0.01,
                                             ms.apex_rt + 0.01))
    assert len(steered) == len(full)
    assert steered['rt'].min() == full['rt'].min()
    assert steered['rt'].max() == full['rt'].max()


def test_quantify_targets_measures_the_shift_once_and_applies_it():
    ms = _MS()
    frags = pd.DataFrame({'target_idx': [0, 0], 'sequence': ['PEPTIDEK'] * 2,
                          'charge': [2, 2], 'precursor_mz': [700.0] * 2,
                          'product_mz': [ms.theory_mz, ms.theory_mz]})
    calls = []
    real = targeted.summed_fragment_xic

    def spy(data, f, **kw):
        calls.append((kw.get('recentre', True), kw.get('mz_shift_ppm')))
        return real(data, f, **kw)

    targeted.summed_fragment_xic, saved = spy, real
    try:
        out = targeted.quantify_targets(
            {'r1': ms}, frags, fragment_filter=None, match='window',
            decoy_offset_da=None, verbose=False)
    finally:
        targeted.summed_fragment_xic = saved
    assert len(out) == 1
    assert calls[0] == (False, None)
    shifts = [s for _, s in calls if s is not None]
    assert shifts, calls
    assert abs(shifts[0] - 8.0) < 5.0, shifts
    assert abs(out['mz_shift_ppm'].iloc[0] - 8.0) < 5.0


def test_measured_shift_is_recorded_on_every_row():
    ms = _MS()
    frags = pd.DataFrame({'target_idx': [0], 'sequence': ['PEPTIDEK'],
                          'charge': [2], 'precursor_mz': [700.0],
                          'product_mz': [ms.theory_mz]})
    out = targeted.quantify_targets({'r1': ms}, frags, fragment_filter=None,
                                    match='window', decoy_offset_da=None,
                                    verbose=False)
    assert 'mz_shift_ppm' in out.columns
    assert np.isfinite(out['mz_shift_ppm']).all()


def test_recentre_false_skips_the_shift():
    ms = _MS()
    frags = pd.DataFrame({'target_idx': [0], 'sequence': ['PEPTIDEK'],
                          'charge': [2], 'precursor_mz': [700.0],
                          'product_mz': [ms.theory_mz]})
    out = targeted.quantify_targets({'r1': ms}, frags, fragment_filter=None,
                                    match='window', decoy_offset_da=None,
                                    recentre=False, verbose=False)
    assert not np.isfinite(out['mz_shift_ppm']).any()


def test_small_shift_is_measured_but_not_applied():
    ms = _MS(offset_ppm=2.0)
    frags = pd.DataFrame({'target_idx': [0], 'sequence': ['PEPTIDEK'],
                          'charge': [2], 'precursor_mz': [700.0],
                          'product_mz': [ms.theory_mz]})
    out = targeted.quantify_targets({'r1': ms}, frags, fragment_filter=None,
                                    match='window', mz_tol=15.0,
                                    decoy_offset_da=None, verbose=False)
    assert np.isfinite(out['mz_shift_ppm'].iloc[0])
    assert not out['mz_shift_applied'].iloc[0]


def test_large_shift_is_applied():
    ms = _MS(offset_ppm=12.0, bg=0.0)
    frags = pd.DataFrame({'target_idx': [0], 'sequence': ['PEPTIDEK'],
                          'charge': [2], 'precursor_mz': [700.0],
                          'product_mz': [ms.theory_mz]})
    out = targeted.quantify_targets({'r1': ms}, frags, fragment_filter=None,
                                    match='window', mz_tol=15.0,
                                    decoy_offset_da=None, verbose=False)
    shift = out['mz_shift_ppm'].iloc[0]
    assert out['mz_shift_applied'].iloc[0]
    assert shift > 0.25 * 15.0, shift
