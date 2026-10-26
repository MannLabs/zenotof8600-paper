"""The targeted-assay defaults: scan matching, blank SD, anchor correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import zenotof8600_analytics as core


def _targeted_file(precursors=(500.0, 500.5), product=800.0, n_cycles=40,
                   amplitudes=(1000.0, 10.0)):
    rt, pmz, inten, start, stop = [], [], [], [], []
    mzs, ints = [], []
    for c in range(n_cycles):
        for p, amp in zip(precursors, amplitudes):
            t = c * 0.02
            y = amp * np.exp(-0.5 * ((t - 0.4) / 0.05) ** 2) + 1.0
            start.append(len(mzs))
            mzs.extend([product - 0.01, product, product + 0.01])
            ints.extend([y * 0.2, y, y * 0.2])
            stop.append(len(mzs))
            rt.append(t)
            pmz.append(p)

    class _MS:
        pass

    ms = _MS()
    ms.spectrum_df = pd.DataFrame({
        'rt': rt, 'ms_level': 2, 'precursor_mz': pmz,
        'isolation_lower_mz': np.array(pmz) - 1.5,
        'isolation_upper_mz': np.array(pmz) + 1.5,
        'peak_start_idx': start, 'peak_stop_idx': stop})
    ms.peak_df = pd.DataFrame({'mz': mzs, 'intensity': ints})
    return ms


def _fragments(precursor_mz, product=800.0):
    return pd.DataFrame([{
        'target_idx': 0, 'sequence': 'PEPTIDEK', 'charge': 2,
        'precursor_mz': precursor_mz, 'product_mz': product,
        'ion_type': 'y', 'ion_number': 4, 'fragment_charge': 1}])


def test_window_matching_merges_neighbouring_targets():
    ms = _targeted_file()
    rt, inten = core.raw.summed_fragment_xic(ms, _fragments(500.0),
                                             match='window', window='union')
    assert len(rt) == 80


def test_central_window_matching_keeps_one_target():
    ms = _targeted_file()
    rt, _ = core.raw.summed_fragment_xic(ms, _fragments(500.0),
                                         match='window')
    assert len(rt) == 40


def test_precursor_matching_keeps_one_target():
    ms = _targeted_file()
    rt, inten = core.raw.summed_fragment_xic(ms, _fragments(500.0),
                                             match='precursor',
                                             match_tol=0.05)
    assert len(rt) == 40


def test_auto_picks_precursor_on_targeted_data_and_says_so():
    ms = _targeted_file()
    with pytest.warns(UserWarning, match='targeted acquisition detected'):
        rt, _ = core.raw.summed_fragment_xic(ms, _fragments(500.0))
    assert len(rt) == 40


def test_merging_inflates_the_weak_partner():
    ms = _targeted_file(amplitudes=(1000.0, 10.0))
    _, merged = core.raw.summed_fragment_xic(ms, _fragments(500.5),
                                             match='window', window='union')
    _, clean = core.raw.summed_fragment_xic(ms, _fragments(500.5),
                                            match='precursor', match_tol=0.05)
    assert merged.max() > 10 * clean.max()


def test_an_unknown_match_mode_is_rejected():
    ms = _targeted_file()
    with pytest.raises(ValueError, match="match must be"):
        core.raw.summed_fragment_xic(ms, _fragments(500.0), match='nearest')


def test_guard_raises_when_the_tolerance_does_not_isolate():
    ms = _targeted_file()
    with pytest.raises(ValueError, match='does not isolate'):
        core.raw.assert_targets_resolved(ms, [500.0, 500.5], tol=1.0,
                                         names=['a', 'b'])


def test_guard_passes_when_it_does():
    ms = _targeted_file()
    core.raw.assert_targets_resolved(ms, [500.0, 500.5], tol=0.05)


def test_guard_names_the_offending_targets():
    ms = _targeted_file()
    with pytest.raises(ValueError) as exc:
        core.raw.assert_targets_resolved(ms, [500.0], tol=1.0,
                                         names=['LGFTDLFSK_2'])
    assert 'LGFTDLFSK_2' in str(exc.value)


def _curve(levels=(20.0, 200.0, 2000.0, 20000.0), slope=35.0, noise=40.0,
           seed=0, n=4):
    rng = np.random.default_rng(seed)
    conc, area = [], []
    for c in levels:
        for _ in range(n):
            conc.append(c)
            area.append(slope * c + rng.normal(0, noise))
    return np.array(conc), np.array(area)


def test_intercept_se_is_the_default():
    c, a = _curve()
    fit = core.raw.calibration_fit(c, a)
    assert fit['sigma'] == 'intercept_se'
    assert fit['lod'] == pytest.approx(3.3 * fit['intercept_se'] / fit['slope'])


def test_both_blank_estimates_are_always_reported():
    c, a = _curve()
    fit = core.raw.calibration_fit(c, a)
    assert np.isfinite(fit['intercept_se'])
    assert np.isfinite(fit['sd_lowest_level'])


def test_lowest_level_remains_available():
    c, a = _curve()
    fit = core.raw.calibration_fit(c, a, sigma='lowest_level')
    assert fit['lod'] == pytest.approx(
        3.3 * fit['sd_lowest_level'] / fit['slope'])


def test_the_default_does_not_depend_on_where_the_curve_starts():
    c_full, a_full = _curve(levels=(20.0, 200.0, 2000.0, 20000.0))
    c_short, a_short = _curve(levels=(200.0, 2000.0, 20000.0))
    lod_full = core.raw.calibration_fit(c_full, a_full)['lod']
    lod_short = core.raw.calibration_fit(c_short, a_short)['lod']
    assert 0.4 < lod_short / lod_full < 2.5


def test_an_unknown_sigma_is_rejected():
    c, a = _curve()
    with pytest.raises(ValueError, match="sigma must be"):
        core.raw.calibration_fit(c, a, sigma='blank')


def test_anchor_correlation_is_high_for_the_same_peak():
    from zenotof8600_analytics.raw.targeted import _anchor_correlation
    rt = np.linspace(0, 1, 60)
    peak = np.exp(-0.5 * ((rt - 0.5) / 0.05) ** 2)
    r = _anchor_correlation(rt, peak * 0.01, (rt, peak), 0.5, 0.15)
    assert r > 0.99


def test_anchor_correlation_rejects_an_interference():
    from zenotof8600_analytics.raw.targeted import _anchor_correlation
    rt = np.linspace(0, 1, 60)
    peak = np.exp(-0.5 * ((rt - 0.5) / 0.05) ** 2)
    rng = np.random.default_rng(3)
    junk = rng.uniform(0, 5, rt.size)
    r = _anchor_correlation(rt, junk, (rt, peak), 0.5, 0.15)
    assert r < 0.5


def test_anchor_correlation_is_nan_when_there_is_nothing_to_compare():
    from zenotof8600_analytics.raw.targeted import _anchor_correlation
    rt = np.linspace(0, 1, 60)
    flat = np.ones_like(rt)
    assert np.isnan(_anchor_correlation(rt, flat, (rt, flat), 0.5, 0.15))
    assert np.isnan(_anchor_correlation(np.array([]), np.array([]),
                                        (rt, flat), 0.5, 0.15))
