"""The XIC window is derived from the peak's own width: +-0.5 FWHM."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from scipy.special import erf

import zenotof8600_analytics as core


def _run(centre=800.0, fwhm_ppm=30.0, n_scans=5, n_points=61, ms_level=1,
         centroided=False):
    sigma = centre * fwhm_ppm / 1e6 / (2 * np.sqrt(2 * np.log(2)))
    if centroided:
        grid = np.array([centre])
        inten = np.array([1000.0])
    else:
        grid = np.linspace(centre - 6 * sigma, centre + 6 * sigma, n_points)
        inten = np.exp(-0.5 * ((grid - centre) / sigma) ** 2) * 1000.0
    mz, it, start, stop = [], [], [], []
    for _ in range(n_scans):
        start.append(len(mz))
        mz.extend(grid)
        it.extend(inten)
        stop.append(len(mz))

    class _MS:
        pass

    ms = _MS()
    ms.spectrum_df = pd.DataFrame({
        'rt': np.linspace(1.0, 1.4, n_scans),
        'ms_level': np.full(n_scans, ms_level, dtype=int),
        'peak_start_idx': start, 'peak_stop_idx': stop})
    ms.peak_df = pd.DataFrame({'mz': mz, 'intensity': it})
    return ms, centre


def test_half_a_fwhm_is_within_one_percent_of_the_sn_optimum():
    def sn(frac):
        k = frac * 2 * np.sqrt(2 * np.log(2))
        return erf(k / np.sqrt(2)) / np.sqrt(k)

    grid = np.arange(0.20, 1.60, 0.005)
    best = grid[int(np.argmax([sn(f) for f in grid]))]
    assert 0.55 <= best <= 0.65
    assert sn(core.raw.XIC_FWHM_FRACTION) / sn(best) > 0.98
    assert sn(1.0) / sn(best) < 0.92
    assert sn(1.5) / sn(best) < 0.80


def test_the_fraction_is_half():
    assert core.raw.XIC_FWHM_FRACTION == 0.5


@pytest.mark.parametrize('fwhm_ppm', [12.0, 30.0, 60.0])
def test_measured_resolution_recovers_a_known_width(fwhm_ppm):
    ms, _ = _run(fwhm_ppm=fwhm_ppm)
    got = core.raw.measure_resolution_ppm(ms, ms_level=1)
    assert got == pytest.approx(fwhm_ppm, rel=0.05)


def test_centroided_data_has_no_measurable_width():
    ms, _ = _run(centroided=True)
    assert np.isnan(core.raw.measure_resolution_ppm(ms, ms_level=1))


def test_tolerance_is_half_the_measured_fwhm():
    ms, _ = _run(fwhm_ppm=30.0)
    assert core.raw.xic_tolerance_ppm(ms, ms_level=1) == pytest.approx(
        15.0, rel=0.05)


def test_tolerance_tracks_the_instrument():
    wide, _ = _run(fwhm_ppm=60.0)
    tight, _ = _run(fwhm_ppm=12.0)
    assert (core.raw.xic_tolerance_ppm(wide, ms_level=1)
            > 3 * core.raw.xic_tolerance_ppm(tight, ms_level=1))


def test_centroided_data_falls_back():
    ms, _ = _run(centroided=True)
    assert core.raw.xic_tolerance_ppm(ms, ms_level=1) == \
        core.raw.XIC_FALLBACK_PPM


def test_the_measurement_is_cached_on_the_file():
    ms, _ = _run()
    first = core.raw.xic_tolerance_ppm(ms, ms_level=1)
    ms.peak_df = ms.peak_df.iloc[:0]
    assert core.raw.xic_tolerance_ppm(ms, ms_level=1) == first


@pytest.mark.parametrize('fn', ['extract_xic',
                                'targeted_xic', 'peak_from_fragments',
                                'summed_fragment_xic', 'quantify_targets'])
def test_every_xic_entry_point_defaults_to_auto(fn):
    assert inspect.signature(
        getattr(core.raw, fn)).parameters['mz_tol'].default == 'auto'


def test_auto_equals_the_number_it_resolves_to():
    ms, centre = _run(fwhm_ppm=30.0)
    tol = core.raw.xic_tolerance_ppm(ms, ms_level=1)
    a = core.raw.extract_xic(ms, centre)['intensity'].to_numpy()
    b = core.raw.extract_xic(ms, centre, mz_tol=tol)['intensity'].to_numpy()
    assert np.allclose(a, b)


def test_an_explicit_number_still_wins():
    ms, centre = _run(fwhm_ppm=30.0)
    auto = core.raw.extract_xic(ms, centre)['intensity'].sum()
    tight = core.raw.extract_xic(ms, centre, mz_tol=3.0)['intensity'].sum()
    wide = core.raw.extract_xic(ms, centre, mz_tol=200.0)['intensity'].sum()
    assert tight < auto < wide


def test_auto_keeps_about_three_quarters_of_the_peak():
    ms, centre = _run(fwhm_ppm=30.0)
    full = core.raw.extract_xic(ms, centre, mz_tol=200.0)['intensity'].sum()
    got = core.raw.extract_xic(ms, centre)['intensity'].sum()
    assert 0.70 <= got / full <= 0.82


def test_a_bad_tolerance_string_is_rejected():
    ms, centre = _run()
    with pytest.raises(ValueError, match="must be a number or 'auto'"):
        core.raw.extract_xic(ms, centre, mz_tol='wide')
    with pytest.raises(ValueError, match='only defined in ppm'):
        core.raw.extract_xic(ms, centre, mz_tol='auto', mz_tol_unit='Da')


def _sparse_run(centre=800.0, fwhm_ppm=30.0, n_signal=3, n_empty=40):
    sigma = centre * fwhm_ppm / 1e6 / (2 * np.sqrt(2 * np.log(2)))
    grid = np.linspace(centre - 6 * sigma, centre + 6 * sigma, 61)
    peak = np.exp(-0.5 * ((grid - centre) / sigma) ** 2) * 10000.0
    rng = np.random.default_rng(0)
    mz, it, start, stop, lvl = [], [], [], [], []
    for i in range(n_signal + n_empty):
        start.append(len(mz))
        mz.extend(grid)
        it.extend(peak if i < n_signal
                  else rng.uniform(0.0, 20.0, grid.size))
        stop.append(len(mz)); lvl.append(2)

    class _MS:
        pass

    ms = _MS()
    ms.spectrum_df = pd.DataFrame({
        'rt': np.linspace(0, 6, n_signal + n_empty),
        'ms_level': lvl, 'peak_start_idx': start, 'peak_stop_idx': stop})
    ms.peak_df = pd.DataFrame({'mz': mz, 'intensity': it})
    return ms, centre


def test_width_comes_from_scans_that_hold_signal():
    ms, _ = _sparse_run(fwhm_ppm=30.0, n_signal=3, n_empty=40)
    assert core.raw.measure_resolution_ppm(ms, ms_level=2) == pytest.approx(
        30.0, rel=0.10)


def test_a_run_that_is_all_noise_reports_no_width():
    ms, _ = _sparse_run(n_signal=0, n_empty=40)
    res = core.raw.measure_resolution_ppm(ms, ms_level=2)
    assert np.isnan(res) or res < 0.5 * 30.0 or res > 30.0


def _shifted_run(theoretical=800.0, shift_ppm=-6.0, fwhm_ppm=30.0,
                 n_scans=9, noise=0.0, seed=0):
    centre = theoretical * (1 + shift_ppm / 1e6)
    sigma = centre * fwhm_ppm / 1e6 / (2 * np.sqrt(2 * np.log(2)))
    grid = np.linspace(theoretical - 8 * sigma, theoretical + 8 * sigma, 121)
    rng = np.random.default_rng(seed)
    mz, it, start, stop = [], [], [], []
    for _ in range(n_scans):
        y = np.exp(-0.5 * ((grid - centre) / sigma) ** 2) * 10000.0
        if noise:
            y = y + rng.uniform(0, noise, grid.size)
        start.append(len(mz)); mz.extend(grid); it.extend(y); stop.append(len(mz))

    class _MS:
        pass

    ms = _MS()
    ms.spectrum_df = pd.DataFrame({
        'rt': np.linspace(1.0, 1.8, n_scans),
        'ms_level': np.full(n_scans, 2, dtype=int),
        'peak_start_idx': start, 'peak_stop_idx': stop})
    ms.peak_df = pd.DataFrame({'mz': mz, 'intensity': it})
    return ms, theoretical, centre


@pytest.mark.parametrize('shift_ppm', [-8.0, -3.0, 0.0, 4.0, 9.0])
def test_the_peak_is_located_where_it_actually_is(shift_ppm):
    ms, theoretical, true_mz = _shifted_run(shift_ppm=shift_ppm)
    got = core.raw.locate_peak_mz(ms, theoretical, ms_level=2)
    assert abs(got - true_mz) / true_mz * 1e6 < 3.0


def test_position_precision_improves_with_signal():
    errs = {}
    for noise in (2000.0, 50.0):
        e = []
        for seed in range(12):
            ms, th, true_mz = _shifted_run(shift_ppm=-5.0, noise=noise,
                                           seed=seed)
            got = core.raw.locate_peak_mz(ms, th, ms_level=2)
            e.append((got - true_mz) / true_mz * 1e6)
        errs[noise] = float(np.std(e))
    assert errs[50.0] < errs[2000.0]
    assert errs[50.0] < 1.0


def test_an_empty_channel_keeps_its_nominal_window():
    ms, th, _ = _shifted_run(shift_ppm=0.0)
    rng = np.random.default_rng(1)
    ms.peak_df = pd.DataFrame({
        'mz': ms.peak_df['mz'],
        'intensity': rng.uniform(0.0, 10.0, len(ms.peak_df))})
    got = core.raw.locate_peak_mz(ms, th, ms_level=2)
    assert abs(got - th) / th * 1e6 < 3.0


def test_recentring_does_not_reach_a_neighbouring_ion():
    ms, th, _ = _shifted_run(shift_ppm=0.0)
    far = th * (1 + 80.0 / 1e6)
    got = core.raw.locate_peak_mz(ms, far, ms_level=2)
    assert got == pytest.approx(far)
