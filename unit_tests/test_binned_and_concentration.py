"""binned_stats, effective_n, gini, and peak_shape."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zenotof8600_analytics.stats import (
    binned_stats,
)
from zenotof8600_analytics.raw.extraction import peak_shape


def test_binned_stats_recovers_a_known_trend():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, 4000)
    y = 2.0 * x + rng.normal(0, 1e-6, x.size)
    out = binned_stats(x, y, bins=10)
    assert list(out.columns) == ['x', 'median', 'q_lo', 'q_hi', 'n']
    assert len(out) == 10
    assert np.allclose(out['median'], 2.0 * out['x'], rtol=1e-3)
    assert out['x'].is_monotonic_increasing


def test_binned_stats_quantile_bins_are_equally_populated():
    rng = np.random.default_rng(1)
    x = rng.exponential(1.0, 3000)
    out_q = binned_stats(x, x, bins=10, mode='quantile')
    out_w = binned_stats(x, x, bins=10, mode='width', min_count=1)
    assert out_q['n'].max() / out_q['n'].min() < 1.2
    assert out_w['n'].max() / out_w['n'].min() > 5


def test_binned_stats_drops_sparse_bins():
    x = np.concatenate([np.zeros(100), [50.0, 50.1]])
    y = np.concatenate([np.ones(100), [9.0, 9.0]])
    out = binned_stats(x, y, bins=4, mode='width', min_count=8)
    assert (out['n'] >= 8).all()
    assert 9.0 not in out['median'].values


def test_binned_stats_iqr_brackets_the_median():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 10, 2000)
    y = rng.normal(5, 2, 2000)
    out = binned_stats(x, y, bins=5)
    assert (out['q_lo'] <= out['median']).all()
    assert (out['median'] <= out['q_hi']).all()


def test_binned_stats_handles_empty_and_nan_input():
    empty = binned_stats([], [])
    assert len(empty) == 0 and list(empty.columns) == ['x', 'median', 'q_lo',
                                                       'q_hi', 'n']
    nans = binned_stats([1.0, np.nan, 3.0], [np.nan, 2.0, np.nan], bins=3)
    assert len(nans) == 0


def test_binned_stats_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        binned_stats([1, 2, 3], [1, 2, 3], mode='nonsense')


def _gaussian(x, centre, fwhm, height=1.0):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return height * np.exp(-0.5 * ((x - centre) / sigma) ** 2)


def test_peak_shape_recovers_a_known_fwhm():
    x = np.linspace(90, 110, 2001)
    y = _gaussian(x, 100.0, 2.0, height=7.0)
    pk = peak_shape(x, y)
    assert pk['apex_x'] == pytest.approx(100.0, abs=0.02)
    assert pk['height'] == pytest.approx(7.0, rel=1e-3)
    assert pk['width'] == pytest.approx(2.0, rel=0.01)


def test_peak_shape_beats_counting_on_a_coarse_grid():
    x = np.linspace(12, 36, 31)
    y = _gaussian(x, 26.3, 2.15)
    step = 0.8
    counted = (y >= 0.5 * y.max()).sum() * step
    pk = peak_shape(x, y)
    assert counted / step == pytest.approx(round(counted / step), abs=1e-9)
    assert counted == pytest.approx(2.4, abs=1e-9)
    assert pk['width'] == pytest.approx(2.15, rel=0.05)
    assert abs(pk['width'] - counted) > 0.1


def test_peak_shape_returns_nan_width_at_the_edge_of_the_range():
    x = np.linspace(0, 10, 101)
    y = np.linspace(1.0, 5.0, 101)
    pk = peak_shape(x, y)
    assert np.isnan(pk['width'])
    assert np.isnan(pk['right_x'])
    assert np.isfinite(pk['height'])


def test_peak_shape_can_search_near_a_given_centre():
    x = np.linspace(0, 100, 4001)
    y = _gaussian(x, 20.0, 1.0, 10.0) + _gaussian(x, 70.0, 1.0, 3.0)
    tall = peak_shape(x, y)
    assert tall['apex_x'] == pytest.approx(20.0, abs=0.1)
    near70 = peak_shape(x, y, centre=70.0, search=5.0)
    assert near70['apex_x'] == pytest.approx(70.0, abs=0.1)
    assert near70['height'] == pytest.approx(3.0, rel=1e-2)


def test_peak_shape_area_is_the_gaussian_integral():
    x = np.linspace(80, 120, 4001)
    y = _gaussian(x, 100.0, 3.0, height=2.0)
    sigma = 3.0 / (2 * np.sqrt(2 * np.log(2)))
    assert peak_shape(x, y)['area'] == pytest.approx(
        2.0 * sigma * np.sqrt(2 * np.pi), rel=1e-3)


def test_peak_shape_height_fraction_controls_the_level():
    x = np.linspace(90, 110, 4001)
    y = _gaussian(x, 100.0, 2.0)
    w50 = peak_shape(x, y)['width']
    w10 = peak_shape(x, y, height_fraction=0.1)['width']
    assert w10 / w50 == pytest.approx(np.sqrt(np.log(10) / np.log(2)), rel=0.01)


def test_peak_shape_rejects_degenerate_input():
    for bad in ([], [1.0], [np.nan] * 5):
        pk = peak_shape(np.arange(len(bad)), bad)
        assert np.isnan(pk['height'])
        assert pk['n_points'] == 0
