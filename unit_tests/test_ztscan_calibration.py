"""Q1 axis calibration from residual precursors."""

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from zenotof8600_analytics.ztscan import (
    binned_median, fit_q1_axis, q1_correction,
)

NEUTRON = 1.00335


def _ladder(mono, z, n, heights):
    return [mono + k * NEUTRON / z for k in range(n)], list(heights)


class TestBinnedMedian:
    def test_drops_sparse_bins_rather_than_drawing_them(self):
        x = np.arange(300.0)
        y = np.zeros(300)
        c, m, e, n = binned_median(x, y, 10, min_n=100)
        assert c.size == 0

    def test_se_shrinks_with_n(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, 20000)
        y = rng.normal(0, 1, 20000)
        _, _, e_few, _ = binned_median(x, y, 20, min_n=10)
        _, _, e_many, _ = binned_median(x, y, 4, min_n=10)
        assert e_many.mean() < e_few.mean()


def _synthetic(n=8000, seed=0, bias=lambda mz: 0.0, rt_slope=0.0,
               background=0.0):
    rng = np.random.default_rng(seed)
    mz = rng.uniform(400, 900, n)
    rt = rng.uniform(10, 30, n)
    dev = bias(mz) + rt_slope * rt + rng.normal(0, 0.05, n)
    k = rng.random(n) < background
    dev[k] = rng.uniform(-1.4, 1.4, int(k.sum()))
    return mz - dev, mz, rt


class TestFitQ1Axis:
    def test_recovers_a_curved_mz_offset(self):
        def bias(mz):
            return -0.25 * np.sin((mz - 400) / 500 * np.pi)

        q1, mz, rt = _synthetic(bias=bias)
        tab = fit_q1_axis(q1, mz, rt)
        corrected = q1 - q1_correction(tab, q1, rt)
        assert np.abs(np.median(mz - corrected)) < 0.02
        assert (np.abs(np.median(mz - corrected))
                < np.abs(np.median(mz - q1)))

    def test_recovers_an_rt_slope(self):
        q1, mz, rt = _synthetic(rt_slope=0.01)
        tab = fit_q1_axis(q1, mz, rt)
        assert tab['slope_rt'] == pytest.approx(-0.01, abs=0.003)

    def test_survives_heavy_co_transmission_background(self):
        def bias(mz):
            return 0.20 * np.ones_like(mz)

        q1, mz, rt = _synthetic(bias=bias, background=0.75, n=20000)
        tab = fit_q1_axis(q1, mz, rt)
        corrected = q1 - q1_correction(tab, q1, rt)
        assert np.abs(np.median(mz - corrected)) < 0.05

    def test_sign_is_subtracted_not_added(self):
        q1, mz, rt = _synthetic(bias=lambda mz: 0.2 * np.ones_like(mz))
        tab = fit_q1_axis(q1, mz, rt)
        assert np.median(tab['mz_corr']) < 0

    def test_table_is_in_q1_correction_form(self):
        q1, mz, rt = _synthetic()
        tab = fit_q1_axis(q1, mz, rt)
        assert tab['kind'] == 'table'
        assert len(tab['mz_knots']) == len(tab['mz_corr'])
        assert np.all(np.diff(tab['mz_knots']) > 0)
        assert np.isfinite(q1_correction(tab, q1, rt)).all()

    def test_refuses_too_few_points(self):
        with pytest.raises(ValueError):
            fit_q1_axis(np.arange(5.0), np.arange(5.0), np.arange(5.0))

    def test_works_without_retention_time(self):
        q1, mz, _ = _synthetic()
        tab = fit_q1_axis(q1, mz, None)
        assert tab['slope_rt'] == 0.0


