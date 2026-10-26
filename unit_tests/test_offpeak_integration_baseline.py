"""The integration baseline must not be measured on the peak."""
import numpy as np
import pytest

from zenotof8600_analytics.raw.peaks import integrate_peak


def _trace(height, base=5.0, n=400):
    rt = np.linspace(0, 1, n)
    return rt, height * np.exp(-0.5 * ((rt - 0.5) / 0.01) ** 2) + base


def test_offpeak_recovers_the_true_baseline():
    rt, y = _trace(100.0)
    got = integrate_peak(rt, y, 0.47, 0.53, baseline='offpeak')
    assert got['baseline'] == pytest.approx(5.0, abs=0.05)


def test_window_local_baselines_are_biased_by_the_peak():
    rt, y = _trace(100.0)
    for b in ('min_boundary', 'p5', 'median'):
        got = integrate_peak(rt, y, 0.47, 0.53, baseline=b)
        assert got['baseline'] > 5.0, b


def test_offpeak_bias_does_not_depend_on_concentration():
    errs = {}
    for b in ('min_boundary', 'offpeak'):
        got = []
        for height in (1e4, 1e3, 1e2, 1e1):
            rt, y = _trace(height)
            r = integrate_peak(rt, y, 0.47, 0.53, baseline=b)
            got.append(r['baseline'] - 5.0)
        errs[b] = max(got) - min(got)
    assert errs['offpeak'] < errs['min_boundary'] / 10, errs


def test_offpeak_falls_back_when_the_trace_is_all_peak():
    rt = np.linspace(0.47, 0.53, 13)
    y = 100 * np.exp(-0.5 * ((rt - 0.5) / 0.01) ** 2) + 5.0
    got = integrate_peak(rt, y, 0.47, 0.53, baseline='offpeak')
    assert np.isfinite(got['baseline'])
    assert got['area'] > 0


def test_mean_not_median_on_sparse_counting_data():
    rng = np.random.default_rng(0)
    rt = np.linspace(0, 1, 400)
    y = rng.poisson(0.3, 400).astype(float)
    y += 50 * np.exp(-0.5 * ((rt - 0.5) / 0.01) ** 2)
    got = integrate_peak(rt, y, 0.47, 0.53, baseline='offpeak')
    assert got['baseline'] > 0.1
