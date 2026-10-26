"""Tests for the DIA isolation-window design subpackage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import zenotof8600_analytics as core
from zenotof8600_analytics.dia_method import _oadia


E367_HEAD = [
    (300.3864250, 321.3959747),
    (321.3959747, 338.4037055),
    (338.4037055, 351.4096172),
    (351.4096172, 363.4150742),
    (363.4150742, 373.4196224),
]
E367_TAIL = [
    (1069.7361283, 1090.7456780),
    (1090.7456780, 1112.7556825),
    (1112.7556825, 1137.7670513),
    (1137.7670513, 1166.7802385),
    (1166.7802385, 1199.7952453),
]


@pytest.fixture(scope='module')
def gpf_like_mz():
    rng = np.random.default_rng(20260901)
    values = np.concatenate([
        rng.normal(520, 110, 60_000),
        rng.normal(700, 190, 30_000),
        rng.lognormal(np.log(450), 0.35, 20_000),
    ])
    return values[(values > 300) & (values < 1500)].tolist()


def test_reproduces_pydiaid_reference():
    for start, end in E367_HEAD + E367_TAIL:
        for edge in (start, end):
            snapped = _oadia.find_closest_forbidden_zone(edge, False)
            assert snapped == pytest.approx(edge, abs=1e-6), (
                f'boundary {edge} no longer sits on a forbidden zone'
            )


def test_forbidden_zone_constants():
    assert _oadia.calculate_forbidden_zones(500, False) == pytest.approx(
        np.ceil(500 / 1.00045475) * 1.00045475 + 0.25)
    assert _oadia.calculate_forbidden_zones(500, True) == pytest.approx(
        np.ceil(500 / 1.00045475) * 1.00045475 + 0.18)
    assert (_oadia.calculate_forbidden_zones(500, False)
            != _oadia.calculate_forbidden_zones(500, True))


def test_dynamic_scheme_is_a_clean_partition(gpf_like_mz):
    scheme = core.design_windows(gpf_like_mz, (300, 1200), 40)
    assert scheme.n_windows == 40
    assert np.allclose(scheme.starts[1:], scheme.ends[:-1])
    assert np.all(scheme.widths > 0)
    assert np.all(np.diff(scheme.starts) > 0)
    scheme.validate()


def test_dynamic_scheme_balances_precursors(gpf_like_mz):
    dynamic = core.design_windows(gpf_like_mz, (300, 1200), 40)
    fixed = core.design_windows(gpf_like_mz, (300, 1200), 40, window_type='fixed')

    dyn_balance = core.evaluate_windows(dynamic, gpf_like_mz)['balance']
    fix_balance = core.evaluate_windows(fixed, gpf_like_mz)['balance']
    assert dyn_balance < fix_balance / 10

    occupancy = core.window_occupancy(dynamic, gpf_like_mz)
    dense = occupancy[occupancy['window_center'].between(450, 650)]['window_width']
    tail = occupancy[occupancy['window_center'] > 1000]['window_width']
    assert dense.median() < tail.median()


def test_max_width_caps_balance_in_a_thin_tail(gpf_like_mz):
    tight = core.design_windows(gpf_like_mz, (300, 1200), 40, max_width=100.0)
    loose = core.design_windows(gpf_like_mz, (300, 1200), 40, max_width=400.0)

    occupancy = core.window_occupancy(tight, gpf_like_mz)
    saturated = occupancy[occupancy['window_width'] >= 100.0 - 1e-6]
    assert not saturated.empty
    assert saturated['n_precursors'].max() < occupancy['n_precursors'].median()

    assert (core.evaluate_windows(loose, gpf_like_mz)['balance']
            < core.evaluate_windows(tight, gpf_like_mz)['balance'])


def test_fixed_scheme_has_equal_widths(gpf_like_mz):
    scheme = core.design_windows(gpf_like_mz, (300, 1200), 45,
                                 window_type='fixed', forbidden_zones=False)
    assert scheme.n_windows == 45
    assert np.allclose(scheme.widths, 900 / 45)
    assert scheme.covered_range == pytest.approx((300.0, 1200.0))


def test_forbidden_zones_move_the_edges_inside(gpf_like_mz):
    snapped = core.design_windows(gpf_like_mz, (300, 1200), 30)
    plain = core.design_windows(gpf_like_mz, (300, 1200), 30,
                                forbidden_zones=False)

    assert snapped.on_forbidden_zones()
    assert not plain.on_forbidden_zones()
    assert plain.covered_range == pytest.approx((300.0, 1200.0))
    lo, hi = snapped.covered_range
    assert lo > 300.0 and hi < 1200.0
    assert lo - 300.0 < 1.01 and 1200.0 - hi < 1.01


def test_max_width_is_enforced(gpf_like_mz):
    scheme = core.design_windows(gpf_like_mz, (300, 1200), 60,
                                 max_width=25.0, forbidden_zones=False)
    assert scheme.widths.max() <= 25.0 + 1e-6


def test_impossible_constraints_raise(gpf_like_mz):
    with pytest.raises(ValueError, match='constraints'):
        core.design_windows(gpf_like_mz, (300, 1200), 10, max_width=20.0)


@pytest.mark.parametrize('kwargs, match', [
    ({'window_type': 'variable'}, 'not supported'),
    ({'n_windows': 0}, 'n_windows'),
])
def test_bad_arguments_raise(gpf_like_mz, kwargs, match):
    call = {'mz_range': (300, 1200), 'n_windows': 20, **kwargs}
    with pytest.raises(ValueError, match=match):
        core.design_windows(gpf_like_mz, **call)


def test_increasing_mz_range_required(gpf_like_mz):
    with pytest.raises(ValueError, match='increasing'):
        core.design_windows(gpf_like_mz, (1200, 300), 20)


def test_empty_range_raises():
    with pytest.raises(ValueError, match='No precursors'):
        core.design_windows([1300.0, 1400.0], (300, 1200), 10)


def test_occupancy_counts_every_covered_precursor(gpf_like_mz):
    scheme = core.design_windows(gpf_like_mz, (300, 1200), 30)
    occupancy = core.window_occupancy(scheme, gpf_like_mz)
    lo, hi = scheme.covered_range
    values = np.asarray(gpf_like_mz)
    expected = int(np.sum((values >= lo) & (values <= hi)))
    assert occupancy['n_precursors'].sum() == expected


def test_occupancy_matches_brute_force(gpf_like_mz):
    scheme = core.design_windows(gpf_like_mz, (300, 1200), 25)
    values = np.asarray(gpf_like_mz)
    fast = core.window_occupancy(scheme, gpf_like_mz)['n_precursors'].to_numpy()
    slow = np.array([
        int(np.sum((values >= s) & (values < e)))
        for s, e in zip(scheme.starts, scheme.ends)
    ])
    slow[-1] += int(np.sum(values == scheme.ends[-1]))
    assert np.array_equal(fast, slow)


def test_evaluate_reports_coverage(gpf_like_mz):
    scheme = core.design_windows(gpf_like_mz, (300, 900), 30)
    stats = core.evaluate_windows(scheme, gpf_like_mz)
    values = np.asarray(gpf_like_mz)
    lo, hi = scheme.covered_range
    expected = 100 * np.mean((values >= lo) & (values <= hi))
    assert stats['coverage_pct'] == pytest.approx(expected, abs=0.01)
    assert stats['coverage_pct'] < 100
    assert stats['on_forbidden_zones'] is True


def test_load_precursor_mz_accepts_many_shapes(tmp_path, gpf_like_mz):
    frame = pd.DataFrame({'mz': gpf_like_mz[:100]})
    path = tmp_path / 'lib.csv'
    frame.to_csv(path, index=False)

    from_frame = core.load_precursor_mz(frame)
    from_series = core.load_precursor_mz(frame['mz'])
    from_list = core.load_precursor_mz(gpf_like_mz[:100])
    assert from_frame == from_series == from_list
    assert core.load_precursor_mz(path) == pytest.approx(from_frame)


@pytest.mark.parametrize('column', ['mz', 'm/z', 'Precursor.Mz', 'precursor_mz'])
def test_load_precursor_mz_finds_engine_column_names(column):
    frame = pd.DataFrame({column: [400.0, 500.0], 'other': ['a', 'b']})
    assert core.load_precursor_mz(frame) == [400.0, 500.0]


def test_load_precursor_mz_drops_nan():
    frame = pd.DataFrame({'mz': [400.0, np.nan, 500.0]})
    assert core.load_precursor_mz(frame) == [400.0, 500.0]


def test_load_precursor_mz_reports_missing_column():
    frame = pd.DataFrame({'retention_time': [1.0, 2.0]})
    with pytest.raises(KeyError, match='No precursor m/z column'):
        core.load_precursor_mz(frame)


