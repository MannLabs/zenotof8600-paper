"""plot_overlapping_bars: the light bar is the UNION, the points the replicates."""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

import zenotof8600_analytics as core


@pytest.fixture
def per_run():
    return pd.DataFrame({
        'condition': ['A'] * 3 + ['B'] * 3,
        'protein_group': [90, 95, 100, 70, 72, 80],
        'total_protein_groups': [120] * 3 + [85] * 3,
        'PG20': [80] * 3 + [60] * 3,
    })


def _bars(ax):
    return [p.get_height() for p in ax.patches]


def _points(ax):
    return [sorted(c.get_offsets()[:, 1]) for c in ax.collections]


@pytest.mark.parametrize('col', ['protein_group', 'total_protein_groups'])
def test_bar_is_union_points_are_runs(per_run, col):
    fig, ax, summary = core.plot_overlapping_bars(
        per_run, x_col='condition', higher_y_col=col, lower_y_col='PG20')
    assert _bars(ax) == [120, 80, 85, 60]
    assert _points(ax) == [[90, 95, 100], [70, 72, 80]]
    assert summary['higher_mean'].tolist() == [120, 85]
    assert summary['higher_is_union'].all()
    assert np.allclose(summary['points_mean'], [95, 74])
    assert ax.get_ylim()[1] >= 120


def test_union_without_per_run_partner_raises(per_run):
    with pytest.raises(ValueError, match='per-run'):
        core.plot_overlapping_bars(
            per_run.drop(columns='protein_group'), x_col='condition',
            higher_y_col='total_protein_groups', lower_y_col='PG20')


def test_union_must_be_constant_per_condition(per_run):
    bad = per_run.copy()
    bad.loc[0, 'total_protein_groups'] = 121
    with pytest.raises(ValueError, match='constant'):
        core.plot_overlapping_bars(bad, x_col='condition',
                                   higher_y_col='protein_group',
                                   lower_y_col='PG20')


def test_per_run_column_without_union_keeps_mean(per_run):
    fig, ax, summary = core.plot_overlapping_bars(
        per_run.drop(columns='total_protein_groups'), x_col='condition',
        higher_y_col='protein_group', lower_y_col='PG20')
    assert summary['higher_mean'].tolist() == [95, 74]
    assert not summary['higher_is_union'].any()


def test_generic_column_keeps_mean(per_run):
    d = per_run.assign(rate=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    fig, ax, summary = core.plot_overlapping_bars(
        d, x_col='condition', higher_y_col='rate', lower_y_col='rate',
        value_fmt=lambda v: f'{v:.1f}')
    assert summary['higher_mean'].tolist() == [2.0, 5.0]
    assert _points(ax) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_explicit_points_col(per_run):
    d = per_run.assign(other=[1, 2, 3, 4, 5, 6])
    fig, ax, _ = core.plot_overlapping_bars(
        d, x_col='condition', higher_y_col='total_protein_groups',
        lower_y_col='PG20', points_col='other')
    assert _points(ax) == [[1, 2, 3], [4, 5, 6]]
