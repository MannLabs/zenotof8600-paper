"""Characterization: every plot_* renders without error and returns a Figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import zenotof8600_analytics as core


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _assert_fig(ret):
    assert isinstance(ret, tuple)
    assert isinstance(ret[0], Figure)
    return ret


DF_SI_PLOTS = [
    'plot_cv_violin',
    'plot_cv_ecdf',
    'plot_cv_stacked_bar_combined',
    'plot_pca',
    'plot_rank',
    'plot_venn',
    'plot_set_overlap',
    'plot_upset',
    'plot_intensity_histogram',
    'plot_peptide_gravy_distribution',
    'plot_peptide_length_distribution',
]


def test_plot_bar(agg):
    _assert_fig(core.plot_bar(agg, x_col='condition2', y_col='protein_group'))


def test_plot_boxplot_with_points(agg):
    _assert_fig(core.plot_boxplot_with_points(agg, x_col='condition2', y_col='protein_group'))


def test_plot_overlapping_bars(agg):
    ret = core.plot_overlapping_bars(
        agg, x_col='run', higher_y_col='protein_group', lower_y_col='PG20')
    _assert_fig(ret)


def test_plot_volcano(df, sample_info):
    fig, ax, vdf = core.plot_volcano(
        df, sample_info, condition_a='500SPD', condition_b='200SPD')
    assert isinstance(fig, Figure)
    assert len(vdf) > 0


def test_cv_violin_stats_nonempty(df, sample_info):
    fig, ax, cv_stats = core.plot_cv_violin(df, sample_info)
    assert len(cv_stats) > 0


