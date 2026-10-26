"""Tests for plot_binned_median and plot_cv_vs_abundance."""

import inspect

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import zenotof8600_analytics as core
from zenotof8600_analytics.plotting import binned, cv as cv_mod


def _trend(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for group, scale in (('fast', 1.0), ('slow', 2.0)):
        x = rng.uniform(4.0, 6.5, n)
        y = scale * (30.0 * np.exp(-(x - 4.0)) + rng.normal(0, 1.0, n))
        rows.append(pd.DataFrame({'group': group, 'x': x, 'y': y}))
    return pd.concat(rows, ignore_index=True)


class TestBinnedMedianPanel:
    def test_source_data_has_one_row_per_drawn_point(self):
        data = _trend()
        fig, ax, src = binned.plot_binned_median(
            data, 'x', 'y', group_col='group', n_bins=8)
        assert list(src.columns) == ['group', 'bin', 'x', 'median',
                                     'q_lo', 'q_hi', 'n']
        assert len(src) == 16
        drawn = [ln for ln in ax.get_lines() if ln.get_label() in
                 ('fast', 'slow')]
        assert len(drawn) == 2
        assert sum(len(ln.get_xdata()) for ln in drawn) == len(src)
        plt.close(fig)

    def test_medians_and_quartiles_are_the_binned_statistics(self):
        data = _trend()
        fig, ax, src = binned.plot_binned_median(
            data, 'x', 'y', group_col='group', n_bins=6)
        for group in ('fast', 'slow'):
            sub = data[data['group'] == group]
            want = core.binned_stats(sub['x'], sub['y'], bins=6)
            got = src[src['group'] == group]
            assert got['median'].to_numpy() == pytest.approx(
                want['median'].to_numpy())
            assert got['q_lo'].to_numpy() == pytest.approx(
                want['q_lo'].to_numpy())
            assert got['x'].to_numpy() == pytest.approx(want['x'].to_numpy())
        plt.close(fig)

    def test_bins_are_per_group_not_shared(self):
        data = _trend()
        data.loc[data['group'] == 'slow', 'x'] *= 0.5
        fig, ax, src = binned.plot_binned_median(
            data, 'x', 'y', group_col='group', n_bins=7)
        assert (src.groupby('group').size() == 7).all()
        lo = src[src['group'] == 'slow']['x'].max()
        hi = src[src['group'] == 'fast']['x'].min()
        assert lo < hi
        plt.close(fig)

    def test_sparse_bins_are_dropped_rather_than_drawn(self):
        data = _trend(n=20)
        fig, ax, src = binned.plot_binned_median(
            data, 'x', 'y', group_col='group', n_bins=10, min_count=8)
        assert len(src) < 20
        assert (src['n'] >= 8).all()
        plt.close(fig)

    def test_no_group_column_draws_one_line_and_no_legend(self):
        data = _trend().drop(columns='group')
        fig, ax, src = binned.plot_binned_median(data, 'x', 'y', n_bins=5)
        assert len(src) == 5
        assert set(src['group']) == {'all'}
        assert ax.get_legend() is None
        plt.close(fig)

    def test_empty_input_returns_the_columns_and_does_not_raise(self):
        data = pd.DataFrame(columns=['group', 'x', 'y'], dtype=float)
        fig, ax, src = binned.plot_binned_median(data, 'x', 'y',
                                                 group_col='group')
        assert src.empty
        assert list(src.columns) == ['group', 'bin', 'x', 'median',
                                     'q_lo', 'q_hi', 'n']
        plt.close(fig)

    def test_threshold_line_is_drawn_only_when_asked(self):
        data = _trend()
        fig, ax, _ = binned.plot_binned_median(data, 'x', 'y',
                                               group_col='group',
                                               threshold=20.0)
        assert any(ln.get_ydata()[0] == 20.0 for ln in ax.get_lines()
                   if len(set(ln.get_ydata())) == 1)
        plt.close(fig)
        fig, ax, _ = binned.plot_binned_median(data, 'x', 'y',
                                               group_col='group')
        assert not [ln for ln in ax.get_lines()
                    if len(ln.get_ydata()) and len(set(ln.get_ydata())) == 1
                    and ln.get_linestyle() == ':']
        plt.close(fig)

    def test_accepts_a_supplied_axis(self):
        data = _trend()
        fig, axes = plt.subplots(1, 2)
        _, ax, _ = binned.plot_binned_median(data, 'x', 'y',
                                             group_col='group', ax=axes[1])
        assert ax is axes[1]
        assert not axes[0].get_lines()
        plt.close(fig)


def _cv_frame(n_entities=200, n_reps=4, seed=0):
    rng = np.random.default_rng(seed)
    rows, runs = [], []
    for cond in ('ctrl', 'treated'):
        for rep in range(n_reps):
            run = f'{cond}{rep}'
            runs.append({'run': run, 'condition2': cond, 'engine': 'peaks'})
            for i in range(n_entities):
                base = 10 ** rng.uniform(4, 6.5)
                rows.append({'run': run, 'engine': 'peaks',
                             'precursor_id': f'p{i}',
                             'precursor_intensity':
                                 base * np.exp(0.15 * rng.standard_normal())})
    return pd.DataFrame(rows), pd.DataFrame(runs)


