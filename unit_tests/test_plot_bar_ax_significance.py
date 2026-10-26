"""`plot_bar` drawing onto a supplied axes, and its significance brackets."""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import zenotof8600_analytics as core


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


@pytest.fixture
def separated():
    return pd.DataFrame({
        'condition': ['ctrl'] * 3 + ['treated'] * 3,
        'value': [1.00, 1.05, 0.95, 10.0, 10.5, 9.5],
    })


@pytest.fixture
def overlapping():
    return pd.DataFrame({
        'condition': ['ctrl'] * 3 + ['treated'] * 3,
        'value': [1.0, 1.2, 0.8, 1.1, 0.9, 1.05],
    })


def _texts(ax):
    return [t.get_text() for t in ax.texts]


def test_ax_draws_on_the_supplied_axes(separated):
    fig, axes = plt.subplots(1, 2)
    ret_fig, ret_ax = core.plot_bar(separated, x_col='condition',
                                    y_col='value', ax=axes[0])
    assert ret_ax is axes[0]
    assert ret_fig is fig
    assert len(axes[0].patches) == 2
    assert not axes[1].patches


def test_ax_does_not_resize_the_callers_figure(separated):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    before = tuple(fig.get_size_inches())
    positions = [tuple(a.get_position().bounds) for a in axes.ravel()]
    core.plot_bar(separated, x_col='condition', y_col='value', ax=axes[1, 1])
    assert tuple(fig.get_size_inches()) == before
    assert [tuple(a.get_position().bounds) for a in axes.ravel()] == positions


def test_figsize_is_honoured_only_when_creating_the_figure(separated):
    fig, _ = core.plot_bar(separated, x_col='condition', y_col='value',
                           figsize=(3, 3))
    assert tuple(fig.get_size_inches()) == (3.0, 3.0)

    host, ax = plt.subplots(figsize=(6, 6))
    out, _ = core.plot_bar(separated, x_col='condition', y_col='value',
                           ax=ax, figsize=(3, 3))
    assert out is host
    assert tuple(host.get_size_inches()) == (6.0, 6.0)


def test_no_brackets_without_the_argument(separated):
    _, ax = core.plot_bar(separated, x_col='condition', y_col='value',
                          show_values=False)
    assert '*' not in ''.join(_texts(ax))


@pytest.mark.parametrize('test', ['welch', 'ttest'])
def test_separated_groups_get_a_star(separated, test):
    _, ax = core.plot_bar(separated, x_col='condition', y_col='value',
                          show_values=False, significance_test=test)
    assert any(t.startswith('*') for t in _texts(ax))


def test_mwu_needs_more_replicates_than_the_t_tests():
    small = pd.DataFrame({'condition': ['ctrl'] * 3 + ['treated'] * 3,
                          'value': [1.0, 1.05, 0.95, 10.0, 10.5, 9.5]})
    big = pd.DataFrame({'condition': ['ctrl'] * 5 + ['treated'] * 5,
                        'value': [1.0, 1.05, 0.95, 1.1, 0.9,
                                  10.0, 10.5, 9.5, 10.2, 9.8]})
    _, ax_small = core.plot_bar(small, x_col='condition', y_col='value',
                                show_values=False, significance_test='mwu')
    _, ax_big = core.plot_bar(big, x_col='condition', y_col='value',
                              show_values=False, significance_test='mwu')
    assert not any(t.startswith('*') for t in _texts(ax_small))
    assert any(t.startswith('*') for t in _texts(ax_big))


def test_overlapping_groups_get_nothing_by_default(overlapping):
    _, ax = core.plot_bar(overlapping, x_col='condition', y_col='value',
                          show_values=False, significance_test='welch')
    assert not any(t.startswith('*') or t == 'ns' for t in _texts(ax))


def test_show_ns_labels_the_non_significant_pair(overlapping):
    _, ax = core.plot_bar(overlapping, x_col='condition', y_col='value',
                          show_values=False, significance_test='welch',
                          significance_show_ns=True)
    assert 'ns' in _texts(ax)


def test_bracket_sits_above_the_tallest_bar(separated):
    _, ax = core.plot_bar(separated, x_col='condition', y_col='value',
                          show_values=False, significance_test='welch')
    top_bar = max(p.get_height() for p in ax.patches)
    star = next(t for t in ax.texts if t.get_text().startswith('*'))
    assert star.get_position()[1] > top_bar
    assert ax.get_ylim()[1] > top_bar


def test_pairs_selects_which_comparisons_are_drawn():
    df = pd.DataFrame({
        'condition': ['a'] * 3 + ['b'] * 3 + ['c'] * 3,
        'value': [1.0, 1.1, 0.9, 10.0, 10.1, 9.9, 20.0, 20.1, 19.9],
    })
    _, ax_all = core.plot_bar(df, x_col='condition', y_col='value',
                              show_values=False, significance_test='welch')
    _, ax_one = core.plot_bar(df, x_col='condition', y_col='value',
                              show_values=False, significance_test='welch',
                              significance_pairs=[(0, 2)])
    n_all = sum(t.startswith('*') for t in _texts(ax_all))
    n_one = sum(t.startswith('*') for t in _texts(ax_one))
    assert n_all == 3
    assert n_one == 1


def test_correction_raises_the_p_values():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'condition': ['a'] * 4 + ['b'] * 4 + ['c'] * 4,
        'value': np.r_[rng.normal(1.0, 0.1, 4), rng.normal(1.3, 0.1, 4),
                       rng.normal(1.0, 0.1, 4)],
    })
    _, raw = core.plot_bar(df, x_col='condition', y_col='value',
                           show_values=False, significance_test='welch')
    _, adj = core.plot_bar(df, x_col='condition', y_col='value',
                           show_values=False, significance_test='welch',
                           significance_correction='fdr_bh')
    n_raw = sum(t.startswith('*') for t in _texts(raw))
    n_adj = sum(t.startswith('*') for t in _texts(adj))
    assert n_adj <= n_raw


def test_tests_the_replicates_not_the_plotted_means(separated):
    means = separated.groupby('condition', as_index=False)['value'].mean()
    _, ax_raw = core.plot_bar(separated, x_col='condition', y_col='value',
                              show_values=False, significance_test='welch')
    _, ax_mean = core.plot_bar(means, x_col='condition', y_col='value',
                               show_values=False, significance_test='welch',
                               significance_show_ns=True)
    assert any(t.startswith('*') for t in _texts(ax_raw))
    assert not any(t.startswith('*') for t in _texts(ax_mean))


def test_significance_is_skipped_and_warns_with_hue():
    df = pd.DataFrame({
        'condition': ['a'] * 6 + ['b'] * 6,
        'engine': (['x'] * 3 + ['y'] * 3) * 2,
        'value': [1, 1.1, 0.9, 2, 2.1, 1.9, 10, 10.1, 9.9, 20, 20.1, 19.9],
    })
    with pytest.warns(UserWarning, match='significance_test ignored'):
        _, ax = core.plot_bar(df, x_col='condition', y_col='value',
                              hue_col='engine', show_values=False,
                              significance_test='welch')
    assert not any(t.startswith('*') for t in _texts(ax))


def test_significance_works_on_a_supplied_axes(separated):
    fig, axes = plt.subplots(2, 2)
    for a in axes.ravel():
        core.plot_bar(separated, x_col='condition', y_col='value', ax=a,
                      show_values=False, significance_test='welch')
    assert all(any(t.get_text().startswith('*') for t in a.texts)
               for a in axes.ravel())
