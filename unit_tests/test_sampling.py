"""Tests for stratified_sample."""

import numpy as np
import pandas as pd
import pytest

from zenotof8600_analytics.raw import stratified_sample


@pytest.fixture
def pool():
    rng = np.random.default_rng(0)
    logint = rng.uniform(3.0, 7.5, 10000)
    mz = 360 + (logint - 3.0) / 4.5 * 400 + rng.normal(0, 60, 10000)
    return pd.DataFrame({'mz': np.clip(mz, 360, 960),
                         'intensity': 10 ** logint})


def test_returns_everything_when_the_pool_is_small():
    d = pd.DataFrame({'mz': [400.0, 500.0], 'intensity': [1e4, 1e5]})
    assert len(stratified_sample(d, 100)) == 2


def test_reaches_the_requested_size():
    d = pd.DataFrame({'mz': np.linspace(360, 960, 5000),
                      'intensity': 10 ** np.linspace(3, 7, 5000)})
    assert len(stratified_sample(d, 1000)) == pytest.approx(1000, abs=30)


def test_every_cell_is_filled_to_what_the_pool_can_offer(pool):
    n = 2500
    mzb_p = pd.qcut(pool.mz, 5, labels=False)
    ab_p = pd.qcut(np.log10(pool.intensity), 5, labels=False)
    avail = pd.crosstab(mzb_p, ab_p)

    out = stratified_sample(pool, n)
    edges_mz = np.quantile(pool.mz, np.linspace(0, 1, 6))
    edges_ab = np.quantile(np.log10(pool.intensity), np.linspace(0, 1, 6))
    got = pd.crosstab(np.clip(np.digitize(out.mz, edges_mz[1:-1]), 0, 4),
                      np.clip(np.digitize(np.log10(out.intensity),
                                          edges_ab[1:-1]), 0, 4))
    got = got.reindex(index=range(5), columns=range(5), fill_value=0)

    quota = n // 25
    for i in range(5):
        for j in range(5):
            a = int(avail.iloc[i, j]) if (i in avail.index
                                          and j in avail.columns) else 0
            g = int(got.iloc[i, j])
            assert g <= a + 1, f'cell ({i},{j}) returned more than it holds'
            if a >= quota:
                assert g >= 0.5 * quota, (
                    f'cell ({i},{j}) holds {a} but only {g} were taken')


def test_it_spans_far_more_abundance_than_top_n(pool):
    strat = stratified_sample(pool, 2000)
    topn = pool.nlargest(2000, 'intensity')
    span = lambda d: np.log10(d.intensity.max()) - np.log10(d.intensity.min())
    assert span(strat) > span(topn) + 1.5
    rng_ = lambda d: d.mz.max() - d.mz.min()
    assert rng_(strat) > rng_(topn)


def test_top_fraction_restricts_before_stratifying(pool):
    out = stratified_sample(pool, 1000, top_fraction=0.3)
    cut = pool.intensity.quantile(0.7)
    assert out.intensity.min() >= cut
    assert np.log10(out.intensity.max()) - np.log10(out.intensity.min()) > 0.5


def test_reproducible_and_seed_changes_it(pool):
    a = stratified_sample(pool, 500)
    b = stratified_sample(pool, 500)
    c = stratified_sample(pool, 500, seed=7)
    assert a.equals(b)
    assert not a.equals(c)


def test_zero_and_missing_intensities_are_dropped():
    d = pd.DataFrame({'mz': [400.0, 500.0, 600.0, 700.0],
                      'intensity': [0.0, np.nan, 1e5, 1e6]})
    out = stratified_sample(d, 10)
    assert len(out) == 2 and out.intensity.min() > 0


def test_empty_pool():
    d = pd.DataFrame({'mz': [], 'intensity': []})
    assert len(stratified_sample(d, 10)) == 0


def test_output_is_sorted_by_mz(pool):
    out = stratified_sample(pool, 800)
    assert out.mz.is_monotonic_increasing
