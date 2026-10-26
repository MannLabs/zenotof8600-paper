"""Every plot drawn on a log intensity axis averages replicates IN LOG SPACE."""

import numpy as np
import pandas as pd
import pytest

import zenotof8600_analytics as core


INTENSITIES = {
    ('a', 'r1', 'P1'): 2.0 ** 10, ('a', 'r2', 'P1'): 2.0 ** 20,
    ('a', 'r1', 'P2'): 2.0 ** 12, ('a', 'r2', 'P2'): 2.0 ** 12,
    ('b', 'r3', 'P1'): 2.0 ** 14, ('b', 'r4', 'P1'): 2.0 ** 16,
    ('b', 'r3', 'P2'): 2.0 ** 13, ('b', 'r4', 'P2'): 2.0 ** 13,
}


@pytest.fixture
def skewed():
    rows = [{'run': run, 'engine': 'diann', 'protein_group': pg,
             'genes': pg, 'pg_intensity': value, 'condition2': cond}
            for (cond, run, pg), value in INTENSITIES.items()]
    df = pd.DataFrame(rows)
    sample_info = (df[['run', 'engine', 'condition2']].drop_duplicates()
                   .reset_index(drop=True))
    return df, sample_info


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close('all')


def test_plot_rank_means_in_log_space(skewed):
    df, sample_info = skewed
    _fig, _ax, rank_df = core.plot_rank(
        df, sample_info, level='protein', group_col='condition2')
    got = (rank_df[rank_df['group'] == 'a']
           .set_index('protein_group')['log10_mean_intensity'])
    assert got['P1'] == pytest.approx(15.0 * np.log10(2.0))
    assert got['P2'] == pytest.approx(12.0 * np.log10(2.0))


def test_plot_volcano_fold_change_is_mean_of_log_difference(skewed):
    df, sample_info = skewed
    _fig, _ax, volcano_df = core.plot_volcano(
        df, sample_info, level='protein', group_col='condition2',
        condition_a='a', condition_b='b', min_valid_per_condition=2,
        method='welch')
    got = volcano_df.set_index('protein_group')['log2_fc']
    assert got['P1'] == pytest.approx(0.0, abs=1e-9)
    assert got['P2'] == pytest.approx(-1.0)
