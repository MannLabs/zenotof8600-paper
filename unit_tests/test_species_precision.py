"""Tests for `species_ratio_precision` / `plot_ratio_precision`."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import zenotof8600_analytics as core
from zenotof8600_analytics import species as sp


def _design(noise_by_species, n_per_species=80, n_reps=4, seed=0):
    rng = np.random.default_rng(seed)
    comp = {'A': {'Human': 50, 'Yeast': 45, 'E. coli': 5},
            'B': {'Human': 50, 'Yeast': 25, 'E. coli': 25},
            'C': {'Human': 50, 'Yeast': 5, 'E. coli': 45}}
    suffix = {'Human': 'HUMAN', 'Yeast': 'YEAST', 'E. coli': 'ECOLI'}
    rows, runs = [], []
    for cond, mix in comp.items():
        for rep in range(n_reps):
            run = f'{cond}{rep}'
            runs.append({'run': run, 'condition2': cond, 'engine': 'peaks'})
            for s, share in mix.items():
                for i in range(n_per_species):
                    base = 1e6 * (i + 1) * share
                    val = base * np.exp(noise_by_species[s]
                                        * rng.standard_normal())
                    rows.append({'run': run, 'engine': 'peaks',
                                 'protein_group': f'{s}_{i}',
                                 'protein_names': f'P{i}_{suffix[s]}',
                                 'pg_intensity': val,
                                 'precursor_intensity': val,
                                 'precursor_id': f'{s}_{i}_pep'})
    df = core.assign_species(pd.DataFrame(rows))
    return df, pd.DataFrame(runs), comp


PAIRS = [('A', 'C'), ('A', 'B'), ('B', 'C')]


class TestRatioPrecision:
    def test_shapes_and_columns(self):
        df, si, comp = _design({'Human': 0.1, 'Yeast': 0.1, 'E. coli': 0.1})
        between, cv = core.species_ratio_precision(df, si, PAIRS, comp,
                                                   verbose=False)
        assert list(between.columns) == ['condition_a', 'condition_b',
                                         'species', 'expected',
                                         'sigma_iqr_log2', 'error_pct', 'n']
        assert len(between) == 9
        assert list(cv.columns) == ['condition', 'species', 'load_pct',
                                    'protein_cv_pct']
        assert len(cv) == 9

    def test_error_pct_tracks_the_injected_noise(self):
        df, si, comp = _design({'Human': 0.05, 'Yeast': 0.30,
                                'E. coli': 0.05}, seed=2)
        between, _ = core.species_ratio_precision(df, si, PAIRS, comp,
                                                  verbose=False)
        med = between.groupby('species')['error_pct'].median()
        assert med['Yeast'] > 3 * med['Human']
        assert med['E. coli'] == pytest.approx(med['Human'], rel=0.5)

    def test_error_pct_is_sigma_iqr_of_the_ratio_expressed_as_percent(self):
        df, si, comp = _design({'Human': 0.2, 'Yeast': 0.2, 'E. coli': 0.2},
                               seed=5)
        between, _ = core.species_ratio_precision(df, si, [('A', 'C')], comp,
                                                  verbose=False)
        table = core.species_ratio_table(df, si, 'A', 'C')
        for _, r in between.iterrows():
            v = table.loc[table['species'] == r['species'], 'log2_ratio']
            assert r['sigma_iqr_log2'] == pytest.approx(core.sigma_iqr(v))
            assert r['error_pct'] == pytest.approx(
                (2 ** r['sigma_iqr_log2'] - 1) * 100)

    def test_zero_noise_gives_zero_spread(self):
        df, si, comp = _design({'Human': 0.0, 'Yeast': 0.0, 'E. coli': 0.0})
        between, cv = core.species_ratio_precision(df, si, PAIRS, comp,
                                                   verbose=False)
        assert between['error_pct'].to_numpy() == pytest.approx(0.0, abs=1e-9)
        assert cv['protein_cv_pct'].to_numpy() == pytest.approx(0.0, abs=1e-9)

    def test_load_pct_is_the_species_share_of_its_condition(self):
        df, si, comp = _design({'Human': 0.1, 'Yeast': 0.1, 'E. coli': 0.1})
        _, cv = core.species_ratio_precision(df, si, PAIRS, comp,
                                             verbose=False)
        got = cv.set_index(['condition', 'species'])['load_pct']
        assert got[('A', 'Yeast')] == pytest.approx(45.0)
        assert got[('A', 'E. coli')] == pytest.approx(5.0)
        assert got[('C', 'Yeast')] == pytest.approx(5.0)
        assert got[('B', 'Human')] == pytest.approx(50.0)

    def test_fractions_and_percentages_give_the_same_load(self):
        df, si, comp = _design({'Human': 0.1, 'Yeast': 0.1, 'E. coli': 0.1})
        frac = {c: {s: v / 100 for s, v in mix.items()}
                for c, mix in comp.items()}
        _, a = core.species_ratio_precision(df, si, PAIRS, comp, verbose=False)
        _, b = core.species_ratio_precision(df, si, PAIRS, frac, verbose=False)
        assert a['load_pct'].to_numpy() == pytest.approx(
            b['load_pct'].to_numpy())

    def test_protein_cv_matches_a_direct_replicate_cv(self):
        df, si, comp = _design({'Human': 0.15, 'Yeast': 0.15,
                                'E. coli': 0.15}, seed=9)
        _, cv = core.species_ratio_precision(df, si, [('A', 'C')], comp,
                                             verbose=False)
        table = core.species_ratio_table(df, si, 'A', 'C')
        for s in ('Human', 'Yeast', 'E. coli'):
            want = table.loc[table['species'] == s, 'cv_a'].median() * 100
            got = cv.loc[(cv['condition'] == 'A') & (cv['species'] == s),
                         'protein_cv_pct'].iloc[0]
            assert got == pytest.approx(want)

    def test_a_condition_absent_from_pairs_is_not_reported(self):
        df, si, comp = _design({'Human': 0.1, 'Yeast': 0.1, 'E. coli': 0.1})
        _, cv = core.species_ratio_precision(df, si, [('A', 'B')], comp,
                                             verbose=False)
        assert set(cv['condition']) == {'A', 'B'}


class TestRatioPrecisionPanel:
    def test_draws_both_panels_and_returns_tidy_source(self):
        df, si, comp = _design({'Human': 0.1, 'Yeast': 0.1, 'E. coli': 0.1})
        between, cv = core.species_ratio_precision(df, si, PAIRS, comp,
                                                   verbose=False)
        fig, axes, src = core.plot_ratio_precision(
            between, cv, species_order=['Human', 'Yeast', 'E. coli'])
        assert len(axes) == 2
        assert set(src['panel']) == {'ratio_error_between_proteins',
                                     'protein_quant_cv'}
        assert len(src) == len(between) + len(cv)
        left = [ln for ln in axes[0].get_lines() if ln.get_label() in
                ('Human', 'Yeast', 'E. coli')]
        assert len(left) == 3
        assert all(ln.get_linestyle() == 'None' for ln in left)
        right = [ln for ln in axes[1].get_lines() if ln.get_label() in
                 ('Human', 'Yeast', 'E. coli')]
        assert len(right) == 3
        for ln in right:
            x = ln.get_xdata()
            assert list(x) == sorted(x)
        plt.close(fig)

    def test_axis_labels_name_the_quantity_only(self):
        df, si, comp = _design({'Human': 0.1, 'Yeast': 0.1, 'E. coli': 0.1})
        between, cv = core.species_ratio_precision(df, si, PAIRS, comp,
                                                   verbose=False)
        fig, axes, _ = core.plot_ratio_precision(between, cv)
        assert axes[0].get_xlabel() == 'Expected log$_2$ ratio'
        assert axes[1].get_xlabel() == 'Species share of mixture [%]'
        assert axes[0].get_ylabel() == 'Precision [%]'
        for ax in axes:
            assert '(' not in ax.get_title()
        assert axes[0].get_ylim()[0] == 0
        plt.close(fig)

    def test_accepts_supplied_axes(self):
        df, si, comp = _design({'Human': 0.1, 'Yeast': 0.1, 'E. coli': 0.1})
        between, cv = core.species_ratio_precision(df, si, PAIRS, comp,
                                                   verbose=False)
        fig, mine = plt.subplots(1, 2)
        _, axes, _ = core.plot_ratio_precision(between, cv, axes=mine)
        assert axes[0] is mine[0] and axes[1] is mine[1]
        plt.close(fig)
