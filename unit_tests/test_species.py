"""Tests for the mixed-species benchmark module."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')

import zenotof8600_analytics as core
from zenotof8600_analytics import species as sp


class TestAssignSpecies:
    def test_entry_name_suffix_maps_to_the_organism(self):
        df = pd.DataFrame({'protein_names': ['ZFP91_HUMAN', 'EF1A_YEAST',
                                             'DNAK_ECOLI']})
        assert list(core.assign_species(df)['species']) == [
            'Human', 'Yeast', 'E. coli']

    def test_group_spanning_two_organisms_is_ambiguous_not_the_first_one(self):
        df = pd.DataFrame({'protein_names': ['ACTB_HUMAN;ACT_YEAST']})
        assert core.assign_species(df)['species'].iloc[0] == sp.AMBIGUOUS

    def test_group_of_one_organism_stays_that_organism(self):
        df = pd.DataFrame({'protein_names': ['RL4_HUMAN;RL4B_HUMAN']})
        assert core.assign_species(df)['species'].iloc[0] == 'Human'

    def test_unrecognised_suffix_is_unknown(self):
        df = pd.DataFrame({'protein_names': ['SOMETHING_MARTIAN', 'nosuffix']})
        assert set(core.assign_species(df)['species']) == {sp.UNKNOWN}

    def test_missing_names_do_not_crash(self):
        df = pd.DataFrame({'protein_names': ['RL4_HUMAN', None]})
        assert list(core.assign_species(df)['species']) == ['Human', sp.UNKNOWN]

    def test_custom_suffix_map(self):
        df = pd.DataFrame({'protein_names': ['X_PIG']})
        out = core.assign_species(df, suffixes={'PIG': 'Pig'})
        assert out['species'].iloc[0] == 'Pig'

    def test_missing_column_names_the_problem(self):
        with pytest.raises(KeyError, match='entry name'):
            core.assign_species(pd.DataFrame({'x': [1]}))


class TestExpectedRatios:
    def test_nominal_ratio_from_two_designs(self):
        out = core.expected_log2_ratios({'Y': 40, 'H': 50, 'E': 10},
                                        {'Y': 20, 'H': 50, 'E': 30})
        assert out['Y'] == pytest.approx(1.0)
        assert out['H'] == pytest.approx(0.0)
        assert out['E'] == pytest.approx(np.log2(1 / 3))

    def test_offset_shifts_every_species(self):
        out = core.expected_log2_ratios({'Y': 40, 'H': 50}, {'Y': 20, 'H': 50},
                                        offset=-0.1)
        assert out['Y'] == pytest.approx(0.9)
        assert out['H'] == pytest.approx(-0.1)

    def test_species_absent_from_one_side_is_undefined(self):
        out = core.expected_log2_ratios({'Y': 40}, {'H': 50})
        assert np.isnan(out['Y']) and np.isnan(out['H'])


def _mixture(fold_by_species, n_per_species=60, n_reps=4, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows, runs = [], []
    suffix = {'Human': 'HUMAN', 'Yeast': 'YEAST', 'E. coli': 'ECOLI'}
    for cond, side in (('A', 0), ('B', 1)):
        for rep in range(n_reps):
            run = f'{cond}{rep}'
            runs.append({'run': run, 'condition2': cond, 'engine': 'peaks'})
            for s, fold in fold_by_species.items():
                for i in range(n_per_species):
                    base = 1e6 * (i + 1) * (fold if side == 0 else 1.0)
                    val = base * np.exp(noise * rng.standard_normal())
                    rows.append({
                        'run': run, 'engine': 'peaks',
                        'protein_group': f'{s}_{i}',
                        'protein_names': f'P{i}_{suffix[s]}',
                        'pg_intensity': val,
                        'precursor_intensity': val,
                        'precursor_id': f'{s}_{i}_pep',
                    })
    df = core.assign_species(pd.DataFrame(rows))
    return df, pd.DataFrame(runs)


class TestRatioTable:
    def test_recovers_the_designed_fold_changes(self):
        df, si = _mixture({'Human': 1.0, 'Yeast': 2.0, 'E. coli': 0.25})
        t = core.species_ratio_table(df, si, 'A', 'B')
        med = t.groupby('species')['log2_ratio'].median()
        assert med['Human'] == pytest.approx(0.0, abs=1e-9)
        assert med['Yeast'] == pytest.approx(1.0, abs=1e-9)
        assert med['E. coli'] == pytest.approx(-2.0, abs=1e-9)

    def test_ratio_is_a_difference_of_mean_logs_not_a_log_of_mean_ratios(self):
        df, si = _mixture({'Human': 1.0}, noise=0.5, seed=3)
        t = core.species_ratio_table(df, si, 'A', 'B')
        wide = df.pivot_table(index='protein_group', columns='run',
                              values='pg_intensity')
        a = [c for c in wide.columns if c.startswith('A')]
        b = [c for c in wide.columns if c.startswith('B')]
        expected = (np.log2(wide[a]).mean(axis=1)
                    - np.log2(wide[b]).mean(axis=1))
        got = t.set_index('protein_group')['log2_ratio']
        assert got.reindex(expected.index).to_numpy() == pytest.approx(
            expected.to_numpy(), abs=1e-9)

    def test_completeness_filter_drops_partly_observed_proteins(self):
        df, si = _mixture({'Human': 1.0}, n_per_species=10)
        df = df[~((df['protein_group'] == 'Human_0') & (df['run'] == 'A0'))]
        full = core.species_ratio_table(df, si, 'A', 'B', min_completeness=1.0)
        loose = core.species_ratio_table(df, si, 'A', 'B', min_completeness=0.5)
        assert 'Human_0' not in set(full['protein_group'])
        assert 'Human_0' in set(loose['protein_group'])

    def test_ambiguous_groups_are_excluded_by_default(self):
        df, si = _mixture({'Human': 1.0}, n_per_species=5)
        df.loc[df['protein_group'] == 'Human_0', 'protein_names'] = 'X_HUMAN;Y_YEAST'
        df = core.assign_species(df.drop(columns='species'))
        t = core.species_ratio_table(df, si, 'A', 'B')
        assert sp.AMBIGUOUS not in set(t['species'])
        assert 'Human_0' not in set(t['protein_group'])

    def test_unknown_condition_names_the_available_ones(self):
        df, si = _mixture({'Human': 1.0}, n_per_species=3)
        with pytest.raises(ValueError, match='available'):
            core.species_ratio_table(df, si, 'A', 'NOPE')


class TestPlots:
    def test_expected_composition_returns_the_plotted_percentages(self):
        comps = {'m1': {'Human': 50, 'Yeast': 45, 'E. coli': 5},
                 'm2': {'Human': 50, 'Yeast': 5, 'E. coli': 45}}
        fig, ax, src = core.plot_expected_composition(comps)
        assert set(src['condition']) == {'m1', 'm2'}
        assert src.groupby('condition')['percentage'].sum().eq(100).all()
        matplotlib.pyplot.close(fig)

    def test_species_counts_detected_exceeds_quantified(self):
        df, si = _mixture({'Human': 1.0}, n_per_species=20)
        df.loc[df['protein_group'].isin([f'Human_{i}' for i in range(10)]),
               'pg_intensity'] = np.nan
        _, _, det = core.plot_species_counts(df, si, count_basis='detected')
        _, _, quant = core.plot_species_counts(df, si, count_basis='quantified')
        assert det['n'].sum() == 2 * 20
        assert quant['n'].sum() == 2 * 10
        matplotlib.pyplot.close('all')

    def test_bad_count_basis_is_rejected(self):
        df, si = _mixture({'Human': 1.0}, n_per_species=3)
        with pytest.raises(ValueError, match='count_basis'):
            core.plot_species_counts(df, si, count_basis='guessed')

    def test_species_ratio_offset_comes_from_the_reference_species(self):
        df, si = _mixture({'Human': 1 / 1.1, 'Yeast': 2 / 1.1})
        fig, axes, stats, ratio_df = core.plot_species_ratio(
            df, si, 'A', 'B',
            composition_a={'Human': 50, 'Yeast': 40},
            composition_b={'Human': 50, 'Yeast': 20},
            reference_species='Human')
        offset = stats['loading_offset_log2'].iloc[0]
        assert offset == pytest.approx(-np.log2(1.1), abs=1e-6)
        exp = stats.set_index('species')['expected_log2_ratio']
        assert exp['Human'] == pytest.approx(offset, abs=1e-9)
        assert exp['Yeast'] == pytest.approx(1.0 + offset, abs=1e-9)
        med = stats.set_index('species')['median']
        assert med['Yeast'] == pytest.approx(exp['Yeast'], abs=1e-6)
        matplotlib.pyplot.close(fig)

    def test_degenerate_species_does_not_crash_the_density_panel(self):
        df, si = _mixture({'Human': 1.0, 'Yeast': 2.0})
        _, _, stats, _ = core.plot_species_ratio(
            df, si, 'A', 'B',
            composition_a={'Human': 50, 'Yeast': 40},
            composition_b={'Human': 50, 'Yeast': 20})
        assert set(stats['species']) == {'Human', 'Yeast'}
        assert stats['sd'].eq(0).all()
        matplotlib.pyplot.close('all')

    def test_species_ratio_without_reference_uses_nominal_ratios(self):
        df, si = _mixture({'Human': 1.0, 'Yeast': 2.0})
        _, _, stats, _ = core.plot_species_ratio(
            df, si, 'A', 'B',
            composition_a={'Human': 50, 'Yeast': 40},
            composition_b={'Human': 50, 'Yeast': 20},
            reference_species=None)
        assert stats['loading_offset_log2'].eq(0).all()
        assert stats.set_index('species')['expected_log2_ratio']['Yeast'] == \
            pytest.approx(1.0)
        matplotlib.pyplot.close('all')

    def test_species_ratio_needs_a_design_or_explicit_ratios(self):
        df, si = _mixture({'Human': 1.0}, n_per_species=3)
        with pytest.raises(ValueError, match='expected_ratios'):
            core.plot_species_ratio(df, si, 'A', 'B')


def _precursor_mixture(fold_by_species, n_per_species=20, n_prec=3, n_reps=4):
    rows, runs = [], []
    suffix = {'Human': 'HUMAN', 'Yeast': 'YEAST', 'E. coli': 'ECOLI'}
    for cond, side in (('A', 0), ('B', 1)):
        for rep in range(n_reps):
            run = f'{cond}{rep}'
            runs.append({'run': run, 'condition2': cond, 'engine': 'peaks'})
            for s, fold in fold_by_species.items():
                for i in range(n_per_species):
                    for p in range(n_prec):
                        base = 1e5 * (i + 1) * (p + 1)
                        rows.append({
                            'run': run, 'engine': 'peaks',
                            'protein_group': f'{s}_{i}',
                            'protein_names': f'P{i}_{suffix[s]}',
                            'precursor_id': f'{s}_{i}_p{p}',
                            'precursor_intensity':
                                base * (fold if side == 0 else 1.0),
                        })
    return core.assign_species(pd.DataFrame(rows)), pd.DataFrame(runs)


class TestSumPrecursors:
    def test_protein_value_is_the_sum_of_its_precursors(self):
        df, si = _precursor_mixture({'Human': 1.0}, n_per_species=2, n_prec=3)
        out = core.sum_precursors_to_protein(df, list(si['run']))
        assert out.loc['Human_0'].iloc[0] == pytest.approx(6e5)

    def test_incomplete_precursors_are_dropped_before_summing(self):
        df, si = _precursor_mixture({'Human': 1.0}, n_per_species=1, n_prec=3)
        runs = list(si['run'])
        df = df[~((df['precursor_id'] == 'Human_0_p2') & (df['run'] == runs[0]))]
        strict = core.sum_precursors_to_protein(df, runs, min_completeness=1.0)
        assert strict.loc['Human_0'].nunique() == 1
        assert strict.loc['Human_0'].iloc[0] == pytest.approx(3e5)

    def test_loose_completeness_keeps_more_precursors(self):
        df, si = _precursor_mixture({'Human': 1.0}, n_per_species=1, n_prec=3)
        runs = list(si['run'])
        df = df[~((df['precursor_id'] == 'Human_0_p2') & (df['run'] == runs[0]))]
        loose = core.sum_precursors_to_protein(df, runs, min_completeness=0.5)
        assert loose.loc['Human_0'].nunique() > 1

    def test_ratio_table_can_use_summed_precursors(self):
        df, si = _precursor_mixture({'Human': 1.0, 'Yeast': 2.0})
        t = core.species_ratio_table(df, si, 'A', 'B', quant='sum_precursors',
                                     value_col='precursor_intensity')
        med = t.groupby('species')['log2_ratio'].median()
        assert med['Yeast'] == pytest.approx(1.0, abs=1e-9)
        assert med['Human'] == pytest.approx(0.0, abs=1e-9)

    def test_unknown_quant_is_rejected(self):
        df, si = _precursor_mixture({'Human': 1.0}, n_per_species=2)
        with pytest.raises(ValueError, match='quant'):
            core.species_ratio_table(df, si, 'A', 'B', quant='magic')


class TestNormalizeSpecies:
    def test_reference_median_removes_a_per_run_loading_shift(self):
        df, si = _precursor_mixture({'Human': 1.0, 'Yeast': 2.0})
        bad = si['run'].iloc[0]
        df.loc[df['run'] == bad, 'precursor_intensity'] *= 0.5
        plain = core.species_ratio_table(df, si, 'A', 'B',
                                         quant='sum_precursors',
                                         value_col='precursor_intensity')
        fixed = core.species_ratio_table(df, si, 'A', 'B',
                                         quant='sum_precursors',
                                         value_col='precursor_intensity',
                                         normalize_species='Human')
        got_plain = plain.loc[plain['species'] == 'Yeast', 'log2_ratio'].median()
        got_fixed = fixed.loc[fixed['species'] == 'Yeast', 'log2_ratio'].median()
        assert abs(got_fixed - 1.0) < abs(got_plain - 1.0)
        assert got_fixed == pytest.approx(1.0, abs=1e-9)

    def test_unknown_reference_species_is_rejected(self):
        df, si = _precursor_mixture({'Human': 1.0}, n_per_species=2)
        with pytest.raises(ValueError, match='normalize_species'):
            core.species_ratio_table(df, si, 'A', 'B',
                                     quant='sum_precursors',
                                     value_col='precursor_intensity',
                                     normalize_species='Martian')


class TestRatioAccuracy:
    def _design(self):
        return {'A': {'Human': 50, 'Yeast': 40}, 'B': {'Human': 50, 'Yeast': 20}}

    def test_recovers_a_known_compression(self):
        df, si = _precursor_mixture({'Human': 1.0, 'Yeast': 2.0 ** 0.8})
        acc, fit = core.species_ratio_accuracy(
            df, si, [('A', 'B')], self._design(), verbose=False,
            quant='sum_precursors', value_col='precursor_intensity')
        assert len(acc) == 2
        assert fit['slope'] == pytest.approx(0.8, abs=1e-6)
        assert fit['r_squared'] == pytest.approx(1.0, abs=1e-9)

    def test_observed_is_the_median_not_the_mean(self):
        acc_cols = {'observed', 'observed_mean'}
        df, si = _precursor_mixture({'Human': 1.0, 'Yeast': 2.0})
        acc, _ = core.species_ratio_accuracy(
            df, si, [('A', 'B')], self._design(), verbose=False,
            quant='sum_precursors', value_col='precursor_intensity')
        assert acc_cols <= set(acc.columns)

    def test_correcting_by_the_fitted_slope_is_tautological(self):
        df, si = _precursor_mixture({'Human': 1.0, 'Yeast': 2.0 ** 0.8})
        acc, fit = core.species_ratio_accuracy(
            df, si, [('A', 'B')], self._design(), verbose=False,
            quant='sum_precursors', value_col='precursor_intensity')
        _, _, src = core.plot_ratio_accuracy(acc, fit, correct=True)
        assert src['fit_slope'].iloc[0] == pytest.approx(1.0, abs=1e-9)
        assert src['compression_slope'].iloc[0] == pytest.approx(0.8, abs=1e-6)
        matplotlib.pyplot.close('all')

    def test_r_squared_is_unchanged_by_the_correction(self):
        df, si = _precursor_mixture({'Human': 1.0, 'Yeast': 2.0 ** 0.9})
        acc, fit = core.species_ratio_accuracy(
            df, si, [('A', 'B')], self._design(), verbose=False,
            quant='sum_precursors', value_col='precursor_intensity')
        before = sp._fit_line(acc['expected'].to_numpy(),
                              acc['observed'].to_numpy())
        after = sp._fit_line(
            acc['expected'].to_numpy(),
            (acc['observed'] / fit['slope']).to_numpy())
        assert after['r_squared'] == pytest.approx(before['r_squared'], abs=1e-9)


class TestSpeciesCvEcdf:
    def test_curves_and_source_cover_every_species(self, capsys):
        df, si = _precursor_mixture({'Human': 1.0, 'Yeast': 1.0},
                                    n_per_species=30)
        prot = core.sum_precursors_to_protein(df, list(si['run']))
        long = (prot.stack().rename('pg_intensity').reset_index()
                .rename(columns={'level_0': 'protein_group'}))
        species = (df[['protein_group', 'species']].drop_duplicates()
                   .set_index('protein_group')['species'])
        long['species'] = long['protein_group'].map(species)
        fig, ax, src = core.plot_species_cv_ecdf(long, si, 'A')
        assert set(src['species']) == {'Human', 'Yeast'}
        assert src['cumulative_fraction'].max() == pytest.approx(1.0)
        matplotlib.pyplot.close(fig)

    def test_unknown_condition_is_rejected(self):
        df, si = _precursor_mixture({'Human': 1.0}, n_per_species=3)
        with pytest.raises(ValueError, match='not in'):
            core.plot_species_cv_ecdf(df, si, 'NOPE')


class TestReplicateScatterColumns:

    def test_sd_log2_is_the_log2_scatter_of_the_replicates(self):
        df, si = _mixture({'Human': 1.0}, n_per_species=8, noise=0.4, seed=11)
        t = core.species_ratio_table(df, si, 'A', 'B').set_index('protein_group')
        wide = df.pivot_table(index='protein_group', columns='run',
                              values='pg_intensity')
        a = [c for c in wide.columns if c.startswith('A')]
        expected = np.log2(wide[a]).std(axis=1, ddof=1)
        assert t['sd_log2_a'].to_numpy() == pytest.approx(
            expected.reindex(t.index).to_numpy(), abs=1e-9)

    def test_n_counts_the_runs_the_entity_was_quantified_in(self):
        df, si = _mixture({'Human': 1.0}, n_per_species=6)
        df = df[~((df['protein_group'] == 'Human_0') & (df['run'] == 'A0'))]
        t = (core.species_ratio_table(df, si, 'A', 'B', min_completeness=0.5)
             .set_index('protein_group'))
        assert t.loc['Human_0', 'n_a'] == 3
        assert t.loc['Human_1', 'n_a'] == 4
        assert t.loc['Human_0', 'completeness_a'] == pytest.approx(3 / 4)

    def test_zero_scatter_when_the_replicates_are_identical(self):
        df, si = _mixture({'Human': 1.0}, n_per_species=4, noise=0.0)
        t = core.species_ratio_table(df, si, 'A', 'B')
        assert t['sd_log2_a'].to_numpy() == pytest.approx(0.0, abs=1e-12)

    def test_standard_error_of_the_ratio_matches_a_direct_computation(self):
        df, si = _mixture({'Yeast': 2.0}, n_per_species=5, noise=0.3, seed=7)
        t = core.species_ratio_table(df, si, 'A', 'B')
        se = np.sqrt(t['sd_log2_a'] ** 2 / t['n_a']
                     + t['sd_log2_b'] ** 2 / t['n_b'])
        wide = df.pivot_table(index='protein_group', columns='run',
                              values='pg_intensity')
        a = [c for c in wide.columns if c.startswith('A')]
        b = [c for c in wide.columns if c.startswith('B')]
        direct = np.sqrt(np.log2(wide[a]).var(axis=1, ddof=1) / len(a)
                         + np.log2(wide[b]).var(axis=1, ddof=1) / len(b))
        assert se.to_numpy() == pytest.approx(
            direct.reindex(t['protein_group']).to_numpy(), abs=1e-9)

    def test_linear_cv_is_still_returned_unchanged(self):
        df, si = _mixture({'Human': 1.0}, n_per_species=6, noise=0.2, seed=5)
        t = core.species_ratio_table(df, si, 'A', 'B').set_index('protein_group')
        wide = df.pivot_table(index='protein_group', columns='run',
                              values='pg_intensity')
        a = [c for c in wide.columns if c.startswith('A')]
        expected = wide[a].std(axis=1, ddof=1) / wide[a].mean(axis=1)
        assert t['cv_a'].to_numpy() == pytest.approx(
            expected.reindex(t.index).to_numpy(), abs=1e-12)
