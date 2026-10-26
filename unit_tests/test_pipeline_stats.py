"""Characterization: process_experiment (per-run stats) + CV table."""

from __future__ import annotations

import pytest

import zenotof8600_analytics as core
from zenotof8600_analytics import stats

EXPECTED_AGG_COLUMNS = [
    'run', 'engine', 'precursor', 'peptide', 'protein_group', 'total_intensity',
    'log2_total_intensity', 'MC0', 'MC1', 'MC2', 'avg_MC', 'mc_rate', 'PG20',
    'Pr20', 'file_path', 'condition1', 'condition2', 'replicate', 'batch',
    'total_peptides', 'total_protein_groups', 'total_precursors',
]

EXPECTED_PER_RUN = {
    '_A4':  dict(peptide=33162, precursor=37792, protein_group=4316),
    '_A5':  dict(peptide=33245, precursor=37819, protein_group=4315),
    '_A6':  dict(peptide=32983, precursor=37327, protein_group=4303),
    '_A10': dict(peptide=54327, precursor=62607, protein_group=5692),
    '_A11': dict(peptide=54406, precursor=62641, protein_group=5688),
    '_A12': dict(peptide=54204, precursor=62396, protein_group=5696),
}

EXPECTED_PG20 = {'500SPD': 3716, '200SPD': 5454}
EXPECTED_PR20 = {'500SPD': 22694, '200SPD': 48898}


def _suffix(run):
    return '_' + run.split('_')[-1]


def test_agg_shape_and_columns(agg):
    assert agg.shape == (6, 22)
    assert list(agg.columns) == EXPECTED_AGG_COLUMNS


def test_agg_per_run_counts(agg):
    for _, row in agg.iterrows():
        exp = EXPECTED_PER_RUN[_suffix(row['run'])]
        assert row['peptide'] == exp['peptide']
        assert row['precursor'] == exp['precursor']
        assert row['protein_group'] == exp['protein_group']


def test_agg_cv20_counts_constant_within_condition(agg):
    for cond, sub in agg.groupby('condition2'):
        assert sub['PG20'].nunique() == 1
        assert sub['Pr20'].nunique() == 1
        assert sub['PG20'].iloc[0] == EXPECTED_PG20[cond]
        assert sub['Pr20'].iloc[0] == EXPECTED_PR20[cond]


def test_cv_table_protein(df, sample_info):
    cv = stats._compute_cv_table(df, sample_info, 'protein', 'condition2', 3)
    assert list(cv.columns) == ['cv', 'group']
    assert cv.shape == (9764, 2)
    med = cv.groupby('group')['cv'].median()
    assert med['200SPD'] == pytest.approx(0.045898, abs=1e-5)
    assert med['500SPD'] == pytest.approx(0.065353, abs=1e-5)


def test_cv_table_precursor(df, sample_info):
    cv = stats._compute_cv_table(df, sample_info, 'precursor', 'condition2', 3)
    med = cv.groupby('group')['cv'].median()
    assert med['200SPD'] == pytest.approx(0.101998, abs=1e-5)
    assert med['500SPD'] == pytest.approx(0.128446, abs=1e-5)
