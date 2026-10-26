"""Characterization: loader output + canonical schema."""

from __future__ import annotations

import pandas as pd
import pytest

import zenotof8600_analytics as core

EXPECTED_DF_COLUMNS = [
    'run', 'protein_group', 'protein_names', 'genes', 'sequence', 'mods',
    'mod_sites', 'precursor_id', 'peptide_id', 'charge', 'mz', 'rt',
    'precursor_intensity', 'peptide_intensity', 'pg_intensity', 'qvalue',
    'score_engine', 'engine',
]

EXPECTED_DTYPES = {
    'run': 'object', 'protein_group': 'object', 'genes': 'object',
    'sequence': 'object', 'mods': 'object', 'mod_sites': 'object',
    'precursor_id': 'object', 'peptide_id': 'object',
    'charge': 'Int64', 'mz': 'float64', 'rt': 'float64',
    'precursor_intensity': 'float64', 'peptide_intensity': 'float64',
    'pg_intensity': 'float64', 'qvalue': 'float64', 'score_engine': 'float64',
    'engine': 'category',
}

EXPECTED_ROWS_500SPD = 41798
EXPECTED_ROWS_200SPD = 66670


def test_df_shape_and_columns(df):
    assert df.shape == (325404, 18)
    assert list(df.columns) == EXPECTED_DF_COLUMNS


def test_per_run_mz_and_rt_are_populated(df):
    obs = df[df['precursor_intensity'].notna()]
    assert obs['mz'].notna().mean() > 0.99
    assert obs['rt'].notna().mean() > 0.99
    assert obs['rt'].between(0, 10).all()

    same_gradient = obs[obs['run'].str.contains('2p3min')]
    spread = (same_gradient.groupby('precursor_id')['rt']
              .agg(lambda s: s.max() - s.min()))
    assert (spread > 0).any(), 'rt is identical in every run — not per-run'
    assert spread.median() * 60 < 10, 'implausible within-gradient RT spread'


def test_df_dtypes(df):
    for col, dtype in EXPECTED_DTYPES.items():
        assert str(df[col].dtype) == dtype, f'{col}: {df[col].dtype} != {dtype}'


def test_required_columns_present(df):
    for col in core.LONG_DF_REQUIRED:
        assert col in df.columns


def test_run_and_engine(df):
    assert df['run'].nunique() == 6
    assert df['engine'].unique().tolist() == ['peaks']


def test_id_counts(df):
    assert df['protein_group'].nunique() == 5906
    assert df['precursor_id'].nunique() == 69645
    assert df['peptide_id'].nunique() == 60023
    assert df['sequence'].nunique() == 58928


def test_qvalue_range(df):
    assert df['qvalue'].min() == pytest.approx(0.0001)
    assert df['qvalue'].max() == pytest.approx(1.0)


def test_rows_per_run(df):
    counts = df.groupby('run').size()
    rows_500 = {n for r, n in counts.items() if '2p3min' in r}
    rows_200 = {n for r, n in counts.items() if '6p4min' in r}
    assert rows_500 == {EXPECTED_ROWS_500SPD}
    assert rows_200 == {EXPECTED_ROWS_200SPD}


def test_intensity_tiers_broadcast(df):
    assert (df.groupby(['run', 'protein_group'])['pg_intensity'].nunique() <= 1).all()
    assert (df.groupby(['run', 'peptide_id'])['peptide_intensity'].nunique() <= 1).all()


def test_sample_info_schema(sample_info):
    assert sample_info.shape == (6, 7)
    assert list(sample_info.columns) == [
        'run', 'file_path', 'condition1', 'condition2', 'replicate', 'batch',
        'engine']
    assert str(sample_info['replicate'].dtype) == 'int64'
    assert sample_info['run'].is_unique
    assert sorted(sample_info['condition2'].unique()) == ['200SPD', '500SPD']
    for col in core.SAMPLE_INFO_REQUIRED:
        assert col in sample_info.columns


def test_validators_pass(df, sample_info):
    core.validate_df(df)
    core.validate_sample_info(sample_info)


@pytest.mark.parametrize('path,expected', [
    ('foo/report.parquet', 'diann'),
    ('foo/lfq.dia.features.csv', 'peaks'),
    ('foo/something.tsv', 'peaks'),
])
def test_detect_engine(path, expected):
    assert core.detect_engine(path) == expected


def test_detect_engine_unknown_raises():
    with pytest.raises(ValueError):
        core.detect_engine('foo/bar.raw')
