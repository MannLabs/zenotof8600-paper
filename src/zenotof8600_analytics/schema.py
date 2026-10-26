"""Canonical column schema and the AlphaBase modification vocabulary."""

from __future__ import annotations

import pandas as pd


LONG_DF_REQUIRED = (
    'run', 'protein_group', 'genes',
    'sequence', 'mods', 'mod_sites',
    'precursor_id', 'peptide_id',
    'precursor_intensity', 'peptide_intensity', 'pg_intensity',
    'qvalue', 'score_engine', 'engine',
)

LONG_DF_OPTIONAL = (
    'protein_names', 'charge', 'mz', 'rt',
)


SAMPLE_INFO_REQUIRED = (
    'run', 'file_path', 'condition1', 'condition2', 'replicate', 'engine',
)

SAMPLE_INFO_OPTIONAL = ('batch', 'group')


ALPHABASE_MODS = (
    'Carbamidomethyl@C',
    'Oxidation@M',
    'Acetyl@Protein_N-term',
    'Acetyl@K',
    'Phospho@S',
    'Phospho@T',
    'Phospho@Y',
    'Deamidated@N',
    'Deamidated@Q',
)


def validate_df(df: pd.DataFrame) -> None:
    """Raise if `df` is not a valid canonical long DataFrame."""
    missing = [c for c in LONG_DF_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f'df missing required columns: {missing}')
    for col in ('mods', 'mod_sites'):
        if df[col].isna().any():
            raise ValueError(f"{col!r} contains NaN; use empty string for unmodified peptides")


def validate_sample_info(sample_info: pd.DataFrame) -> None:
    """Raise if `sample_info` is not a valid sample-metadata DataFrame."""
    missing = [c for c in SAMPLE_INFO_REQUIRED if c not in sample_info.columns]
    if missing:
        raise ValueError(f'sample_info missing required columns: {missing}')
    if sample_info.duplicated(['run', 'engine']).any():
        dup = sample_info[sample_info.duplicated(['run', 'engine'])][['run', 'engine']]
        raise ValueError(f'sample_info has duplicate (run, engine) rows:\n{dup}')
