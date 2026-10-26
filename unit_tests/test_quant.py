"""Characterization: DirectLFQ protein-group quantification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import zenotof8600_analytics as core


@pytest.fixture(scope='module')
def dlfq(loaded):
    df, _ = loaded
    return core.compute_directlfq_pg_intensity(df, num_cores=1)


@pytest.mark.slow
def test_dlfq_series_shape(dlfq, df):
    assert isinstance(dlfq, pd.Series)
    assert dlfq.name == 'pg_intensity'
    assert len(dlfq) == len(df)
    assert int(dlfq.notna().sum()) == 324315


@pytest.mark.slow
def test_dlfq_protein_group_coverage(dlfq, df):
    n_pg = df.loc[dlfq.notna().to_numpy(), 'protein_group'].nunique()
    assert n_pg == 5855


@pytest.mark.slow
def test_dlfq_spot_value(dlfq, df):
    work = df.assign(dlfq=dlfq.to_numpy())
    first_pg = sorted(work['protein_group'].unique())[0]
    assert first_pg == 'A0A024R1R8;Q9Y2S6'
    per_run = work[work['protein_group'] == first_pg].groupby('run')['dlfq'].first().sort_index()
    assert float(per_run.iloc[0]) == pytest.approx(3476110.4715, rel=1e-4)
    assert float(np.log2(per_run.iloc[0])) == pytest.approx(21.729043, abs=1e-4)
