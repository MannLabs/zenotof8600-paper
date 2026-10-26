"""Characterization: filtering + reshape helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import zenotof8600_analytics as core


def test_filter_outlier_runs_keep_all(df, sample_info):
    dff, sif, summary = core.filter_outlier_runs(
        df, sample_info, cohort_col='condition2', min_runs_for_fit=4, plot=False)
    assert list(summary.columns) == [
        'cohort', 'n_total', 'n_kept', 'n_dropped', 'threshold', 'dropped_runs']
    assert len(summary) == 2
    assert int(summary['n_dropped'].sum()) == 0
    assert dff['run'].nunique() == 6
    assert sif['run'].nunique() == 6


def test_filter_outlier_runs_counts_detected_precursors_only():
    rng = np.random.default_rng(0)
    n_prec = 2000
    depth = {f'run{i:02d}': int(rng.normal(1800, 40)) for i in range(10)}
    depth['run08'], depth['run09'] = 150, 220
    rows = []
    for run, n in depth.items():
        inten = np.full(n_prec, np.nan)
        inten[:n] = rng.lognormal(10, 1, n)
        rows.append(pd.DataFrame({
            'run': run, 'precursor_id': [f'P{k}' for k in range(n_prec)],
            'precursor_intensity': inten}))
    long = pd.concat(rows, ignore_index=True)
    si = pd.DataFrame({'run': list(depth)})
    _, kept, summary = core.filter_outlier_runs(long, si, plot=False)
    assert set(summary['dropped_runs'].iloc[0]) == {'run08', 'run09'}
    assert kept['run'].nunique() == 8


