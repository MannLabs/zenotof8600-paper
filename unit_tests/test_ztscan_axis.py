"""Tests for the ZT Scan Q1 axis correction."""

import json

import numpy as np
import pytest

from zenotof8600_analytics.ztscan import (
    q1_correction,
)


@pytest.fixture
def table_axis(tmp_path):
    c = {'kind': 'table',
         'mz_knots': [400.0, 600.0, 800.0],
         'mz_corr': [0.10, -0.05, 0.20],
         'slope_rt': 0.0075,
         'n_precursors': 1600, 'n_slices': 8,
         'residual_sd_th': 0.021, 'affine_residual_sd_th': 0.041}
    p = tmp_path / 'q1_axis_A5.json'
    p.write_text(json.dumps(c))
    return str(p), c


@pytest.fixture
def affine_file(tmp_path):
    p = tmp_path / 'q1_axis_affine.json'
    p.write_text(json.dumps({'intercept': 0.2, 'slope_mz': -1e-4,
                             'slope_rt': 0.0075}))
    return str(p)


def test_no_axis_means_no_correction():
    corr = q1_correction(None, np.array([500.0, 900.0]), np.array([10.0, 20.]))
    assert np.allclose(corr, 0.0)
