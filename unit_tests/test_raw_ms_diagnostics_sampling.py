"""raw_ms_diagnostics.sample_precursors must run with the script's own config."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import raw_ms_diagnostics as rmd


def _table(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'mz_measured': rng.uniform(400, 1200, n),
        'rt': rng.uniform(1, 30, n),
        'intensity': 10 ** rng.uniform(3, 8, n),
        'sequence': [f'PEPTIDE{i}K' for i in range(n)],
    })


def test_sample_precursors_runs_with_module_config():
    out = rmd.sample_precursors(_table(), n=200)
    assert 0 < len(out) <= 200
    cut = _table()['intensity'].quantile(1 - rmd.TOP_INTENSITY_FRAC)
    assert (out['intensity'] >= cut * 0.999).all()


def test_bin_constants_defined():
    assert isinstance(rmd.ABUNDANCE_BINS, int) and rmd.ABUNDANCE_BINS > 0
    assert isinstance(rmd.MZ_BINS, int) and rmd.MZ_BINS > 0


def _targets(n=20000, seed=2):
    rng = np.random.default_rng(seed)
    mz = rng.uniform(360, 1200, n)
    logi = rng.normal(5 + (mz - 780) / 600, 0.8)
    return pd.DataFrame({'mz_theoretical': mz, 'intensity': 10 ** logi,
                         'sequence': [f'P{i}K' for i in range(n)]})


def test_calibration_subset_covers_mz_and_abundance():
    t = _targets()
    sub = rmd.calibration_subset(t, n=1000, top_fraction=1.0)
    assert 900 <= len(sub) <= 1000
    deciles = np.quantile(t['mz_theoretical'], np.linspace(0, 1, 11))
    hit = np.histogram(sub['mz_theoretical'], bins=deciles)[0]
    assert (hit > 0).all()
    q = np.quantile(np.log10(t['intensity']), np.linspace(0, 1, 6))
    hit = np.histogram(np.log10(sub['intensity']), bins=q)[0]
    assert hit.min() >= 0.5 * hit.mean()


def test_calibration_subset_respects_top_fraction():
    t = _targets()
    sub = rmd.calibration_subset(t, n=500, top_fraction=0.5)
    cut = t['intensity'].quantile(0.5)
    assert (sub['intensity'] >= cut * 0.999).all()
    assert len(sub) == 500


def test_ms1_calibration_fits_monoisotopic_peaks_only():
    rng = np.random.default_rng(3)
    rows = []
    for k in range(3000):
        mz = rng.uniform(400, 1000)
        rows.append({'level': 'MS1 precursor', 'label': 'precursor',
                     'mz_theoretical': mz, 'ppm': 1.0 + rng.normal(0, 0.3)})
        for iso in ('M+1', 'M+2', 'M+3'):
            rows.append({'level': 'MS1 precursor', 'label': iso,
                         'mz_theoretical': mz, 'ppm': -2.0 + rng.normal(0, 0.3)})
    cals = rmd._fit_calibrations(rows, verbose=False)
    cal = cals['MS1 precursor']
    assert abs(float(np.median([cal(m) for m in (500.0, 700.0, 900.0)])) - 1.0) < 0.2
