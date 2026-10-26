"""Scan selection for targeted acquisitions — see raw/targeted_scans.py."""
import numpy as np
import pandas as pd
import pytest

from zenotof8600_analytics.raw.extraction import extract_xic
from zenotof8600_analytics.raw.targeted_scans import (DEFAULT_MATCH_TH,
                                                     is_targeted,
                                                     target_scans,
                                                     targeted_xic)


class _MS:
    def __init__(self, spectrum_df, peak_df):
        self.spectrum_df = spectrum_df
        self.peak_df = peak_df


LIGHT_Q1 = 1072.9405
HEAVY_Q1 = 1073.9423

FAKE_HALF_TH = 1.5


def _two_target_run(n_cycles=6):
    rows, mzs, ints = [], [], []
    for c in range(n_cycles):
        for q1, frag in ((LIGHT_Q1, 500.0), (HEAVY_Q1, 501.0)):
            start = len(mzs)
            mzs.append(frag)
            ints.append(100.0)
            rows.append({
                'rt': 0.01 * len(rows), 'ms_level': 2, 'nce': 12.0,
                'precursor_mz': q1,
                'isolation_lower_mz': q1 - FAKE_HALF_TH,
                'isolation_upper_mz': q1 + FAKE_HALF_TH,
                'peak_start_idx': start, 'peak_stop_idx': start + 1,
            })
    return _MS(pd.DataFrame(rows),
               pd.DataFrame({'mz': mzs, 'intensity': ints}))


def test_isolation_containment_merges_neighbouring_targets():
    ms = _two_target_run()
    dia = extract_xic(ms, 501.0, mz_tol=20.0, ms_level=2,
                      precursor_mz=HEAVY_Q1, window='union')
    assert len(dia) == len(ms.spectrum_df) == 12
    assert (dia['intensity'] == 0).sum() == 6


def test_central_window_keeps_only_the_targets_own_scans():
    ms = _two_target_run()
    dia = extract_xic(ms, 501.0, mz_tol=20.0, ms_level=2,
                      precursor_mz=HEAVY_Q1)
    assert len(dia) == 6
    assert (dia['intensity'] > 0).all()


def test_target_scans_keeps_only_its_own():
    ms = _two_target_run()
    view = target_scans(ms, HEAVY_Q1)
    assert len(view.spectrum_df) == 6
    assert np.allclose(view.spectrum_df['precursor_mz'], HEAVY_Q1)


def test_targeted_xic_has_no_interleaved_zeros():
    ms = _two_target_run()
    x = targeted_xic(ms, HEAVY_Q1, 501.0, mz_tol=20.0)
    assert len(x) == 6
    assert (x['intensity'] > 0).all()
    assert x['intensity'].sum() == pytest.approx(600.0)


def test_light_and_heavy_are_separated():
    ms = _two_target_run()
    light = targeted_xic(ms, LIGHT_Q1, 500.0, mz_tol=20.0)
    heavy = targeted_xic(ms, HEAVY_Q1, 500.0, mz_tol=20.0)
    assert light['intensity'].sum() == pytest.approx(600.0)
    assert heavy['intensity'].sum() == 0.0


def test_all_transitions_share_one_rt_grid():
    ms = _two_target_run()
    a = targeted_xic(ms, HEAVY_Q1, 501.0, mz_tol=20.0)
    b = targeted_xic(ms, HEAVY_Q1, 900.0, mz_tol=20.0)
    assert np.array_equal(a['rt'].to_numpy(), b['rt'].to_numpy())


def test_match_tolerance_is_below_sil_spacing():
    assert DEFAULT_MATCH_TH < (HEAVY_Q1 - LIGHT_Q1) / 2


def test_nce_selects_one_activation():
    ms = _two_target_run()
    ms.spectrum_df.loc[1::4, 'nce'] = 50.0
    view = target_scans(ms, HEAVY_Q1, nce=12.0)
    assert set(view.spectrum_df['nce']) == {12.0}
    assert len(view.spectrum_df) == 3


def test_rt_range_restricts():
    ms = _two_target_run()
    view = target_scans(ms, HEAVY_Q1, rt_range=(0.0, 0.05))
    assert view.spectrum_df['rt'].max() <= 0.05


def test_missing_target_raises_and_names_what_is_there():
    ms = _two_target_run()
    with pytest.raises(ValueError, match='no scan within'):
        target_scans(ms, 1234.5678)


def test_no_precursor_column_raises():
    ms = _two_target_run()
    ms.spectrum_df = ms.spectrum_df.drop(columns=['precursor_mz'])
    with pytest.raises(ValueError, match='not a targeted acquisition'):
        target_scans(ms, HEAVY_Q1)


def test_is_targeted_distinguishes_from_dia():
    assert is_targeted(_two_target_run())
    dia = _two_target_run()
    dia.spectrum_df['precursor_mz'] = np.linspace(400, 1200,
                                                  len(dia.spectrum_df))
    assert not is_targeted(dia, max_windows=5)


def _windows_run(lowers, uppers, n_cycles=3):
    lo = np.tile(np.asarray(lowers, float), n_cycles)
    hi = np.tile(np.asarray(uppers, float), n_cycles)
    spec = pd.DataFrame({'ms_level': 2, 'precursor_mz': (lo + hi) / 2,
                         'isolation_lower_mz': lo, 'isolation_upper_mz': hi,
                         'rt': np.arange(len(lo)) * 0.01})
    return _MS(spec, pd.DataFrame({'mz': [], 'intensity': []}))


def test_variable_window_swath_is_not_targeted():
    edges = np.concatenate([[360.0], 360 + np.cumsum(np.r_[np.full(60, 4.0),
                                                           np.full(40, 9.0)])])
    assert not is_targeted(_windows_run(edges[:-1], edges[1:]))


def test_overlapping_dia_windows_are_not_targeted():
    lower = 400 + 12.0 * np.arange(30)
    assert not is_targeted(_windows_run(lower, lower + 13.0))


def test_isolated_targets_are_targeted():
    q1 = np.array([500.0, 501.0, 640.2, 712.9, 830.4, 905.6])
    assert is_targeted(_windows_run(q1 - FAKE_HALF_TH, q1 + FAKE_HALF_TH))
