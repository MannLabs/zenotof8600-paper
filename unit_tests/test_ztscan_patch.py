"""Tests for building a ZT Scan patch out of an already-read Q1 window."""

import numpy as np
import pytest

from zenotof8600_analytics.ztscan.patch import patch_from_window


def window(species, n_q1=7, n_cycles=5, q1_lo=500.0, d_q1=0.5,
           rt0=10.0, d_rt=0.02):
    mzs, ins, counts, s_rt, s_q1 = [], [], [], [], []
    for c in range(n_cycles):
        for q in range(n_q1):
            here = [(m, h) for m, qi, ci, h in species
                    if qi == q and ci == c]
            mzs.extend(m for m, _ in here)
            ins.extend(h for _, h in here)
            counts.append(len(here))
            s_rt.append(rt0 + c * d_rt)
            s_q1.append(q1_lo + q * d_q1)
    return dict(mz=np.array(mzs, float), inten=np.array(ins, float),
                off=np.concatenate(([0], np.cumsum(counts))).astype(np.int64),
                s_rt=np.array(s_rt, float), s_q1=np.array(s_q1, float),
                rt1=rt0 + np.arange(n_cycles) * d_rt)


def test_one_peak_lands_in_one_cell():
    w = window([(600.0, 3, 2, 1000.0)])
    p = patch_from_window(w, 10.04, n_cycles=5)
    assert len(p['mz']) == 1
    assert p['mz'][0] == pytest.approx(600.0, abs=0.01)
    q, r = p['q1_prof'][0], p['rt_prof'][0]
    assert np.argmax(q) == 3 and np.argmax(r) == 2
    assert q.sum() == pytest.approx(1000.0) == r.sum()


def test_a_shared_mz_keeps_both_species_apart_on_the_q1_axis():
    p = patch_from_window(
        window([(600.0, 1, 1, 400.0), (600.0, 5, 3, 900.0)]),
        10.04, n_cycles=5)
    assert len(p['mz']) == 1, 'one m/z is one peak identity'
    q = p['q1_prof'][0]
    assert q[1] == pytest.approx(400.0) and q[5] == pytest.approx(900.0)


def test_max_peaks_keeps_the_most_intense():
    sp = [(600.0 + i, 3, 2, float(10 * (i + 1))) for i in range(6)]
    p = patch_from_window(window(sp), 10.04, n_cycles=5, max_peaks=2)
    assert len(p['mz']) == 2
    assert set(np.round(p['mz'])) == {605.0, 604.0}


def test_an_empty_window_returns_none_rather_than_an_empty_patch():
    w = window([])
    assert patch_from_window(w, 10.04, n_cycles=5) is None


def test_the_patch_is_centred_on_the_requested_retention_time():
    sp = [(600.0, 3, c, 100.0) for c in range(5)]
    w = window(sp, n_cycles=9)
    p = patch_from_window(w, w['rt1'][4], n_cycles=3)
    assert len(p['rt_axis']) == 3
    assert p['rt_axis'][1] == pytest.approx(w['rt1'][4])


def gauss(x, c, h, w=0.004):
    return h * np.exp(-0.5 * ((x - c) / w) ** 2)


def test_two_overlapping_peaks_merge_without_valley_splitting():
    from zenotof8600_analytics.raw import centroid_profile
    x = np.arange(500.0, 500.7, 0.0012)
    y = gauss(x, 500.25, 1000.0) + gauss(x, 500.45, 800.0)
    mzc, val, _, _ = centroid_profile(x, y, valley_frac=None)
    assert len(mzc) == 1, 'the two peaks are not separated'
    assert 500.25 < mzc[0] < 500.45, 'and the centroid lands between them'


def test_valley_splitting_separates_them():
    from zenotof8600_analytics.raw import centroid_profile
    x = np.arange(500.0, 500.7, 0.0012)
    y = gauss(x, 500.25, 1000.0) + gauss(x, 500.45, 800.0)
    mzc, val, _, _ = centroid_profile(x, y)
    assert len(mzc) == 2
    assert mzc[0] == pytest.approx(500.25, abs=0.002)
    assert mzc[1] == pytest.approx(500.45, abs=0.002)
    assert val.sum() == pytest.approx(y.sum(), rel=1e-9)


def test_a_shoulder_below_the_valley_test_is_not_split():
    from zenotof8600_analytics.raw import centroid_profile
    x = np.arange(500.0, 500.7, 0.0012)
    y = gauss(x, 500.30, 1000.0) + gauss(x, 500.3055, 950.0)
    assert len(centroid_profile(x, y)[0]) == 1


def test_position_threshold_tightens_against_a_neighbour():
    from zenotof8600_analytics.raw import centroid_profile
    x = np.arange(500.0, 500.7, 0.0012)
    y = gauss(x, 500.30, 1000.0) + gauss(x, 500.3065, 600.0)
    half = centroid_profile(x, y, position_thresh=0.5)[0][0]
    top30 = centroid_profile(x, y)[0][0]
    assert abs(top30 - 500.30) < abs(half - 500.30)
