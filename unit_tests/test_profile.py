"""Tests for centroid_profile - profile samples to peaks."""

import numpy as np
import pytest

from zenotof8600_analytics.raw import (
    centroid_profile, lattice_step, LATTICE_AT_500, GAP_FACTOR)


def gaussian(mz, centre, fwhm, height):
    s = fwhm / 2.3548200450309493
    return height * np.exp(-0.5 * ((mz - centre) / s) ** 2)


def sampled_peak(centre, fwhm=0.014, height=1000.0, n_sigma=3.0):
    step = float(lattice_step(centre))
    half = n_sigma * fwhm / 2.3548200450309493
    mz = np.arange(centre - half, centre + half, step)
    return mz, gaussian(mz, centre, fwhm, height)


def test_lattice_step_scales_as_sqrt_mz():
    a, b = lattice_step(500.0), lattice_step(2000.0)
    assert a == pytest.approx(LATTICE_AT_500, rel=1e-12)
    assert b / a == pytest.approx(2.0, rel=1e-12)


def test_lattice_step_accepts_arrays():
    out = lattice_step([500.0, 500.0])
    assert out.shape == (2,) and np.all(out > 0)


def test_one_peak_collapses_to_one_centroid():
    mz, it = sampled_peak(500.05)
    assert mz.size > 5, 'test needs a genuinely sampled peak'
    c, tot = centroid_profile(mz, it)[:2]
    assert c.size == 1, f'{mz.size} samples became {c.size} peaks'
    assert tot[0] == pytest.approx(it.sum())


def test_two_resolved_peaks_stay_two():
    m1, i1 = sampled_peak(500.05)
    m2, i2 = sampled_peak(500.20)
    c, _ = centroid_profile(np.concatenate([m1, m2]),
                            np.concatenate([i1, i2]))[:2]
    assert c.size == 2
    assert c[0] == pytest.approx(500.05, abs=0.002)
    assert c[1] == pytest.approx(500.20, abs=0.002)


def test_gap_below_the_threshold_does_not_split_a_peak():
    step = float(lattice_step(500.0))
    mz = np.array([500.0, 500.0 + step, 500.0 + step * 2.5])
    c, _ = centroid_profile(mz, np.ones(3))[:2]
    assert c.size == 1


def test_a_single_missing_sample_sits_close_to_the_split_threshold():
    lo = 500.0
    step = float(lattice_step(lo))
    thr = GAP_FACTOR * float(lattice_step(lo + step))

    one_missing = centroid_profile(
        np.array([lo, lo + step, lo + step + 2 * step]), np.ones(3))[0]
    assert one_missing.size == 1, 'one dropped sample should not split a peak'

    clear = centroid_profile(
        np.array([lo, lo + step, lo + step + thr * 1.01]), np.ones(3))[0]
    assert clear.size == 2, 'a gap past the threshold must split'

    assert 2 * step < thr, 'the margin this relies on has gone'
    assert (thr - 2 * step) / thr < 1e-3, 'margin is thinner than 0.1%'


def test_gap_above_the_threshold_splits():
    step = float(lattice_step(500.0))
    mz = np.array([500.0, 500.0 + step,
                   500.0 + step * (GAP_FACTOR + 1.5),
                   500.0 + step * (GAP_FACTOR + 2.5)])
    c, _ = centroid_profile(mz, np.ones(4))[:2]
    assert c.size == 2


def test_position_comes_from_the_half_maximum_region():
    mz, it = sampled_peak(500.05, height=1000.0, n_sigma=6.0)
    clean = centroid_profile(mz, it)[0][0]
    tailed = it.copy()
    below_half = it < 0.5 * it.max()
    one_side = below_half & (mz < 500.05)
    tailed[one_side] += 200.0
    dirty = centroid_profile(mz, tailed)[0][0]
    assert abs(dirty - clean) < 0.001, (
        f'a sub-half-max tail moved the centroid by {abs(dirty-clean)*1e3:.2f} '
        f'mTh; the position should ignore it')

    naive = float(np.average(mz, weights=tailed))
    assert abs(naive - clean) > 0.002, (
        'the tail is too small to distinguish the two rules - this test '
        'would pass for a full-profile weighted mean too')


def test_intensity_integrates_the_whole_peak_not_just_the_core():
    mz, it = sampled_peak(500.05)
    _, tot = centroid_profile(mz, it)[:2]
    assert tot[0] == pytest.approx(it.sum(), rel=1e-12)


def test_centroid_is_accurate_on_a_symmetric_peak():
    for centre in (400.0, 700.0, 950.0):
        mz, it = sampled_peak(centre)
        c = centroid_profile(mz, it)[0][0]
        assert c == pytest.approx(centre, abs=float(lattice_step(centre)) / 2)


def test_empty_input():
    c, tot = centroid_profile(np.array([]), np.array([]))[:2]
    assert c.size == 0 and tot.size == 0


def test_zero_intensity_group_is_dropped_not_nan():
    mz, it = sampled_peak(500.05)
    c, tot = centroid_profile(mz, np.zeros_like(it))[:2]
    assert np.all(np.isfinite(c)) and c.size == tot.size


def test_extra_arrays_are_summed_over_the_same_groups():
    mz, it = sampled_peak(500.05)
    prof = np.tile(np.arange(4.0), (mz.size, 1))
    out = centroid_profile(mz, it, extra=(prof,))
    ex = out[-1][0]
    assert ex.shape == (1, 4)
    assert ex[0] == pytest.approx(prof.sum(axis=0))
