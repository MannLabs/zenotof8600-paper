"""Tests for merge_peaks - the one primitive every spectrum sum goes through."""

import numpy as np
import pytest

from zenotof8600_analytics.raw import (
    merge_peaks,
)

ATOL = 1e-9


def same_mz(a, b, atol=ATOL):
    return np.allclose(a, b, rtol=0.0, atol=atol)


def gaussian(mz, centre, fwhm, height):
    s = fwhm / 2.3548200450309493
    return height * np.exp(-0.5 * ((mz - centre) / s) ** 2)


def centroid(mz, inten):
    return float(np.sum(mz * inten) / np.sum(inten))


def test_empty():
    m, i = merge_peaks([], [])
    assert m.size == 0 and i.size == 0


def test_single_sample():
    m, i = merge_peaks([500.0], [7.0])
    assert m == pytest.approx([500.0]) and i == pytest.approx([7.0])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        merge_peaks([1.0, 2.0], [1.0])


def test_nonfinite_raises():
    with pytest.raises(ValueError):
        merge_peaks([1.0, np.nan], [1.0, 1.0])


def test_output_is_sorted_and_input_order_irrelevant():
    mz = np.array([600.02, 500.0, 550.01, 500.0])
    it = np.array([3.0, 1.0, 2.0, 4.0])
    a = merge_peaks(mz, it)
    p = np.array([2, 0, 3, 1])
    b = merge_peaks(mz[p], it[p])
    assert np.all(np.diff(a[0]) > 0)
    assert same_mz(a[0], b[0]) and np.allclose(a[1], b[1], rtol=0, atol=1e-9)


def test_total_intensity_is_always_conserved():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mz = np.sort(rng.uniform(400, 900, 200))
        it = rng.uniform(0, 1e6, 200)
        _, out = merge_peaks(mz, it)
        assert np.isclose(out.sum(), it.sum(), rtol=1e-12)


def test_identical_grids_sum_and_keep_exact_position():
    grid = np.arange(500.0, 500.1, 0.00476)
    a = gaussian(grid, 500.05, 0.018, 1000.0)
    b = gaussian(grid, 500.05, 0.018, 250.0)
    mz, it = merge_peaks(np.concatenate([grid, grid]), np.concatenate([a, b]))
    assert mz.size == grid.size
    assert same_mz(mz, grid, atol=0.0)
    assert np.allclose(it, a + b)


def test_separate_lattice_points_are_never_merged():
    step = 0.00476
    grid = np.arange(500.0, 500.05, step)
    mz, _ = merge_peaks(grid, np.ones_like(grid))
    assert mz.size == grid.size


def test_interleaved_combs_are_merged_not_sawtoothed():
    step, offset = 0.00476, 0.0001
    g1 = np.arange(500.0, 500.1, step)
    g2 = g1 + offset
    y1 = gaussian(g1, 500.05, 0.018, 1000.0)
    y2 = gaussian(g2, 500.05, 0.018, 6000.0)
    mz, it = merge_peaks(np.concatenate([g1, g2]),
                           np.concatenate([y1, y2]))
    assert mz.size == g1.size, 'combs did not merge'
    d = np.diff(mz)
    assert np.all(d > 0.9 * step)


def test_position_does_not_depend_on_intensity_ratio():
    step, offset = 0.00476, 0.0001
    g1 = np.arange(500.0, 500.1, step)
    g2 = g1 + offset
    axes = []
    for ratio in (0.01, 1.0, 100.0):
        y1 = gaussian(g1, 500.05, 0.018, 1000.0)
        y2 = gaussian(g2, 500.05, 0.018, 1000.0 * ratio)
        mz, _ = merge_peaks(np.concatenate([g1, g2]),
                              np.concatenate([y1, y2]))
        axes.append(mz)
    for other in axes[1:]:
        assert same_mz(axes[0], other, atol=0.0), \
            'merged axis moved with the intensity ratio'


def test_position_is_the_median_of_distinct_positions():
    d = 1.0e-4
    base = 500.0
    mz, it = merge_peaks([base, base + d, base + 4 * d], [5.0, 1.0, 900.0])
    assert mz.size == 1, 'the three samples should form one group'
    assert it[0] == pytest.approx(906.0)
    assert mz[0] == pytest.approx(base + d, abs=1e-9), (
        f'expected the median {base + d!r}, got {mz[0]!r} - mean would be '
        f'{base + 5 * d / 3!r}, first sample {base!r}')


def test_repeated_positions_do_not_outvote_a_distinct_one():
    d = 1.0e-4
    base = 500.0
    mz, _ = merge_peaks([base] * 9 + [base + d], [1.0] * 10)
    assert mz[0] == pytest.approx(base + d / 2, abs=1e-9)


def test_merged_axis_is_the_midpoint_of_two_combs():
    step, offset = 0.00476, 0.0001
    g1 = np.arange(500.0, 500.05, step)
    g2 = g1 + offset
    mz, _ = merge_peaks(np.concatenate([g1, g2]),
                          np.concatenate([np.ones(g1.size),
                                          np.full(g2.size, 50.0)]))
    assert same_mz(mz, g1 + offset / 2, atol=1e-12)


def test_peak_width_is_not_narrowed_by_merging():
    step, offset, fwhm = 0.00476, 0.0012, 0.018
    g1 = np.arange(499.9, 500.2, step)
    g2 = g1 + offset
    y1 = gaussian(g1, 500.05, fwhm, 1000.0)
    y2 = gaussian(g2, 500.05, fwhm, 1000.0)
    mz, it = merge_peaks(np.concatenate([g1, g2]),
                           np.concatenate([y1, y2]))
    half = it.max() / 2
    above = mz[it >= half]
    measured = above.max() - above.min()
    assert abs(measured - fwhm) < step, f'width {measured} vs {fwhm}'


def test_centroid_is_preserved_within_half_a_step():
    step, fwhm, true_mz = 0.00476, 0.018, 500.05
    rng = np.random.default_rng(3)
    for offset in (0.0, 0.0001, 0.0005, 0.001):
        g1 = np.arange(499.9, 500.2, step)
        g2 = g1 + offset
        y1 = gaussian(g1, true_mz, fwhm, 1000.0)
        y2 = gaussian(g2, true_mz, fwhm, 1000.0 * rng.uniform(0.2, 5))
        mz, it = merge_peaks(np.concatenate([g1, g2]),
                               np.concatenate([y1, y2]))
        assert abs(centroid(mz, it) - true_mz) < step / 2


def test_many_scans_collapse_to_one_axis():
    step = 0.00476
    base = np.arange(500.0, 500.1, step)
    rng = np.random.default_rng(7)
    mzs, its = [], []
    for k in range(12):
        g = base + rng.uniform(-0.2, 0.2) * 1e-3
        mzs.append(g)
        its.append(gaussian(g, 500.05, 0.018, rng.uniform(100, 10000)))
    mz, it = merge_peaks(np.concatenate(mzs), np.concatenate(its))
    assert mz.size == base.size
    assert np.isclose(it.sum(), np.concatenate(its).sum(), rtol=1e-12)


def test_sparse_scans_keep_points_the_others_lack():
    step = 0.00476
    g1 = np.arange(500.0, 500.05, step)
    g2 = g1[::2]
    mz, it = merge_peaks(np.concatenate([g1, g2]),
                           np.concatenate([np.ones(g1.size),
                                           np.ones(g2.size)]))
    assert mz.size == g1.size
    assert np.isclose(it.sum(), g1.size + g2.size)


def test_tolerance_boundary():
    mz0 = 500.0
    for factor, should_merge in ((0.5, True), (2.0, False)):
        d = mz0 * 1.0e-6 * factor
        mz, _ = merge_peaks([mz0, mz0 + d], [1.0, 1.0])
        assert (mz.size == 1) is should_merge


def test_zero_intensity_samples_do_not_break_position():
    mz, it = merge_peaks([500.0, 500.0000005], [0.0, 0.0])
    assert mz.size == 1 and np.isfinite(mz[0]) and it[0] == 0.0


def test_weighted_mode_is_the_intensity_weighted_mean():
    mz, it = merge_peaks([500.0, 500.0004], [1.0, 3.0],
                         tol=1.0, tol_unit='ppm', position='weighted')
    assert mz.size == 1
    assert mz[0] == pytest.approx((500.0 * 1 + 500.0004 * 3) / 4, abs=1e-12)
    assert it[0] == pytest.approx(4.0)


def test_weighted_and_median_differ_on_profile_data():
    d = 1.0e-4
    args = ([500.0, 500.0 + d], [1.0, 99.0])
    a, _ = merge_peaks(*args, position='median')
    b, _ = merge_peaks(*args, position='weighted')
    assert abs(a[0] - b[0]) > 1e-6


def test_weighted_group_of_zeros_does_not_return_zero_mz():
    mz, it = merge_peaks([500.0, 500.0000005], [0.0, 0.0],
                         position='weighted')
    assert mz.size == 1 and mz[0] == pytest.approx(500.0, abs=1e-6)
    assert it[0] == 0.0


def test_bad_position_rejected():
    with pytest.raises(ValueError):
        merge_peaks([1.0], [1.0], position='nonsense')


def test_bad_tol_unit_rejected():
    with pytest.raises(ValueError):
        merge_peaks([1.0, 2.0], [1.0, 1.0], tol_unit='furlongs')


def test_da_tolerance():
    mz, _ = merge_peaks([500.0, 500.05, 501.0], [1.0, 1.0, 1.0],
                        tol=0.1, tol_unit='Da')
    assert mz.size == 2


def test_chaining_actually_happens_at_a_too_large_tolerance():
    step = 0.00476
    grid = np.arange(500.0, 500.2, step)
    mz, _ = merge_peaks(grid, np.ones_like(grid), tol=20.0, tol_unit='ppm')
    assert mz.size == 1, 'expected the whole profile to collapse'


def _lattice(start, step, n):
    return start + step * np.arange(n)


def test_scan_id_shared_grid_identical_to_default():
    g = _lattice(500.0, 0.0042, 40)
    rng = np.random.default_rng(1)
    mz = np.concatenate([g, g, g])
    inten = rng.uniform(1, 100, mz.size)
    scan = np.repeat([0, 1, 2], g.size)
    a = merge_peaks(mz, inten)
    b = merge_peaks(mz, inten, scan_id=scan)
    assert same_mz(a[0], b[0]) and np.allclose(a[1], b[1])


def test_scan_id_offset_grids_identical_to_default():
    g = _lattice(500.0, 0.0042, 40)
    mz = np.concatenate([g, g + 0.0001])
    inten = np.ones(mz.size)
    scan = np.repeat([0, 1], g.size)
    a = merge_peaks(mz, inten)
    b = merge_peaks(mz, inten, scan_id=scan)
    assert same_mz(a[0], b[0]) and np.allclose(a[1], b[1])
    assert a[0].size == g.size


def test_scan_id_drifting_grids_never_add_one_scans_samples():
    step, n, n_scans = 0.0042, 30, 12
    drift = 500.0 * 0.8e-6
    mz = np.concatenate([_lattice(500.0 + k * drift, step, n)
                         for k in range(n_scans)])
    inten = np.ones(mz.size)
    scan = np.repeat(np.arange(n_scans), n)
    m0, i0 = merge_peaks(mz, inten, tol=1.0)
    m1, i1 = merge_peaks(mz, inten, tol=1.0, scan_id=scan)
    assert i0.max() > n_scans
    assert i1.max() <= n_scans
    assert np.isclose(i1.sum(), mz.size)
    assert m1.size <= n + 1
    assert np.sum(i1 == n_scans) >= n - 2


def test_scan_id_keeps_nearest_partner():
    base = 500.0
    p = base * 1e-6
    mz = np.array([base, base + 0.95 * p, base + 1.8 * p])
    inten = np.array([1.0, 10.0, 100.0])
    scan = np.array([0, 1, 0])
    m, i = merge_peaks(mz, inten, tol=1.0, scan_id=scan)
    assert np.allclose(np.sort(i), [1.0, 110.0])


def test_scan_id_length_mismatch_raises():
    with pytest.raises(ValueError):
        merge_peaks([1.0, 2.0], [1.0, 1.0], scan_id=[0])
