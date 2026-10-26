"""PeakIndex and window_indices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zenotof8600_analytics.raw.extraction import (
    window_indices,
)


def test_window_indices_scalar_and_array_agree():
    mz = np.array([100.0, 100.1, 100.2, 200.0, 300.0])
    a, b = window_indices(mz, 100.05, 100.25)
    assert (a, b) == (1, 3)
    aa, bb = window_indices(mz, np.array([100.05, 199.9]),
                            np.array([100.25, 200.1]))
    assert list(aa) == [1, 3] and list(bb) == [3, 4]


def test_window_indices_does_not_promote_the_haystack():
    mz32 = np.linspace(100.0, 1000.0, 1000).astype(np.float32)
    lo = np.array([499.5], dtype=np.float64)
    hi = np.array([500.5], dtype=np.float64)
    a, b = window_indices(mz32, lo, hi)
    assert b[0] > a[0]
    assert np.all(mz32[a[0]:b[0]] >= np.float32(499.5))
    assert np.all(mz32[a[0]:b[0]] <= np.float32(500.5))
    assert mz32.dtype == np.float32


def test_window_indices_matches_naive_result_on_float32():
    rng = np.random.default_rng(0)
    mz = np.sort(rng.uniform(200, 900, 5000).astype(np.float32))
    targets = rng.uniform(250, 850, 50)
    half = targets * 20e-6
    a, b = window_indices(mz, targets - half, targets + half)
    for i, t in enumerate(targets):
        naive = np.nonzero((mz >= np.float32(t - half[i]))
                           & (mz <= np.float32(t + half[i])))[0]
        assert (b[i] - a[i]) == len(naive)


class _FakeMS:

    def __init__(self, n_cycles=6, n_exp=4, peaks_per_scan=5):
        rows, mz, inten = [], [], []
        k = 0
        for c in range(n_cycles):
            for e in range(n_exp):
                start = k
                for j in range(peaks_per_scan):
                    mz.append(500.0 + e / 100.0 + j)
                    inten.append(float(10 * (e + 1) + c))
                    k += 1
                rows.append(dict(spec_idx=len(rows), peak_start_idx=start,
                                 peak_stop_idx=k, rt=c * 1.0 + e * 0.01,
                                 ms_level=1 if e == 0 else 2))
        self.spectrum_df = pd.DataFrame(rows)
        self.peak_df = pd.DataFrame({'mz': np.array(mz, dtype=np.float32),
                                     'intensity': np.array(inten, dtype=np.float32)})


