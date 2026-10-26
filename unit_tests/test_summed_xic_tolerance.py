"""The XIC tolerance is a property of the FILE, not of the caller's window."""
import numpy as np
import pandas as pd
import pytest

from zenotof8600_analytics.raw import targeted
from zenotof8600_analytics.raw.targeted import summed_fragment_xic


class _MS:

    def __init__(self, n_scans=60, fwhm_ppm=30.0, mz=500.0):
        self.fwhm_ppm = fwhm_ppm
        rows, peaks = [], []
        start = 0
        for i in range(n_scans):
            grid = mz * (1 + np.linspace(-4, 4, 21) * fwhm_ppm / 1e6)
            amp = np.exp(-0.5 * ((i - n_scans / 2) / 4.0) ** 2) * 1e4
            inten = amp * np.exp(-4 * np.log(2)
                                 * ((grid - mz) / (mz * fwhm_ppm / 1e6)) ** 2)
            peaks.append(pd.DataFrame({'mz': grid, 'intensity': inten}))
            rows.append({'rt': i * 0.01, 'ms_level': 2,
                         'precursor_mz': 700.0,
                         'isolation_lower_mz': 698.5,
                         'isolation_upper_mz': 701.5,
                         'peak_start_idx': start,
                         'peak_stop_idx': start + len(grid)})
            start += len(grid)
        self.spectrum_df = pd.DataFrame(rows)
        self.peak_df = pd.concat(peaks, ignore_index=True)


def _frags(mz=500.0):
    return pd.DataFrame({'target_idx': [0], 'sequence': ['PEPTIDEK'],
                         'charge': [2], 'precursor_mz': [700.0],
                         'product_mz': [mz]})


def test_auto_tolerance_is_resolved_before_the_view(monkeypatch):
    ms = _MS()
    seen = []
    real = targeted.extract_xic

    def spy(data, mz, **kw):
        seen.append(kw.get('mz_tol'))
        return real(data, mz, **kw)

    monkeypatch.setattr(targeted, 'extract_xic', spy)
    summed_fragment_xic(ms, _frags(), match='precursor', match_tol=0.05)
    assert seen, 'extract_xic was not called'
    assert all(isinstance(t, float) for t in seen), seen


def test_tolerance_does_not_depend_on_rt_range(monkeypatch):
    ms = _MS()
    got = {}
    real = targeted.extract_xic

    def spy(data, mz, **kw):
        got.setdefault(len(got), kw.get('mz_tol'))
        return real(data, mz, **kw)

    monkeypatch.setattr(targeted, 'extract_xic', spy)
    summed_fragment_xic(ms, _frags(), match='precursor', match_tol=0.05)
    summed_fragment_xic(ms, _frags(), match='precursor', match_tol=0.05,
                        rt_range=(0.25, 0.35))
    assert len(got) == 2
    assert got[0] == pytest.approx(got[1]), got


def test_light_and_heavy_get_the_same_tolerance(monkeypatch):
    ms = _MS()
    seen = []
    real = targeted.extract_xic

    def spy(data, mz, **kw):
        seen.append(kw.get('mz_tol'))
        return real(data, mz, **kw)

    monkeypatch.setattr(targeted, 'extract_xic', spy)
    for product in (500.0, 502.0):
        summed_fragment_xic(ms, _frags(product), match='precursor',
                            match_tol=0.05, rt_range=(0.2, 0.4))
    assert seen[0] == pytest.approx(seen[1]), seen


def test_explicit_tolerance_is_passed_through(monkeypatch):
    ms = _MS()
    seen = []
    real = targeted.extract_xic

    def spy(data, mz, **kw):
        seen.append(kw.get('mz_tol'))
        return real(data, mz, **kw)

    monkeypatch.setattr(targeted, 'extract_xic', spy)
    summed_fragment_xic(ms, _frags(), mz_tol=17.0, match='precursor',
                        match_tol=0.05)
    assert seen == [17.0]
