"""Raw manual-mode extraction: XIC building, peak picking, peptide m/z."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zenotof8600_analytics.raw import (
    extract_xic, sum_spectra, pick_peak, peak_from_fragments, match_peaks,
)


class _FakeMS:
    def __init__(self, spectrum_df, peak_df):
        self.spectrum_df = spectrum_df
        self.peak_df = peak_df


TARGET = 500.25
APEX_SHAPE = [0.0, 100.0, 500.0, 300.0, 50.0, 0.0]
RTS = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5]


@pytest.fixture
def ms():
    spec_rows, peak_mz, peak_int = [], [], []
    start = 0
    for rt, ta in zip(RTS, APEX_SHAPE):
        mzs = [400.10, TARGET, 700.90]
        ints = [7.0, ta, 3.0]
        peak_mz += mzs
        peak_int += ints
        spec_rows.append(dict(rt=rt, ms_level=1, peak_start_idx=start,
                              peak_stop_idx=start + len(mzs)))
        start += len(mzs)
    peak_mz += [TARGET]
    peak_int += [9999.0]
    spec_rows.append(dict(rt=10.25, ms_level=2, peak_start_idx=start,
                          peak_stop_idx=start + 1))

    spectrum_df = pd.DataFrame(spec_rows)
    peak_df = pd.DataFrame({'mz': peak_mz, 'intensity': peak_int})
    return _FakeMS(spectrum_df, peak_df)


def test_extract_xic_basic(ms):
    xic = extract_xic(ms, TARGET, mz_tol=10, mz_tol_unit='ppm')
    assert list(xic.columns) == ['rt', 'intensity']
    assert len(xic) == 6
    assert xic['rt'].tolist() == RTS
    assert xic['intensity'].tolist() == APEX_SHAPE


def test_extract_xic_ignores_other_ms_level(ms):
    xic = extract_xic(ms, TARGET, ms_level=1)
    assert xic['intensity'].max() == 500.0


def test_extract_xic_rt_range(ms):
    xic = extract_xic(ms, TARGET, rt_range=(10.15, 10.35))
    assert xic['rt'].tolist() == [10.2, 10.3]
    assert xic['intensity'].tolist() == [500.0, 300.0]


def test_extract_xic_tolerance_excludes_off_target(ms):
    xic = extract_xic(ms, 555.55, mz_tol=5, mz_tol_unit='ppm')
    assert xic['intensity'].sum() == 0.0


def test_extract_xic_multi_mz_sums(ms):
    xic = extract_xic(ms, [TARGET, 400.10], mz_tol=50, mz_tol_unit='ppm')
    assert xic['intensity'].tolist() == [a + 7.0 for a in APEX_SHAPE]


def test_pick_peak_apex_and_area(ms):
    xic = extract_xic(ms, TARGET)
    pk = pick_peak(xic['rt'].to_numpy(), xic['intensity'].to_numpy())
    assert pk['apex_rt'] == pytest.approx(10.2)
    assert pk['apex_intensity'] == pytest.approx(500.0)
    assert pk['left_rt'] < 10.2 < pk['right_rt']
    assert pk['fwhm'] > 0
    assert pk['area'] > 0


def test_pick_peak_empty_is_nan():
    pk = pick_peak(np.array([1.0, 2.0, 3.0]), np.zeros(3))
    assert np.isnan(pk['apex_rt'])
    assert pk['area'] == 0.0
    assert pk['n_points'] == 0


@pytest.fixture
def ms_ms2():
    rows, pmz, pint = [], [], []
    start = 0

    def add(rt, ilo, ihi, mzs, ints):
        nonlocal start
        rows.append(dict(rt=rt, ms_level=2, isolation_lower_mz=ilo,
                         isolation_upper_mz=ihi, peak_start_idx=start,
                         peak_stop_idx=start + len(mzs)))
        pmz.extend(mzs); pint.extend(ints)
        start += len(mzs)

    add(10.20, 500, 510, [200.100, 300.200], [10.0, 4.0])
    add(10.30, 500, 510, [200.101, 300.199], [6.0, 5.0])
    add(10.25, 600, 610, [200.100], [999.0])
    add(11.00, 500, 510, [200.100], [999.0])

    return _FakeMS(pd.DataFrame(rows),
                   pd.DataFrame({'mz': pmz, 'intensity': pint}))


def test_sum_spectra_dia_window_and_rt(ms_ms2):
    spec = sum_spectra(ms_ms2, rt_range=(10.15, 10.35), ms_level=2,
                       precursor_mz=505.0, bin_tol=50, bin_tol_unit='ppm')
    assert list(spec.columns) == ['mz', 'intensity']
    assert len(spec) == 2
    d = dict(zip(spec['mz'].round(1), spec['intensity']))
    assert d[200.1] == pytest.approx(16.0)
    assert d[300.2] == pytest.approx(9.0)
    frag = spec.loc[(spec['mz'] - 200.1).abs() < 0.01, 'mz'].iloc[0]
    assert frag == pytest.approx(200.1005, abs=1e-9)

    wt = sum_spectra(ms_ms2, rt_range=(10.15, 10.35), ms_level=2,
                     precursor_mz=505.0, bin_tol=50, bin_tol_unit='ppm',
                     position='weighted')
    frag_w = wt.loc[(wt['mz'] - 200.1).abs() < 0.01, 'mz'].iloc[0]
    assert frag_w == pytest.approx((200.100 * 10 + 200.101 * 6) / 16, abs=1e-9)
    assert frag != pytest.approx(frag_w, abs=1e-9)


def test_sum_spectra_excludes_wrong_window_and_rt(ms_ms2):
    spec = sum_spectra(ms_ms2, rt_range=(10.15, 10.35), precursor_mz=505.0)
    assert spec['intensity'].max() < 100


def test_match_peaks_max_and_miss():
    smz = np.array([200.100, 200.101, 300.200])
    sint = np.array([10.0, 6.0, 4.0])
    inten, mz = match_peaks(smz, sint, [200.1005, 250.0, 300.2],
                            tol=20, tol_unit='ppm')
    assert inten[0] == pytest.approx(10.0)
    assert inten[1] == 0.0 and np.isnan(mz[1])
    assert inten[2] == pytest.approx(4.0)


def test_match_peaks_sum():
    smz = np.array([200.100, 200.101])
    sint = np.array([10.0, 6.0])
    inten, mz = match_peaks(smz, sint, 200.1005, tol=20, tol_unit='ppm', reduce='sum')
    assert inten[0] == pytest.approx(16.0)


@pytest.fixture
def ms_dia():
    shape = [0.0, 50.0, 200.0, 120.0, 20.0, 0.0]
    rts = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5]
    rows, pmz, pint = [], [], []
    start = 0
    for rt, a in zip(rts, shape):
        rows.append(dict(rt=rt, ms_level=2, isolation_lower_mz=500, isolation_upper_mz=510,
                         peak_start_idx=start, peak_stop_idx=start + 2))
        pmz += [600.300, 700.400]; pint += [a, a * 0.5]
        start += 2
    rows.append(dict(rt=10.2, ms_level=2, isolation_lower_mz=600, isolation_upper_mz=610,
                     peak_start_idx=start, peak_stop_idx=start + 1))
    pmz += [600.300]; pint += [9999.0]
    return _FakeMS(pd.DataFrame(rows), pd.DataFrame({'mz': pmz, 'intensity': pint}))


def test_extract_xic_ms2_isolation_filter(ms_dia):
    xic = extract_xic(ms_dia, 600.300, ms_level=2, precursor_mz=505.0, mz_tol=50)
    assert xic['intensity'].tolist() == [0.0, 50.0, 200.0, 120.0, 20.0, 0.0]


def test_peak_from_fragments(ms_dia):
    peak, trace, used = peak_from_fragments(
        ms_dia, precursor_mz=505.0, fragment_mz=[600.300, 700.400],
        rt_range=(9.9, 10.6), mz_tol=50)
    assert peak['apex_rt'] == pytest.approx(10.2)
    assert peak['fwhm'] > 0
    assert len(used) == 2
    assert trace['intensity'].max() == pytest.approx(300.0)


def test_peak_from_fragments_no_signal(ms_dia):
    peak, trace, used = peak_from_fragments(
        ms_dia, precursor_mz=505.0, fragment_mz=[999.999], rt_range=(9.9, 10.6))
    assert np.isnan(peak['apex_rt'])
    assert len(used) == 0


def test_load_ms_data_bad_extension():
    pytest.importorskip('alpharaw')
    with pytest.raises(ValueError):
        __import__('zenotof8600_analytics.raw', fromlist=['load_ms_data'])
        from zenotof8600_analytics.raw import load_ms_data
        load_ms_data('nonexistent.txt', cache=False)


def _gaussian_trace(half_window_min, fwhm_min=0.05, baseline=100.0,
                    apex=10_000.0, dt_min=0.005):
    n = int(round(2 * half_window_min / dt_min)) + 1
    rt = np.linspace(-half_window_min, half_window_min, n)
    sigma = fwhm_min / 2.3548200450309493
    return rt, baseline + apex * np.exp(-0.5 * (rt / sigma) ** 2)


class TestOffpeakBaseline:

    def test_width_is_stable_across_windows(self):
        widths = {}
        for base in ('median', 'p5', 'offpeak'):
            got = []
            for half in (0.08, 0.12, 0.20, 0.40):
                rt, y = _gaussian_trace(half)
                got.append(pick_peak(rt, y, baseline=base,
                                     interpolate_points=2000)['fwhm'])
            widths[base] = np.array(got)
        spread = {b: (v.max() - v.min()) / np.median(v)
                  for b, v in widths.items()}
        assert spread['offpeak'] < 0.02
        assert spread['offpeak'] < spread['median']

    def test_it_recovers_the_true_width_in_a_tight_window(self):
        rt, y = _gaussian_trace(0.08, fwhm_min=0.05)
        off = pick_peak(rt, y, baseline='offpeak', interpolate_points=500)
        med = pick_peak(rt, y, baseline='median', interpolate_points=500)
        assert off['fwhm'] == pytest.approx(0.05, rel=0.05)
        assert med['fwhm'] < off['fwhm']

    def test_it_finds_the_real_baseline_level(self):
        rt, y = _gaussian_trace(0.30, baseline=250.0)
        pk = pick_peak(rt, y, baseline='offpeak', interpolate_points=500)
        assert pk['baseline'] == pytest.approx(250.0, rel=0.05)

    def test_it_falls_back_when_the_window_is_all_peak(self):
        rt, y = _gaussian_trace(0.02, fwhm_min=0.05)
        pk = pick_peak(rt, y, baseline='offpeak', interpolate_points=500)
        assert np.isfinite(pk['baseline'])
        assert pk['fwhm'] >= 0

    def test_an_unknown_baseline_spec_still_raises(self):
        rt, y = _gaussian_trace(0.20)
        with pytest.raises(ValueError, match='baseline must be'):
            pick_peak(rt, y, baseline='nonsense')
