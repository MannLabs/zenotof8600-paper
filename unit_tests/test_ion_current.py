"""TIC per spectrum and as a chromatogram, across acquisition types."""
import warnings

import numpy as np
import pandas as pd
import pytest

import zenotof8600_analytics as core
from zenotof8600_analytics.ion_current import tic, tic_per_spectrum


class _Run:

    def __init__(self, scans, shuffle_peaks=False):
        rows, chunks, start = [], [], 0
        for rt, lev, inten in scans:
            inten = np.asarray(inten, dtype=np.float32)
            rows.append({'rt': rt, 'ms_level': lev, 'peak_start_idx': start,
                         'peak_stop_idx': start + inten.size})
            chunks.append(inten)
            start += inten.size
        sd = pd.DataFrame(rows)
        inten = (np.concatenate(chunks) if chunks
                 else np.array([], np.float32))
        if shuffle_peaks:
            blocks = list(zip(sd['peak_start_idx'], sd['peak_stop_idx']))
            new, pos, out = [], 0, []
            for a, b in reversed(blocks):
                out.append(inten[a:b])
                new.append((pos, pos + (b - a)))
                pos += b - a
            new = new[::-1]
            sd['peak_start_idx'] = [a for a, _ in new]
            sd['peak_stop_idx'] = [b for _, b in new]
            inten = np.concatenate(out)
        self.spectrum_df = sd
        self.peak_df = pd.DataFrame({'mz': np.arange(inten.size, dtype=float),
                                     'intensity': inten})


def _dia(n_cycles=10, n_windows=4):
    scans, dt = [], 0.01
    for c in range(n_cycles):
        scans.append((c * dt, 1, [100.0 + c, 1.0]))
        for w in range(n_windows):
            scans.append((c * dt + (w + 1) * dt / 10, 2, [10.0 * (w + 1)]))
    return _Run(scans)


def test_empty_spectra_sum_to_zero():
    ms = _Run([(0.0, 1, [5.0, 5.0]), (0.1, 2, []), (0.2, 2, [3.0]),
               (0.3, 2, []), (0.4, 1, [1.0])])
    np.testing.assert_array_equal(tic_per_spectrum(ms),
                                  [10.0, 0.0, 3.0, 0.0, 1.0])


def test_non_contiguous_peak_layout_gives_the_same_sums():
    scans = [(0.0, 1, [1.0, 2.0]), (0.1, 2, []), (0.2, 2, [4.0, 8.0, 16.0])]
    np.testing.assert_array_equal(tic_per_spectrum(_Run(scans)),
                                  tic_per_spectrum(_Run(scans,
                                                        shuffle_peaks=True)))


def test_dia_cycles_sum_each_level_and_do_not_warn():
    ms = _dia()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        full = tic(ms)
        ms1 = tic(ms, ms_level=1)
        ms2 = tic(ms, ms_level=2)
    assert len(full) == len(ms1) == len(ms2) == 10
    np.testing.assert_allclose(ms2['tic'], 100.0)
    np.testing.assert_allclose(ms1['tic'], 101.0 + np.arange(10))
    np.testing.assert_allclose(full['tic'], ms1['tic'] + ms2['tic'])
    assert (ms2['n_spectra'] == 4).all()
    assert ms2.attrs['ms2_per_cycle'] == (4, 4)
    np.testing.assert_allclose(ms1['rt'], np.arange(10) * 0.01)


def test_per_spectrum_is_a_comb_over_the_windows():
    ms2 = tic(_dia(), ms_level=2, per='spectrum')
    assert len(ms2) == 40
    np.testing.assert_allclose(ms2['tic'].to_numpy()[:4], [10, 20, 30, 40])


def test_rt_range_and_unknown_per():
    out = tic(_dia(), ms_level=1, rt_range=(0.02, 0.05))
    np.testing.assert_allclose(out['rt'], [0.02, 0.03, 0.04, 0.05])
    with pytest.raises(ValueError, match='per must be'):
        tic(_dia(), per='window')


def test_dda_variable_topn_warns_and_reports_the_mean():
    scans, rng = [], np.random.default_rng(0)
    counts = [3, 10, 1, 7, 10, 2]
    for c, n in enumerate(counts):
        scans.append((c, 1, [1000.0]))
        scans += [(c + (k + 1) / 20, 2, [50.0]) for k in range(n)]
    with pytest.warns(UserWarning, match='MS2 scans per cycle vary'):
        out = tic(_Run(scans), ms_level=2)
    np.testing.assert_array_equal(out['n_spectra'], counts)
    np.testing.assert_allclose(out['tic'], 50.0 * np.array(counts))
    np.testing.assert_allclose(out['tic_per_spectrum'], 50.0)
    assert out.attrs['ms2_per_cycle'] == (1, 10)


def test_dda_ms1_tic_does_not_warn():
    scans = []
    for c, n in enumerate([3, 10, 1, 7]):
        scans.append((c, 1, [1000.0]))
        scans += [(c + (k + 1) / 20, 2, [50.0]) for k in range(n)]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = tic(_Run(scans), ms_level=1)
    np.testing.assert_allclose(out['tic'], 1000.0)


def test_scheduled_targeted_steps_show_in_the_sum_not_the_mean():
    scans = []
    for c in range(12):
        n = 2 if c < 6 else 4
        scans.append((c * 0.02, 1, [500.0]))
        scans += [(c * 0.02 + (k + 1) * 0.001, 2, [25.0]) for k in range(n)]
    with pytest.warns(UserWarning, match='scheduled targeted'):
        out = tic(_Run(scans), ms_level=2)
    assert out['tic'].iloc[7] == 2 * out['tic'].iloc[2]
    np.testing.assert_allclose(out['tic_per_spectrum'], 25.0)


def test_targeted_without_survey_scan_falls_back_to_spectra():
    scans = [(k * 0.01, 2, [float(k)]) for k in range(1, 9)]
    with pytest.warns(UserWarning, match='no MS1 scans'):
        out = tic(_Run(scans))
    assert out.attrs['per'] == 'spectrum'
    np.testing.assert_allclose(out['tic'], np.arange(1, 9))


def test_exported_at_top_level_and_from_raw():
    assert core.tic is tic
    assert core.raw.tic is tic
