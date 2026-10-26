"""Tests for raw/metadata.py — the parts that need no vendor DLL."""

import warnings

import numpy as np
import pandas as pd
import pytest

from zenotof8600_analytics.raw import metadata as md


def _windows(start, stop, width):
    centres = np.arange(start + width / 2, stop, width)
    return np.full(centres.size, float(width)), centres


class TestClassify:

    def test_no_fragment_experiments_is_ms1(self):
        assert md._classify([], 0) == 'ms1'

    def test_variable_window_dia(self):
        assert md._classify([True] * 100, 100) == 'dia'

    def test_scanning_dia_is_swath_with_many_windows(self):
        n = 1256
        assert n >= md._SCANNING_MIN_WINDOWS
        assert md._classify([True] * n, n) == 'scanning_dia'

    def test_targeted_list_is_not_dia_despite_dia_like_spacing(self):
        assert md._classify([False] * 30, 30) == 'targeted'

    def test_isotope_envelope_of_one_peptide_is_targeted(self):
        assert md._classify([False] * 6, 6) == 'targeted'

    def test_a_single_swath_window_still_reads_as_dia(self):
        assert md._classify([True], 1) == 'dia'

    def test_any_swath_product_experiment_makes_it_dia(self):
        assert md._classify([False] * 99 + [True], 100) == 'dia'


class TestCoverage:

    def test_tiled_windows_cover_their_span(self):
        widths, centres = _windows(400, 900, 2.0)
        assert md._coverage(widths, centres) == pytest.approx(1.0, abs=0.01)

    def test_sparse_targets_cover_little(self):
        widths = np.ones(4)
        centres = np.array([400.0, 500.0, 600.0, 700.0])
        assert md._coverage(widths, centres) == pytest.approx(4 / 301, abs=1e-4)

    def test_undefined_without_two_centres(self):
        assert np.isnan(md._coverage(np.array([1.0]), np.array([500.0])))

    def test_all_nan_widths_give_nan_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert np.isnan(md._coverage(np.array([np.nan, np.nan]),
                                         np.array([500.0, 600.0])))


class TestCentreFromName:
    def test_targeted_name_gives_the_precursor(self):
        assert md._centre_from_name('TOF PI of 496.7 (100 - 2000)') == 496.7

    def test_variable_window_name_gives_the_midpoint(self):
        assert md._centre_from_name(
            'TOF PI of 380.4 to 394.4 (130 - 2000)') == pytest.approx(387.4)

    def test_ms1_name_has_no_precursor(self):
        assert md._centre_from_name('TOF MS (400 - 1500)') is None

    def test_unparseable_name_returns_none_rather_than_raising(self):
        assert md._centre_from_name('TOF PI of everything (100 - 2000)') is None


class TestResolve:
    def test_directory_becomes_a_wiff_glob(self, tmp_path):
        (tmp_path / 'b.wiff').write_bytes(b'')
        (tmp_path / 'a.wiff').write_bytes(b'')
        (tmp_path / 'notes.txt').write_text('ignored')
        found = [p.rsplit('\\', 1)[-1].rsplit('/', 1)[-1]
                 for p in md._resolve(str(tmp_path))]
        assert found == ['a.wiff', 'b.wiff']

    def test_explicit_list_passes_through(self):
        assert md._resolve(['x.wiff', 'y.wiff']) == ['x.wiff', 'y.wiff']


def test_unsupported_extension_raises_with_a_useful_message():
    with pytest.raises(NotImplementedError, match=r'\.wiff'):
        md.read_acquisition_metadata('run.raw')

class _FakeDateTime:
    def __init__(self, iso):
        self._iso = iso

    def ToString(self, _fmt):
        return self._iso


class _FakeDetails:
    def __init__(self, iso):
        self.AcquisitionDateTime = _FakeDateTime(iso)


class _FakeSample:
    def __init__(self, iso):
        self.Details = _FakeDetails(iso)


class _FakeReader:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class TestReadAcquisitionTime:

    def test_reads_the_timestamp_without_touching_the_experiments(self, monkeypatch):
        reader = _FakeReader()
        ms_sample = object()

        monkeypatch.setattr(md, '_open_wiff',
                            lambda path: (reader, _FakeSample('2026-09-02T14:47:09.0000000'), ms_sample))

        def _boom(_ms):
            raise AssertionError('read_acquisition_time must not enumerate '
                                 'MS experiments — that is the 41.5 s cost')

        monkeypatch.setattr(md, '_experiment_rows', _boom)

        got = md.read_acquisition_time('x.wiff')
        assert got == pd.Timestamp('2026-09-02T14:47:09')

    def test_closes_the_reader(self, monkeypatch):
        reader = _FakeReader()
        monkeypatch.setattr(md, '_open_wiff',
                            lambda path: (reader, _FakeSample('2026-09-02T14:47:09.0000000'), object()))
        md.read_acquisition_time('x.wiff')
        assert reader.closed

    def test_rejects_non_wiff(self):
        with pytest.raises(NotImplementedError):
            md.read_acquisition_time('run.raw')

    def test_acquisition_times_uses_the_cheap_path_and_keys_by_stem(self, monkeypatch):
        stamps = {'b.wiff': '2026-09-02T02:00:00.0000000',
                  'a.wiff': '2026-09-02T01:00:00.0000000'}
        monkeypatch.setattr(md, 'read_acquisition_time',
                            lambda path: pd.Timestamp(stamps[path]))

        def _boom(_path, **_kw):
            raise AssertionError('acquisition_times must not build the full '
                                 'method summary for one timestamp')

        monkeypatch.setattr(md, 'read_acquisition_metadata', _boom)

        out = md.acquisition_times(['b.wiff', 'a.wiff'])
        assert list(out.index) == ['a', 'b']
        assert out['a'] == pd.Timestamp('2026-09-02T01:00:00')

    def test_a_file_that_cannot_be_read_is_skipped_not_fatal(self, monkeypatch):
        def _half(path):
            if path == 'bad.wiff':
                raise RuntimeError('vendor DLL said no')
            return pd.Timestamp('2026-09-02T01:00:00')

        monkeypatch.setattr(md, 'read_acquisition_time', _half)
        out = md.acquisition_times(['bad.wiff', 'good.wiff'])
        assert list(out.index) == ['good']
