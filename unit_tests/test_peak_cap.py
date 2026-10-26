"""Tests for the per-spectrum peak cap and the cache stamp that protects it."""

from __future__ import annotations

import numpy as np
import pytest

from zenotof8600_analytics.raw import extraction as ex


class TestDefaultLiftsTheCap:

    def test_sciex_defaults_to_unlimited(self):
        opts = ex._reader_opts('sciex', None)
        assert opts['keep_k_peaks_per_spec'] == ex.KEEP_ALL_PEAKS

    def test_the_limit_is_above_any_real_spectrum(self):
        assert ex.KEEP_ALL_PEAKS > 10 ** 9

    def test_an_explicit_option_still_wins(self):
        opts = ex._reader_opts('sciex', {'keep_k_peaks_per_spec': 2000})
        assert opts['keep_k_peaks_per_spec'] == 2000

    def test_other_readers_get_no_sciex_settings(self):
        assert ex._reader_opts('thermo', None) == {}

    def test_unrelated_caller_options_pass_through(self):
        opts = ex._reader_opts('thermo', {'some_setting': 3})
        assert opts == {'some_setting': 3}

    def test_the_sentinel_is_reported_as_unlimited(self):
        assert ex._fmt_limit(ex.KEEP_ALL_PEAKS) == 'unlimited'
        assert ex._fmt_limit(2000) == '2,000'


class TestCachePeakLimitStamp:

    def _hdf(self, tmp_path):
        h5py = pytest.importorskip('h5py')
        path = str(tmp_path / 'x.ms_data.hdf')
        with h5py.File(path, 'w') as h:
            h.create_dataset('d', data=np.arange(3))
        return path

    def test_round_trips_the_limit(self, tmp_path):
        path = self._hdf(tmp_path)
        ex._stamp_peak_limit(path, ex.KEEP_ALL_PEAKS)
        assert ex._cached_peak_limit(path) == ex.KEEP_ALL_PEAKS

    def test_an_unstamped_cache_reads_as_unknown(self, tmp_path):
        path = self._hdf(tmp_path)
        have = ex._cached_peak_limit(path)
        assert have is None
        assert have != int(ex.KEEP_ALL_PEAKS)

    def test_a_capped_cache_does_not_satisfy_an_unlimited_request(self, tmp_path):
        path = self._hdf(tmp_path)
        ex._stamp_peak_limit(path, 2000)
        assert ex._cached_peak_limit(path) != int(ex.KEEP_ALL_PEAKS)

    def test_a_missing_file_reads_as_unknown_rather_than_raising(self, tmp_path):
        assert ex._cached_peak_limit(str(tmp_path / 'nope.hdf')) is None

    def test_stamping_a_missing_file_is_not_fatal(self, tmp_path):
        ex._stamp_peak_limit(str(tmp_path / 'nope.hdf'), 10)

    def test_none_is_stamped_distinguishably(self, tmp_path):
        path = self._hdf(tmp_path)
        ex._stamp_peak_limit(path, None)
        assert ex._cached_peak_limit(path) == -1


def test_uncapped_limits_share_a_cache():
    from zenotof8600_analytics.raw.loading import (KEEP_ALL_PEAKS,
                                                  _same_peak_limit)
    assert _same_peak_limit(10 ** 7, KEEP_ALL_PEAKS)
    assert _same_peak_limit(KEEP_ALL_PEAKS, 10 ** 7)
    assert _same_peak_limit(2000, 2000)


def test_a_real_cap_still_forces_a_reparse():
    from zenotof8600_analytics.raw.loading import (KEEP_ALL_PEAKS,
                                                  _same_peak_limit)
    assert not _same_peak_limit(2000, KEEP_ALL_PEAKS)
    assert not _same_peak_limit(KEEP_ALL_PEAKS, 2000)
    assert not _same_peak_limit(None, KEEP_ALL_PEAKS)
