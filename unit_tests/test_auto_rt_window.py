"""Tests for core.auto_rt_window and the extraction-window search."""

from __future__ import annotations

import pathlib
import sys

import warnings

import numpy as np
import pytest

import zenotof8600_analytics as core

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestAutoRtWindow:

    def test_converges_from_a_measurable_start(self):
        win, trace = core.auto_rt_window(lambda w: 0.05, start=0.05,
                                         verbose=False)
        assert win == pytest.approx(0.15)
        assert len(trace) >= 1

    def test_expands_an_unmeasurable_first_probe(self):
        seen = []

        def measure(w):
            seen.append(w)
            return 0.05 if w >= 0.12 else np.nan

        win, trace = core.auto_rt_window(measure, start=0.05, verbose=False)
        assert seen[0] == pytest.approx(0.05)
        assert max(seen) >= 0.12
        assert np.isfinite(win)
        assert win != pytest.approx(0.05)
        assert len(trace) >= 1

    def test_returns_nan_when_no_window_is_measurable(self):
        win, trace = core.auto_rt_window(lambda w: np.nan, start=0.05,
                                         verbose=False)
        assert np.isnan(win)
        assert trace.empty

    def test_does_not_expand_past_the_upper_bound(self):
        probed = []

        def measure(w):
            probed.append(w)
            return np.nan

        win, _ = core.auto_rt_window(measure, start=0.05, bounds=(0.02, 0.20),
                                     verbose=False)
        assert np.isnan(win)
        assert max(probed) <= 0.20 + 1e-12

    def test_a_failure_after_a_good_measurement_keeps_the_anchored_window(self):
        calls = {'n': 0}

        def measure(w):
            calls['n'] += 1
            return 0.05 if calls['n'] == 1 else np.nan

        win, trace = core.auto_rt_window(measure, start=0.05, verbose=False)
        assert np.isfinite(win)
        assert len(trace) == 1

    def test_expansion_is_reported(self, capsys):
        core.auto_rt_window(lambda w: 0.05 if w >= 0.12 else np.nan,
                            start=0.05, verbose=True, label='RT_WINDOW')
        assert 'unmeasurable' in capsys.readouterr().out


class TestCycleAwareFirstProbe:

    def _resolve(self, monkeypatch, cycle_s, capture):
        import lc_peak_width_analysis as lcw
        import pandas as pd

        monkeypatch.setattr(lcw, 'RT_WINDOW', 'auto')
        monkeypatch.setattr(lcw, '_cycle_time_s', lambda path: cycle_s)

        def fake_measure_run(raw_path, targets, *, verbose=True, rt_window=None):
            capture.append(rt_window)
            return pd.DataFrame({'fwhm_s': [5.4] * 180}), cycle_s

        monkeypatch.setattr(lcw, 'measure_run', fake_measure_run)
        targets = pd.DataFrame({'mz': [500.0] * 200, 'rt': [10.0] * 200})
        return lcw.resolve_rt_window('x.wiff', targets, verbose=False)

    def test_a_slow_cycle_raises_the_first_probe(self, monkeypatch):
        import lc_peak_width_analysis as lcw
        probes = []
        self._resolve(monkeypatch, 1.48, probes)
        assert probes[0] == pytest.approx(12 * 1.48 / 60 / 2)
        assert probes[0] > lcw.RT_WINDOW_START

    def test_a_fast_cycle_keeps_the_default(self, monkeypatch):
        import lc_peak_width_analysis as lcw
        probes = []
        self._resolve(monkeypatch, 0.40, probes)
        assert probes[0] == pytest.approx(lcw.RT_WINDOW_START)

    def test_an_unresolvable_window_raises_rather_than_measuring(self, monkeypatch):
        import lc_peak_width_analysis as lcw
        import pandas as pd

        monkeypatch.setattr(lcw, 'RT_WINDOW', 'auto')
        monkeypatch.setattr(lcw, '_cycle_time_s', lambda path: 1.48)
        monkeypatch.setattr(lcw, 'measure_run',
                            lambda *a, **k: (pd.DataFrame({'fwhm_s': []}), 1.48))
        targets = pd.DataFrame({'mz': [500.0] * 200, 'rt': [10.0] * 200})
        with pytest.raises(RuntimeError, match='could not resolve an RT window'):
            lcw.resolve_rt_window('x.wiff', targets, verbose=False)


class TestSeedRtWindow:

    def test_a_fast_cycle_leaves_the_start_alone(self):
        assert core.seed_rt_window(0.3, start=0.05, verbose=False) == \
            pytest.approx(0.05)

    def test_a_slow_cycle_raises_the_start(self):
        got = core.seed_rt_window(1.48, start=0.05, verbose=False)
        assert got == pytest.approx(12 * 1.48 / 60 / 2)
        assert got > 0.05
        assert 2 * got * 60 / 1.48 == pytest.approx(12)

    def test_an_unknown_cycle_leaves_the_start_alone(self):
        for bad in (np.nan, 0.0, -1.0):
            assert core.seed_rt_window(bad, start=0.05, verbose=False) == \
                pytest.approx(0.05)

    def test_it_reports_when_it_intervenes(self, capsys):
        core.seed_rt_window(1.48, start=0.05, verbose=True, label='rt_tol')
        out = capsys.readouterr().out
        assert 'rt_tol' in out and '1.48' in out


class TestThinProbeIsNotAnAnchor:

    def test_a_low_yield_probe_is_expanded_not_anchored(self):
        seen = []

        def measure(w):
            seen.append(w)
            if w < 0.12:
                return 0.02, {'ok_frac': 0.05}
            return 0.05, {'ok_frac': 0.9}

        win, trace = core.auto_rt_window(measure, start=0.05, verbose=False)
        assert win == pytest.approx(0.15)
        assert max(seen) >= 0.12
        assert (trace['ok_frac'] >= 0.5).all()

    def test_a_healthy_probe_is_still_accepted(self):
        win, _ = core.auto_rt_window(lambda w: (0.05, {'ok_frac': 0.9}),
                                     start=0.05, verbose=False)
        assert win == pytest.approx(0.15)

    def test_measures_without_ok_frac_are_unaffected(self):
        win, _ = core.auto_rt_window(lambda w: 0.05, start=0.05, verbose=False)
        assert win == pytest.approx(0.15)


class TestConvergenceIsReported:

    def test_a_converged_search_is_flagged_and_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _win, trace = core.auto_rt_window(lambda w: 0.05, start=0.05,
                                              verbose=False)
        assert trace.attrs['converged'] is True

    def test_a_search_that_runs_out_of_iterations_warns(self):
        with pytest.warns(UserWarning, match='did not converge'):
            win, trace = core.auto_rt_window(
                lambda w: w * 0.5, start=0.05, max_iter=3, max_expand=0,
                bounds=(0.02, 10.0), verbose=False)
        assert trace.attrs['converged'] is False
        assert np.isfinite(win)

    def test_the_warning_says_which_way_the_bias_runs(self):
        with pytest.warns(UserWarning, match='NARROW'):
            core.auto_rt_window(lambda w: w * 0.5, start=0.05, max_iter=3,
                                max_expand=0, bounds=(0.02, 10.0),
                                verbose=False)

    def test_only_thin_probes_degrades_to_a_flagged_estimate_not_nan(self):
        with pytest.warns(UserWarning, match='did not converge'):
            win, trace = core.auto_rt_window(
                lambda w: (0.05, {'ok_frac': 0.05}), start=0.05,
                verbose=False)
        assert np.isfinite(win)
        assert trace.attrs['converged'] is False
        assert len(trace) == 1

    def test_an_anchored_early_stop_is_not_a_non_convergence(self):
        calls = {'n': 0}

        def measure(w):
            calls['n'] += 1
            return 0.05 if calls['n'] == 1 else np.nan

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _win, trace = core.auto_rt_window(measure, start=0.05,
                                              verbose=False)
        assert trace.attrs['converged'] is True
