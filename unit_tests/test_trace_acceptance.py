"""trace_acceptance / expected_trace_correlation: a trace judged by counting."""
import numpy as np
import pandas as pd
import pytest

from zenotof8600_analytics.raw import targeted

from test_quantify_targets_ms1 import _MS, _ms1_transitions

RT = np.arange(60) * 0.01
APEX, SIGMA = 0.30, 0.03


def _shape(shift=0.0):
    return np.exp(-0.5 * ((RT - APEX - shift) / SIGMA) ** 2)


def _trace(signal_ions, baseline_ions, shift=0.0, seed=0):
    s = _shape(shift)
    mu = baseline_ions + signal_ions * s / s.sum()
    return np.random.default_rng(seed).poisson(mu) / 0.05


def test_quantify_targets_returns_traces_on_request():
    kw = dict(ms_level=1, mz_tol=30.0, decoy_offset_da=None, verbose=False)
    plain = targeted.quantify_targets({'r1': _MS()}, _ms1_transitions(), **kw)
    table, traces = targeted.quantify_targets(
        {'r1': _MS()}, _ms1_transitions(), return_traces=True, **kw)
    pd.testing.assert_frame_equal(plain, table)
    assert set(traces) == {('r1', 0), ('r1', 1)}
    rt, inten = traces[('r1', 0)]
    assert len(rt) == len(inten) > 0
