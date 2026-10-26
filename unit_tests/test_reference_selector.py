"""Tests for naming the reference run explicitly."""
import numpy as np
import pandas as pd
import pytest

from zenotof8600_analytics.raw.targeted import _pick_reference


def _meta():
    return pd.DataFrame(
        {'sequence': ['PEPTIDEK', 'PEPTIDEK'],
         'charge': [2, 2], 'precursor_mz': [700.0, 703.0],
         'label': ['light', 'heavy']},
        index=pd.Index([0, 1], name='target_idx'))


def _traces(light_max, heavy_max):
    return {0: (np.arange(5.0), np.full(5, light_max)),
            1: (np.arange(5.0), np.full(5, heavy_max))}


def test_max_signal_follows_whichever_is_brighter():
    meta, members = _meta(), [0, 1]
    assert _pick_reference('max_signal', members, _traces(10, 5), meta,
                           'PEPTIDEK') == 0
    assert _pick_reference('max_signal', members, _traces(5, 10), meta,
                           'PEPTIDEK') == 1


def test_named_anchor_is_stable_whatever_the_sample_holds():
    meta, members = _meta(), [0, 1]
    sel = {'label': 'heavy'}
    assert _pick_reference(sel, members, _traces(10, 5), meta,
                           'PEPTIDEK') == 1
    assert _pick_reference(sel, members, _traces(5, 10), meta,
                           'PEPTIDEK') == 1


def test_int_selector():
    assert _pick_reference(1, [0, 1], _traces(10, 5), _meta(),
                           'PEPTIDEK') == 1


def test_callable_selector():
    sel = (lambda members, traces, meta, key:
           min(members, key=lambda t: meta.loc[t, 'precursor_mz']))
    assert _pick_reference(sel, [0, 1], _traces(1, 1), _meta(),
                           'PEPTIDEK') == 0


def test_ambiguous_named_anchor_raises():
    meta = _meta()
    meta['label'] = ['heavy', 'heavy']
    with pytest.raises(ValueError, match='matched 2 targets'):
        _pick_reference({'label': 'heavy'}, [0, 1], _traces(1, 1), meta,
                        'PEPTIDEK')


def test_missing_named_anchor_raises():
    with pytest.raises(ValueError, match='matched 0 targets'):
        _pick_reference({'label': 'medium'}, [0, 1], _traces(1, 1), _meta(),
                        'PEPTIDEK')
