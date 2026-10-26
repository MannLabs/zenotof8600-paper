"""Tests for reading a plate position back out of a run name."""

from __future__ import annotations

import pandas as pd
import pytest

import zenotof8600_analytics as core


def test_trailing_well_and_quadrant_token():
    pos = core.parse_run_position(
        '20260801_Zeno2_Eno0_2p3min_TiHe_SA_H032_E349_DrugP2_Q1_B10')
    assert pos == {'plate_group': 'Q1', 'plate_well': 'B10',
                   'plate_row': 'B', 'plate_col': 10, 'plate_quadrant': 'Q1'}


def test_bare_well_has_no_group_and_no_quadrant():
    pos = core.parse_run_position('sample_A7')
    assert pos['plate_well'] == 'A7'
    assert pos['plate_group'] is None
    assert pos['plate_quadrant'] is None


def test_group_may_itself_look_like_a_well():
    pos = core.parse_run_position('20260801_x_plate_B1_H12')
    assert (pos['plate_group'], pos['plate_well']) == ('B1', 'H12')
    assert pos['plate_quadrant'] is None


def test_384_well_address_derives_its_quadrant():
    assert core.parse_run_position('run_I13')['plate_quadrant'] == 'Q1'
    assert core.parse_run_position('run_P24')['plate_quadrant'] == 'Q4'
    assert core.parse_run_position('run_I14')['plate_quadrant'] == 'Q2'
    assert core.parse_run_position('run_P23')['plate_quadrant'] == 'Q3'


@pytest.mark.parametrize('name', [
    'no_well_here',
    'sample_10',
    'expt_H032_E349',
    'x_Z9',
    'x_A25',
])
def test_names_without_a_plate_position(name):
    pos = core.parse_run_position(name)
    assert all(v is None for v in pos.values()), pos


def test_plate_metadata_frame_is_indexed_by_run():
    runs = ['x_Q1_A1', 'x_Q2_A1', 'no_position']
    meta = core.plate_metadata(runs)
    assert list(meta.index) == runs
    assert list(meta.columns) == list(core.PLATE_FIELDS)
    assert list(meta['plate_quadrant']) == ['Q1', 'Q2', None]
    assert pd.isna(meta.loc['no_position', 'plate_well'])
