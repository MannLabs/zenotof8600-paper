"""Characterization: in-silico digestion + sequence property helpers."""

from __future__ import annotations

import math

import pytest

import zenotof8600_analytics as core

SEQ = 'MKWVTFISLLFLFSSAYSR'


def test_count_missed_cleavages():
    assert core.count_missed_cleavages(SEQ) == 1
    assert core.count_missed_cleavages('WVTFISLLFLFSSAYSR') == 0


def test_build_peptide_id():
    assert core.build_peptide_id('PEPTIDEK', 'Oxidation@M', '3') == 'PEP(Oxidation@M)TIDEK'
    assert core.build_peptide_id('PEPTIDEK', '', '') == 'PEPTIDEK'
