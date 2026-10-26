"""Characterization: sequence-coverage pipeline + plot_coverage_histogram."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import zenotof8600_analytics as core

EXPECTED_PI_COLUMNS = [
    'protein_group', 'gene', 'group', 'representative_protein',
    'protein_length', 'n_peptides', 'coverage_pct',
]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def test_load_protein_sequences(protein_sequences):
    assert len(protein_sequences) == 20590
    assert 'A0A0A0MS01' in protein_sequences
    assert protein_sequences['A0A0A0MS01'].isalpha()


def test_compute_protein_info_shape(protein_info):
    assert protein_info.shape == (10032, 7)
    assert list(protein_info.columns) == EXPECTED_PI_COLUMNS
    assert protein_info.groupby('group').size().to_dict() == {
        '200SPD': 5700, '500SPD': 4332}


def test_compute_protein_info_all_leaders_resolved(protein_info):
    assert int(protein_info['coverage_pct'].isna().sum()) == 0
    assert (protein_info['coverage_pct'].between(0, 100)).all()


def test_compute_protein_info_median_coverage(protein_info):
    med = protein_info.groupby('group')['coverage_pct'].median()
    assert med['200SPD'] == pytest.approx(21.2121, abs=1e-3)
    assert med['500SPD'] == pytest.approx(18.3651, abs=1e-3)


