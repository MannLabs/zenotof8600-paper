"""Characterization: protein-group inference rules."""

from __future__ import annotations

import zenotof8600_analytics as core


def test_group_by_shared_peptides_connected_components():
    sets = [['A', 'B'], ['B', 'C'], ['D'], ['E', 'F']]
    out = core.group_proteins_by_shared_peptides(sets)
    assert out == {
        'A': ('A', 'B', 'C'), 'B': ('A', 'B', 'C'), 'C': ('A', 'B', 'C'),
        'D': ('D',), 'E': ('E', 'F'), 'F': ('E', 'F'),
    }


def test_group_by_signature_splits_on_distinct_peptides():
    sets = [['A', 'B'], ['B', 'C'], ['D'], ['E', 'F']]
    out = core.group_proteins_by_signature(sets)
    assert out == {
        'A': ('A',), 'B': ('B',), 'C': ('C',), 'D': ('D',),
        'E': ('E', 'F'), 'F': ('E', 'F'),
    }


def test_group_by_signature_vs_shared_diverge():
    sets = [['A', 'B'], ['A', 'B'], ['A'], ['C']]
    assert core.group_proteins_by_signature(sets) == {
        'A': ('A',), 'B': ('B',), 'C': ('C',)}
    assert core.group_proteins_by_shared_peptides(sets) == {
        'A': ('A', 'B'), 'B': ('A', 'B'), 'C': ('C',)}
