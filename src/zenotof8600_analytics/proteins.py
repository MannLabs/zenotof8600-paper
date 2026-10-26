"""Protein-group inference rules and the peptidoform id builder."""

from __future__ import annotations

from typing import Iterable


def group_proteins_by_shared_peptides(
    accession_sets: Iterable[Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    """Connected-components grouping ("CC", PEAKS' rule)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
            return x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for accs in accession_sets:
        accs = list(accs)
        if not accs:
            continue
        find(accs[0])
        for a in accs[1:]:
            union(accs[0], a)

    components: dict[str, list[str]] = {}
    for acc in parent:
        components.setdefault(find(acc), []).append(acc)

    out: dict[str, tuple[str, ...]] = {}
    for members in components.values():
        group = tuple(sorted(members))
        for m in members:
            out[m] = group
    return out


def group_proteins_by_signature(
    accession_sets: Iterable[Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    """Signature-based grouping (DIA-NN's rule)."""
    accession_set_keys = []
    protein_to_sig: dict[str, set] = {}
    for accs in accession_sets:
        accs = tuple(sorted(set(accs)))
        if not accs:
            continue
        accession_set_keys.append(accs)
        for a in accs:
            protein_to_sig.setdefault(a, set()).add(accs)

    sig_to_proteins: dict[frozenset, list[str]] = {}
    for prot, sig in protein_to_sig.items():
        sig_to_proteins.setdefault(frozenset(sig), []).append(prot)

    out: dict[str, tuple[str, ...]] = {}
    for members in sig_to_proteins.values():
        group = tuple(sorted(members))
        for m in members:
            out[m] = group
    return out


def build_peptide_id(sequence: str, mods: str, mod_sites: str) -> str:
    """Render the peptidoform identifier as an inline modified-sequence string in alphabase mod-name format."""
    if not mods:
        return sequence
    mod_list = mods.split(';')
    site_list = [int(s) for s in mod_sites.split(';')]
    pairs = sorted(zip(site_list, mod_list), key=lambda p: p[0])

    out: list[str] = []
    for s, m in pairs:
        if s == 0:
            out.append(f'({m})')
    internal = [(s, m) for s, m in pairs if s > 0]
    j = 0
    for i, aa in enumerate(sequence, start=1):
        out.append(aa)
        while j < len(internal) and internal[j][0] == i:
            out.append(f'({internal[j][1]})')
            j += 1
    return ''.join(out)
