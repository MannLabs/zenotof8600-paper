"""Fragment content of the crowded CID spectrum."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg
import pathlib

import numpy as np
import pandas as pd

import zenotof8600_analytics as core
from alphabase.constants.aa import AA_ASCII_MASS

from annotate_spectra import precursor_species, summed_spectrum
from score_fragment_coverage import (FRAGMENT_TOL_PPM, FRAG_TYPES,
                                     PHOSPHO_SITE, SEQUENCE, activation_of,
                                     load, targets)

OUT_DIR = pathlib.Path(cfg.cache('H032_E391', 'analysis'))

PROTON = 1.00727646
H2O = 18.01056468
NH3 = 17.02654910
CO = 27.99491462
PHOSPHO = 79.96633089
H3PO4 = 97.97689580

SPECIES = 'heavy pS129'

RUNS = ('A1', 'A2', 'A5', 'A7', 'A6', 'A8')
MAX_CHARGE = 3

MAX_INTERNAL_LEN = 12

LOSSES = {'-H2O': H2O, '-NH3': NH3, '-H3PO4': H3PO4}


def residue_mass(aa):
    return float(AA_ASCII_MASS[ord(aa)])


def mods_in(lo, hi, phospho_site):
    """Modification mass carried by peptide residues lo..hi (1-based)."""
    return PHOSPHO if lo <= phospho_site <= hi else 0.0


def internal_ions(sequence, phospho_site, max_len=MAX_INTERNAL_LEN,
                  charges=(1, 2)):
    """b- and a-type internal fragments: a stretch with neither terminus."""
    n = len(sequence)
    rows = []
    for lo in range(2, n):
        for hi in range(lo, min(lo + max_len, n)):
            neutral = sum(residue_mass(c) for c in sequence[lo - 1:hi])
            neutral += mods_in(lo, hi, phospho_site)
            for z in charges:
                rows.append(('internal', (neutral + z * PROTON) / z, z))
                rows.append(('internal', (neutral - CO + z * PROTON) / z, z))
    return rows


H_ATOM = 1.00782503


def families(sequence, frags, phospho_site):
    """Candidate m/z per family, added cumulatively in a fixed order."""
    out = {}
    term = frags[frags['fragment_charge'] <= 2]
    out['terminal 1-2+'] = [(r.ion_type, float(r.product_mz), int(r.fragment_charge))
                            for r in term.itertuples()]
    shift = []
    for r in term.itertuples():
        z = int(r.fragment_charge)
        if r.ion_type == 'c':
            shift.append(('c-dot', float(r.product_mz) - H_ATOM / z, z))
        elif r.ion_type == 'z':
            shift.append(('z+1', float(r.product_mz) + H_ATOM / z, z))
            shift.append(('z', float(r.product_mz) - H_ATOM / z, z))
    out['c/z H transfer'] = shift
    loss = []
    for r in term.itertuples():
        z = int(r.fragment_charge)
        for name, dm in LOSSES.items():
            loss.append((f'{r.ion_type}{name}', float(r.product_mz) - dm / z, z))
    out['neutral losses'] = loss
    out['internal'] = internal_ions(sequence, phospho_site)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    core.raw.register_sil_modifications()
    tg = targets()
    i = tg.index[tg['species'] == SPECIES][0]

    rows = []
    for run in RUNS:
        ms = load(run)
        act, _ = activation_of(ms)
        frags = core.raw.target_fragments(tg, frag_types=FRAG_TYPES[act],
                                          min_mz=100.0, max_mz=4000.0)
        sub = frags[frags['target_idx'] == i]
        q1 = float(sub['precursor_mz'].iloc[0])
        spec, _, _, n_spectra = summed_spectrum(
            ms, q1, sub[sub['fragment_charge'] <= 2]['product_mz'].astype(float))
        mz = spec['mz'].to_numpy()
        inten = spec['intensity'].to_numpy()
        total = float(inten.sum())

        fam = families(SEQUENCE, sub, PHOSPHO_SITE)
        fam['precursor'] = [('M', float(m), int(z)) for _, m, z in
                            [(r.ion_type, r.product_mz, r.fragment_charge)
                             for r in precursor_species(
                                 (q1 - PROTON) * 4).itertuples()]]

        order = ['terminal 1-2+', 'precursor', 'c/z H transfer',
                 'neutral losses', 'internal']
        claimed = np.zeros(len(mz), dtype=bool)
        print(f'\n=== {run}: {act.upper()} · {n_spectra} spectra · '
              f'{len(mz)} peaks · TIC {total:.3g} ===')
        print(f'{"family":16s} {"candidates":>11s} {"new peaks":>10s} '
              f'{"TIC added":>10s} {"cumulative":>11s}')
        for name in order:
            targets_mz = np.array([m for _, m, _ in fam[name]], dtype=float)
            hit = np.zeros(len(mz), dtype=bool)
            for t in targets_mz:
                half = t * FRAGMENT_TOL_PPM * 1e-6
                hit |= np.abs(mz - t) <= half
            new = hit & ~claimed
            claimed |= hit
            add = float(inten[new].sum()) / total
            cum = float(inten[claimed].sum()) / total
            rows.append({'run': run, 'activation': act, 'family': name,
                         'n_candidates': len(targets_mz),
                         'new_peaks': int(new.sum()),
                         'tic_added': round(add, 4),
                         'tic_cumulative': round(cum, 4)})
            print(f'{name:16s} {len(targets_mz):11d} {int(new.sum()):10d} '
                  f'{add:9.1%} {cum:10.1%}')
        rest = ~claimed
        print(f'{"unexplained":16s} {"":11s} {int(rest.sum()):10d} '
              f'{float(inten[rest].sum()) / total:9.1%}')
        rows.append({'run': run, 'activation': act, 'family': 'unexplained',
                     'n_candidates': 0, 'new_peaks': int(rest.sum()),
                     'tic_added': round(float(inten[rest].sum()) / total, 4),
                     'tic_cumulative': 1.0})

    pd.DataFrame(rows).to_csv(OUT_DIR / 'cid_noise_sourcedata.csv', index=False)
    print(f'\nwrote {OUT_DIR / "cid_noise_sourcedata.csv"}')


if __name__ == '__main__':
    main()
