"""Scheduled EAD electron-energy sweep for the alpha-synuclein assay."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg
import pathlib

import pandas as pd

import zenotof8600_analytics as core

from score_fragment_coverage import (CHARGE, LABEL, LABEL_SITE, MET_SITES,
                                     PHOSPHO_SITE, SEQUENCE)

OUT = pathlib.Path(cfg.data('H032_E391', 'mrmhr_energy_sweep_E391.txt'))

ELECTRON_ENERGIES_EV = (0.0, 7.0, 12.0, 15.0, 17.0, 20.0)

RETENTION_MIN = {
    'heavy unmod +32': 4.516, 'heavy pS129 +32': 4.647,
    'heavy unmod +16': 5.132, 'heavy pS129 +16': 5.289,
    'heavy unmod': 5.337, 'heavy pS129': 5.430,
}

LAYOUT = 'ead_scheduled'
GROUP = 'Group 1'
Q1_ISOTOPE = 'most_abundant'
TOF_RANGE = (150.0, 3000.0)
ACCUMULATION_S = 0.040
QJET_DP = 20
COLLISION_ENERGY = 12
CE_SPREAD = 0
REACTION_TIME_MS = 15.0
ETC_PERCENT = 100.0
TIME_BINS_TO_SUM = 8
RT_TOLERANCE_S = 8


def species_targets():
    """One alphabase precursor row per species, mods sorted by site."""
    rows = []
    for phospho in (False, True):
        for n_ox in (0, 1, 2):
            mods = [(s, 'Oxidation@M') for s in MET_SITES[:n_ox]]
            if phospho:
                mods.append((PHOSPHO_SITE, 'Phospho@S'))
            mods.append((LABEL_SITE, LABEL))
            mods.sort()
            name = 'pS129' if phospho else 'unmod'
            rows.append({
                'species': f'heavy {name}'
                           + {0: '', 1: ' +16', 2: ' +32'}[n_ox],
                'short': name + {0: '', 1: '_ox1', 2: '_ox2'}[n_ox],
                'sequence': SEQUENCE,
                'mods': ';'.join(m for _, m in mods),
                'mod_sites': ';'.join(str(s) for s, _ in mods),
                'charge': CHARGE})
    return pd.DataFrame(rows)


def main():
    core.raw.register_sil_modifications()
    species = core.precursor_q1_mz(species_targets(), isotope=Q1_ISOTOPE)
    species['rt_min'] = species['species'].map(RETENTION_MIN)
    if species['rt_min'].isna().any():
        missing = species.loc[species['rt_min'].isna(), 'species'].tolist()
        raise SystemExit(f'no measured retention time for {missing}')

    rows = []
    for ke in ELECTRON_ENERGIES_EV:
        for r in species.itertuples():
            rows.append({'compound_id': f'{r.short}_{ke:g}eV',
                         'precursor_mz': r.q1_mz,
                         'rt_min': r.rt_min,
                         'electron_ke_ev': ke})
    targets = pd.DataFrame(rows)

    table = core.write_sciex_mrmhr_table(
        targets, OUT, layout=LAYOUT, group=GROUP, tof_range=TOF_RANGE,
        accumulation_s=ACCUMULATION_S, qjet_dp=QJET_DP,
        collision_energy=COLLISION_ENERGY, ce_spread=CE_SPREAD,
        reaction_time_ms=REACTION_TIME_MS, fragmentation_mode='EAD',
        etc_percent=ETC_PERCENT, time_bins_to_sum=TIME_BINS_TO_SUM,
        rt_tolerance_s=RT_TOLERANCE_S)

    print(f'{len(targets)} targets = {len(species)} species '
          f'x {len(ELECTRON_ENERGIES_EV)} energies, Q1 on the '
          f'{Q1_ISOTOPE.replace("_", " ")} isotope\n')
    for r in species.itertuples():
        print(f'  {r.short:12s} Q1 {r.q1_mz:9.4f}  RT {r.rt_min:.3f} '
              f'+-{RT_TOLERANCE_S}s')
    print(f'\nenergies: {", ".join(f"{e:g}" for e in ELECTRON_ENERGIES_EV)} eV')
    print(f'\nwrote {OUT}\n')
    print(table.to_csv(sep='\t', index=False, lineterminator='\n'))


if __name__ == '__main__':
    main()
