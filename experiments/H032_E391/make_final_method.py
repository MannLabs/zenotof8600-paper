"""Final assay method: EAD 15 eV, six heavy and six light targets."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg
import pathlib

import pandas as pd

import zenotof8600_analytics as core

from make_energy_sweep_method import RETENTION_MIN
from score_fragment_coverage import (CHARGE, LABEL, LABEL_SITE, MET_SITES,
                                     PHOSPHO_SITE, SEQUENCE)

OUT = pathlib.Path(cfg.data('H032_E391', 'mrmhr_final_ead15_E391.txt'))

ELECTRON_KE_EV = 15.0
LAYOUT = 'ead_scheduled'
GROUP = 'Group 1'
Q1_ISOTOPE = 'most_abundant'
TOF_RANGE = (150.0, 3000.0)
ACCUMULATION_S = 0.050
QJET_DP = 20
COLLISION_ENERGY = 12
CE_SPREAD = 0
REACTION_TIME_MS = 15.0
ETC_PERCENT = 100.0
TIME_BINS_TO_SUM = 8
RT_TOLERANCE_S = 8


def all_species():
    """Light and heavy, phospho and not, three oxidation states each."""
    rows = []
    for form in ('heavy', 'light'):
        for phospho in (False, True):
            for n_ox in (0, 1, 2):
                mods = [(s, 'Oxidation@M') for s in MET_SITES[:n_ox]]
                if phospho:
                    mods.append((PHOSPHO_SITE, 'Phospho@S'))
                if form == 'heavy':
                    mods.append((LABEL_SITE, LABEL))
                mods.sort()
                name = 'pS129' if phospho else 'unmod'
                suffix = {0: '', 1: '_ox1', 2: '_ox2'}[n_ox]
                rows.append({
                    'rt_key': f'heavy {name}'
                              + {0: '', 1: ' +16', 2: ' +32'}[n_ox],
                    'compound_id': f'aSyn_{form[0]}_{name}{suffix}',
                    'sequence': SEQUENCE,
                    'mods': ';'.join(m for _, m in mods),
                    'mod_sites': ';'.join(str(s) for s, _ in mods),
                    'charge': CHARGE})
    return pd.DataFrame(rows)


def main():
    core.raw.register_sil_modifications()
    sp = core.precursor_q1_mz(all_species(), isotope=Q1_ISOTOPE)
    sp['rt_min'] = sp['rt_key'].map(RETENTION_MIN)
    if sp['rt_min'].isna().any():
        raise SystemExit(f'no retention time for '
                         f'{sp.loc[sp["rt_min"].isna(), "rt_key"].tolist()}')

    targets = sp.rename(columns={'q1_mz': 'precursor_mz'})[
        ['compound_id', 'precursor_mz', 'rt_min']]
    table = core.write_sciex_mrmhr_table(
        targets, OUT, layout=LAYOUT, group=GROUP, tof_range=TOF_RANGE,
        accumulation_s=ACCUMULATION_S, qjet_dp=QJET_DP,
        collision_energy=COLLISION_ENERGY, ce_spread=CE_SPREAD,
        reaction_time_ms=REACTION_TIME_MS, fragmentation_mode='EAD',
        electron_ke_ev=ELECTRON_KE_EV, etc_percent=ETC_PERCENT,
        time_bins_to_sum=TIME_BINS_TO_SUM, rt_tolerance_s=RT_TOLERANCE_S)

    print(f'{len(targets)} targets, EAD {ELECTRON_KE_EV:g} eV, '
          f'{ACCUMULATION_S * 1000:g} ms accumulation\n')
    for r in sp.itertuples():
        print(f'  {r.compound_id:20s} Q1 {r.q1_mz:9.4f}  RT {r.rt_min:.2f} '
              f'+-{RT_TOLERANCE_S}s')
    conc = 3 * 2
    cycle = core.ZENOTOF_8600.fixed_s + conc * (ACCUMULATION_S + 0.0043)
    print(f'\nconcurrency {conc} targets -> cycle {cycle:.3f} s '
          f'-> {2.2 / cycle * 1.70:.1f} points per peak on a 2.2 s FWHM')
    print(f'\nwrote {OUT}  ({len(table)} data rows)\n')
    print(table.to_csv(sep='\t', index=False, lineterminator='\n'))


if __name__ == '__main__':
    main()
