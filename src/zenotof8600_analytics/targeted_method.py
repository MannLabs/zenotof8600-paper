"""Targeted-method export: SCIEX OS Zeno MRM-HR mass tables."""
from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    'SCIEX_MRMHR_LAYOUTS',
    'SCIEX_MRMHR_FORMATS',
    'Q1_ISOTOPE_CHOICES',
    'CE_RULES',
    'precursor_q1_mz',
    'sciex_mrmhr_table',
    'write_sciex_mrmhr_table',
]

SCIEX_MRMHR_LAYOUTS: dict[str, tuple[str, ...]] = {
    'cid': (
        'Compound ID', 'Group name', 'Precursor Ion (Da)',
        'TOF Start Mass (Da)', 'TOF Stop Mass (Da)',
        'Accumulation Time (sec)', 'QJet DP (V)', 'Collision energy (V)',
        'CE Spread (V)',
    ),
    'cid_scheduled': (
        'Compound ID', 'Group name', 'Precursor Ion (Da)',
        'TOF Start Mass (Da)', 'TOF Stop Mass (Da)',
        'Accumulation Time (sec)', 'QJet DP (V)', 'Collision energy (V)',
        'CE Spread (V)', 'Retention time (min)',
        'Retention time tolerance (sec)',
    ),
    'ead': (
        'Compound ID', 'Group name', 'Precursor Ion (Da)',
        'TOF Start Mass (Da)', 'TOF Stop Mass (Da)', 'Reaction time (ms)',
        'Accumulation Time (sec)', 'QJet DP (V)', 'Collision energy (V)',
        'CE Spread (V)', 'Fragmentation mode', 'Electron KE (eV)', 'ETC (%)',
    ),
    'ead_scheduled': (
        'Compound ID', 'Group name', 'Precursor Ion (Da)',
        'TOF Start Mass (Da)', 'TOF Stop Mass (Da)', 'Reaction time (ms)',
        'Accumulation Time (sec)', 'QJet DP (V)', 'Collision energy (V)',
        'CE Spread (V)', 'Fragmentation mode', 'Electron KE (eV)', 'ETC (%)',
        'Retention time (min)', 'Retention time tolerance (sec)',
        'Time bins to sum',
    ),
}

SCIEX_MRMHR_FORMATS: dict[str, str] = {
    'Precursor Ion (Da)': '{:.5f}',
    'TOF Start Mass (Da)': '{:.5f}',
    'TOF Stop Mass (Da)': '{:.5f}',
    'Reaction time (ms)': '{:.1f}',
    'Accumulation Time (sec)': '{:.4f}',
    'QJet DP (V)': '{:g}',
    'Collision energy (V)': '{:.1f}',
    'CE Spread (V)': '{:g}',
    'Retention time (min)': '{:.2f}',
    'Retention time tolerance (sec)': '{:g}',
    'Electron KE (eV)': '{:.3f}',
    'ETC (%)': '{:.1f}',
    'Time bins to sum': '{:g}',
}

Q1_ISOTOPE_CHOICES = ('monoisotope', 'most_abundant')

CE_RULES: dict[str, Callable[[float], float]] = {
    'sciex_2plus': lambda mz: 0.0625 * mz - 3.5,
    'linear_0048': lambda mz: 0.048 * mz - 2.0,
}

_REQUIRED = ('compound_id', 'precursor_mz')
_PER_ROW = ('group', 'collision_energy', 'accumulation_s', 'qjet_dp',
            'ce_spread', 'rt_min', 'rt_tolerance_s', 'reaction_time_ms',
            'tof_start', 'tof_stop', 'fragmentation_mode', 'electron_ke_ev',
            'etc_percent', 'time_bins_to_sum')


def precursor_q1_mz(precursors: pd.DataFrame, *,
                    isotope: str = 'monoisotope',
                    max_isotope: int = 8) -> pd.DataFrame:
    """Q1 m/z per precursor, at the monoisotope or the envelope apex."""
    if isotope not in Q1_ISOTOPE_CHOICES:
        raise ValueError(f'isotope must be one of {Q1_ISOTOPE_CHOICES}, '
                         f'got {isotope!r}')
    from alphabase.constants.atom import MASS_ISOTOPE
    from alphabase.constants.isotope import IsotopeDistribution
    from alphabase.peptide.precursor import (get_mod_seq_formula,
                                             update_precursor_mz)

    df = precursors.copy()
    if 'nAA' not in df.columns:
        df['nAA'] = df['sequence'].str.len()
    update_precursor_mz(df)

    if isotope == 'monoisotope':
        df['q1_isotope_index'] = 0
        df['q1_mz'] = df['precursor_mz']
        return df

    dist = IsotopeDistribution()
    cache: dict[tuple, int] = {}
    idx = []
    for seq, mods in zip(df['sequence'], df['mods'].fillna('')):
        formula = tuple(get_mod_seq_formula(seq, mods))
        if formula not in cache:
            abundances, mono_at = dist.calc_formula_distribution(list(formula))
            above = np.asarray(abundances, dtype=float)[mono_at:][:max_isotope]
            cache[formula] = int(np.argmax(above))
        idx.append(cache[formula])
    df['q1_isotope_index'] = idx
    df['q1_mz'] = (df['precursor_mz']
                   + df['q1_isotope_index'] * MASS_ISOTOPE / df['charge'])
    return df


def _resolve_ce(rule, mz):
    if rule is None:
        raise ValueError(
            'collision_energy is required: pass a number, a callable of m/z, '
            f'or one of {sorted(CE_RULES)} — there is no default')
    if isinstance(rule, str):
        try:
            rule = CE_RULES[rule]
        except KeyError:
            raise ValueError(f'unknown CE rule {rule!r}; '
                             f'known: {sorted(CE_RULES)}') from None
    return float(rule(mz)) if callable(rule) else float(rule)


def sciex_mrmhr_table(targets, *, layout: str = 'cid',
                      group: str = 'Group 1',
                      tof_range: tuple[float, float] = (150.0, 3000.0),
                      accumulation_s: float = 0.05,
                      qjet_dp: float = 20,
                      collision_energy=None,
                      ce_spread: float = 0,
                      rt_tolerance_s: float = 30,
                      reaction_time_ms=None,
                      fragmentation_mode=None,
                      electron_ke_ev=None,
                      etc_percent=None,
                      time_bins_to_sum=None) -> pd.DataFrame:
    """Build the SCIEX OS mass table as a frame of already-formatted strings."""
    try:
        columns = SCIEX_MRMHR_LAYOUTS[layout]
    except KeyError:
        raise ValueError(f'unknown layout {layout!r}; '
                         f'known: {sorted(SCIEX_MRMHR_LAYOUTS)}') from None

    df = pd.DataFrame(list(targets)) if not isinstance(targets, pd.DataFrame) \
        else targets.copy()
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f'targets is missing {missing}')
    if df['compound_id'].duplicated().any():
        dup = df.loc[df['compound_id'].duplicated(), 'compound_id'].tolist()
        raise ValueError(f'compound_id must be unique; repeated: {dup}')

    scheduled = 'Retention time (min)' in columns
    if scheduled and 'rt_min' not in df.columns:
        raise ValueError(f"layout {layout!r} is scheduled, so targets needs an "
                         "'rt_min' column")

    defaults = {'group': group, 'accumulation_s': accumulation_s,
                'qjet_dp': qjet_dp, 'ce_spread': ce_spread,
                'rt_tolerance_s': rt_tolerance_s,
                'reaction_time_ms': reaction_time_ms,
                'fragmentation_mode': fragmentation_mode,
                'electron_ke_ev': electron_ke_ev,
                'etc_percent': etc_percent,
                'time_bins_to_sum': time_bins_to_sum,
                'tof_start': tof_range[0], 'tof_stop': tof_range[1]}

    def _absent(v):
        return v is None or (isinstance(v, float) and np.isnan(v))

    def get(row, key):
        v = row.get(key, None)
        return defaults[key] if _absent(v) else v

    rows = []
    for _, r in df.iterrows():
        row = r.to_dict()
        mz = float(row['precursor_mz'])
        ce_raw = row.get('collision_energy')
        if _absent(ce_raw):
            ce_raw = collision_energy
        ce = _resolve_ce(ce_raw, mz)
        values = {
            'Compound ID': str(row['compound_id']),
            'Group name': str(get(row, 'group')),
            'Precursor Ion (Da)': mz,
            'TOF Start Mass (Da)': float(get(row, 'tof_start')),
            'TOF Stop Mass (Da)': float(get(row, 'tof_stop')),
            'Reaction time (ms)': get(row, 'reaction_time_ms'),
            'Accumulation Time (sec)': float(get(row, 'accumulation_s')),
            'QJet DP (V)': float(get(row, 'qjet_dp')),
            'Collision energy (V)': ce,
            'CE Spread (V)': float(get(row, 'ce_spread')),
            'Fragmentation mode': get(row, 'fragmentation_mode'),
            'Electron KE (eV)': get(row, 'electron_ke_ev'),
            'ETC (%)': get(row, 'etc_percent'),
            'Retention time (min)': row.get('rt_min'),
            'Retention time tolerance (sec)': float(get(row, 'rt_tolerance_s')),
            'Time bins to sum': get(row, 'time_bins_to_sum'),
        }
        out = {}
        for c in columns:
            v = values[c]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out[c] = ''
            elif c in SCIEX_MRMHR_FORMATS and not isinstance(v, str):
                out[c] = SCIEX_MRMHR_FORMATS[c].format(v)
            else:
                out[c] = str(v)
        rows.append(out)
    return pd.DataFrame(rows, columns=list(columns))


def write_sciex_mrmhr_table(targets, path, *, header: bool = True,
                            **kwargs) -> pd.DataFrame:
    """Write :func:`sciex_mrmhr_table` as a tab-separated file."""
    table = sciex_mrmhr_table(targets, **kwargs)
    text = table.to_csv(sep='\t', index=False, header=header,
                        lineterminator='\n')
    with open(path, 'w', encoding='utf8', newline='') as fh:
        fh.write(text)
    return table
