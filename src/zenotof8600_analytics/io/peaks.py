"""PEAKS output reader."""

from __future__ import annotations

import os
import re
import warnings

import numpy as np
import pandas as pd

from ..proteins import (
    build_peptide_id,
    group_proteins_by_shared_peptides,
    group_proteins_by_signature,
)
from ..quant import compute_directlfq_pg_intensity
from ..schema import validate_df, validate_sample_info


_PEAKS_MOD_LOOKUP = {
    'Carbamidomethylation':         ('Carbamidomethyl@C',     'C', None),
    'Oxidation (M)':                ('Oxidation@M',           'M', None),
    'Acetylation (Protein N-term)': ('Acetyl@Protein_N-term', 'M', 0),
    'Acetylation (K)':              ('Acetyl@K',              'K', None),
    'Phosphorylation (S)':          ('Phospho@S',             'S', None),
    'Phosphorylation (T)':          ('Phospho@T',             'T', None),
    'Phosphorylation (Y)':          ('Phospho@Y',             'Y', None),
    'Deamidation (N)':              ('Deamidated@N',          'N', None),
    'Deamidation (Q)':              ('Deamidated@Q',          'Q', None),
}

_PEAKS_MASS_TOL = 0.02


def _peaks_recover_mass(mass: float, sequence: str, site: int) -> tuple[str, int] | None:
    """Map (mass, AA, site) to AlphaBase."""
    aa = sequence[site - 1] if 0 < site <= len(sequence) else None
    near = lambda target: abs(mass - target) < _PEAKS_MASS_TOL
    if near(57.0215) and aa == 'C':
        return 'Carbamidomethyl@C', site
    if near(15.9949) and aa == 'M':
        return 'Oxidation@M', site
    if near(42.0106):
        if site == 1 and aa == 'M':
            return 'Acetyl@Protein_N-term', 0
        if aa == 'K':
            return 'Acetyl@K', site
    if near(79.9663) and aa in ('S', 'T', 'Y'):
        return f'Phospho@{aa}', site
    if near(0.9840):
        if aa == 'N': return 'Deamidated@N', site
        if aa == 'Q': return 'Deamidated@Q', site
    return None


_ANY_PAREN_RE = re.compile(r'\([^()]*\)')
_MASS_RE = re.compile(r'\(([+-]?\d+\.\d+)\)')


def parse_peaks_modified_peptide(peptide: str, ptm_string: str) -> tuple[str, str, str]:
    """Parse a PEAKS Peptide string + PTM column into AlphaBase columns."""
    sequence = _ANY_PAREN_RE.sub('', peptide)

    positions: list[int] = []
    masses: list[float | None] = []
    pos = 0
    i = 0
    n = len(peptide)
    while i < n:
        ch = peptide[i]
        if ch == '(':
            j = peptide.find(')', i)
            positions.append(pos if pos > 0 else 0)
            inside = peptide[i + 1:j]
            try:
                masses.append(float(inside))
            except ValueError:
                masses.append(None)
            i = j + 1
        else:
            pos += 1
            i += 1

    if pd.isna(ptm_string) or not str(ptm_string).strip():
        names: list[str] = []
    else:
        names = [n.strip() for n in str(ptm_string).split(';') if n.strip()]

    if not positions and not names:
        return sequence, '', ''

    if len(positions) != len(names):
        warnings.warn(
            f'PEAKS: annotation/PTM count mismatch in {peptide!r} | {ptm_string!r}: '
            f'{len(positions)} annotation(s) vs {len(names)} PTM name(s); dropped'
        )
        return sequence, '', ''

    mods: list[str] = []
    sites: list[str] = []
    for site, mass, name in zip(positions, masses, names):
        if name == 'more':
            if mass is None:
                continue
            recovered = _peaks_recover_mass(mass, sequence, site)
            if recovered is None:
                warnings.warn(
                    f"PEAKS: 'more' truncation with un-recoverable mass {mass:+.4f} "
                    f"at site {site} in {peptide!r}; dropped"
                )
                continue
            alphabase_name, corrected_site = recovered
            mods.append(alphabase_name)
            sites.append(str(corrected_site))
            continue

        entry = _PEAKS_MOD_LOOKUP.get(name)
        if entry is None:
            warnings.warn(
                f'PEAKS: unknown PTM name {name!r} in {peptide!r}; dropped'
            )
            continue
        alphabase_name, expected_aa, override = entry
        if override is not None:
            site = override
        if expected_aa == 'N-term':
            ok = (site == 0)
        elif site == 0:
            ok = True
        else:
            aa = sequence[site - 1] if 1 <= site <= len(sequence) else ''
            ok = (aa == expected_aa)
        if not ok:
            warnings.warn(
                f'PEAKS: {name!r} expects {expected_aa!r} but found '
                f'{sequence[site-1] if 0<site<=len(sequence) else "?"!r} at site {site} '
                f'in {peptide!r}; mod kept anyway'
            )
        mods.append(alphabase_name)
        sites.append(str(site))

    return sequence, ';'.join(mods), ';'.join(sites)


def _ten_lgp_to_qvalue(s: pd.Series) -> pd.Series:
    """Vectorised pseudo-q-value from PEAKS `-10LgP`."""
    out = pd.Series(1.0, index=s.index, dtype=float)
    out.loc[s >= 15] = 0.05
    out.loc[s >= 20] = 0.01
    out.loc[s >= 30] = 0.001
    out.loc[s >= 40] = 0.0001
    return out


def _attach_protein_csv_intensities(
    long: pd.DataFrame, protein_csv_path: str, runs: list[str]
) -> pd.DataFrame:
    """Replace `long['pg_intensity']` with MaxLFQ values from PEAKS' protein CSV."""
    pr = pd.read_csv(protein_csv_path)
    area_cols = {}
    for c in pr.columns:
        if c.endswith(' Area') and 'Group' not in c:
            run = c[:-len(' Area')]
            if run in runs:
                area_cols[run] = c
    missing = [r for r in runs if r not in area_cols]
    if missing:
        raise ValueError(
            f'protein CSV missing area columns for runs: {missing}'
        )

    bare_to_intensities: dict[str, dict[str, float]] = {}
    has_top = 'Top' in pr.columns
    if has_top:
        pr_sorted = pr.sort_values('Top', ascending=False)
    else:
        pr_sorted = pr
    for _, row in pr_sorted.iterrows():
        acc = str(row['Accession'])
        bare = acc.split('|', 1)[0] if '|' in acc else acc
        if bare in bare_to_intensities:
            continue
        bare_to_intensities[bare] = {
            run: (None if pd.isna(row[col]) else float(row[col]))
            for run, col in area_cols.items()
        }

    n_unmatched = 0
    pg_values: list[float | None] = []
    for pg, run in zip(long['protein_group'].astype(str), long['run']):
        first_bare = pg.split(';', 1)[0]
        m = bare_to_intensities.get(first_bare)
        if m is None:
            n_unmatched += 1
            pg_values.append(None)
        else:
            pg_values.append(m.get(run))
    if n_unmatched:
        warnings.warn(
            f'PEAKS MaxLFQ: {n_unmatched} peptide rows had no matching '
            f'protein-CSV row; pg_intensity left as NaN for those.'
        )
    long = long.copy()
    long['pg_intensity'] = pg_values
    return long


def _build_protein_csv_lookup(protein_csv_path: str) -> dict[str, dict]:
    """Build {individual_accession_with_pipe: {gene, protein_name}} from protein CSV."""
    pr = pd.read_csv(protein_csv_path, usecols=['Accession', 'Gene'])
    out: dict[str, dict] = {}
    for _, row in pr.iterrows():
        acc = str(row['Accession'])
        gene = '' if pd.isna(row['Gene']) else str(row['Gene'])
        if '|' in acc:
            _, name = acc.split('|', 1)
        else:
            name = ''
        out[acc] = {'gene': gene, 'protein_name': name}
    return out


def load_peaks(
    path: str,
    sample_info: pd.DataFrame,
    *,
    protein_csv: str | None = None,
    qvalue_filter: float | None = None,
    pg_intensity_method: str = 'directlfq',
    protein_grouping: str = 'cc',
    directlfq_num_cores: int = 8,
    directlfq_use_inmemory: bool = True,
) -> pd.DataFrame:
    """Load a PEAKS DIA result into the canonical long DataFrame.

    Parameters:
      path:           path to either `lfq.dia.features.csv` or
                      `lfq.dia.peptides.csv`. The other file is auto-located
                      as a sibling when present.
      sample_info:    canonical sample metadata. Only its `run` values are used
                      for filtering; other columns travel with downstream joins.
      protein_csv:    OPTIONAL path to PEAKS protein CSV. Used to populate
                      `genes` / `protein_names` via accession lookup, and is
                      REQUIRED when `pg_intensity_method='maxlfq'`.
      qvalue_filter:  threshold on the pseudo-qvalue mapped from -10LgP. None
                      disables filtering. Default: None.
      pg_intensity_method:
                      'directlfq' (default) -> compute pg_intensity via
                          DirectLFQ from precursor-level intensities.
                      'maxlfq' -> use the MaxLFQ-style values from the PEAKS
                          protein CSV. Raises if `protein_csv` is None.
                      'auto'   -> use 'maxlfq' if `protein_csv` is provided,
                          else fall back to 'directlfq' with a warning. The
                          resolved choice is stored in
                          `df.attrs['pg_intensity_method']`; the original
                          request is preserved in
                          `df.attrs['pg_intensity_method_requested']`.
      protein_grouping:
                      'cc' (default) -> connected-components grouping. Matches
                          PEAKS' own `protein.csv` exactly.
                      'signature' -> signature-based grouping (DIA-NN-style).
                          ~99.5% exact match with DIA-NN's actual output.
      directlfq_num_cores:
                      number of cores for DirectLFQ when
                      `pg_intensity_method='directlfq'`. Default 8. Ignored
                      otherwise.
    """
    validate_sample_info(sample_info)

    base = os.path.basename(path).lower()
    if 'features.csv' in base:
        features_csv = path if os.path.exists(path) else None
        candidate = path.replace('features.csv', 'peptides.csv')
        peptide_csv = candidate if os.path.exists(candidate) else None
    elif 'peptides.csv' in base:
        peptide_csv = path if os.path.exists(path) else None
        candidate = path.replace('peptides.csv', 'features.csv')
        features_csv = candidate if os.path.exists(candidate) else None
    else:
        raise ValueError(
            f'PEAKS path must end in "features.csv" or "peptides.csv", got {path!r}'
        )

    if features_csv is not None:
        raw = pd.read_csv(features_csv)
        required = {'Peptide', 'Accession', '-10LgP', 'PTM', 'z'}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(
                f'PEAKS features.csv at {features_csv} missing required columns: {sorted(missing)}'
            )
        area_suffix = ' Normalized Area'
        granularity = 'precursor'
    elif peptide_csv is not None:
        warnings.warn(
            f'PEAKS: features.csv not found alongside {path}; falling back to '
            f'peptides.csv (peptide-level granularity). Charge-resolved '
            f'precursors are unavailable.'
        )
        raw = pd.read_csv(peptide_csv)
        required = {'Peptide', 'Accession', '-10LgP', 'PTM'}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(
                f'PEAKS peptide CSV at {peptide_csv} missing required columns: {sorted(missing)}'
            )
        area_suffix = None
        granularity = 'peptide'
    else:
        raise ValueError(
            f'PEAKS: neither features.csv nor peptides.csv found at {path}'
        )

    keep_runs = list(sample_info['run'])
    area_cols: dict[str, str] = {}
    mz_cols: dict[str, str] = {}
    rt_cols: dict[str, str] = {}
    if area_suffix is not None:
        for c in raw.columns:
            if c.endswith(area_suffix) and 'Group' not in c:
                run = c[:-len(area_suffix)]
                if run in keep_runs:
                    area_cols[run] = c
            elif c.endswith(' m/z') and 'Group' not in c:
                run = c[:-len(' m/z')]
                if run in keep_runs:
                    mz_cols[run] = c
            elif c.endswith(' RT mean') and 'Group' not in c:
                run = c[:-len(' RT mean')]
                if run in keep_runs:
                    rt_cols[run] = c
    else:
        for c in raw.columns:
            if c.startswith('Area ') and 'Group' not in c:
                run = c[len('Area '):]
                if run in keep_runs:
                    area_cols[run] = c
    missing_runs = [r for r in keep_runs if r not in area_cols]
    if missing_runs:
        src = features_csv if features_csv else peptide_csv
        raise ValueError(f'sample_info references runs absent from {src}: {missing_runs}')

    parsed = [
        parse_peaks_modified_peptide(pep, ptm)
        for pep, ptm in zip(raw['Peptide'].astype(str), raw['PTM'])
    ]
    raw['sequence'] = [t[0] for t in parsed]
    raw['mods'] = [t[1] for t in parsed]
    raw['mod_sites'] = [t[2] for t in parsed]

    accession_sets = [
        str(s).split(':') for s in raw['Accession'].dropna().unique()
    ]
    if protein_grouping == 'cc':
        full_acc_to_group = group_proteins_by_shared_peptides(accession_sets)
    elif protein_grouping == 'signature':
        full_acc_to_group = group_proteins_by_signature(accession_sets)
    else:
        raise ValueError(
            f"protein_grouping must be 'cc' or 'signature', got {protein_grouping!r}"
        )

    if protein_csv is not None:
        csv_lookup = _build_protein_csv_lookup(protein_csv)
    else:
        warnings.warn(
            'PEAKS: protein_csv not provided; `genes` will be empty. '
            'Pass protein_csv=... to populate gene names.'
        )
        csv_lookup = {}

    def _gene_name_for(full_acc: str) -> tuple[str, str]:
        info = csv_lookup.get(full_acc)
        if info is not None:
            return info['gene'], info['protein_name']
        if '|' in full_acc:
            _, n = full_acc.split('|', 1)
            return '', n
        return '', ''

    def _row_to_group_strings(acc_string) -> tuple[str, str, str]:
        if pd.isna(acc_string) or not str(acc_string):
            return '', '', ''
        accs = str(acc_string).split(':')
        members = full_acc_to_group.get(accs[0])
        if members is None:
            members = tuple(accs)
        bares: list[str] = []
        names: list[str] = []
        genes: list[str] = []
        for m in members:
            bare = m.split('|', 1)[0] if '|' in m else m
            gene, name = _gene_name_for(m)
            bares.append(bare)
            names.append(name)
            genes.append(gene)
        return ';'.join(bares), ';'.join(names), ';'.join(genes)

    parsed = raw['Accession'].map(_row_to_group_strings)
    raw['protein_group'] = parsed.map(lambda t: t[0])
    raw['protein_names'] = parsed.map(lambda t: t[1])
    raw['genes'] = parsed.map(lambda t: t[2])

    raw['peptide_id'] = [
        build_peptide_id(s, m, ms)
        for s, m, ms in zip(raw['sequence'], raw['mods'], raw['mod_sites'])
    ]
    if granularity == 'precursor':
        raw['precursor_id'] = raw['peptide_id'] + '_z' + raw['z'].astype(int).astype(str)
        raw['_charge'] = raw['z'].astype(int)
    else:
        raw['precursor_id'] = raw['peptide_id']
        raw['_charge'] = pd.array([pd.NA] * len(raw), dtype='Int64')

    keep_meta = [
        'sequence', 'mods', 'mod_sites',
        'protein_group', 'protein_names', 'genes',
        'precursor_id', 'peptide_id', '_charge',
        'Accession', '-10LgP',
    ]
    melt_cols = list(area_cols.values())
    long = raw[keep_meta + melt_cols].melt(
        id_vars=keep_meta, value_vars=melt_cols,
        var_name='_area_col', value_name='precursor_intensity',
    )
    col_to_run = {col: run for run, col in area_cols.items()}
    long['run'] = long['_area_col'].map(col_to_run)
    long = long.drop(columns='_area_col')
    long.loc[long['precursor_intensity'] == 0, 'precursor_intensity'] = np.nan

    for cols, name in ((mz_cols, 'mz'), (rt_cols, 'rt')):
        if not cols:
            long[name] = np.nan
            continue
        melted = (raw[['precursor_id'] + list(cols.values())]
                  .melt(id_vars=['precursor_id'], value_vars=list(cols.values()),
                        var_name='_col', value_name=name))
        melted['run'] = melted['_col'].map({col: run for run, col in cols.items()})
        long = long.merge(melted[['precursor_id', 'run', name]],
                          on=['precursor_id', 'run'], how='left')
        long[name] = pd.to_numeric(long[name], errors='coerce')
        long.loc[long[name] == 0, name] = np.nan

    long['qvalue'] = _ten_lgp_to_qvalue(long['-10LgP'])
    if qvalue_filter is not None:
        n_before = len(long)
        long = long[long['qvalue'] < qvalue_filter].copy()
        pct = 100 * len(long) / n_before if n_before else 0
        print(
            f'[peaks] qvalue<{qvalue_filter}: kept {len(long):,} of {n_before:,} '
            f'rows ({pct:.1f}%)'
        )

    long = long.reset_index(drop=True)

    long['peptide_intensity'] = (long.groupby(['peptide_id', 'run'])['precursor_intensity']
                                  .transform('sum'))
    long.loc[long['peptide_intensity'] == 0, 'peptide_intensity'] = np.nan

    requested_pg_method = pg_intensity_method
    if pg_intensity_method == 'auto':
        if protein_csv is not None:
            pg_intensity_method = 'maxlfq'
        else:
            warnings.warn(
                f"PEAKS pg_intensity_method='auto': proteins.csv not found "
                f"alongside {path}; falling back to DirectLFQ.",
                stacklevel=2,
            )
            pg_intensity_method = 'directlfq'

    if pg_intensity_method == 'directlfq':
        print(
            f'[peaks] running DirectLFQ on {len(long):,} precursor rows '
            f'({long["protein_group"].nunique():,} protein groups, '
            f'{long["run"].nunique()} runs) ...'
        )
        pg_intensity = compute_directlfq_pg_intensity(
            long,
            num_cores=directlfq_num_cores,
            use_inmemory=directlfq_use_inmemory,
        )
        long['pg_intensity'] = pg_intensity.values
    elif pg_intensity_method == 'maxlfq':
        if protein_csv is None:
            raise ValueError(
                "pg_intensity_method='maxlfq' requires protein_csv= argument "
                "(MaxLFQ values are read from PEAKS' protein CSV). Pass "
                "pg_intensity_method='auto' to fall back to DirectLFQ when "
                "proteins.csv is missing."
            )
        print(f'[peaks] reading MaxLFQ from {protein_csv}')
        long = _attach_protein_csv_intensities(long, protein_csv, list(area_cols.keys()))
    else:
        raise ValueError(
            f"pg_intensity_method must be 'directlfq', 'maxlfq', or 'auto', "
            f"got {requested_pg_method!r}"
        )

    long.loc[long['pg_intensity'] == 0, 'pg_intensity'] = np.nan

    df = pd.DataFrame({
        'run':                long['run'].astype(str),
        'protein_group':      long['protein_group'].astype(str),
        'protein_names':      long['protein_names'].astype(str),
        'genes':              long['genes'].astype(str),
        'sequence':           long['sequence'].astype(str),
        'mods':               long['mods'].astype(str),
        'mod_sites':          long['mod_sites'].astype(str),
        'precursor_id':       long['precursor_id'].astype(str),
        'peptide_id':         long['peptide_id'].astype(str),
        'charge':             long['_charge'].astype('Int64'),
        'mz':                 long['mz'].astype(float),
        'rt':                 long['rt'].astype(float),
        'precursor_intensity': long['precursor_intensity'].astype(float),
        'peptide_intensity':  long['peptide_intensity'].astype(float),
        'pg_intensity':       long['pg_intensity'].astype(float),
        'qvalue':             long['qvalue'].astype(float),
        'score_engine':       long['-10LgP'].astype(float),
        'engine':             pd.Categorical(['peaks'] * len(long), categories=['diann', 'peaks']),
    })

    df.attrs['source_path'] = path
    df.attrs['features_csv'] = features_csv
    df.attrs['peptide_csv'] = peptide_csv
    df.attrs['protein_csv'] = protein_csv
    df.attrs['qvalue_filter'] = qvalue_filter
    df.attrs['pg_intensity_method'] = pg_intensity_method
    df.attrs['pg_intensity_method_requested'] = requested_pg_method
    df.attrs['protein_grouping'] = protein_grouping
    df.reset_index(drop=True, inplace=True)
    validate_df(df)
    return df
