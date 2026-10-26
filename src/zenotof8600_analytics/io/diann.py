"""DIA-NN report reader."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from ..proteins import build_peptide_id
from ..quant import compute_directlfq_pg_intensity
from ..schema import validate_df, validate_sample_info


_UNIMOD_TO_ALPHABASE = {
    (1, 'N-term'): 'Acetyl@Protein_N-term',
    (1, 'K'): 'Acetyl@K',
    (4, 'C'): 'Carbamidomethyl@C',
    (7, 'N'): 'Deamidated@N',
    (7, 'Q'): 'Deamidated@Q',
    (21, 'S'): 'Phospho@S',
    (21, 'T'): 'Phospho@T',
    (21, 'Y'): 'Phospho@Y',
    (35, 'M'): 'Oxidation@M',
}


_UNIMOD_RE = re.compile(r'\(UniMod:(\d+)\)')


def parse_diann_modified_sequence(modseq: str) -> tuple[str, str, str]:
    """Parse a DIA-NN `Modified.Sequence` string into AlphaBase columns."""
    sequence = _UNIMOD_RE.sub('', modseq)
    if '(' not in modseq:
        return sequence, '', ''

    mods: list[str] = []
    sites: list[str] = []
    pos = 0
    i = 0
    n = len(modseq)
    while i < n:
        ch = modseq[i]
        if ch == '(':
            j = modseq.find(')', i)
            inside = modseq[i + 1:j]
            m = re.fullmatch(r'UniMod:(\d+)', inside)
            if m:
                unimod_id = int(m.group(1))
                if pos == 0:
                    key = (unimod_id, 'N-term')
                    site = 0
                else:
                    aa = sequence[pos - 1]
                    key = (unimod_id, aa)
                    site = pos
                mod_name = _UNIMOD_TO_ALPHABASE.get(key)
                if mod_name is None:
                    warnings.warn(
                        f'DIA-NN: unknown modification UniMod:{unimod_id} at '
                        f'{key[1]} (site {site}) in {modseq!r}; dropped'
                    )
                else:
                    mods.append(mod_name)
                    sites.append(str(site))
            i = j + 1
        else:
            pos += 1
            i += 1

    return sequence, ';'.join(mods), ';'.join(sites)


_DIANN_READ_COLS = (
    'Run', 'Protein.Group', 'Protein.Names', 'Genes',
    'Stripped.Sequence', 'Modified.Sequence',
    'Precursor.Id', 'Precursor.Charge', 'Precursor.Mz', 'Ms1.Apex.Mz.Delta',
    'RT', 'Precursor.Normalised',
    'PG.MaxLFQ', 'Q.Value', 'PG.Q.Value',
)


def load_diann(
    path: str,
    sample_info: pd.DataFrame,
    *,
    qvalue_filter: float | None = 0.01,
    pg_intensity_method: str = 'directlfq',
    directlfq_num_cores: int = 8,
    directlfq_use_inmemory: bool = True,
) -> pd.DataFrame:
    """Load a DIA-NN parquet report into the canonical long DataFrame.

    Parameters:
      path:           DIA-NN report.parquet path
      sample_info:    canonical sample metadata. Only its `run` values are used
                      for filtering; other columns travel with downstream joins.
      qvalue_filter:  keep rows with `PG.Q.Value < qvalue_filter`. Pass None
                      to disable filtering.
      pg_intensity_method:
                      'directlfq' -> recompute `pg_intensity` via DirectLFQ
                                     from `Precursor.Normalised` (default,
                                     uniform across engines).
                      'maxlfq'    -> use DIA-NN's native `PG.MaxLFQ`.
      directlfq_num_cores:
                      number of cores for DirectLFQ when
                      `pg_intensity_method='directlfq'`. Default 8. Ignored
                      otherwise.
    """
    validate_sample_info(sample_info)

    raw = pd.read_parquet(path, columns=list(_DIANN_READ_COLS))

    keep_runs = set(sample_info['run'])
    have_runs = set(raw['Run'].unique())
    missing = keep_runs - have_runs
    if missing:
        raise ValueError(
            f'sample_info references runs absent from {path}: {sorted(missing)}'
        )
    raw = raw[raw['Run'].isin(keep_runs)].copy()
    if raw.empty:
        raise ValueError(f'No rows left after filtering by sample_info runs at {path}')

    if qvalue_filter is not None:
        n_before = len(raw)
        raw = raw[raw['PG.Q.Value'] < qvalue_filter].copy()
        pct = 100 * len(raw) / n_before if n_before else 0
        print(
            f'[diann] qvalue<{qvalue_filter}: kept {len(raw):,} of {n_before:,} '
            f'rows ({pct:.1f}%)'
        )

    parsed = raw['Modified.Sequence'].map(parse_diann_modified_sequence)
    raw['sequence'] = parsed.map(lambda t: t[0])
    raw['mods'] = parsed.map(lambda t: t[1])
    raw['mod_sites'] = parsed.map(lambda t: t[2])

    raw['peptide_id'] = [
        build_peptide_id(s, m, ms)
        for s, m, ms in zip(raw['sequence'], raw['mods'], raw['mod_sites'])
    ]

    df = pd.DataFrame({
        'run':              raw['Run'].astype(str),
        'protein_group':    raw['Protein.Group'].astype(str),
        'protein_names':    raw['Protein.Names'].fillna('').astype(str),
        'genes':            raw['Genes'].fillna('').astype(str),
        'sequence':         raw['sequence'],
        'mods':             raw['mods'],
        'mod_sites':        raw['mod_sites'],
        'precursor_id':     raw['Precursor.Id'].astype(str),
        'peptide_id':       raw['peptide_id'].astype(str),
        'charge':           raw['Precursor.Charge'].astype('Int64'),
        'mz':               (raw['Precursor.Mz'].astype(float)
                             + raw['Ms1.Apex.Mz.Delta'].astype(float).fillna(0.0)),
        'rt':               raw['RT'].astype(float),
        'precursor_intensity': raw['Precursor.Normalised'].astype(float),
        'pg_intensity':     raw['PG.MaxLFQ'].astype(float),
        'qvalue':           raw['PG.Q.Value'].astype(float),
        'score_engine':     raw['Q.Value'].astype(float),
        'engine':           pd.Categorical(['diann'] * len(raw), categories=['diann', 'peaks']),
    })

    for c in ('precursor_intensity', 'pg_intensity'):
        df.loc[df[c] == 0, c] = np.nan

    df['peptide_intensity'] = (df.groupby(['peptide_id', 'run'])['precursor_intensity']
                               .transform('sum'))
    df.loc[df['peptide_intensity'] == 0, 'peptide_intensity'] = np.nan

    if pg_intensity_method == 'directlfq':
        print(
            f'[diann] running DirectLFQ on {len(df):,} precursor rows '
            f'({df["protein_group"].nunique():,} protein groups, '
            f'{df["run"].nunique()} runs) ...'
        )
        df = df.reset_index(drop=True)
        df['pg_intensity'] = compute_directlfq_pg_intensity(
            df,
            num_cores=directlfq_num_cores,
            use_inmemory=directlfq_use_inmemory,
        ).values
        df.loc[df['pg_intensity'] == 0, 'pg_intensity'] = np.nan
    elif pg_intensity_method != 'maxlfq':
        raise ValueError(
            f"pg_intensity_method must be 'maxlfq' or 'directlfq', "
            f"got {pg_intensity_method!r}"
        )

    df.attrs['source_path'] = path
    df.attrs['qvalue_filter'] = qvalue_filter
    df.attrs['pg_intensity_method'] = pg_intensity_method
    df.reset_index(drop=True, inplace=True)
    validate_df(df)
    return df
