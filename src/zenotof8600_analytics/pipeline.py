"""High-level per-run summary orchestration (process_experiment)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .filters import filter_runs
from .schema import validate_df, validate_sample_info
from .sequences import count_missed_cleavages


def process_experiment(
    df,
    sample_info,
    *,
    group_col='condition2',
    hue_col=None,
    protease='trypsin',
    max_missed_cleavages=2,
    min_values_for_cv=3,
):
    """Per-run summary statistics on a canonical long DataFrame."""
    validate_df(df)
    validate_sample_info(sample_info)

    keys = ['run', 'engine']
    partition_cols = [group_col] + ([hue_col] if hue_col else [])

    cache_key = ('_run_stats', protease, max_missed_cleavages)
    if cache_key in df.attrs:
        run_stats = df.attrs[cache_key]
    else:
        run_stats = _compute_per_run_stats(df, protease, max_missed_cleavages)
        try:
            df.attrs[cache_key] = run_stats
        except Exception:
            pass

    si_pairs = pd.MultiIndex.from_frame(sample_info[keys])
    rs_pairs = pd.MultiIndex.from_frame(run_stats[keys])
    out = run_stats[rs_pairs.isin(si_pairs)].reset_index(drop=True)
    if out.empty:
        raise ValueError(
            'no runs in sample_info found in df; check (run, engine) pairs'
        )

    df_filtered = filter_runs(df, sample_info)
    detected = df_filtered[df_filtered['precursor_intensity'].notna()]

    df_part = df_filtered.merge(sample_info[keys + partition_cols],
                                on=keys, how='inner')

    pg_stats = (df_part.drop_duplicates(keys + partition_cols + ['protein_group'])
                       .groupby(partition_cols + ['protein_group'])['pg_intensity']
                       .agg(['mean', 'std', 'count']))
    pg_stats = pg_stats[pg_stats['count'] >= min_values_for_cv]
    pg_stats['cv'] = pg_stats['std'] / pg_stats['mean']
    pg_stats = pg_stats[np.isfinite(pg_stats['cv'])]
    pg20_per_partition = (
        (pg_stats['cv'] < 0.2)
        .groupby(level=list(range(len(partition_cols))))
        .sum().astype(int)
    )

    pr_stats = (df_part.groupby(partition_cols + ['precursor_id'])
                        ['precursor_intensity']
                        .agg(['mean', 'std', 'count']))
    pr_stats = pr_stats[pr_stats['count'] >= min_values_for_cv]
    pr_stats['cv'] = pr_stats['std'] / pr_stats['mean']
    pr_stats = pr_stats[np.isfinite(pr_stats['cv'])]
    pr20_per_partition = (
        (pr_stats['cv'] < 0.2)
        .groupby(level=list(range(len(partition_cols))))
        .sum().astype(int)
    )

    si_indexed = sample_info.set_index(keys)
    if len(partition_cols) == 1:
        run_to_partition = si_indexed[partition_cols[0]]
    else:
        run_to_partition = si_indexed[partition_cols].apply(tuple, axis=1)
    out_keys = list(zip(out['run'], out['engine']))
    out_parts = [run_to_partition.loc[k] for k in out_keys]
    out['PG20'] = [int(pg20_per_partition.get(p, 0)) for p in out_parts]
    out['Pr20'] = [int(pr20_per_partition.get(p, 0)) for p in out_parts]

    partition_keys_df = sample_info[keys + partition_cols]
    detected_with_part = detected.merge(partition_keys_df, on=keys, how='left')
    partition_totals = (detected_with_part.groupby(partition_cols)
                        .agg(total_peptides=('peptide_id', 'nunique'),
                             total_protein_groups=('protein_group', 'nunique'),
                             total_precursors=('precursor_id', 'nunique'))
                        .reset_index())

    out = out.merge(sample_info, on=keys, how='left')
    out = out.merge(partition_totals, on=partition_cols, how='left')
    return out


def _compute_per_run_stats(df, protease, max_missed_cleavages):
    """Compute per-run identification counts, total intensity, and MC distribution."""
    keys = ['run', 'engine']
    df = df.reset_index(drop=True)
    detected = df[df['precursor_intensity'].notna()]
    counts = (detected.groupby(keys, sort=False, observed=False).agg(
        precursor=('precursor_id', 'nunique'),
        peptide=('peptide_id', 'nunique'),
        protein_group=('protein_group', 'nunique'),
        total_intensity=('precursor_intensity', 'sum'),
    ).reset_index())
    counts['log2_total_intensity'] = np.log2(
        counts['total_intensity'].where(counts['total_intensity'] > 0)
    )

    if 'sequence' in df.columns:
        unique_seqs = df['sequence'].dropna().astype(str).unique()
        mc_lookup = {
            s: min(count_missed_cleavages(s, protease), max_missed_cleavages)
            for s in unique_seqs if s
        }
    else:
        mc_lookup = {}

    def _mc_per_run(g):
        det = g.loc[g['precursor_intensity'].notna(),
                    ['peptide_id', 'sequence']].drop_duplicates('peptide_id')
        det = det[det['sequence'].str.len() > 0]
        if det.empty:
            return pd.Series({f'MC{i}': 0.0
                              for i in range(max_missed_cleavages + 1)})
        mc_capped = det['sequence'].map(mc_lookup)
        dist = mc_capped.value_counts(normalize=True)
        return pd.Series({f'MC{i}': float(dist.get(i, 0.0))
                          for i in range(max_missed_cleavages + 1)})

    mc = (df.groupby(keys, group_keys=False, sort=False, observed=False)
            .apply(_mc_per_run, include_groups=False).reset_index())
    out = counts.merge(mc, on=keys, how='left')
    out['avg_MC'] = sum(out[f'MC{i}'] * i for i in range(max_missed_cleavages + 1))
    out['mc_rate'] = 1.0 - out['MC0']
    return out
