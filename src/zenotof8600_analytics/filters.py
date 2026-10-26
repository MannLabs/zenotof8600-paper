"""Row/column and run-level filtering of the canonical DataFrame."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def filter_runs(df: pd.DataFrame, sample_info: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose (run, engine) appears in sample_info."""
    keys = sample_info[['run', 'engine']].drop_duplicates()
    keys['_keep'] = True
    out = df.merge(keys, on=['run', 'engine'], how='left')
    out = out[out['_keep'] == True].drop(columns='_keep')
    return out.reset_index(drop=True)


def filter_outlier_runs(
    df,
    sample_info,
    *,
    cohort_col=None,
    p_good_threshold=0.5,
    min_runs_for_fit=4,
    plot=True,
    figsize=None,
    palette=None,
):
    """Drop technical-outlier runs via a 2-component Gaussian mixture on log10(precursor count per run).

    Parameters:
      cohort_col:        sample_info column defining per-cohort fits, or
                         None for a single global fit across all runs.
      p_good_threshold:  posterior probability cutoff for the "good"
                         cluster (default 0.5 = equal-posterior boundary).
      min_runs_for_fit:  cohorts with fewer runs than this keep everything.
                         Default 4.
      plot:              when True (default), draw one diagnostic panel per
                         cohort with the histogram, fitted Gaussians, and
                         threshold line.
    """
    from .plotting._style import PALETTE_SINGLE, _hide_top_right_spines
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture
    from scipy.optimize import brentq

    if cohort_col is not None and cohort_col not in sample_info.columns:
        raise ValueError(
            f'cohort_col={cohort_col!r} not in sample_info columns '
            f'({list(sample_info.columns)})'
        )

    detected = df
    if 'precursor_intensity' in df.columns:
        detected = df[df['precursor_intensity'].fillna(0) > 0]
    pr_per_run = detected.groupby('run', sort=False)['precursor_id'].nunique()

    if cohort_col is None:
        run_to_cohort = pd.Series(
            'all', index=sample_info['run'].astype(str), name='cohort')
    else:
        run_to_cohort = (sample_info.set_index('run')[cohort_col]
                                    .astype(str).rename('cohort'))
    cohorts = list(dict.fromkeys(run_to_cohort.loc[
        run_to_cohort.index.intersection(pr_per_run.index)
    ].values))

    if plot:
        if figsize is None:
            figsize = (max(7, 4 * len(cohorts)), 4)
        fig, axes = plt.subplots(1, len(cohorts), figsize=figsize,
                                 squeeze=False)
        axes = axes[0]
        if palette is None:
            palette = PALETTE_SINGLE

    keep_runs = []
    summary_rows = []
    for i, cohort in enumerate(cohorts):
        runs_in_cohort = [r for r in pr_per_run.index
                          if run_to_cohort.get(r) == cohort]
        vals = pr_per_run.loc[runs_in_cohort].values
        n_runs = len(vals)
        title_base = cohort if cohort != 'all' else 'all samples'

        if n_runs < min_runs_for_fit or vals.std() == 0:
            warnings.warn(
                f'filter_outlier_runs: cohort {cohort!r} has {n_runs} runs '
                f'(need >= {min_runs_for_fit}); keeping all.'
            )
            keep_runs.extend(runs_in_cohort)
            summary_rows.append({
                'cohort': cohort, 'n_total': n_runs, 'n_kept': n_runs,
                'n_dropped': 0, 'threshold': float('nan'), 'dropped_runs': [],
            })
            if plot:
                ax = axes[i]
                ax.hist(np.log10(vals + 1), bins=max(5, n_runs), alpha=0.5,
                        color=palette[i % len(palette)])
                ax.set_title(f'{title_base}\n(kept all — n={n_runs} too few to fit)',
                             fontsize=10)
                ax.set_xlabel('log10(precursor count)')
                _hide_top_right_spines(ax)
            continue

        x = np.log10(vals).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=0).fit(x)
        good_label = int(np.argmax(gmm.means_.ravel()))
        p_good = gmm.predict_proba(x)[:, good_label]
        try:
            boundary = brentq(
                lambda t: gmm.predict_proba([[t]])[0, good_label] - p_good_threshold,
                x.min(), x.max(),
            )
            threshold = 10 ** boundary
        except ValueError:
            boundary = float('nan')
            threshold = float('nan')
            p_good = np.ones(n_runs)
            warnings.warn(
                f'filter_outlier_runs: cohort {cohort!r} GMM has no '
                f'equal-posterior point in range; keeping all.'
            )

        mask = p_good >= p_good_threshold
        cohort_keep = [r for r, m in zip(runs_in_cohort, mask) if m]
        cohort_drop = [r for r, m in zip(runs_in_cohort, mask) if not m]
        keep_runs.extend(cohort_keep)

        if min(gmm.weights_) < 0.02:
            warnings.warn(
                f'filter_outlier_runs: cohort {cohort!r} minority cluster '
                f'weight {min(gmm.weights_):.2%} — distribution may not be '
                f'genuinely bimodal; review the diagnostic plot.'
            )

        summary_rows.append({
            'cohort': cohort,
            'n_total': n_runs,
            'n_kept': len(cohort_keep),
            'n_dropped': len(cohort_drop),
            'threshold': threshold,
            'dropped_runs': cohort_drop,
        })

        if plot:
            ax = axes[i]
            color = palette[i % len(palette)]
            ax.hist(x.ravel(), bins=min(40, max(5, n_runs)),
                    density=True, alpha=0.45, color=color)
            xx = np.linspace(x.min() - 0.05, x.max() + 0.05, 400)
            for k in range(2):
                var = gmm.covariances_[k, 0, 0]
                pdf = gmm.weights_[k] * (
                    np.exp(-0.5 * ((xx - gmm.means_[k, 0]) / np.sqrt(var)) ** 2)
                    / np.sqrt(2 * np.pi * var)
                )
                label = 'good' if k == good_label else 'fail'
                ax.plot(xx, pdf,
                        color='black' if k == good_label else '#888',
                        linestyle='-' if k == good_label else '--',
                        label=label)
            if not np.isnan(boundary):
                ax.axvline(boundary, color='red', linestyle='--', linewidth=1.2,
                           label=f'thr ≈ {int(threshold):,}')
            ax.set_title(
                f'{title_base}\n{len(cohort_keep)}/{n_runs} kept '
                f'({len(cohort_drop)} dropped)',
                fontsize=10,
            )
            ax.set_xlabel('log10(precursor count)')
            ax.legend(fontsize=8, frameon=False)
            _hide_top_right_spines(ax)

    if plot:
        axes[0].set_ylabel('density')
        plt.tight_layout()
        plt.show()

    summary = pd.DataFrame(summary_rows)

    df_filtered = df[df['run'].isin(keep_runs)].reset_index(drop=True)
    si_filtered = sample_info[sample_info['run'].isin(keep_runs)].reset_index(drop=True)

    n_total = len(sample_info)
    n_kept = len(si_filtered)
    print(f'filter_outlier_runs: kept {n_kept}/{n_total} runs '
          f'(dropped {n_total - n_kept}) across {len(cohorts)} cohort(s).')
    for row in summary_rows:
        for r in row['dropped_runs']:
            print(f'  dropped [{row["cohort"]}] {r}: '
                  f'{int(pr_per_run.loc[r]):,} precursors')

    return df_filtered, si_filtered, summary
