"""PCA scores plot."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._style import (PALETTE_SINGLE, _empty_plot_with_message, _hide_top_right_spines, _filter_pivot_by_validity, _confidence_ellipse, _LEVEL_COLS, _resolve_panel_size)
from ..filters import filter_runs


def plot_pca(
    df,
    sample_info,
    *,
    level='protein',
    color_by='condition2',
    style_by=None,
    label_by='replicate',
    min_valid_fraction=0.5,
    min_valid_fraction_in_any_group=None,
    valid_group_col=None,
    log_transform=True,
    scale=True,
    n_components=2,
    pc_x=1,
    pc_y=2,
    batch_correct=None,
    palette=None,
    figsize=None,
    point_size=15,
    point_alpha=0.85,
    show_labels=False,
    show_ellipses=True,
    ellipse_confidence=0.95,
    ellipse_alpha=0.40,
    label_fontsize=10,
    tick_fontsize=10,
    legend_fontsize=8,
    title=None,
    title_fontsize=11,
):
    """PCA on the (run x feature) wide pivot at the chosen quantitation level.

    Parameters:
      level: 'precursor' | 'peptide' | 'protein'  (default 'protein').
      min_valid_fraction: a feature must have a valid value in at least this
        fraction of all runs (default 0.5).
      min_valid_fraction_in_any_group: if set (e.g. 0.75), a feature passes
        when at least one group has this fraction of valid replicates. Useful
        to rescue features that are only quantified in some conditions but
        well-measured there. Off by default. Combined with `min_valid_fraction`
        as AND — a feature has to clear every active filter.
      valid_group_col: which sample_info column defines the groups for the
        per-group filter. Defaults to `color_by`.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if level not in _LEVEL_COLS:
        raise ValueError(
            f"level must be one of {sorted(_LEVEL_COLS)}, got {level!r}"
        )
    id_col, intensity_col = _LEVEL_COLS[level]

    if pc_x < 1 or pc_y < 1 or pc_x == pc_y:
        raise ValueError(
            f'pc_x and pc_y must be distinct, >=1 (got pc_x={pc_x}, pc_y={pc_y})'
        )
    n_components = max(n_components, pc_x, pc_y)
    x_col = f'PC{pc_x}'
    y_col = f'PC{pc_y}'

    if df['engine'].nunique() > 1:
        raise ValueError('plot_pca: df contains multiple engines; '
                         'use core.split_by_engine first')

    df = filter_runs(df, sample_info)

    pivot = df.pivot_table(
        index='run', columns=id_col, values=intensity_col, aggfunc='first',
    )

    n_before = pivot.shape[1]
    pivot = _filter_pivot_by_validity(
        pivot, sample_info,
        min_valid_fraction=min_valid_fraction,
        min_valid_fraction_in_any_group=min_valid_fraction_in_any_group,
        group_col=valid_group_col or color_by,
        runs_axis='index',
        log_prefix=f'[pca:{level}]',
    )
    if pivot.shape[1] == 0 or pivot.shape[0] < 2:
        warnings.warn(
            f'plot_pca: not enough data — {pivot.shape[1]} {level}-level '
            f'features and {pivot.shape[0]} runs after filtering. PCA needs '
            f'>= 2 runs and >= 1 feature; rendering an empty plot.'
        )
        fig, ax = _empty_plot_with_message(
            f'PCA: insufficient data\n'
            f'({pivot.shape[1]} features x {pivot.shape[0]} runs)',
            figsize=_resolve_panel_size(figsize), title=title)
        return fig, ax, {
            'pca': None, 'transformed': pd.DataFrame(),
            'variance_explained': np.zeros(n_components),
            'loadings': pd.DataFrame(), 'sample_info': sample_info,
            'matrix': pivot,
        }
    if pivot.isna().any().any():
        pivot = pivot.fillna(pivot.min())

    if log_transform:
        pivot = np.log2(pivot + 1)

    pivot_corrected = pivot.copy()
    correction_label = 'none'
    if batch_correct == 'median':
        sample_med = pivot_corrected.median(axis=1)
        global_med = float(np.nanmedian(pivot_corrected.values))
        pivot_corrected = pivot_corrected.sub(sample_med, axis=0) + global_med
        correction_label = 'median centering'
    elif batch_correct == 'zscore':
        means = pivot_corrected.mean(axis=1)
        stds = pivot_corrected.std(axis=1)
        pivot_corrected = pivot_corrected.sub(means, axis=0).div(stds, axis=0)
        correction_label = 'z-score per sample'
    elif batch_correct == 'total':
        totals = pivot_corrected.sum(axis=1)
        global_total = totals.mean()
        pivot_corrected = pivot_corrected.div(totals, axis=0) * global_total
        correction_label = 'total intensity normalisation'
    elif batch_correct == 'quantile':
        sorted_vals = np.sort(pivot_corrected.values, axis=0)
        ref = sorted_vals.mean(axis=1)
        ranks = pivot_corrected.rank(method='min').values.astype(int) - 1
        pivot_corrected = pd.DataFrame(
            ref[ranks], index=pivot_corrected.index, columns=pivot_corrected.columns,
        )
        correction_label = 'quantile normalisation'
    elif batch_correct in ('subtract_pc1', 'subtract_pc1_rescale'):
        col_mean = pivot_corrected.mean()
        centered = pivot_corrected - col_mean
        pca_tmp = PCA(n_components=min(centered.shape))
        pca_tmp.fit(centered)
        pc1_scores = pca_tmp.transform(centered)[:, 0]
        pc1_loadings = pca_tmp.components_[0, :]
        corrected = centered.values - np.outer(pc1_scores, pc1_loadings)
        if batch_correct == 'subtract_pc1_rescale':
            corrected = corrected + col_mean.values
        pivot_corrected = pd.DataFrame(
            corrected, index=pivot_corrected.index, columns=pivot_corrected.columns,
        )
        correction_label = batch_correct.replace('_', ' ')
    elif batch_correct is not None:
        raise ValueError(f'unknown batch_correct: {batch_correct!r}')

    matrix = StandardScaler().fit_transform(pivot_corrected) if scale else pivot_corrected.values
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(matrix)
    var = pca.explained_variance_ratio_ * 100

    pca_df = pd.DataFrame(
        transformed,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=pivot_corrected.index,
    ).reset_index().merge(sample_info, on='run', how='left')

    if palette is None:
        palette = PALETTE_SINGLE
    data_groups = set(pca_df[color_by].astype(str).unique())
    if isinstance(palette, dict):
        groups = [str(g) for g in palette.keys() if str(g) in data_groups]
        missing = [g for g in data_groups if g not in groups]
        groups += missing
        color_map = {g: palette.get(g, palette.get(type(next(iter(palette)))(g), '#bdbdbd'))
                     for g in groups}
        for g in missing:
            color_map[g] = '#bdbdbd'
    else:
        groups = list(pca_df[color_by].astype(str).unique())
        color_map = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    if style_by is None:
        for grp in groups:
            sub = pca_df[pca_df[color_by].astype(str) == grp]
            ax.scatter(sub[x_col], sub[y_col], s=point_size, alpha=point_alpha,
                       c=color_map[grp], label=grp,
                       edgecolors='black', linewidth=1)
            if show_ellipses and len(sub) >= 3:
                try:
                    import matplotlib.colors as _mcolors
                    fill_rgba = list(_mcolors.to_rgba(color_map[grp]))
                    fill_rgba[3] = ellipse_alpha
                    edge_rgba = list(_mcolors.to_rgba(color_map[grp]))
                    edge_rgba[3] = 1.0
                    _confidence_ellipse(
                        sub[x_col].values, sub[y_col].values, ax,
                        n_std=float(np.sqrt(-2 * np.log(1 - ellipse_confidence))),
                        facecolor=fill_rgba,
                        edgecolor=edge_rgba, linewidth=1.2,
                    )
                except (ValueError, np.linalg.LinAlgError):
                    pass
    else:
        markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', '*']
        styles = list(pca_df[style_by].astype(str).unique())
        marker_map = {s: markers[i % len(markers)] for i, s in enumerate(styles)}
        for grp in groups:
            for sty in styles:
                sub = pca_df[(pca_df[color_by].astype(str) == grp)
                             & (pca_df[style_by].astype(str) == sty)]
                if sub.empty:
                    continue
                ax.scatter(sub[x_col], sub[y_col], s=point_size,
                           alpha=point_alpha, c=color_map[grp],
                           marker=marker_map[sty],
                           edgecolors='black', linewidth=1,
                           label=f'{grp} / {sty}')
            if show_ellipses:
                grp_sub = pca_df[pca_df[color_by].astype(str) == grp]
                if len(grp_sub) >= 3:
                    try:
                        import matplotlib.colors as _mcolors
                        fill_rgba = list(_mcolors.to_rgba(color_map[grp]))
                        fill_rgba[3] = ellipse_alpha
                        edge_rgba = list(_mcolors.to_rgba(color_map[grp]))
                        edge_rgba[3] = 1.0
                        _confidence_ellipse(
                            grp_sub[x_col].values, grp_sub[y_col].values, ax,
                            n_std=float(np.sqrt(-2 * np.log(1 - ellipse_confidence))),
                            facecolor=fill_rgba,
                            edgecolor=edge_rgba, linewidth=1.2,
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        pass
    if show_labels and label_by in pca_df.columns:
        for _, row in pca_df.iterrows():
            ax.annotate(
                str(row[label_by]),
                (row[x_col], row[y_col]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7,
            )

    ax.set_xlabel(f'{x_col} ({var[pc_x - 1]:.1f}% variance)',
                  fontsize=label_fontsize)
    ax.set_ylabel(f'{y_col} ({var[pc_y - 1]:.1f}% variance)',
                  fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    if title is None:
        title = (f'PCA — coloured by {color_by}'
                 + (f' ({correction_label})' if batch_correct else ''))
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=8)
    ax.legend(loc='best', frameon=False, fontsize=legend_fontsize)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)
    _hide_top_right_spines(ax)
    plt.tight_layout()

    return fig, ax, {
        'pca': pca,
        'transformed': pca_df,
        'variance_explained': pca.explained_variance_ratio_,
        'loadings': pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=pivot_corrected.columns,
        ),
        'matrix': pivot_corrected,
        'batch_correction': correction_label,
    }
