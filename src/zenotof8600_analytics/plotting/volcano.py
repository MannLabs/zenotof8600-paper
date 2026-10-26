"""Volcano plot (limma moderated t-test)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._style import (_empty_plot_with_message, _hide_top_right_spines, _filter_pivot_by_validity, _LEVEL_COLS, _resolve_panel_size)
from ..stats import _ebayes_moderated_p
from ..filters import filter_runs


def _annotate_without_overlap(ax, points, *, fontsize=8, pad_pt=1.5,
                              max_shift_pt=90):
    """Label (x, y, text) points, nudging each one up until it stops colliding."""
    if not points:
        return
    fig = ax.figure
    px_per_pt = fig.dpi / 72.0
    line_pt = fontsize * 1.25

    def place():
        """(x, y, text, dx, dy) per point, plus the highest top reached."""
        out, placed, top = [], [], -np.inf
        for x, y, text in sorted(points, key=lambda p: -p[1]):
            anchor = ax.transData.transform((x, y)) / px_per_pt
            w = len(text) * 0.6 * fontsize
            dx = 5 if x > 0 else -5
            left0 = anchor[0] + (dx if x > 0 else dx - w)
            dy = 5
            while dy <= max_shift_pt:
                bottom = anchor[1] + dy
                box = (left0, bottom, left0 + w, bottom + line_pt)
                if not any(box[0] < o[2] + pad_pt and o[0] < box[2] + pad_pt
                           and box[1] < o[3] + pad_pt and o[1] < box[3] + pad_pt
                           for o in placed):
                    placed.append(box)
                    top = max(top, box[3])
                    break
                dy += line_pt * 0.75
            out.append((x, y, text, dx, dy))
        return out, top

    labels, top_pt = place()
    for _ in range(3):
        y_top = ax.transData.inverted().transform(
            (0, (top_pt + line_pt * 1.2) * px_per_pt))[1]
        lo, hi = ax.get_ylim()
        if y_top <= hi:
            break
        ax.set_ylim(lo, y_top)
        labels, top_pt = place()

    for x, y, text, dx, dy in labels:
        ax.annotate(
            text, (x, y), fontsize=fontsize, alpha=0.8,
            ha='left' if x > 0 else 'right', va='bottom',
            xytext=(dx, dy), textcoords='offset points',
            arrowprops=(dict(arrowstyle='-', linewidth=0.4, color='0.55',
                             shrinkA=0, shrinkB=1.5, alpha=0.8)
                        if dy > 5 else None),
        )


def plot_volcano(
    df,
    sample_info,
    *,
    level='protein',
    group_col='condition2',
    condition_a=None,
    condition_b=None,
    fc_threshold=0.585,
    padj_threshold=0.05,
    min_valid_per_condition=3,
    min_valid_fraction=None,
    min_valid_fraction_in_any_group=None,
    palette=None,
    highlight_genes=None,
    highlight_color='#FFD700',
    highlight_size=80,
    figsize=None,
    point_size_sig=12,
    point_size_ns=6,
    alpha_sig=0.85,
    alpha_ns=0.4,
    label_top_n=0,
    label_highlighted=True,
    label_fontsize=10,
    tick_fontsize=10,
    legend_fontsize=8,
    legend_loc='upper right',
    title_fontsize=11,
    title=None,
    method='welch',
):
    """Volcano plot comparing two values of `group_col` (default `condition2`) at the chosen quantitation level.

    Parameters:
      level: 'precursor' | 'peptide' | 'protein'  (default 'protein').
        At non-protein levels the row id column is named after the level
        (`peptide_id` or `precursor_id`); the gene column still resolves via
        the feature's parent protein_group, so `highlight_genes=` works at any
        level (highlighting every feature of the matching gene).
      min_valid_per_condition: integer count of non-zero replicates required
        in BOTH compared conditions for the t-test (default 3, mathematically
        the minimum for a Welch test).
      min_valid_fraction: optional overall-fraction filter applied across the
        runs of the two compared conditions before the t-test.
      min_valid_fraction_in_any_group: optional per-group fraction filter
        (over `group_col`); a feature passes if at least one group hits this
        fraction. Useful for many-condition datasets — filters out features
        that are unreliable everywhere even if they squeak past the
        condition-A/B count requirement.
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    if level not in _LEVEL_COLS:
        raise ValueError(
            f"level must be one of {sorted(_LEVEL_COLS)}, got {level!r}"
        )
    id_col, intensity_col = _LEVEL_COLS[level]

    if df['engine'].nunique() > 1:
        raise ValueError('plot_volcano: df contains multiple engines; '
                         'use core.split_by_engine first')
    if group_col not in sample_info.columns:
        raise ValueError(f'group_col {group_col!r} missing from sample_info')

    available = list(sample_info[group_col].drop_duplicates())
    if condition_a is None or condition_b is None:
        if len(available) != 2:
            raise ValueError(
                f'sample_info[{group_col!r}] has {len(available)} unique values '
                f'{available!r}; pass condition_a=, condition_b=  explicitly'
            )
        if condition_a is None:
            condition_a = available[0]
        if condition_b is None:
            condition_b = available[1]

    runs_a = sample_info.loc[sample_info[group_col] == condition_a, 'run'].tolist()
    runs_b = sample_info.loc[sample_info[group_col] == condition_b, 'run'].tolist()
    if not runs_a or not runs_b:
        raise ValueError('one or both conditions matched zero runs')

    df = filter_runs(df, sample_info)
    pivot = df.pivot_table(
        index=id_col, columns='run',
        values=intensity_col, aggfunc='first',
    )
    cols_a = [r for r in runs_a if r in pivot.columns]
    cols_b = [r for r in runs_b if r in pivot.columns]

    if min_valid_fraction is not None or min_valid_fraction_in_any_group is not None:
        compared_runs = cols_a + cols_b
        pivot_compared = pivot[compared_runs]
        pivot_compared = _filter_pivot_by_validity(
            pivot_compared, sample_info,
            min_valid_fraction=min_valid_fraction,
            min_valid_fraction_in_any_group=min_valid_fraction_in_any_group,
            group_col=group_col,
            runs_axis='columns',
            log_prefix=f'[volcano:{level}]',
        )
        pivot = pivot.loc[pivot_compared.index]

    gene_map = (df.assign(_g=df['genes'].astype(str).str.split(';').str[0])
                  .groupby(id_col)['_g'].agg(
                      lambda s: next((v for v in s if v), '')))

    mat_a = pivot[cols_a].astype(float).to_numpy()
    mat_b = pivot[cols_b].astype(float).to_numpy()
    mat_a = np.where(mat_a > 0, mat_a, np.nan)
    mat_b = np.where(mat_b > 0, mat_b, np.nan)

    n_a = np.sum(~np.isnan(mat_a), axis=1)
    n_b = np.sum(~np.isnan(mat_b), axis=1)
    enough = (n_a >= min_valid_per_condition) & (n_b >= min_valid_per_condition)

    if method not in ('welch', 'moderated'):
        raise ValueError(f"method must be 'welch' or 'moderated', got {method!r}")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean_a = np.nanmean(mat_a, axis=1)
        mean_b = np.nanmean(mat_b, axis=1)
        if method == 'moderated':
            log2_fc, p_val, _prior_df = _ebayes_moderated_p(
                mat_a, mat_b, n_a, n_b, enough)
        else:
            log2_fc = (np.nanmean(np.log2(mat_a), axis=1)
                       - np.nanmean(np.log2(mat_b), axis=1))
            t_result = stats.ttest_ind(np.log2(mat_a), np.log2(mat_b),
                                       axis=1, equal_var=False,
                                       nan_policy='omit')
            p_val = np.asarray(t_result.pvalue)

    ids = pivot.index.to_numpy()
    keep = enough & np.isfinite(p_val) & np.isfinite(log2_fc)
    if not keep.any():
        warnings.warn(
            f'plot_volcano: no {level}-level features passed validity '
            f'filters (min_valid_per_condition={min_valid_per_condition}). '
            f'Either too few replicates per condition or too sparse data. '
            f'Rendering an empty plot.'
        )
        fig, ax = _empty_plot_with_message(
            f'volcano: no features passed validity filters\n'
            f'(min_valid_per_condition={min_valid_per_condition})',
            figsize=_resolve_panel_size(figsize),
            title=title or f'Volcano: {condition_a} vs {condition_b}')
        empty = pd.DataFrame(columns=[
            id_col, 'gene', 'mean_a', 'mean_b', 'log2_fc', 'p_value',
            'p_adj', 'neg_log10_padj', 'n_a', 'n_b',
            'significance', 'highlighted',
        ])
        return fig, ax, empty

    kept_ids = ids[keep]
    volcano_df = pd.DataFrame({
        id_col: kept_ids,
        'gene': [gene_map.get(i, '') for i in kept_ids],
        'mean_a': mean_a[keep],
        'mean_b': mean_b[keep],
        'log2_fc': log2_fc[keep],
        'p_value': p_val[keep],
        'n_a': n_a[keep].astype(int),
        'n_b': n_b[keep].astype(int),
    })

    _, p_adj, *_ = multipletests(volcano_df['p_value'], method='fdr_bh')
    volcano_df['p_adj'] = p_adj
    volcano_df['neg_log10_padj'] = -np.log10(p_adj)

    def classify(r):
        if r['p_adj'] < padj_threshold:
            if r['log2_fc'] > fc_threshold:
                return 'Up'
            if r['log2_fc'] < -fc_threshold:
                return 'Down'
        return 'NS'
    volcano_df['significance'] = volcano_df.apply(classify, axis=1)

    if highlight_genes:
        upper = {g.upper() for g in highlight_genes}
        volcano_df['highlighted'] = volcano_df['gene'].fillna('').astype(str).str.upper().isin(upper)
    else:
        volcano_df['highlighted'] = False

    palette = palette if palette is not None else ['#E74C3C', '#3498DB']
    colors = {'Up': palette[0], 'Down': palette[1], 'NS': '#CCCCCC'}
    counts = volcano_df['significance'].value_counts()

    fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    for cat in ('NS', 'Down', 'Up'):
        sub = volcano_df[(volcano_df['significance'] == cat) & (~volcano_df['highlighted'])]
        if sub.empty:
            continue
        is_sig = cat != 'NS'
        ax.scatter(
            sub['log2_fc'], sub['neg_log10_padj'],
            c=colors[cat],
            s=point_size_ns if cat == 'NS' else point_size_sig,
            alpha=alpha_ns if cat == 'NS' else alpha_sig,
            label=f'{cat} (n={int(counts.get(cat, 0))})',
            edgecolors='black' if is_sig else 'none',
            linewidth=0.4 if is_sig else 0,
        )
    if highlight_genes:
        sub = volcano_df[volcano_df['highlighted']]
        if not sub.empty:
            ax.scatter(
                sub['log2_fc'], sub['neg_log10_padj'],
                c=highlight_color, s=highlight_size, alpha=1.0,
                label=f'Highlighted (n={len(sub)})',
                edgecolors='black', linewidth=1, zorder=10,
            )
    ax.axhline(-np.log10(padj_threshold), color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(fc_threshold, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(-fc_threshold, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    ratio = f'[{condition_a} / {condition_b}]'
    xlabel = f'log₂(fold change)  {ratio}'
    if len(xlabel) * 0.6 * label_fontsize / 72 > fig.get_size_inches()[0]:
        xlabel = f'log₂(fold change)\n{ratio}'
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel('−log₁₀(adjusted p-value)', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    if title is None:
        title = f'Volcano: {condition_a} vs {condition_b}'
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=8)
    ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=False)

    if (label_top_n > 0) or (label_highlighted and highlight_genes):
        ax.margins(x=0.10, y=0.06)

    if label_highlighted and highlight_genes:
        for _, row in volcano_df[volcano_df['highlighted']].iterrows():
            ax.annotate(
                str(row['gene']) or str(row[id_col]),
                (row['log2_fc'], row['neg_log10_padj']),
                fontsize=9, fontweight='bold',
                ha='left' if row['log2_fc'] > 0 else 'right', va='bottom',
                xytext=(8 if row['log2_fc'] > 0 else -8, 5),
                textcoords='offset points',
            )
    if label_top_n > 0:
        sig = volcano_df[(volcano_df['significance'] != 'NS')
                         & (~volcano_df['highlighted'])]
        ranked = sig.assign(_abs_fc=sig['log2_fc'].abs())
        picks = [g.sort_values(['p_adj', '_abs_fc'], ascending=[True, False])
                  .head(label_top_n)
                 for _, g in ranked.groupby('significance', sort=False)]
        chosen = pd.concat(picks) if picks else sig.iloc[:0]
        _annotate_without_overlap(
            ax,
            [(float(r['log2_fc']), float(r['neg_log10_padj']),
              str(r['gene']) or str(r[id_col]))
             for _, r in chosen.iterrows()],
            fontsize=8,
        )

    _hide_top_right_spines(ax)
    plt.tight_layout()
    return fig, ax, volcano_df
