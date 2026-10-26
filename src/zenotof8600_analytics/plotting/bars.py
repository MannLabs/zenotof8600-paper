"""Bar / box / scatter count plots: plot_bar, plot_boxplot_with_points, plot_median_scatter, plot_overlapping_bars."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._style import (PALETTE, PALETTE_DOUBLE, _hide_top_right_spines, _annotate_bar_values,
                     _replicate_point_size, _area_to_diameter, _resolve_panel_size)
from ..stats import _annotate_significance

BAR_WIDTH_IN = 0.38
BAR_PITCH_IN = 0.50
BAR_GROUP_GAP_IN = 0.28


def fix_bar_geometry(fig, ax, *, bar_in=None, unit_in=None, h_in=None):
    """Rescale the axes so one x unit is `unit_in`, then set bars to `bar_in`."""
    from ._style import set_axes_size_inches

    bar_in = BAR_WIDTH_IN if bar_in is None else float(bar_in)
    unit_in = BAR_PITCH_IN if unit_in is None else float(unit_in)

    x0, x1 = ax.get_xlim()
    set_axes_size_inches(fig, ax, w_in=(x1 - x0) * unit_in, h_in=h_in)

    w_data = bar_in / unit_in
    for patch in ax.patches:
        centre = patch.get_x() + patch.get_width() / 2
        patch.set_width(w_data)
        patch.set_x(centre - w_data / 2)


def _legend_above(ax, handles, labels, fontsize, ncol=2):
    """House bar-plot key: one compact horizontal row just above the axes."""
    ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 1.0),
              ncol=ncol, frameon=False, fontsize=fontsize, handlelength=0.8,
              handleheight=0.9, handletextpad=0.35, borderpad=0.0,
              columnspacing=0.8)


_TOTAL_LABEL_OFFSET = 0.045


def plot_bar(
    df,
    x_col,
    y_col,
    *,
    hue_col=None,
    ax=None,
    figsize=None,
    palette=None,
    y_label=None,
    x_label=None,
    title=None,
    ylim=None,
    bar_width=0.8,
    bar_gap=0.0,
    show_values=True,
    value_format='int',
    value_fontsize=None,
    show_points=False,
    point_size=None,
    point_alpha=0.75,
    point_color='black',
    label_fontsize=11,
    tick_fontsize=10,
    legend_fontsize=10,
    edgecolor='black',
    linewidth=1.0,
    significance_test=None,
    significance_correction=None,
    significance_show_ns=False,
    significance_pairs=None,
    significance_fontsize=12,
):
    """Bar plot with one bar per category (mean of `y_col` per `x_col`)."""
    import matplotlib.pyplot as plt
    palette = palette if palette is not None else PALETTE
    vfont = value_fontsize if value_fontsize is not None else label_fontsize
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    else:
        fig = ax.figure
    cats = list(pd.unique(df[x_col]))
    xs = np.arange(len(cats))

    if hue_col is None:
        plot_df = df.groupby(x_col, as_index=False)[y_col].mean()
        plot_df = plot_df.set_index(x_col).reindex(cats).reset_index()
        values = plot_df[y_col].values
        bar_colors = [palette[i % len(palette)] for i in range(len(cats))]
        bars = ax.bar(xs, values, width=bar_width, color=bar_colors,
                      edgecolor=edgecolor, linewidth=linewidth)
        if show_points:
            psize = point_size if point_size is not None else _replicate_point_size(len(cats))
            for i, cat in enumerate(cats):
                cat_data = df.loc[df[x_col] == cat, y_col].dropna().values
                if not len(cat_data):
                    continue
                jitter = np.random.uniform(-bar_width * 0.3, bar_width * 0.3, len(cat_data))
                pt_color = bar_colors[i] if point_color == 'match' else point_color
                ax.scatter(xs[i] + jitter, cat_data, s=psize, color=pt_color,
                           alpha=point_alpha, edgecolor='white', linewidth=0.5, zorder=3)
        if show_values:
            tops = ([df.loc[df[x_col] == cat, y_col].max() for cat in cats]
                    if show_points else None)
            _annotate_bar_values(ax, bars, values, value_format, vfont,
                                 tops=tops)
        max_val = (df[y_col].max() if show_points else (max(values) if len(values) else 0))
    else:
        hues = list(pd.unique(df[hue_col]))
        n_hue = len(hues)
        usable = max(bar_width - bar_gap * max(n_hue - 1, 0), 0.0)
        sub_width = usable / max(n_hue, 1)
        offsets = (np.arange(n_hue) - (n_hue - 1) / 2.0) * (sub_width + bar_gap)
        hue_colors = [palette[i % len(palette)] for i in range(n_hue)]
        max_val = 0
        for hi, hue in enumerate(hues):
            sub = df[df[hue_col] == hue]
            means = sub.groupby(x_col, as_index=False)[y_col].mean()
            vals = means.set_index(x_col).reindex(cats)[y_col].values
            present_mask = ~np.isnan(vals)
            if not present_mask.any():
                continue
            bars = ax.bar(xs[present_mask] + offsets[hi], vals[present_mask],
                          width=sub_width, color=hue_colors[hi],
                          edgecolor=edgecolor, linewidth=linewidth,
                          label=str(hue))
            if show_values:
                tops = None
                if show_points:
                    tops = [sub.loc[sub[x_col] == cat, y_col].max()
                            for i, cat in enumerate(cats) if present_mask[i]]
                _annotate_bar_values(ax, bars, vals[present_mask], value_format,
                                     vfont, tops=tops)
            if show_points:
                psize = (point_size if point_size is not None
                         else _replicate_point_size(len(cats) * n_hue))
                for i, cat in enumerate(cats):
                    if not present_mask[i]:
                        continue
                    cat_data = sub.loc[sub[x_col] == cat, y_col].dropna().values
                    if not len(cat_data):
                        continue
                    jitter = np.random.uniform(-sub_width * 0.25, sub_width * 0.25,
                                               len(cat_data))
                    pt_color = hue_colors[hi] if point_color == 'match' else point_color
                    ax.scatter(xs[i] + offsets[hi] + jitter, cat_data,
                               s=psize, color=pt_color, alpha=point_alpha,
                               edgecolor='white', linewidth=0.5, zorder=3)
            max_val = max(max_val, np.nanmax(vals[present_mask]))
        ax.legend(fontsize=legend_fontsize, frameon=False,
                  loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    ax.set_xticks(xs)
    ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=tick_fontsize)
    ax.set_ylabel(y_label if y_label else y_col, fontsize=label_fontsize)
    if x_label:
        ax.set_xlabel(x_label, fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 2, fontweight='bold')

    ax.set_ylim(bottom=0)
    if ylim is not None:
        if isinstance(ylim, (int, float)):
            ax.set_ylim(top=ylim)
        else:
            ax.set_ylim(ylim)
    else:
        ax.set_ylim(top=max_val * 1.15 if max_val else 1)

    if df[y_col].max() > 1000:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.tick_params(axis='y', labelsize=tick_fontsize)

    if significance_test is not None:
        if hue_col is not None:
            warnings.warn('significance_test ignored when hue_col is set '
                          '(only supported on the simple x-only layout)')
        else:
            _annotate_significance(
                ax, df, x_col, y_col,
                test=significance_test,
                correction=significance_correction,
                show_ns=significance_show_ns,
                pairs=significance_pairs,
                fontsize=significance_fontsize,
            )

    _hide_top_right_spines(ax)
    if created_fig:
        plt.tight_layout()
    return fig, ax


def plot_boxplot_with_points(
    df,
    x_col,
    y_col,
    *,
    hue_col=None,
    figsize=None,
    palette=None,
    y_label=None,
    x_label=None,
    title=None,
    ylim=None,
    ymin=None,
    box_width=0.6,
    show_points=True,
    point_size=None,
    point_alpha=1.0,
    point_color='match',
    point_edgecolor='black',
    point_edge_linewidth=0.4,
    xtick_rotation=45,
    showfliers=False,
    show_median_labels=False,
    median_label_decimals=1,
    median_label_fontsize=8,
    label_fontsize=10,
    tick_fontsize=10,
    legend_fontsize=8,
    significance_test=None,
    significance_correction=None,
    significance_show_ns=False,
    significance_pairs=None,
    significance_fontsize=12,
):
    """Boxplot with optional jittered-point overlay."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    palette = palette if palette is not None else PALETTE
    palette_is_dict = isinstance(palette, dict)
    fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))

    same_col = hue_col is not None and x_col == hue_col
    n_x = df[x_col].nunique()
    box_kwargs = dict(
        x=x_col, y=y_col, data=df, ax=ax, width=box_width, showfliers=showfliers,
        saturation=1.0,
        boxprops={'edgecolor': 'black', 'linewidth': 1},
        whiskerprops={'color': 'black', 'linewidth': 1},
        capprops={'color': 'black', 'linewidth': 1},
        medianprops={'color': 'black', 'linewidth': 1.5},
        flierprops={'marker': 'o', 'markersize': 3,
                    'markerfacecolor': 'black', 'markeredgecolor': 'none',
                    'alpha': 0.5},
    )
    if hue_col and not same_col:
        n_hue = df[hue_col].nunique()
        box_kwargs['hue'] = hue_col
        box_kwargs['palette'] = palette if palette_is_dict else palette[:n_hue]
    else:
        box_kwargs['hue'] = x_col
        box_kwargs['palette'] = palette if palette_is_dict else palette[:n_x]
        box_kwargs['legend'] = False
    sns.boxplot(**box_kwargs)

    if show_points:
        if point_size is not None:
            psize = point_size
        else:
            n_bars = n_x * (df[hue_col].nunique() if (hue_col and not same_col) else 1)
            psize = _area_to_diameter(_replicate_point_size(n_bars))
        strip = dict(x=x_col, y=y_col, data=df, ax=ax, size=psize,
                     alpha=point_alpha, jitter=True,
                     edgecolor=point_edgecolor, linewidth=point_edge_linewidth)
        if point_color == 'match':
            if hue_col and not same_col:
                strip.update(hue=hue_col, dodge=True,
                             palette=palette if palette_is_dict else palette,
                             legend=False)
            else:
                strip.update(hue=x_col,
                             palette=palette if palette_is_dict else palette[:n_x],
                             legend=False)
        else:
            strip['color'] = point_color
            if hue_col and not same_col:
                strip.update(hue=hue_col, dodge=True,
                             palette=[point_color] * df[hue_col].nunique(), legend=False)
        sns.stripplot(**strip)

    if show_median_labels and not (hue_col and not same_col):
        x_levels = list(pd.unique(df[x_col]))
        tick_locs = ax.get_xticks()
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        if not tick_labels or any(not lab for lab in tick_labels):
            tick_labels = [str(lv) for lv in x_levels]
        for k, lab in enumerate(tick_labels):
            sub = df.loc[df[x_col].astype(str) == str(lab), y_col].dropna()
            if sub.empty:
                continue
            med = float(np.median(sub))
            if palette_is_dict:
                color = palette.get(lab, palette.get(str(lab), '#444444'))
            else:
                color = palette[k % len(palette)]
            ax.annotate(f'{med:.{median_label_decimals}f}',
                        xy=(tick_locs[k], med),
                        xytext=(0, 8),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        color=color, fontsize=median_label_fontsize,
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', edgecolor='none',
                                  alpha=0.8))

    ax.set_ylabel(y_label if y_label else y_col, fontsize=label_fontsize)
    ax.set_xlabel(x_label if x_label else '', fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 2, fontweight='bold')
    if ylim is not None:
        if isinstance(ylim, (int, float)):
            ax.set_ylim(top=ylim)
        else:
            ax.set_ylim(ylim)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    if xtick_rotation != 0:
        plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha='right')
    if df[y_col].max() > 1000:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    if hue_col and not same_col:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=legend_fontsize,
                  frameon=False, borderaxespad=0)

    if significance_test is not None:
        if hue_col and not same_col:
            warnings.warn('significance_test ignored when hue_col is set '
                          '(only supported on the simple x-only layout)')
        else:
            _annotate_significance(
                ax, df, x_col, y_col,
                test=significance_test,
                correction=significance_correction,
                show_ns=significance_show_ns,
                pairs=significance_pairs,
                fontsize=significance_fontsize,
            )

    _hide_top_right_spines(ax)
    plt.tight_layout()
    return fig, ax


UNION_TO_PER_RUN = {
    'total_precursors': 'precursor',
    'total_peptides': 'peptide',
    'total_protein_groups': 'protein_group',
}
_PER_RUN_TO_UNION = {v: k for k, v in UNION_TO_PER_RUN.items()}


def _resolve_bar_and_points(df, higher_y_col, points_col):
    """(bar column, points column, bar is a union) for plot_overlapping_bars."""
    if points_col not in (None, 'auto'):
        union = higher_y_col in UNION_TO_PER_RUN
        return higher_y_col, points_col, union
    if higher_y_col in _PER_RUN_TO_UNION \
            and _PER_RUN_TO_UNION[higher_y_col] in df.columns:
        return _PER_RUN_TO_UNION[higher_y_col], higher_y_col, True
    if higher_y_col in UNION_TO_PER_RUN:
        per_run = UNION_TO_PER_RUN[higher_y_col]
        if per_run not in df.columns:
            raise ValueError(
                f'{higher_y_col!r} is a union across runs and the frame has no '
                f'per-run {per_run!r} column for the replicate points; pass '
                f'points_col= explicitly')
        return higher_y_col, per_run, True
    return higher_y_col, higher_y_col, False


def plot_overlapping_bars(
    df,
    x_col,
    higher_y_col,
    lower_y_col,
    *,
    points_col='auto',
    hue_col=None,
    higher_label='IDs',
    lower_label='< 20% CV',
    figsize=(8, 6),
    palette=None,
    y_label='Count',
    x_label=None,
    title=None,
    bar_width=0.8,
    bar_gap=0.02,
    show_points=True,
    point_color='black',
    point_size=None,
    point_alpha=0.75,
    label_fontsize=11,
    value_fontsize=None,
    tick_fontsize=10,
    legend_fontsize=7,
    legend_loc='top',
    lighten_factor=0.45,
    value_fmt=None,
    constant_bar_width=True,
    ax=None,
):
    """Two overlapping bars per (x, hue): a light "total" and a darker subset."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    palette = palette if palette is not None else PALETTE_DOUBLE
    if value_fmt is None:
        def value_fmt(v):
            return f'{int(v):,}'
    vfont = value_fontsize if value_fontsize is not None else label_fontsize
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    bar_col, points_col, bar_is_union = _resolve_bar_and_points(
        df, higher_y_col, points_col)
    y_span = float(max(df[bar_col].max(), df[points_col].max()) or 0)

    def lighten(color, factor=0.4):
        c = mcolors.to_rgb(color)
        return tuple(1 - (1 - ch) * (1 - factor) for ch in c)

    if legend_loc == 'upper left':
        legend_loc = 'top'

    x_categories = list(df[x_col].unique())
    same_col = hue_col is not None and x_col == hue_col
    if hue_col and not same_col:
        hue_groups = list(df[hue_col].unique())
    else:
        hue_groups = [None]
    n_groups = len(hue_groups)
    group_width = bar_width / n_groups
    x_positions = np.arange(len(x_categories))
    psize = (point_size if point_size is not None
             else _replicate_point_size(len(x_categories) * n_groups))

    summary, legend_handles, legend_labels = [], [], []

    for i, hue in enumerate(hue_groups):
        hue_df = df[df[hue_col] == hue] if (hue_col and not same_col) else df
        offset = (i - (n_groups - 1) / 2) * (group_width + bar_gap)
        positions = x_positions + offset
        for j, x_cat in enumerate(x_categories):
            cat_df = hue_df[hue_df[x_col] == x_cat]
            if cat_df.empty:
                continue
            if isinstance(palette, dict):
                key = hue if (hue_col and not same_col) else x_cat
                color = palette.get(key, palette.get(str(key), '#444444'))
            else:
                color = (palette[i % len(palette)]
                         if (hue_col and not same_col)
                         else palette[j % len(palette)])
            light = lighten(color, lighten_factor)
            if bar_is_union:
                vals = cat_df[bar_col].dropna().unique()
                if len(vals) > 1:
                    raise ValueError(
                        f'{bar_col!r} takes {len(vals)} values within '
                        f'{x_cat!r}; a union column must be constant per '
                        f'condition')
                higher = float(vals[0]) if len(vals) else 0.0
            else:
                higher = cat_df[bar_col].mean()
            lower = cat_df[lower_y_col].mean()
            pos = positions[j]

            ax.bar(pos, higher, width=group_width, color=light, edgecolor='darkgray', linewidth=1.0)
            ax.bar(pos, lower, width=group_width, color=color, edgecolor='black', linewidth=1.0)
            if show_points:
                jitter = np.random.uniform(-group_width * 0.25, group_width * 0.25, len(cat_df))
                pt = color if point_color == 'match' else point_color
                ax.scatter(pos + jitter, cat_df[points_col].values, c=pt,
                           s=psize, alpha=point_alpha, zorder=5,
                           edgecolors='white', linewidths=0.5)
            ax.text(pos, higher + y_span * _TOTAL_LABEL_OFFSET,
                    value_fmt(higher), ha='center', va='bottom',
                    fontsize=vfont)
            ax.text(pos, lower, value_fmt(lower), ha='center', va='top',
                    fontsize=vfont,
                    color='white' if lower > higher * 0.1 else 'black')

            summary.append({
                'x_category': x_cat,
                'hue_group': hue if (hue_col and not same_col) else 'all',
                'higher_mean': higher, 'lower_mean': lower,
                'higher_is_union': bar_is_union,
                'points_mean': cat_df[points_col].mean(),
                'higher_std': cat_df[points_col].std(),
                'lower_std': cat_df[lower_y_col].std(),
                'n_samples': len(cat_df),
                'percent_lower': lower / higher * 100 if higher > 0 else 0,
            })
        if hue_col and not same_col:
            color = palette[i % len(palette)]
            light = lighten(color, lighten_factor)
            legend_handles.append(Patch(facecolor=light, edgecolor='darkgray', linewidth=1.0))
            legend_labels.append(str(hue))
            legend_handles.append(Patch(facecolor=color, edgecolor='black', linewidth=1.0))
            legend_labels.append(f'{hue} ({lower_label})')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_categories, rotation=45, ha='right', fontsize=tick_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    if x_label:
        ax.set_xlabel(x_label, fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 2, fontweight='bold',
                     pad=legend_fontsize + 12 if legend_loc == 'top' else 6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.set_ylim(bottom=0, top=y_span * 1.15)
    if not (hue_col and not same_col):
        grey_dark = '#595959'
        grey_light = lighten(grey_dark, 0.55)
        legend_handles = [
            Patch(facecolor=grey_light, edgecolor='darkgray', linewidth=0.8),
            Patch(facecolor=grey_dark, edgecolor='black', linewidth=0.8),
        ]
        legend_labels = [higher_label, lower_label]
        if legend_loc == 'none':
            pass
        elif legend_loc == 'top':
            _legend_above(ax, legend_handles, legend_labels, legend_fontsize)
        elif legend_loc == 'outside':
            ax.legend(legend_handles, legend_labels, loc='upper left',
                      bbox_to_anchor=(1.02, 1), fontsize=legend_fontsize,
                      frameon=False, borderaxespad=0)
        else:
            ax.legend(legend_handles, legend_labels, loc=legend_loc,
                      fontsize=legend_fontsize, frameon=False)
    elif legend_handles:
        if legend_loc == 'none':
            pass
        elif legend_loc == 'top':
            _legend_above(ax, legend_handles, legend_labels, legend_fontsize,
                          ncol=min(len(legend_labels), 4))
        else:
            ax.legend(legend_handles, legend_labels, loc='upper left',
                      bbox_to_anchor=(1.02, 1), fontsize=legend_fontsize,
                      frameon=False, borderaxespad=0)
    _hide_top_right_spines(ax)
    if created_fig:
        plt.tight_layout()
        if constant_bar_width:
            fix_bar_geometry(fig, ax)
    return fig, ax, pd.DataFrame(summary)


