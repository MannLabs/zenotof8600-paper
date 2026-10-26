"""Shared plotting style: palettes, init_plotting (rcParams), and the small drawing/validation helpers used across plot families."""

from __future__ import annotations

import numpy as np
import pandas as pd


PALETTE = [
    '#FBA08D', '#C79EEA', '#9ECCEA',
    '#FA7A61', '#B178E2', '#78B8E2',
    '#F95534', '#9B52DA', '#52A4DA',
    '#ED2E07', '#7E2AC7', '#2A88C7',
    '#CB2706', '#6D25AD', '#2576AD',
    '#9E1E05', '#551D87', '#1D5C87',
]
PALETTE_DOUBLE = [
    '#FBA08D', '#C79EEA', '#9ECCEA', '#9EDCB8', '#FBE49C', '#F4ADCA',
    '#FA7A61', '#B178E2', '#78B8E2', '#6BC88F', '#F5C842', '#E85D9A',
]
PALETTE_SINGLE = ['#FA7A61', '#B178E2', '#78B8E2', '#6BC88F', '#F5C842', '#E85D9A']
PALETTE_CV_CATEGORIES = ['#5cb85c', '#f0ad4e', '#d9534f']


DEFAULT_PANEL_SIZE = (4.0, 4.0)


def set_default_panel_size(width, height=None):
    """Set the global default atomic-panel ``figsize`` used by plot_pca / plot_volcano / plot_boxplot_with_points whenever they're called without an explicit ``figsize`` (i.e. ``figsize=None``)."""
    global DEFAULT_PANEL_SIZE
    if height is None:
        if not (isinstance(width, (tuple, list)) and len(width) == 2):
            raise ValueError(
                'pass width and height, or a single (width, height) tuple')
        w, h = width
    else:
        w, h = width, height
    DEFAULT_PANEL_SIZE = (float(w), float(h))
    return DEFAULT_PANEL_SIZE


def set_axes_size_inches(fig, axes, *, w_in=None, h_in=None):
    """Resize the figure so the axes rectangle is exactly ``w_in`` x ``h_in``."""
    axes = [axes] if not isinstance(axes, (list, tuple, np.ndarray)) else list(axes)
    fig.canvas.draw()
    fig_w, fig_h = fig.get_size_inches()
    pos = axes[0].get_position()
    left_in, right_in = pos.x0 * fig_w, (1.0 - pos.x1) * fig_w
    bottom_in, top_in = pos.y0 * fig_h, (1.0 - pos.y1) * fig_h
    axes_w = pos.width * fig_w if w_in is None else float(w_in)
    axes_h = pos.height * fig_h if h_in is None else float(h_in)

    new_fig_w = left_in + axes_w + right_in
    new_fig_h = bottom_in + axes_h + top_in
    fig.set_size_inches(new_fig_w, new_fig_h)
    rect = [left_in / new_fig_w, bottom_in / new_fig_h,
            axes_w / new_fig_w, axes_h / new_fig_h]
    for ax in axes:
        ax.set_position(rect)
    return (axes_w, axes_h)


def _resolve_panel_size(figsize):
    """Return ``figsize`` when given, else the current ``DEFAULT_PANEL_SIZE``."""
    return DEFAULT_PANEL_SIZE if figsize is None else figsize


def _empty_plot_with_message(message, *, figsize=(6, 4), title=None):
    """Render a placeholder axes with a centered grey message — used when there's not enough data to draw the requested plot."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, transform=ax.transAxes,
            ha='center', va='center', fontsize=11, color='gray',
            wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig, ax


def _hide_top_right_spines(ax):
    """Defensive equivalent of seaborn's despine — keeps plot frames clean even if `init_plotting()` rcParams have been reset (e.g. by a third-party plot import) or never applied."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def init_plotting():
    """Set matplotlib defaults for the figures in this module."""
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'


POINT_DIAMETER_FRAC = 0.45
POINT_DIAMETER_MIN_PT = 4.5
POINT_DIAMETER_MAX_PT = 9.0


def set_replicate_point_scale(frac=None, min_pt=None, max_pt=None):
    """Change the global replicate-point sizing rule used by ``replicate_point_size``."""
    global POINT_DIAMETER_FRAC, POINT_DIAMETER_MIN_PT, POINT_DIAMETER_MAX_PT
    frac = POINT_DIAMETER_FRAC if frac is None else float(frac)
    min_pt = POINT_DIAMETER_MIN_PT if min_pt is None else float(min_pt)
    max_pt = POINT_DIAMETER_MAX_PT if max_pt is None else float(max_pt)
    if min_pt > max_pt:
        raise ValueError('min_pt must not exceed max_pt')
    POINT_DIAMETER_FRAC, POINT_DIAMETER_MIN_PT, POINT_DIAMETER_MAX_PT = (
        frac, min_pt, max_pt)
    return (POINT_DIAMETER_FRAC, POINT_DIAMETER_MIN_PT, POINT_DIAMETER_MAX_PT)


def replicate_point_size(bar_width_in, *, frac=None, min_pt=None, max_pt=None):
    """Marker area (matplotlib scatter ``s``, points²) for replicate dots overlaid on a bar ``bar_width_in`` inches wide."""
    frac = POINT_DIAMETER_FRAC if frac is None else frac
    min_pt = POINT_DIAMETER_MIN_PT if min_pt is None else min_pt
    max_pt = POINT_DIAMETER_MAX_PT if max_pt is None else max_pt
    diameter = np.clip(float(frac) * float(bar_width_in) * 72.0,
                       float(min_pt), float(max_pt))
    return float(diameter ** 2)


def drawn_bar_width_inches(ax):
    """Median drawn width, in inches, of the bar patches on ``ax``."""
    from matplotlib.patches import Rectangle

    fig = ax.figure
    fig.canvas.draw()
    widths = []
    for patch in ax.patches:
        if not isinstance(patch, Rectangle):
            continue
        x0 = patch.get_x()
        x1 = x0 + patch.get_width()
        (px0, _), (px1, _) = ax.transData.transform([(x0, 0.0), (x1, 0.0)])
        widths.append(abs(px1 - px0) / fig.dpi)
    return float(np.median(widths)) if widths else None


def scale_replicate_points(ax, collections=None, **kwargs):
    """Resize replicate-point collections on ``ax`` to match its drawn bars."""
    bar_in = drawn_bar_width_inches(ax)
    if bar_in is None:
        return None
    size = replicate_point_size(bar_in, **kwargs)
    for collection in (ax.collections if collections is None else collections):
        collection.set_sizes([size])
    return size


def _replicate_point_size(n_bars):
    """Fallback marker area (scatter ``s``, points²) for callers that do not know their bar width: scaled inversely with the number of bars/boxes so the dots stay proportionate as a panel gets more crowded, clamped to a readable range."""
    return float(np.clip(180.0 / max(int(n_bars), 1), 5, 24))


def _area_to_diameter(area):
    """Convert a matplotlib scatter area (`s`, points²) to the equivalent marker diameter in points — the unit seaborn's stripplot `size` expects — so point sizing is consistent between scatter- and stripplot-based plots."""
    return float(2.0 * np.sqrt(max(area, 0.0) / np.pi))


BAR_VALUE_PAD_PT = 2.0

BAR_VALUE_POINT_PAD_PT = POINT_DIAMETER_MAX_PT / 2 + BAR_VALUE_PAD_PT


def _annotate_bar_values(ax, bars, values, value_format, fontsize, tops=None):
    """Helper for plot_bar — write the per-bar value labels above each bar."""
    tops = [None] * len(bars) if tops is None else list(tops)
    for bar, val, top in zip(bars, values, tops):
        if not np.isfinite(val):
            continue
        if value_format == 'int':
            text = f'{int(val):,}'
        elif value_format == 'percent':
            text = f'{val:.1f}%'
        elif value_format == 'scientific':
            text = f'{val:.2e}'
        elif value_format == 'float':
            text = f'{val:.2f}'
        else:
            text = f'{val:,}'
        height = bar.get_height()
        anchor, pad = height, BAR_VALUE_PAD_PT
        if top is not None and np.isfinite(top):
            anchor, pad = max(height, float(top)), BAR_VALUE_POINT_PAD_PT
        ax.annotate(text, (bar.get_x() + bar.get_width() / 2, anchor),
                    xytext=(0, pad), textcoords='offset points',
                    ha='center', va='bottom', fontsize=fontsize)


def _filter_pivot_by_validity(
    pivot,
    sample_info,
    *,
    min_valid_fraction,
    min_valid_fraction_in_any_group,
    group_col,
    runs_axis,
    log_prefix,
):
    """Drop columns (or rows) of `pivot` that fail one or more validity filters.

    Parameters:
      pivot: a DataFrame whose `runs_axis` is run-indexed; the orthogonal axis
        is the feature axis.
      runs_axis: 'index' if rows are runs (PCA layout), 'columns' if columns
        are runs (volcano layout).
      min_valid_fraction: fraction of all runs that must have valid values.
        None to skip.
      min_valid_fraction_in_any_group: fraction of replicates within at least
        one group that must be valid. None to skip.
      group_col: sample_info column used to partition runs into groups (for
        the per-group filter).
      log_prefix: short prefix for the printed kept-count line.
    """
    if runs_axis not in ('index', 'columns'):
        raise ValueError("runs_axis must be 'index' or 'columns'")
    is_runs_index = (runs_axis == 'index')
    runs = pivot.index if is_runs_index else pivot.columns
    n_runs = len(runs)
    n_before = pivot.shape[1] if is_runs_index else pivot.shape[0]

    keep = pd.Series(True, index=(pivot.columns if is_runs_index else pivot.index))

    if min_valid_fraction is not None:
        min_required = max(2, int(n_runs * min_valid_fraction))
        valid_count = (pivot.notna().sum(axis=0)
                       if is_runs_index
                       else pivot.notna().sum(axis=1))
        keep &= (valid_count >= min_required)

    if min_valid_fraction_in_any_group is not None:
        if group_col not in sample_info.columns:
            raise ValueError(
                f'group_col {group_col!r} missing from sample_info '
                f'(needed for min_valid_fraction_in_any_group)'
            )
        run_to_group = sample_info.set_index('run')[group_col]
        run_groups = run_to_group.reindex(runs)
        any_pass = pd.Series(False, index=keep.index)
        for grp, grp_runs in run_groups.dropna().groupby(run_groups.dropna()).groups.items():
            grp_runs = list(grp_runs)
            if not grp_runs:
                continue
            if is_runs_index:
                sub = pivot.loc[grp_runs]
                valid = sub.notna().sum(axis=0)
            else:
                sub = pivot[grp_runs]
                valid = sub.notna().sum(axis=1)
            min_required = max(1, int(len(grp_runs) * min_valid_fraction_in_any_group))
            any_pass |= (valid >= min_required)
        keep &= any_pass

    if is_runs_index:
        out = pivot.loc[:, keep]
    else:
        out = pivot.loc[keep]
    n_kept = (out.shape[1] if is_runs_index else out.shape[0])
    parts = []
    if min_valid_fraction is not None:
        parts.append(f'overall>={min_valid_fraction:g}')
    if min_valid_fraction_in_any_group is not None:
        parts.append(f'any-{group_col}>={min_valid_fraction_in_any_group:g}')
    filt_desc = ', '.join(parts) or 'no filter'
    print(f'{log_prefix} kept {n_kept:,} of {n_before:,} features ({filt_desc})')
    return out


def _confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """Covariance confidence ellipse helper for PCA scatter plots."""
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as mtransforms
    if x.size != y.size:
        raise ValueError('x and y must be the same size')
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_rx = np.sqrt(1 + pearson)
    ell_ry = np.sqrt(1 - pearson)
    ell = Ellipse((0, 0), width=ell_rx * 2, height=ell_ry * 2,
                  facecolor=facecolor, **kwargs)
    sx = np.sqrt(cov[0, 0]) * n_std
    sy = np.sqrt(cov[1, 1]) * n_std
    transf = (mtransforms.Affine2D()
              .scale(sx, sy)
              .translate(np.mean(x), np.mean(y)))
    ell.set_transform(transf + ax.transData)
    return ax.add_patch(ell)


_LEVEL_COLS = {
    'protein':   ('protein_group', 'pg_intensity'),
    'peptide':   ('peptide_id',    'peptide_intensity'),
    'precursor': ('precursor_id',  'precursor_intensity'),
}


def _resolve_highlights(
    df: pd.DataFrame,
    id_col: str,
    highlight_genes: list | None,
    highlight_ids: list | None,
) -> tuple[set, dict]:
    """Resolve gene-name and id-based highlights against `df`."""
    highlight_set: set = set()
    label_map: dict = {}
    if highlight_genes:
        upper_targets = {str(g).upper() for g in highlight_genes}
        gene_lookup = (df.dropna(subset=[id_col, 'genes'])
                         .drop_duplicates(id_col)
                         .set_index(id_col)['genes'].astype(str))
        for ent, gene in gene_lookup.items():
            tokens = [t.strip() for t in gene.split(';') if t.strip()]
            hit = next((t for t in tokens if t.upper() in upper_targets), None)
            if hit is not None:
                highlight_set.add(ent)
                label_map[ent] = hit
    if highlight_ids:
        for ent in highlight_ids:
            highlight_set.add(ent)
            label_map.setdefault(ent, str(ent))
    return highlight_set, label_map
