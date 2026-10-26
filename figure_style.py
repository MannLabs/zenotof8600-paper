"""Shared panel conventions for the ZenoTOF 8600 paper figures."""

from __future__ import annotations

import datetime as _dt
import os
import re

import build_manifest

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import zenotof8600_analytics as core

PANEL_W_IN = 4.20
AXES_H_IN = 2.50
FONTSIZE = 10
LEGEND_FONTSIZE = 10

BAR_IN = 0.42
UNIT_IN = 0.52
POINT_SIZE = core.replicate_point_size(BAR_IN)

SHOW_VALUE_LABELS = False

LEGEND_DARK = '#595959'

PER_RUN_ID_COL = {
    'total_precursors': 'precursor',
    'total_peptides': 'peptide',
    'total_protein_groups': 'protein_group',
}


def lighten(color, factor: float = 0.45):
    """Blend toward white — the house light variant of a bar's own hue."""
    return tuple(1 - (1 - ch) * (1 - factor) for ch in mcolors.to_rgb(color))


def mix(color, target, factor: float):
    """Blend `color` `factor` of the way toward `target`."""
    a, b = mcolors.to_rgb(color), mcolors.to_rgb(target)
    return tuple(x * (1 - factor) + y * factor for x, y in zip(a, b))


def set_axes_size_inches(fig, ax, *, w_in=None, h_in=None) -> None:
    """Resize the figure so the axes rectangle is exactly `w_in` x `h_in`."""
    fig.canvas.draw()
    fig_w, fig_h = fig.get_size_inches()
    pos = ax.get_position()
    left_in, right_in = pos.x0 * fig_w, (1.0 - pos.x1) * fig_w
    bottom_in, top_in = pos.y0 * fig_h, (1.0 - pos.y1) * fig_h
    axes_w = pos.width * fig_w if w_in is None else w_in
    axes_h = pos.height * fig_h if h_in is None else h_in

    new_fig_w = left_in + axes_w + right_in
    new_fig_h = bottom_in + axes_h + top_in
    fig.set_size_inches(new_fig_w, new_fig_h)
    ax.set_position([left_in / new_fig_w, bottom_in / new_fig_h,
                     axes_w / new_fig_w, axes_h / new_fig_h])


def fit_panel_width(fig, ax, *, total_w_in: float, h_in=AXES_H_IN,
                    min_axes_w_in: float = 0.70) -> float:
    """Size the axes so the whole FIGURE comes out `total_w_in` wide."""
    set_axes_size_inches(fig, ax, h_in=h_in)
    fig.canvas.draw()
    fig_w = fig.get_size_inches()[0]
    margins = fig_w - ax.get_position().width * fig_w
    axes_w = max(total_w_in - margins, min_axes_w_in)
    set_axes_size_inches(fig, ax, w_in=axes_w, h_in=h_in)
    return axes_w


def set_bar_geometry_inches(fig, ax, *, bar_in=BAR_IN, unit_in=UNIT_IN,
                            h_in=AXES_H_IN,
                            min_axes_w_in: float = 0.0) -> None:
    """Resize the axes so one x data unit is `unit_in`, then set bars to `bar_in`."""
    x0, x1 = ax.get_xlim()
    span = x1 - x0
    unit_in = max(unit_in, min_axes_w_in / span)
    set_axes_size_inches(fig, ax, w_in=span * unit_in, h_in=h_in)

    w_data = bar_in / unit_in
    for patch in ax.patches:
        centre = patch.get_x() + patch.get_width() / 2
        patch.set_width(w_data)
        patch.set_x(centre - w_data / 2)


_VALUE_LABEL_RE = re.compile(r'^\s*[+-]?[\d,.]+\s*%?\s*$')


def strip_value_labels(ax, *, keep_annotations: bool = False) -> int:
    """Remove per-bar value labels a library plot drew."""
    texts = list(ax.texts)
    if keep_annotations:
        texts = [t for t in texts if _VALUE_LABEL_RE.match(t.get_text())]
    for t in texts:
        t.remove()
    return len(texts)


def label_pct(ax, items, *, fontsize=None, pad=0.025, headroom=1.15):
    """Write each bar's percent difference from its reference above the bar."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    for x, top, pct in items:
        ax.text(x, top + span * pad, '%+.1f%%' % pct, ha='center', va='bottom',
                fontsize=fontsize or FONTSIZE, fontweight='bold')
    tallest = max([p.get_height() for p in ax.patches] +
                  [t for _, t, _ in items], default=hi)
    ax.set_ylim(lo, max(hi, tallest * headroom))


def set_categorical_width_inches(fig, ax, *, unit_in=UNIT_IN, h_in=AXES_H_IN):
    """Give a categorical panel the same per-category width as a bar panel."""
    x0, x1 = ax.get_xlim()
    set_axes_size_inches(fig, ax, w_in=(x1 - x0) * unit_in, h_in=h_in)


def tidy(ax) -> None:
    """Drop the top/right spines and the library's bold axis labels."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.xaxis.label.set_fontweight('normal')
    ax.yaxis.label.set_fontweight('normal')


_BUILD = {'fp': None}


def start_build(*, script: str, inputs, out_dir: str = None) -> None:
    """Arm panel provenance for this run."""
    _BUILD['fp'] = build_manifest.fingerprint(script, list(inputs))
    print(f'build: data {_BUILD["fp"]["data"]} '
          f'({_BUILD["fp"]["n_inputs"]} input files), '
          f'code {_BUILD["fp"]["code"]}')


def report_stale(out_dir: str, *, prefix: str = 'figure') -> list:
    """Print which panels in `out_dir` no longer match this run."""
    return build_manifest.report(out_dir, _BUILD['fp'], prefix=prefix)


def save_anchors(fig, out_dir: str, name: str, points: dict) -> None:
    """Register named points for the composer (e.g. a zoom box's corners)."""
    pending = getattr(fig, '_fs_anchor_points', {})
    pending.update({k: (float(v[0]) / fig.dpi, float(v[1]) / fig.dpi)
                    for k, v in points.items()})
    fig._fs_anchor_points = pending


def _write_layout_record(fig, out_dir, name):
    """`<name>.anchors.json`: crop size, every DATA axes box, named points."""
    import functools
    import json
    from matplotlib.backend_bases import _get_renderer
    pad = plt.rcParams['savefig.pad_inches']
    with fig.canvas._switch_canvas_and_return_print_method('pdf') as pm:
        renderer = _get_renderer(fig, functools.partial(
            pm, orientation='portrait'))
        bb = fig.get_tightbbox(renderer).padded(pad)
        dpi = fig.dpi
        to_in = lambda x, y: [x / dpi - bb.x0, y / dpi - bb.y0]
        axes = []
        for ax in fig.axes:
            if not ax.get_visible() or not ax.axison                     or getattr(ax, '_colorbar', None) is not None:
                continue
            e = ax.get_window_extent(renderer)
            axes.append(to_in(e.x0, e.y0) + to_in(e.x1, e.y1))
    out = {'crop_w_in': bb.width, 'crop_h_in': bb.height, 'axes': axes,
           'points': {k: [v[0] - bb.x0, v[1] - bb.y0] for k, v in
                      getattr(fig, '_fs_anchor_points', {}).items()}}
    with open(os.path.join(out_dir, f'{name}.anchors.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)


def saved_width_in(fig):
    """Width of the file `save_panel` would write: tight crop + pad, PDF renderer."""
    import functools
    from matplotlib.backend_bases import _get_renderer
    with fig.canvas._switch_canvas_and_return_print_method('pdf') as pm:
        renderer = _get_renderer(fig, functools.partial(
            pm, orientation='portrait'))
        return fig.get_tightbbox(renderer).padded(
            plt.rcParams['savefig.pad_inches']).width


def fit_saved_width(fig, target_w_in, *, iters=4, tol_in=0.01, layout=True):
    """Resize the canvas until the SAVED (tight-cropped) panel is target wide."""
    h = fig.get_size_inches()[1]
    for _ in range(iters):
        got = saved_width_in(fig)
        if abs(got - target_w_in) < tol_in:
            break
        w = fig.get_size_inches()[0] + (target_w_in - got)
        fig.set_size_inches(w, h)
        if layout:
            fig.tight_layout()
    return saved_width_in(fig)


def save_panel(fig, out_dir: str, name: str, source_df, *, dpi: int = 600):
    """Write `<name>.pdf`, `<name>.png` and `<name>_sourcedata.csv`."""
    os.makedirs(out_dir, exist_ok=True)
    for ax in fig.axes:
        title = ax.get_title()
        if title:
            print(f'    [caption] {name}: {title}')
            ax.set_title('')

    try:
        _write_layout_record(fig, out_dir, name)
    except Exception as exc:
        print(f'    note: no layout record for {name} ({exc})')
    fig.savefig(os.path.join(out_dir, f'{name}.pdf'),
                bbox_inches='tight', dpi=dpi)
    fig.savefig(os.path.join(out_dir, f'{name}.png'),
                dpi=300, bbox_inches='tight')
    if source_df is not None:
        source_df.to_csv(os.path.join(out_dir, f'{name}_sourcedata.csv'),
                         index=False)

    build_manifest.record(out_dir, name, _BUILD['fp'],
                          _dt.datetime.now().strftime('%Y-%m-%d %H:%M'))

    w, h = fig.get_size_inches()
    print(f'  wrote panels/{name}.pdf|.png|_sourcedata.csv  '
          f'({w:.2f} x {h:.2f} in)')
    return w, h


def report_printed_sizes(panel_w_in: float = PANEL_W_IN,
                         panels_per_row: int = 2,
                         column_mm: float = 180.0) -> None:
    """State the printed font sizes for an assumed assembly, and flag the floor."""
    row_in = panel_w_in * panels_per_row
    scale = (column_mm / 25.4) / row_in
    print(f'\nprinted size check: {panels_per_row} panels x {panel_w_in:.2f} in '
          f'= {row_in:.2f} in -> {column_mm:.0f} mm  (x{scale:.3f})')
    for label, pt in [('axis + tick', FONTSIZE), ('legend', LEGEND_FONTSIZE)]:
        printed = pt * scale
        flag = 'OK' if printed >= 8.0 else 'BELOW 8 pt FLOOR'
        print(f'  {label:12s} {pt:4.1f} pt -> {printed:.1f} pt   {flag}')


def init():
    """House rcParams plus the panel default used by this paper."""
    core.init_plotting()
    core.set_default_panel_size(PANEL_W_IN, 4.0)
    plt.rcParams['axes.grid'] = False
