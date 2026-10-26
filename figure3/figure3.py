"""Figure 3 — quantitative accuracy on a mixed-species (HYE) design, H032_E378."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import logging
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(cfg.DATA_ROOT, 'H032_E378')
OUT = cfg.panels(__file__)

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

for _noisy in ('directlfq', 'fontTools', 'matplotlib'):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

LETTER = {
    'design': 'a',
    'species_counts': 'b',
    'method_pg': 'c',
    'method_pep': 'd',
    'species_ratio': 'e',
    'accuracy': 'f',
    'between_protein': 'g',
    'protein_cv': 'h',
}

ROW_W_IN = 8.40
ROW_B_W_IN = ROW_W_IN
ROW_C_W_IN = ROW_W_IN / 3
ROW_A_AXES_H_IN = 1.70
ROW_C_AXES_H_IN = 2.05

BAR_IN = 0.20
UNIT_IN = 0.24

MIN_BAR_AXES_W_IN = 0.86
LABEL_SPREAD_IN = 0.035

MARKER_AREA = 34
MARKER_MS = 6

RASTERIZE_OVER = 500
RASTER_DPI = 600


def bar_data_width(ax, *, min_axes_w_in=0.0):
    """The drawn bar's width in DATA units, reproducing the helper's rule."""
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    return BAR_IN / max(UNIT_IN, min_axes_w_in / span)


def thousands_axis(ax, label):
    """Count axis in thousands, with the factor in the label."""
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1e3:,.0f}'))
    ax.set_ylabel(rf'{label} [$\times$10$^3$]', fontsize=fs.FONTSIZE)

COND_ORDER = [f'HYE{i}' for i in range(1, 7)]
COMPOSITION = {
    'HYE1': {'Yeast': 45, 'Human': 50, 'E. coli': 5},
    'HYE2': {'Yeast': 40, 'Human': 50, 'E. coli': 10},
    'HYE3': {'Yeast': 30, 'Human': 50, 'E. coli': 20},
    'HYE4': {'Yeast': 20, 'Human': 50, 'E. coli': 30},
    'HYE5': {'Yeast': 10, 'Human': 50, 'E. coli': 40},
    'HYE6': {'Yeast':  5, 'Human': 50, 'E. coli': 45},
}

PAIRS = [('HYE1', 'HYE6'), ('HYE2', 'HYE5'), ('HYE2', 'HYE4'),
         ('HYE3', 'HYE5'), ('HYE3', 'HYE4')]

SCATTER_PAIR = ('HYE1', 'HYE6')

SPECIES_ORDER = ['Human', 'Yeast', 'E. coli']
SPECIES_COLORS = {'Human': core.PALETTE_SINGLE[1],
                  'Yeast': core.PALETTE_SINGLE[0],
                  'E. coli': core.PALETTE_SINGLE[2]}

QUALITY_MIN = 50
QUANT_KW = dict(value_col='pg_intensity', quant='column',
                min_completeness=1.0, normalize_species='Human')

METHODS = {
    'ZT Scan': dict(dir='HYE_All', layout='row'),
    'Zeno SWATH': dict(dir='HYE_all_DIA', layout='column'),
}
SWATH_WELLS = {
    'HYE1': ['A1', 'B1', 'C1', 'D1'], 'HYE2': ['A2', 'B2', 'C2', 'D2'],
    'HYE3': ['A3', 'B3', 'C3', 'D3'], 'HYE4': ['A4', 'B4', 'C4', 'D4'],
    'HYE5': ['E1', 'E3', 'F1', 'F3'], 'HYE6': ['E2', 'E4', 'F2', 'F4'],
}
METHOD_COLOR = {'ZT Scan': core.PALETTE_SINGLE[4],
                'Zeno SWATH': core.PALETTE_SINGLE[3]}

METHOD_MARKER = {'ZT Scan': 'o', 'Zeno SWATH': '^'}

BAR_ORDER = ['Zeno SWATH', 'ZT Scan']
METHOD_LINESTYLE = {'ZT Scan': '-', 'Zeno SWATH': '--'}

FIT_COLOR = '#333333'


def save(fig, name, source_df, dpi=300, out=None):
    return fs.save_panel(fig, out or OUT, name, source_df, dpi=dpi)


def legend_on_top(ax, handles, *, ncol=3, **kw):
    """The house placement for these panels: above the axes, spanning them."""
    opts = dict(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=ncol,
                frameon=False, fontsize=fs.LEGEND_FONTSIZE, handlelength=1.2,
                handletextpad=0.4, borderpad=0.0, columnspacing=0.9,
                labelspacing=0.25)
    opts.update(kw)
    ax.legend(handles=handles, **opts)


def species_method_handles(*, line=False, species_line=None,
                           method_line=None, methods=METHODS):
    """The two legend groups of a ratio panel, SEPARATELY."""
    import matplotlib.lines as mlines
    species_line = line if species_line is None else species_line
    method_line = line if method_line is None else method_line
    species = [mlines.Line2D([], [], marker='o', ms=6,
                             ls='-' if species_line else 'none', lw=1.4,
                             color=SPECIES_COLORS[s], mec='black', mew=0.7,
                             label=s) for s in SPECIES_ORDER]
    method = [mlines.Line2D([], [], marker=METHOD_MARKER[m], ms=6,
                            ls=(METHOD_LINESTYLE[m] if method_line else 'none'),
                            lw=1.2, color=FIT_COLOR if method_line else '#BBBBBB',
                            mec='black', mew=0.7, label=m) for m in methods]
    return species, method


def legend_two_rows(ax, top, bottom, *, gap=0.015):
    """Two stacked legends above the axes, each sizing its own columns."""
    opts = dict(loc='lower center', frameon=False,
                fontsize=fs.LEGEND_FONTSIZE, handlelength=1.4,
                handletextpad=0.4, borderpad=0.0, columnspacing=1.1,
                labelspacing=0.25)
    if len(bottom) < 2:
        ax.legend(handles=top, ncol=len(top),
                  bbox_to_anchor=(0.5, 1.02), **opts)
        return
    low = ax.legend(handles=bottom, ncol=len(bottom),
                    bbox_to_anchor=(0.5, 1.02), **opts)
    ax.add_artist(low)
    ax.figure.canvas.draw()
    h = low.get_window_extent().transformed(ax.transAxes.inverted()).height
    ax.legend(handles=top, ncol=len(top),
              bbox_to_anchor=(0.5, 1.02 + h + gap), **opts)


def _runs(features):
    return [c[:-len(' m/z')] for c in pd.read_csv(features, nrows=0).columns
            if c.endswith(' m/z') and c != 'm/z']


def _cache_path(kind, key):
    return os.path.join(cfg.cache('figure3'), f'{kind}_{key}.pkl')


def _fingerprint(*parts):
    import hashlib
    h = hashlib.sha1()
    for p in parts:
        h.update(repr(p).encode())
    return h.hexdigest()[:16]


def load(method):
    """load_uncached(method), cached on the export's size and modification time."""
    features = os.path.join(DATA, METHODS[method]['dir'], 'lfq.dia.features.csv')
    st = os.stat(features)
    path = _cache_path('load', _fingerprint(method, st.st_size, st.st_mtime_ns))
    if os.path.exists(path):
        return pd.read_pickle(path)
    out = load_uncached(method)
    pd.to_pickle(out, path)
    return out


def directlfq_cached(d):
    """compute_directlfq_pg_intensity(d), cached on the content of its input."""
    cols = ['run', 'protein_group', 'precursor_id', 'precursor_intensity']
    key = _fingerprint(int(pd.util.hash_pandas_object(d[cols], index=True).sum()),
                       len(d))
    path = _cache_path('directlfq', key)
    if os.path.exists(path):
        return pd.read_pickle(path)
    out = core.compute_directlfq_pg_intensity(d)
    pd.to_pickle(out, path)
    return out


def load_uncached(method):
    """Long frame + sample_info for one acquisition method, UNFILTERED."""
    spec = METHODS[method]
    d = os.path.join(DATA, spec['dir'])
    features = os.path.join(d, 'lfq.dia.features.csv')
    runs = _runs(features)

    experiments = []
    for i, cond in enumerate(COND_ORDER):
        if spec['layout'] == 'row':
            tags = [f'_{"ABCDEF"[i]}{r}' for r in range(1, 5)]
        else:
            tags = [r for r in runs
                    if any(r.split('E378_')[-1].split('_')[0] == w
                           for w in SWATH_WELLS[cond])]
            if len(tags) != 4:
                raise ValueError(f'{method} {cond}: matched {len(tags)} runs, '
                                 f'expected 4 -- {tags}')
        experiments.append({'path': features, 'file_tags': tags,
                            'condition1': 'HYE', 'condition2': cond})

    df, sample_info = core.load_experiments(experiments)
    df = core.assign_species(df)

    feat = pd.read_csv(features, usecols=['Peptide', 'z', 'Quality'])
    feat['stripped'] = (feat['Peptide']
                        .str.replace(r'\([^)]*\)', '', regex=True)
                        .str.replace(r'[^A-Z]', '', regex=True))
    q = feat.groupby(['stripped', 'z'])['Quality'].max()
    df['quality'] = df.set_index(['sequence', 'charge']).index.map(q)
    return df, sample_info


def pair_quant(df, sample_info, quality_min=None, *, label=''):
    """{pair: long frame with `pg_intensity`}, one DirectLFQ per mixture pair."""
    d0 = df[df['precursor_intensity'] > 0]
    if quality_min is not None:
        d0 = d0[d0['quality'].fillna(-1) >= quality_min]
    out = {}
    for a, b in PAIRS:
        runs = sample_info.loc[sample_info['condition2'].isin([a, b]), 'run']
        d = d0[d0['run'].isin(runs)]
        n = d.groupby('precursor_id')['run'].nunique()
        d = d[d['precursor_id'].isin(n.index[n == len(runs)])].copy()
        d['pg_intensity'] = directlfq_cached(d)
        out[(a, b)] = d
        print(f'  [{label}{a}/{b}] {d["precursor_id"].nunique():,} complete '
              f'precursors, {d["protein_group"].nunique():,} protein groups'
              + (f' (Quality >= {quality_min})' if quality_min else ''))
    return out


def ratio_tables(pq, sample_info, *, verbose=True):
    """Accuracy and both precisions, pooled over the per-pair quantifications."""
    from zenotof8600_analytics.species import _fit_line
    acc, between, cv = [], [], []
    for pair, d in pq.items():
        a, _ = core.species_ratio_accuracy(
            d, sample_info, [pair], COMPOSITION, species_order=SPECIES_ORDER,
            verbose=False, **QUANT_KW)
        b, c = core.species_ratio_precision(
            d, sample_info, [pair], COMPOSITION, species_order=SPECIES_ORDER,
            verbose=False, **QUANT_KW)
        acc.append(a)
        between.append(b)
        cv.append(c)
    accuracy = pd.concat(acc, ignore_index=True)
    fit = _fit_line(accuracy['expected'].to_numpy(),
                    accuracy['observed'].to_numpy())
    between = pd.concat(between, ignore_index=True)
    cv = (pd.concat(cv, ignore_index=True)
          .groupby(['condition', 'species'], as_index=False)
          .agg(load_pct=('load_pct', 'first'),
               protein_cv_pct=('protein_cv_pct', 'mean')))
    if verbose:
        print(f'  slope {fit["slope"]:.3f}  R2 {fit["r_squared"]:.4f}  '
              f'between-protein error median {between["error_pct"].median():.1f}%'
              f'  protein CV median {cv["protein_cv_pct"].median():.1f}%')
    return dict(accuracy=accuracy, fit=fit, between=between, cv=cv)


def panel_species_ratio(df, sample_info, *, out=None, name=None,
                        size=(ROW_B_W_IN, 4), target_w=None):
    """e — one pair, every protein group placed by ratio and abundance."""
    a, b = SCATTER_PAIR
    fig, axes, stats, ratios = core.plot_species_ratio(
        df, sample_info, a, b,
        composition_a=COMPOSITION[a], composition_b=COMPOSITION[b],
        reference_species=None, species_order=SPECIES_ORDER,
        palette=SPECIES_COLORS, figsize=size,
        legend_fontsize=fs.LEGEND_FONTSIZE, rasterize_over=RASTERIZE_OVER,
        **QUANT_KW)
    axes = np.atleast_1d(axes).ravel()
    n_raster = 0
    for ax in axes:
        for coll in ax.collections:
            if len(coll.get_offsets()) > RASTERIZE_OVER:
                coll.set_rasterized(True)
                n_raster += len(coll.get_offsets())
    for ax in axes:
        ax.tick_params(labelsize=fs.FONTSIZE)
        ax.xaxis.label.set_fontsize(fs.FONTSIZE)
        ax.yaxis.label.set_fontsize(fs.FONTSIZE)
        fs.tidy(ax)
    lo, hi = axes[1].get_xlim()
    if hi >= 1:
        axes[1].set_xticks([0, int(np.floor(hi))])
    handles, labels = axes[0].get_legend_handles_labels()
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    axes[0].legend(handles, labels, loc='lower left',
                   bbox_to_anchor=(0, 1.01, 1, 0.1), mode='expand',
                   ncol=len(labels), frameon=False,
                   fontsize=fs.LEGEND_FONTSIZE, handlelength=1.0,
                   handletextpad=0.5, borderpad=0.0, columnspacing=1.2)
    print(f'    {a} / {b}, n = {len(ratios):,} protein groups')
    print(f'    rasterized {n_raster:,} points at {RASTER_DPI} dpi; '
          f'text, axes, curves and boxes stay vector')
    print(stats.to_string())
    fig.tight_layout()
    if target_w is None:
        import make_main_figures as mf
        target_w = mf.full_row_width(3, LETTER['species_ratio'])
    got = fs.fit_saved_width(fig, target_w)
    print(f'    saved width {got:.2f} in (row target {target_w:.2f})')
    save(fig, name or (f'figure3{LETTER["species_ratio"]}_species_ratio_'
                       f'{a.lower()}_vs_{b.lower()}'),
         ratios, dpi=RASTER_DPI, out=out)
    plt.close(fig)


def panel_design():
    """a — the designed composition of each mixture, as stacked bars."""
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    bottom = np.zeros(len(COND_ORDER))
    rows = []
    for sp in SPECIES_ORDER:
        vals = np.array([COMPOSITION[c][sp] for c in COND_ORDER], float)
        ax.bar(range(len(COND_ORDER)), vals, bottom=bottom,
               color=SPECIES_COLORS[sp], edgecolor='black', linewidth=0.6,
               label=sp, zorder=2)
        for c, v, b in zip(COND_ORDER, vals, bottom):
            rows.append({'condition': c, 'species': sp, 'percent': v,
                         'cumulative_bottom': b})
        bottom += vals
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels([c.replace('HYE', '') for c in COND_ORDER],
                       fontsize=fs.FONTSIZE)
    ax.set_xlabel('HYE mixture', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Share of mixture [%]', fontsize=fs.FONTSIZE)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=fs.FONTSIZE)
    legend_on_top(ax, ax.get_legend_handles_labels()[0], ncol=3,
                  handlelength=0.9, handleheight=0.9)
    fs.tidy(ax)
    fig.tight_layout()
    fs.set_bar_geometry_inches(fig, ax, bar_in=BAR_IN, unit_in=UNIT_IN,
                               h_in=ROW_A_AXES_H_IN)
    save(fig, f'figure3{LETTER["design"]}_design_composition',
         pd.DataFrame(rows))
    plt.close(fig)


def panel_species_counts(df, sample_info):
    """b — protein groups identified per species, per condition."""
    fig, ax, src = core.plot_species_counts(
        df, sample_info, condition_order=COND_ORDER,
        species_order=SPECIES_ORDER, palette=SPECIES_COLORS,
        figsize=(fs.PANEL_W_IN, 4), legend_fontsize=fs.LEGEND_FONTSIZE)
    n = fs.strip_value_labels(ax)
    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(0.6)
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels([c.replace('HYE', '') for c in COND_ORDER],
                       rotation=0, ha='center', fontsize=fs.FONTSIZE)
    ax.set_xlabel('HYE mixture', fontsize=fs.FONTSIZE)
    thousands_axis(ax, 'Protein groups')
    ax.tick_params(labelsize=fs.FONTSIZE)
    ax.xaxis.label.set_fontsize(fs.FONTSIZE)
    ax.yaxis.label.set_fontsize(fs.FONTSIZE)
    legend_on_top(ax, ax.get_legend_handles_labels()[0], ncol=3,
                  handlelength=0.9, handleheight=0.9)
    fs.tidy(ax)
    fig.tight_layout()
    fs.set_bar_geometry_inches(fig, ax, bar_in=BAR_IN, unit_in=UNIT_IN,
                               h_in=ROW_A_AXES_H_IN)
    print(f'    dropped {n} in-bar count labels')
    save(fig, f'figure3{LETTER["species_counts"]}_species_counts', src)
    plt.close(fig)


def panel_method_ids(depth):
    """c, d — depth of the two acquisition methods on the identical design."""
    methods = list(BAR_ORDER)
    for key, col, ylabel in [('method_pg', 'pg', 'Protein groups'),
                             ('method_pep', 'pep', 'Peptides')]:
        fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
        vals = [depth[m][col] for m in methods]
        heights = [depth[m].attrs['union'][col] for m in methods]
        ax.bar(range(len(methods)), heights,
               color=[METHOD_COLOR[m] for m in methods],
               edgecolor='black', linewidth=0.6, zorder=2)
        rng = np.random.default_rng(0)
        w = bar_data_width(ax, min_axes_w_in=MIN_BAR_AXES_W_IN)
        for i, v in enumerate(vals):
            ax.scatter(i + rng.uniform(-0.30 * w, 0.30 * w, len(v)), v,
                       s=core.replicate_point_size(BAR_IN), color='black',
                       alpha=0.65, linewidth=0.3, edgecolor='white', zorder=5)
        h = dict(zip(methods, heights))
        gain = (h['ZT Scan'] - h['Zeno SWATH']) / h['Zeno SWATH'] * 100
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(' ', '\n') for m in methods],
                           fontsize=fs.FONTSIZE)
        ax.set_ylim(0, max(heights) * 1.15)
        thousands_axis(ax, ylabel)
        ax.tick_params(labelsize=fs.FONTSIZE)
        fs.tidy(ax)
        fig.tight_layout()
        fs.set_bar_geometry_inches(fig, ax, bar_in=BAR_IN, unit_in=UNIT_IN,
                                   h_in=ROW_A_AXES_H_IN,
                                   min_axes_w_in=MIN_BAR_AXES_W_IN)
        for lab, sign in zip(ax.get_xticklabels(), (-1, 1)):
            lab.set_transform(lab.get_transform() + ScaledTranslation(
                sign * LABEL_SPREAD_IN, 0, fig.dpi_scale_trans))
        rows = [{'level': col, 'method': m, 'union': h,
                 'mean_per_run': float(np.mean(v)),
                 'sd': float(np.std(v, ddof=1)), 'n_runs': len(v)}
                for m, h, v in zip(methods, heights, vals)]
        print(f'    {ylabel}: ' + '  '.join(
            f'{m} {v:,.0f}' for m, v in zip(methods, heights))
            + f'   ({gain:+.1f}% for ZT Scan over Zeno SWATH)')
        save(fig, f'figure3{LETTER[key]}_method_{col}', pd.DataFrame(rows))
        plt.close(fig)


def panel_accuracy(accuracy, fit, dia=None, *, out=None, name=None,
                   total_w_in=ROW_C_W_IN, h_in=ROW_C_AXES_H_IN):
    """f — measured against expected over all five pairs, UNCORRECTED."""
    fig, ax, src = core.plot_ratio_accuracy(
        accuracy, fit, correct=False, species_order=SPECIES_ORDER,
        palette=SPECIES_COLORS, figsize=(fs.PANEL_W_IN, 4),
        legend_fontsize=fs.LEGEND_FONTSIZE)
    for coll in ax.collections:
        coll.set_sizes([MARKER_AREA])
    for ln in ax.lines:
        if ln.get_linestyle() in ('--', (0, (6.4, 1.6))):
            ln.set_linestyle(':')
            ln.set_color('#999999')

    lo, hi = ax.get_xlim()
    fit_lines = []

    def draw_fit(f, method):
        x = np.array([lo, hi])
        ax.plot(x, f['slope'] * x + f['intercept'],
                ls=METHOD_LINESTYLE[method], color=FIT_COLOR, lw=1.3,
                zorder=2, solid_capstyle='butt')
        fit_lines.append(method)

    draw_fit(fit, 'ZT Scan')
    src = src.assign(method='ZT Scan')
    print(f'    ZT Scan     slope {fit["slope"]:.3f}  '
          f'intercept {fit["intercept"]:+.3f}  R2 {fit["r_squared"]:.4f}  '
          f'(n = {fit["n_points"]} points)')

    if dia is not None:
        acc_dia, fit_dia = dia
        for s in SPECIES_ORDER:
            sub = acc_dia[acc_dia['species'] == s]
            ax.scatter(sub['expected'], sub['observed'], s=MARKER_AREA,
                       marker=METHOD_MARKER['Zeno SWATH'],
                       color=SPECIES_COLORS[s], edgecolor='black',
                       linewidth=0.6, zorder=3)
        draw_fit(fit_dia, 'Zeno SWATH')
        src = pd.concat([src, acc_dia.assign(
            method='Zeno SWATH', fit_slope=fit_dia['slope'],
            fit_intercept=fit_dia['intercept'],
            compression_slope=fit_dia['slope'])], ignore_index=True)
        print(f'    Zeno SWATH  slope {fit_dia["slope"]:.3f}  '
              f'intercept {fit_dia["intercept"]:+.3f}  '
              f'R2 {fit_dia["r_squared"]:.4f}  '
              f'(n = {fit_dia["n_points"]} points)')

    for t in list(ax.texts):
        t.remove()

    fits = [('ZT Scan', fit)] + ([('Zeno SWATH', dia[1])] if dia else [])
    lines = [(f'{m}  ' if len(fits) > 1 else '')
             + f'slope {f["slope"]:.3f}' for m, f in fits]
    if len(fits) == 1:
        lines.append(f'R$^2$ {fit["r_squared"]:.4f}')
    ax.annotate('\n'.join(lines), xy=(0.97, 0.03), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=fs.LEGEND_FONTSIZE,
                linespacing=1.35)

    ax.tick_params(labelsize=fs.FONTSIZE)
    ax.xaxis.label.set_fontsize(fs.FONTSIZE)
    ax.yaxis.label.set_fontsize(fs.FONTSIZE)
    fs.tidy(ax)
    legend_two_rows(ax, *species_method_handles(
        method_line=True, methods=[m for m, _f in fits]))
    fig.tight_layout()
    ax.set_aspect('auto')
    fs.fit_panel_width(fig, ax, total_w_in=total_w_in, h_in=h_in)
    ax.set_aspect('equal', adjustable='box')
    save(fig, name or f'figure3{LETTER["accuracy"]}_ratio_accuracy', src,
         out=out)
    plt.close(fig)


def panel_precision(between, cv, dia=None):
    """g, h — the two precisions, SEPARATE panels."""
    between_dia, cv_dia = (dia if dia is not None else (None, None))

    for key, name in [('between_protein', 'between_protein_error'),
                      ('protein_cv', 'protein_cv')]:
        fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
        other = plt.figure().add_subplot(111)
        axes = ([ax, other] if key == 'between_protein' else [other, ax])
        _, _, src = core.plot_ratio_precision(
            between, cv, species_order=SPECIES_ORDER, palette=SPECIES_COLORS,
            axes=np.array(axes), legend_fontsize=fs.LEGEND_FONTSIZE)
        plt.close(other.figure)
        for ln in ax.lines:
            ln.set_markersize(MARKER_MS)
        src = src.assign(method='ZT Scan')
        if key == 'protein_cv':
            ax.set_ylabel('Protein quant CV [%]')
            ax.set_ylim(0, None)
            ax.set_xticks([0, 10, 20, 30, 40, 50])

        if dia is not None:
            for s in SPECIES_ORDER:
                mk = dict(marker=METHOD_MARKER['Zeno SWATH'], ms=MARKER_MS,
                          mec='black', mew=0.8, color=SPECIES_COLORS[s],
                          zorder=3)
                if key == 'between_protein':
                    b = between_dia[between_dia['species'] == s]
                    ax.plot(b['expected'], b['error_pct'], ls='none', **mk)
                else:
                    c = cv_dia[cv_dia['species'] == s].sort_values('load_pct')
                    ax.plot(c['load_pct'], c['protein_cv_pct'],
                            ls=METHOD_LINESTYLE['Zeno SWATH'], lw=1.2, **mk)
            add = (between_dia.assign(panel='ratio_error_between_proteins')
                   if key == 'between_protein'
                   else cv_dia.assign(panel='protein_quant_cv'))
            src = pd.concat([src, add.assign(method='Zeno SWATH')],
                            ignore_index=True)

        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_title('')
        legend_two_rows(ax, *species_method_handles(
            line=(key == 'protein_cv'),
            methods=list(METHODS)[:None if dia is not None else 1]))
        ax.tick_params(labelsize=fs.FONTSIZE)
        ax.xaxis.label.set_fontsize(fs.FONTSIZE)
        ax.yaxis.label.set_fontsize(fs.FONTSIZE)
        fs.tidy(ax)
        fig.tight_layout()
        fs.fit_panel_width(fig, ax, total_w_in=ROW_C_W_IN,
                           h_in=ROW_C_AXES_H_IN)
        keep = src[src['panel'] == ('ratio_error_between_proteins'
                                    if key == 'between_protein'
                                    else 'protein_quant_cv')]
        save(fig, f'figure3{LETTER[key]}_{name}', keep)
        plt.close(fig)
    return between, cv


def depth_table():
    """Per-run identification depth for both methods, unfiltered."""
    out = {}
    for m in METHODS:
        df, si = load(m)
        d = df[df['precursor_intensity'] > 0]
        per = d.groupby('run').agg(pg=('protein_group', 'nunique'),
                                   pep=('peptide_id', 'nunique'),
                                   pre=('precursor_id', 'nunique'))
        per.attrs['union'] = {'pg': d['protein_group'].nunique(),
                              'pep': d['peptide_id'].nunique(),
                              'pre': d['precursor_id'].nunique()}
        out[m] = per
        print(f'  [{m}] {len(per)} runs, {per.pg.mean():,.0f} PG/run, '
              f'union {per.attrs["union"]["pg"]:,} PG')
    return out


def main(*letters):
    """Build all panels, or only the letters asked for."""
    wanted = set(letters) or set(LETTER.values())
    unknown = wanted - set(LETTER.values())
    if unknown:
        raise SystemExit(f'unknown panel(s) {sorted(unknown)}; this figure '
                         f'has {sorted(set(LETTER.values()))}')
    fs.init()
    fs.start_build(script=__file__, inputs=[DATA], out_dir=OUT)
    if wanted != set(LETTER.values()):
        print(f'building only: {" ".join(sorted(wanted))}')

    cache = {}

    def once(key, label, make):
        if key not in cache:
            print(f'\nloading {label} ...')
            cache[key] = make()
        return cache[key]

    def zt_raw():
        return once('zt_raw', 'ZT Scan', lambda: load('ZT Scan'))

    def zt_pairs():
        return once('zt_pairs', 'ZT Scan, per-pair DirectLFQ',
                    lambda: pair_quant(*zt_raw(), label='ZT '))

    def zt():
        return once('zt_tables', 'ZT Scan ratio tables',
                    lambda: ratio_tables(zt_pairs(), zt_raw()[1]))

    def swath():
        def make():
            df, si = load('Zeno SWATH')
            return ratio_tables(pair_quant(df, si, label='SWATH '), si)
        return once('swath_tables', 'Zeno SWATH ratio tables', make)

    if LETTER['design'] in wanted:
        print('\npanel a -- designed composition ...')
        panel_design()
    if LETTER['species_counts'] in wanted:
        print('\npanel b -- species counts ...')
        panel_species_counts(*zt_raw())
    if {LETTER['method_pg'], LETTER['method_pep']} & wanted:
        print('\npanels c, d -- method depth ...')
        panel_method_ids(depth_table())
    if LETTER['accuracy'] in wanted:
        print('\npanel f -- ratio accuracy ...')
        t = zt()
        print('  Zeno SWATH (measured, not drawn -- see supplementary fig 3):')
        swath()
        panel_accuracy(t['accuracy'], t['fit'])

    if {LETTER['between_protein'], LETTER['protein_cv']} & wanted:
        print('\npanels g, h -- precision ...')
        print('  Zeno SWATH (measured, not drawn):')
        swath()
        t = zt()
        panel_precision(t['between'], t['cv'])

    if LETTER['species_ratio'] in wanted:
        print('\npanel e -- species ratio ...')
        panel_species_ratio(zt_pairs()[SCATTER_PAIR], zt_raw()[1])

    for n in (4, 3):
        fs.report_printed_sizes(ROW_W_IN / n, n)
    fs.report_stale(OUT)


if __name__ == '__main__':
    main(*sys.argv[1:])
