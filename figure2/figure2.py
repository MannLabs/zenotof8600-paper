"""Figure 2 — ZT Scan: choosing the Q1 isolation width, H032_E376."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter
import pandas as pd

from scipy.spatial import cKDTree

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(cfg.DATA_ROOT, 'H032_E376')
OUT = cfg.panels(__file__)

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

CONDITIONS = [
    {'label': 'DIA', 'dir': '40SPD_100vw12ms', 'width_th': None},
    {'label': '1', 'dir': '40SPD_1Th_1p48s', 'width_th': 1.0},
    {'label': '2.8', 'dir': '40SPD_2p8Th_1p48s', 'width_th': 2.8},
    {'label': '4.3', 'dir': '40SPD_4p3Th_1p48s', 'width_th': 4.3},
    {'label': '5.7', 'dir': '40SPD_5p7Th_1p48s', 'width_th': 5.7},
    {'label': '7.2', 'dir': '40SPD_7p2Th_1p48s', 'width_th': 7.2},
]
COND_ORDER = [c['label'] for c in CONDITIONS]

COLOR = {c: core.PALETTE_SINGLE[i % len(core.PALETTE_SINGLE)]
         for i, c in enumerate(COND_ORDER)}

CHOSEN = '2.8'


LETTER = {
    'schematic': 'a',
    'ids_prec': 'b',
    'ids_pg': 'c',
    'rt_q1_map': 'd',
    'q1_window': 'e',
    'dilution_prec': 'f',
    'dilution_pg': 'g',
}

SUPP_OUT = os.path.join(os.path.dirname(HERE), 'supplementary_figure2',
                        'panels')
SUPP_LETTER = {'coisolation': 'd'}


def load():
    """Canonical long df + sample_info for the six 40 SPD arms."""
    experiments = []
    for c in CONDITIONS:
        d = os.path.join(DATA, c['dir'])
        f = os.path.join(d, 'lfq.dia.features.csv')
        runs = [x[:-len(' m/z')] for x in pd.read_csv(f, nrows=0).columns
                if x.endswith(' m/z') and x != 'm/z']
        experiments.append({
            'path': f, 'protein_path': os.path.join(d, 'lfq.dia.proteins.csv'),
            'file_tags': runs, 'condition1': '250 ng HEK, 40 SPD, 1.48 s',
            'condition2': c['label']})
    df, sample_info = core.load_experiments(experiments)
    sample_info['replicate'] = sample_info.groupby('condition2').cumcount() + 1
    return df, sample_info


def save(fig, name, source_df):
    return fs.save_panel(fig, OUT, name, source_df, dpi=300)


def panel_ids(agg):
    """b, c — identifications against Q1 width, with the CV<20% subset."""
    for name, total_col, cv_col, ylabel in [
        (f'figure2{LETTER["ids_prec"]}_precursor_ids', 'total_precursors', 'Pr20',
         '# precursors'),
        (f'figure2{LETTER["ids_pg"]}_protein_group_ids', 'total_protein_groups', 'PG20',
         '# protein groups'),
    ]:
        fig, ax, summary = core.plot_overlapping_bars(
            agg, x_col='condition2',
            higher_y_col=total_col,
            lower_y_col=cv_col,
            higher_label='IDs', lower_label='< 20% CV',
            palette=[COLOR[c] for c in COND_ORDER],
            figsize=(fs.PANEL_W_IN, 4), y_label=ylabel,
            label_fontsize=fs.FONTSIZE, tick_fontsize=fs.FONTSIZE,
            legend_fontsize=fs.LEGEND_FONTSIZE,
            value_fontsize=fs.FONTSIZE,
            point_size=core.replicate_point_size(IDS_BAR_IN))
        if not fs.SHOW_VALUE_LABELS:
            fs.strip_value_labels(ax)
        ax.set_xlabel('Q1 isolation width [Th]', fontsize=fs.FONTSIZE)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        fs.tidy(ax)
        ax.tick_params(labelsize=fs.FONTSIZE)
        ax.xaxis.label.set_fontsize(fs.FONTSIZE)
        ax.yaxis.label.set_fontsize(fs.FONTSIZE)

        tot = (agg.groupby('condition2')[total_col].first().reindex(COND_ORDER))
        q20 = agg.groupby('condition2')[cv_col].first().reindex(COND_ORDER)
        print(f'    {ylabel}')
        for c in COND_ORDER:
            mark = '  <- chosen' if c == CHOSEN else ''
            print(f'      {c.replace(chr(10), " "):10s} '
                  f'{tot[c]:>9,.0f}   <20% CV {q20[c]:>9,.0f}{mark}')
        print(f'      totals peak at {tot.idxmax().replace(chr(10), " ")!r}, '
              f'quantifiable peak at {q20.idxmax().replace(chr(10), " ")!r}')

        fig.tight_layout()
        fs.set_bar_geometry_inches(fig, ax, bar_in=IDS_BAR_IN,
                                   unit_in=IDS_UNIT_IN,
                                   h_in=RESULT_AXES_H_IN)
        save(fig, name, summary)
        plt.close(fig)


HEATMAP_CANDIDATES = [('A9', '2.8 Th'), ('A5', '1 Th')]

ZOOM_RT = 25.41
ZOOM_Q1 = (636.886, 639.682)
ZOOM_PAD_RT = 1.2
ZOOM_PAD_Q1 = 12.0


def heatmap_grid():
    """(label, width, npz) for the best available RT x Q1 grid."""
    for run, width in HEATMAP_CANDIDATES:
        p = os.path.join(DATA, f'ztscan_rt_q1_map_{run}_grid.npz')
        if os.path.exists(p):
            return run, width, np.load(p)
    raise FileNotFoundError(
        f'no ztscan_rt_q1_map_*_grid.npz in {DATA}; build one with '
        f'ztscan_api\\ztscan_rt_q1_map.py <hdf> <run> ztscan_rt_q1_map')


def _edges(c):
    """Cell edges from sample CENTRES, midpoints between neighbours."""
    c = np.asarray(c, float)
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate(([c[0] - (mid[0] - c[0])], mid,
                           [c[-1] + (c[-1] - mid[-1])]))


def panel_rt_q1_map():
    """c — summed MS2 ion current over retention time and Q1 position."""
    run, width, z = heatmap_grid()
    tic, seen, rt, q1 = z['tic'], z['seen'], z['rt'], z['q1']
    print(f'    grid {run} ({width}): {tic.shape[0]:,} cycles x '
          f'{tic.shape[1]:,} Q1 positions, {100 * seen.mean():.2f}% acquired')

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    M = np.ma.masked_where(~seen | (tic <= 0), tic)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('white')
    pos = tic[seen & (tic > 0)]
    im = ax.pcolormesh(_edges(rt), _edges(q1), M.T,
                       norm=LogNorm(vmin=float(np.percentile(pos, 0.5)),
                                    vmax=float(pos.max())),
                       cmap=cmap, shading='auto', rasterized=True)
    zoom_box = ax.add_patch(Rectangle(
        (ZOOM_RT - ZOOM_PAD_RT, ZOOM_Q1[0] - ZOOM_PAD_Q1),
        2 * ZOOM_PAD_RT, (ZOOM_Q1[1] - ZOOM_Q1[0]) + 2 * ZOOM_PAD_Q1,
        fill=False, edgecolor='white', lw=1.2))
    ax.set_xlabel('Retention time [min]', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Q1 dimension [m/z]', fontsize=fs.FONTSIZE)
    ax.tick_params(labelsize=fs.FONTSIZE)
    cb = fig.colorbar(im, ax=ax, label='summed MS2 intensity')
    cb.ax.tick_params(labelsize=fs.FONTSIZE)
    cb.set_label('summed MS2 intensity', fontsize=fs.FONTSIZE)
    fs.tidy(ax)

    source = pd.DataFrame({'run': [run], 'isolation_width': [width],
                           'n_cycles': [tic.shape[0]],
                           'n_q1_positions': [tic.shape[1]],
                           'rt_min': [rt.min()], 'rt_max': [rt.max()],
                           'q1_min': [q1.min()], 'q1_max': [q1.max()],
                           'fraction_acquired': [float(seen.mean())],
                           'zoom_rt': [ZOOM_RT],
                           'zoom_q1_lo': [ZOOM_Q1[0]],
                           'zoom_q1_hi': [ZOOM_Q1[1]],
                           'grid_npz': [f'ztscan_rt_q1_map_{run}_grid.npz']})
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=3.20, h_in=ROW_BC_AXES_H_IN)
    x0, y0 = zoom_box.get_x(), zoom_box.get_y()
    x1, y1 = x0 + zoom_box.get_width(), y0 + zoom_box.get_height()
    fs.save_anchors(fig, OUT, f'figure2{LETTER["rt_q1_map"]}_rt_q1_map', {
        'zoom_top_right': ax.transData.transform((x1, y1)),
        'zoom_bottom_right': ax.transData.transform((x1, y0))})
    save(fig, f'figure2{LETTER["rt_q1_map"]}_rt_q1_map', source)
    plt.close(fig)
    if run != 'A9':
        print(f'    WARNING: heatmap is {run} ({width}), not the 2.8 Th run '
              f'panel f zooms into -- say so in the caption or build A9')


DILUTION = os.path.join(cfg.DATA_ROOT, 'H032_E377')

LOADS = [('0.25', '0p25ng'), ('0.833', '0p833ng'), ('2.5', '2p5ng'),
         ('8.3', '8p3ng'), ('25', '25ng'), ('83.3', '83p3ng'),
         ('250', '250ng'), ('500', '500ng'), ('1000', '1000ng')]


def panel_dilution():
    """Identifications against HEK load, precursors then protein groups."""
    experiments, order, ng = [], [], []
    for lab, d in LOADS:
        p = os.path.join(DILUTION, d)
        f = os.path.join(p, 'lfq.dia.features.csv')
        runs = [x[:-len(' m/z')] for x in pd.read_csv(f, nrows=0).columns
                if x.endswith(' m/z') and x != 'm/z']
        experiments.append({
            'path': f, 'protein_path': os.path.join(p, 'lfq.dia.proteins.csv'),
            'file_tags': runs, 'condition1': 'HEK dilution',
            'condition2': lab})
        order.append(lab)
        ng.append(float(lab))

    df, si = core.load_experiments(experiments)
    si['replicate'] = si.groupby('condition2').cumcount() + 1
    agg = core.process_experiment(df, si, group_col='condition2')

    for key, total_col, cv_col, ylabel, stem in [
        ('dilution_prec', 'total_precursors', 'Pr20',
         '# precursors', 'dilution_precursors'),
        ('dilution_pg', 'total_protein_groups', 'PG20',
         '# protein groups', 'dilution_protein_groups'),
    ]:
        t = (agg.groupby('condition2')[[total_col, cv_col]]
             .first().reindex(order))
        fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
        color = COLOR[CHOSEN]
        ax.plot(ng, t[total_col], '-o', color=color, lw=2.0, ms=5,
                label='identified')
        ax.plot(ng, t[cv_col], '--o', color=color, lw=2.0, ms=5,
                markerfacecolor='white', label='< 20% CV')
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.set_xlabel('HEK digest loaded [ng]', fontsize=fs.FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=fs.FONTSIZE)
        ax.set_ylim(0, float(t[total_col].max()) * 1.15)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.tick_params(labelsize=fs.FONTSIZE)
        ax.legend(loc='lower right', frameon=False,
                  fontsize=fs.LEGEND_FONTSIZE, handlelength=1.6,
                  handletextpad=0.5, borderpad=0.2, labelspacing=0.3)
        fs.tidy(ax)

        source = t.reset_index().rename(
            columns={'condition2': 'load_ng_label'})
        source['load_ng'] = ng
        print(f'    {ylabel}')
        print('      ' + '  '.join(f'{a}ng={b:,.0f}' for a, b in
                                   zip(order, t[total_col])))
        print(f'      identified peak at '
              f'{order[int(np.argmax(t[total_col]))]} ng, quantifiable peak '
              f'at {order[int(np.argmax(t[cv_col]))]} ng')
        fig.tight_layout()
        fs.set_axes_size_inches(fig, ax, w_in=3.10, h_in=fs.AXES_H_IN)
        save(fig, f'figure2{LETTER[key]}_{stem}', source)
        plt.close(fig)


COELUTION_MIN = 0.05

Q1_PRECISION_P32_CSV = os.path.join(DATA, 'q1_precision_by_width_p32.csv')
Q1_PRECISION_CSV = Q1_PRECISION_P32_CSV


Q1_WHOLEFILE_DIR = os.path.join(DATA, 'q1_wholefile')
Q1_WHOLEFILE_TAGS = {'A5': 1.0, 'A9': 2.8, 'B1': 4.3, 'B5': 5.7, 'B9': 7.2}
Q1_MIN_PCT = 50.0
Q1_GATES = 'strict'
_WHOLEFILE_CACHE = {}


def q1_wholefile(gates=None, min_pct=Q1_MIN_PCT):
    """Placed precursors from the whole-file measurement, all five arms."""
    gates = gates or Q1_GATES
    key = (gates, min_pct)
    if key in _WHOLEFILE_CACHE:
        return _WHOLEFILE_CACHE[key]
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from measure_q1_precision import detrend
    parts = []
    for tag, width in Q1_WHOLEFILE_TAGS.items():
        d = pd.read_csv(os.path.join(Q1_WHOLEFILE_DIR, f'{tag}.csv'))
        d = d[d[f'err_raw_{gates}'].notna() & (d['area_pct'] >= min_pct)]
        e = detrend(d[f'err_raw_{gates}'].to_numpy(), d['mz'].to_numpy())
        e = detrend(e, d['rt'].to_numpy())
        parts.append(d.assign(width_th=width, err=e, gates=gates))
    out = pd.concat(parts, ignore_index=True)
    _WHOLEFILE_CACHE[key] = out
    return out


OLD_AXIS_DIR = os.path.join(DATA, 'q1_calibration_old')

Q1_AXIS_WHOLEFILE = os.path.join(DATA, 'q1_axis_A9_wholefile.json')

Q1_AXIS_LOESS_FRAC = 0.10


def uncalibrated_q1(d, tag):
    """Raw Q1 position of each whole-file placement, with the provisional axis removed."""
    import json
    from zenotof8600_analytics.ztscan.axis import q1_correction
    q1 = (d['mz'] + d['err_raw_strict']).to_numpy()
    p = os.path.join(OLD_AXIS_DIR, f'q1_axis_{tag}.json')
    if not os.path.exists(p):
        return q1
    with open(p) as fh:
        old = json.load(fh)
    rt_c = d['tile'].str.extract(r'^rt([0-9.]+)_')[0].astype(float).to_numpy()
    raw = q1 + q1_correction(old, q1, rt_c)
    return q1 + q1_correction(old, raw, rt_c)


def wholefile_placements(tag, min_pct=Q1_MIN_PCT):
    """One arm's strict placements at/above `min_pct`, with raw `q1`."""
    d = pd.read_csv(os.path.join(Q1_WHOLEFILE_DIR, f'{tag}.csv'))
    d = d[d['err_raw_strict'].notna() & (d['area_pct'] >= min_pct)]
    return d.assign(q1=uncalibrated_q1(d, tag))


def q1_sigma_by_width(path=None):
    """{width_th: across-precursor sigma_IQR}."""
    t = pd.read_csv(path) if path else q1_wholefile()
    out = {}
    for w, g in t.groupby('width_th'):
        e = g['err'].dropna()
        out[float(w)] = float((e.quantile(0.75) - e.quantile(0.25)) / 1.349)
    return out


K_SIGMA = 1.0


def n_within_tol(mz, rt, half_width, rt_tol):
    """Co-eluting precursors inside +-half_width m/z of each precursor."""
    pts = np.column_stack([mz / half_width, rt / rt_tol])
    return cKDTree(pts).query_ball_point(pts, r=1.0, p=np.inf,
                                         return_length=True)


ABUNDANT_PCT = 0.50


def abundant_reference(df, sample_info, *, min_pct=ABUNDANT_PCT):
    """The union precursor set restricted to the reasonably abundant ones."""
    d = df[df['precursor_intensity'] > 0].merge(
        sample_info[['run', 'condition2']], on='run')
    per_arm = (d.groupby(['condition2', 'precursor_id'])['precursor_intensity']
               .median().rename('inten').reset_index())
    per_arm['pct'] = per_arm.groupby('condition2')['inten'].rank(pct=True)
    pct = per_arm.groupby('precursor_id')['pct'].median()
    ref = (d.groupby('precursor_id').agg(mz=('mz', 'median'),
                                         rt=('rt', 'median')).dropna())
    ref['abundance_pct'] = pct
    n_all = len(ref)
    ref = ref[ref['abundance_pct'] >= min_pct].sort_values('mz')
    print(f'    reference set: {len(ref):,} of {n_all:,} precursors (union of '
          f'all arms, >= {100 * min_pct:.0f}th abundance percentile)')
    return ref


def panel_coisolation(df, sample_info):
    """e — spectral complexity with and without the Q1 dimension."""
    ref = abundant_reference(df, sample_info)
    mz = ref['mz'].to_numpy()
    rt = ref['rt'].to_numpy()

    def n_within(half_width):
        return n_within_tol(mz, rt, half_width, COELUTION_MIN)

    sig = q1_sigma_by_width()
    widths = [w for w in (c['width_th'] for c in CONDITIONS) if w]

    print(f'    sensitivity (median co-eluting precursors)')
    print('               RT tol ->' + ''.join(f'{t:>8.2f}'
                                               for t in (0.02, 0.05, 0.10)))
    for w in widths:
        vals = [np.median(n_within_tol(mz, rt, w / 2, t))
                for t in (0.02, 0.05, 0.10)]
        print(f'      window {w:>4} Th   ' + ''.join(f'{v:>8.0f}' for v in vals))
    s_chosen = sig[float(CHOSEN)]
    for k in (0.5, 1.0, 2.0, 3.0):
        vals = [np.median(n_within_tol(mz, rt, k * s_chosen, t))
                for t in (0.02, 0.05, 0.10)]
        print(f'      Q1 +-{k:>3}sig     ' + ''.join(f'{v:>8.0f}' for v in vals)
              + f'   ({2 * k * s_chosen:.2f} Th at {CHOSEN} Th)')
    print(f'    chosen: RT +-{COELUTION_MIN} min, Q1 +-{K_SIGMA:g} sigma '
          f'(element {2 * K_SIGMA * sig[min(sig)]:.2f}-'
          f'{2 * K_SIGMA * sig[max(sig)]:.2f} Th over the series)')

    rows = []
    for w in widths:
        n = n_within(w / 2)
        q1_half = K_SIGMA * sig[w]
        n_q1 = n_within(q1_half)
        alone = n <= 1
        rows.append({'width_th': w,
                     'mean_without_q1': float(n.mean()),
                     'mean_with_q1': float(n_q1.mean()),
                     'median_without_q1': float(np.median(n)),
                     'median_with_q1': float(np.median(n_q1)),
                     'fraction_alone_without_q1': float(alone.mean()),
                     'q1_sigma_th': sig[w],
                     'q1_resolving_width_th': 2 * q1_half,
                     'n_precursors': len(mz)})
        print(f'      {w:>4} Th   sigma {sig[w]:.3f}   '
              f'without Q1 {n.mean():>5.1f}   '
              f'with Q1 {n_q1.mean():>4.1f}   '
              f'({n.mean() / n_q1.mean():.1f}x)   '
              f'{100 * alone.mean():4.1f}% alone')
    src = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    ax.plot(src['width_th'], src['mean_without_q1'], '-o',
            color=core.PALETTE_SINGLE[0], lw=2.2, ms=5,
            label='one isolation window')
    ax.plot(src['width_th'], src['mean_with_q1'], '-o',
            color=core.PALETTE_SINGLE[1], lw=2.2, ms=5,
            label=f'resolved by Q1 centroid ($\\pm${K_SIGMA:g}$\\sigma$)')

    ax.set_xlabel('Q1 isolation width [Th]', fontsize=fs.FONTSIZE)
    ax.set_ylabel('co-isolated precursors', fontsize=fs.FONTSIZE)
    ax.set_ylim(0, None)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=1,
              frameon=False, fontsize=fs.LEGEND_FONTSIZE,
              handlelength=1.6, handletextpad=0.5, borderpad=0.15,
              labelspacing=0.3)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=2.60, h_in=fs.AXES_H_IN)
    plt.close(fig)


Q1_WINDOW_NPZ = os.path.join(DATA, 'q1_window_A9_wholefile.npz')

ZOOM_ROWS = [0.80, 1.05, 1.55, 0.90]

ZOOM_SPACER = 0.55

ZOOM_HSPACE = 0.16

ROW_BC_AXES_H_IN = 4.00

RESULT_AXES_H_IN = 1.85

IDS_BAR_IN = 0.21
IDS_UNIT_IN = 0.27


def panel_q1_window():
    """f — one real 2.8 Th window holding four co-eluting peptides."""
    if not os.path.exists(Q1_WINDOW_NPZ):
        print(f'    SKIPPING: no {os.path.basename(Q1_WINDOW_NPZ)} -- build it '
              f'once with extract_q1_window.py')
        return
    z = np.load(Q1_WINDOW_NPZ)
    qs, mzs = z['qs'], z['mzs']
    kept = list(z['kept'])
    labels = z['labels']
    win_lo, win_hi = float(z['win_lo']), float(z['win_hi'])
    q_lo, q_hi = float(z['q_lo']), float(z['q_hi'])
    cols = {k: core.PALETTE_SINGLE[i % len(core.PALETTE_SINGLE)]
            for i, k in enumerate(kept)}

    fig, all_axes = plt.subplots(
        5, 1, figsize=(3.55, 4.80),
        gridspec_kw=dict(height_ratios=ZOOM_ROWS[:3] + [ZOOM_SPACER]
                         + ZOOM_ROWS[3:]))
    all_axes[3].set_visible(False)
    axes = [all_axes[0], all_axes[1], all_axes[2], all_axes[4]]
    axes[0].sharex(axes[2])
    axes[1].sharex(axes[2])
    rows = []

    ax = axes[0]
    ax.axvspan(win_lo, win_hi, color='0.55', alpha=0.16, lw=0, zorder=0)
    if 'ms1_mz' in z:
        pm, pv = z['ms1_mz'], z['ms1_int']
        ax.plot(pm, pv, color='0.35', lw=0.9, zorder=2)
        ax.fill_between(pm, 0, pv, color='0.75', alpha=0.35, zorder=1)
    for k in kept:
        ax.axvline(mzs[k], color=cols[k], lw=1.1, ls='--', alpha=0.9, zorder=3)
    ax.set_ylabel('TIC MS1', fontsize=fs.FONTSIZE)
    ax.set_xlim(q_lo, q_hi)

    ax = axes[1]
    ax.axvspan(win_lo, win_hi, color='0.55', alpha=0.10, lw=0, zorder=0)
    for k in kept:
        sub = z[f'p{k}_profiles']
        for row in sub:
            ax.plot(qs, row / row.max(), color=cols[k], lw=0.6, alpha=0.28)
        prof = (sub / sub.max(axis=1)[:, None]).mean(axis=0)
        ax.plot(qs, prof / prof.max(), color=cols[k], lw=2.2)
        ax.axvline(mzs[k], color=cols[k], lw=1.0, ls='--', alpha=0.7)
    ax.set_ylabel('Q1 profile', fontsize=fs.FONTSIZE)
    ax.set_xlim(q_lo, q_hi)
    ax.set_ylim(0, 1.08)

    ax = axes[2]
    rng = np.random.default_rng(0)
    ax.axvspan(win_lo, win_hi, color='0.55', alpha=0.10, lw=0, zorder=0)
    for i, k in enumerate(kept):
        y = len(kept) - 1 - i
        cc = z[f'p{k}_centroids']
        ax.axhline(y, color='0.9', lw=0.6, zorder=0)
        ax.scatter(cc, y + rng.uniform(-0.15, 0.15, cc.size), s=16,
                   color=cols[k], alpha=0.85, linewidth=0.3,
                   edgecolor='white', zorder=3)
        ax.plot([np.median(cc)], [y], marker='|', ms=16, color=cols[k],
                mew=2.4, zorder=4)
        ax.axvline(mzs[k], color=cols[k], lw=1.0, ls='--', alpha=0.7, zorder=1)
        rows.append({'sub_panel': 'iii centroids', 'peptide': labels[k],
                     'precursor_mz': mzs[k], 'median_centroid':
                     float(np.median(cc)), 'n_fragments': len(cc)})
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(kept) + 0.15)
    ax.set_xlim(q_lo, q_hi)
    ax.set_ylabel('Q1 centroid', fontsize=fs.FONTSIZE)

    ax = axes[3]
    ax.vlines(z['spec_mz'], 0, z['spec_int'], color='0.72', lw=0.6, zorder=1)
    tot = float(z['spec_int'].sum())
    claimed = 0.0
    for k in kept:
        sm, si = z[f'p{k}_spec_mz'], z[f'p{k}_spec_int']
        ax.vlines(sm, 0, si, color=cols[k], lw=1.5, zorder=3)
        claimed += float(si.sum())
    ax.set_xlim(float(z['frag_mz_lo']), float(z['frag_mz_hi']))
    ax.set_xlabel('Fragment m/z', fontsize=fs.FONTSIZE)
    ax.set_ylabel('TIC MS2', fontsize=fs.FONTSIZE)

    axes[2].set_xlabel('Q1 dimension [m/z]', fontsize=fs.FONTSIZE)
    for ax in axes:
        ax.tick_params(labelsize=fs.FONTSIZE)
        fs.tidy(ax)
    for ax in (axes[0], axes[1]):
        plt.setp(ax.get_xticklabels(), visible=False)

    med = np.array([np.median(z[f'p{k}_centroids']) for k in kept])
    gap = np.diff(np.sort(med))
    err = med - mzs[kept]
    across = float((np.percentile(err, 75) - np.percentile(err, 25)) / 1.349)
    scale = max(across, q1_sigma_by_width()[2.8])
    print(f'    {len(kept)} peptides in one {float(z["dia_win"]):g} Th window, '
          f'span {mzs[kept].max() - mzs[kept].min():.3f} Th')
    print(f'    closest measured pair {gap.min():.3f} Th = '
          f'{gap.min() / scale:.1f} sigma against {scale:.3f} Th')
    print(f'    order preserved: '
          f'{bool(np.all(np.argsort(med) == np.argsort(mzs[kept])))}')
    print(f'    chimeric spectrum: {len(z["spec_mz"]):,} peaks, '
          f'{claimed / tot:.1%} of intensity assigned')

    source = pd.DataFrame(rows)
    source['window_lo'] = win_lo
    source['window_hi'] = win_hi
    source['closest_pair_th'] = gap.min()
    source['across_precursor_sigma_th'] = scale
    fig.tight_layout()
    fig.subplots_adjust(hspace=ZOOM_HSPACE)
    fs.save_anchors(fig, OUT, f'figure2{LETTER["q1_window"]}_q1_window', {
        'zoom_top_left': axes[0].transAxes.transform((0, 1)),
        'zoom_bottom_left': axes[2].transAxes.transform((0, 0))})
    save(fig, f'figure2{LETTER["q1_window"]}_q1_window', source)
    plt.close(fig)


def main():
    fs.init()
    fs.start_build(script=__file__, inputs=[DATA], out_dir=OUT)
    print(f'loading figure 2 inputs from {DATA} ...')
    df, sample_info = load()
    print(f'  {df.shape[0]:,} rows, {len(sample_info)} runs')
    agg = core.process_experiment(df, sample_info, group_col='condition2')
    import schematic
    print('\npanel a -- acquisition schematic ...')
    schematic.panel(LETTER['schematic'])
    print('\npanels b, c -- identifications ...')
    panel_ids(agg)
    print('\npanel d -- RT x Q1 map ...')
    panel_rt_q1_map()
    print('\npanel e -- Q1 window zoom ...')
    panel_q1_window()
    print('\npanels f, g -- dilution series ...')
    panel_dilution()
    print('\nco-isolation -> supplementary figure 2 ...')
    panel_coisolation(df, sample_info)
    fs.report_printed_sizes()
    fs.report_stale(OUT, prefix='figure')


if __name__ == '__main__':
    main()
