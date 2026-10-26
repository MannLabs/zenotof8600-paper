"""Supplementary figure 6 -- the 6x5 standard quantified at MS1 against MS2."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = cfg.panels(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import figure_style as fs

E396 = cfg.data('H032_E396')
MS2_LEVELS = os.path.join(E396, '6x5_levels_sourcedata.csv')
MS1_LEVELS = os.path.join(E396, 'ms1', '6x5_levels_sourcedata.csv')
MS2_CAL = os.path.join(E396, '6x5_calibration.csv')
E396_DIR = cfg.data('H032_E396')

PEPTIDE_ORDER = ['LASVSVSR', 'LGFTDLFSK', 'LLSLGAGEFK', 'VVGGLVALR',
                 'YVYVADVAAK']
COLORS = {p: core.PALETTE_SINGLE[i % len(core.PALETTE_SINGLE)]
          for i, p in enumerate(PEPTIDE_ORDER)}
LEVEL_STYLE = {'MS2': '-', 'MS1': (0, (4, 2.5))}
LEVEL_MARKER = {'MS2': 'o', 'MS1': 's'}
AMOL_TICKS = [2, 20, 200, 2000, 20000]
LETTER = {'traces': 'd', 'trace_r': 'e', 'response': 'f', 'lowest': 'g',
          'cv': 'h'}
TRACE_PEPTIDE, TRACE_RUN = 'LGFTDLFSK', 'A1'
TRACE_HALF_FWHM = 6.0


def _rules():
    """figure6's rules_6x5: one acceptance for MS1 and MS2 (2026-10-06)."""
    for p in (E396_DIR, os.path.join(ROOT, 'figure6')):
        if p not in sys.path:
            sys.path.insert(0, p)
    import rules_6x5
    return rules_6x5


def per_level(path, level_name, rules):
    """Mean area, CV and acceptance per (peptide, level)."""
    d = pd.read_csv(path)
    acc = {(pep, int(lv)): bool(ok) for pep, g0 in d.groupby('peptide')
           for lv, ok in rules.accepted_from_table(g0).items()}
    g = d.groupby(['peptide', 'level', 'amol'])
    t = g.agg(mean_area=('ions', 'mean'),
              sd_area=('ions', lambda x: x.std(ddof=1)),
              trace_r=('trace_r', 'mean'),
              n_pos=('ions', lambda x: int((x > 0).sum())),
              n=('ions', 'size')).reset_index()
    t['cv_pct'] = 100 * t['sd_area'] / t['mean_area']
    t['accepted'] = [acc[(p_, int(lv))] and m > 0 for p_, lv, m in
                     zip(t['peptide'], t['level'], t['mean_area'])]
    t = t.sort_values(['peptide', 'level'], ascending=[True, False])
    top = t[t['level'] == t.groupby('peptide')['level'].transform('max')]
    t = t.merge(top[['peptide', 'mean_area']].rename(
        columns={'mean_area': 'top_area'}), on='peptide')
    t['relative'] = t['mean_area'] / t['top_area']
    t['ms_level'] = level_name
    return t.sort_values(['peptide', 'level']).reset_index(drop=True)


MS_COLOR = {'MS2': core.PALETTE_SINGLE[2], 'MS1': core.PALETTE_SINGLE[0]}
POINT_ALPHA = 0.45


def _log_amol_axis(ax):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(AMOL_TICKS))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f'{v / 1000:g}k' if v >= 1000 else f'{v:g}'))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(1.2, 33000)
    ax.set_xlabel('Spiked amount [amol]', fontsize=fs.FONTSIZE)


def _ms_legend(ax, *, line=True, extra=()):
    h = [plt.Line2D([], [], color=MS_COLOR[m], lw=2.0,
                    ls=LEVEL_STYLE[m] if line else 'none',
                    marker=LEVEL_MARKER[m], ms=5.5, markeredgecolor='black',
                    markeredgewidth=0.5, label=m) for m in ('MS2', 'MS1')]
    h += list(extra)
    ax.legend(handles=h, loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2,
              frameon=False, fontsize=fs.LEGEND_FONTSIZE, handlelength=1.6,
              handletextpad=0.4, columnspacing=0.8, borderpad=0.1)


def _points_and_median(ax, t, col, *, dodge=1.12):
    """Every peptide as a faint point, the median over peptides as a line."""
    rng = np.random.default_rng(0)
    rows = []
    for name, dx in (('MS2', 1 / dodge), ('MS1', dodge)):
        g = t[(t['ms_level'] == name) & np.isfinite(t[col])
              & (t[col] > 0)]
        c = MS_COLOR[name]
        jit = 10 ** rng.uniform(-0.035, 0.035, len(g))
        x = g['amol'].to_numpy() * dx * jit
        face = mcolors.to_rgba(c, POINT_ALPHA)
        for acc in (True, False):
            m = (g['accepted'] == acc).to_numpy()
            ax.scatter(x[m], g[col].to_numpy()[m], s=20,
                       marker=LEVEL_MARKER[name],
                       facecolor=face if acc else 'white',
                       edgecolor='black' if acc else c,
                       linewidth=0.5 if acc else 1.0, zorder=3)
        med = g.groupby('amol')[col].median()
        ax.plot(med.index * dx, med.values, color=c, lw=2.2,
                ls=LEVEL_STYLE[name], marker=LEVEL_MARKER[name], ms=5.5,
                markeredgecolor='black', markeredgewidth=0.5, zorder=4)
        rows += [{'ms_level': name, 'amol': a, f'median_{col}': v}
                 for a, v in med.items()]
    return pd.DataFrame(rows)


def ms_limits():
    """{level: (amol, legend label)} for the dashed limit lines in b and d."""
    lod = float(pd.read_csv(MS2_CAL)['LOD_amol'].median())
    return {'MS2': (lod, 'MS2 LOD'), 'MS1': (2000.0, 'MS1 lowest')}


def _limit_lines(ax):
    """Everything left of a level's line is that level's noise."""
    handles = []
    for name, (x, label) in ms_limits().items():
        ax.axvline(x, color=MS_COLOR[name], lw=1.1, ls=(0, (5, 3)),
                   zorder=1.5)
        handles.append(plt.Line2D([], [], color=MS_COLOR[name], lw=1.1,
                                  ls=(0, (5, 3)), label=label))
    return handles


def panel_response(t):
    """b -- measured over expected signal against amount."""
    t = t.copy()
    t['measured_over_expected'] = t['relative'] / (t['amol'] / 20000.0)
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    ax.axhline(1.0, color='#999999', lw=0.9, ls=':', zorder=1)
    lim = _limit_lines(ax)
    med = _points_and_median(ax, t, 'measured_over_expected')
    _log_amol_axis(ax)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100, 1000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}x'))
    ax.set_ylabel('Measured / expected signal', fontsize=fs.FONTSIZE)
    ax.tick_params(labelsize=fs.FONTSIZE)
    _ms_legend(ax, extra=lim)
    fs.tidy(ax)
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=1.45, h_in=fs.AXES_H_IN)
    print(med.pivot(index='amol', columns='ms_level').round(2).to_string())
    fs.save_panel(fig, OUT, f'supplementary_figure6{LETTER["response"]}'
                  f'_response_ms1_vs_ms2', t, dpi=300)
    plt.close(fig)


def panel_lowest(t):
    """c -- lowest accepted level per peptide, MS2 against MS1."""
    low = (t[t['accepted']].groupby(['peptide', 'ms_level'])['amol'].min()
           .unstack('ms_level').reindex(PEPTIDE_ORDER))
    print(low.to_string())
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    y = np.arange(len(PEPTIDE_ORDER))[::-1]
    for yi, pep in zip(y, PEPTIDE_ORDER):
        v2, v1 = low.loc[pep, 'MS2'], low.loc[pep, 'MS1']
        ax.plot([v2, v1], [yi, yi], color='#BBBBBB', lw=1.2, zorder=1)
        for name, v in (('MS2', v2), ('MS1', v1)):
            ax.scatter([v], [yi], marker=LEVEL_MARKER[name], s=52,
                       color=MS_COLOR[name], edgecolor='black',
                       linewidth=0.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(PEPTIDE_ORDER, fontsize=fs.FONTSIZE)
    _log_amol_axis(ax)
    ticks = sorted(set(low.to_numpy().ravel()) - {np.nan})
    ticks = [v for v in ticks if v == v]
    ax.set_xlim(min(ticks) / 2.2, max(ticks) * 2.2)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f'{v/1000:g}k' if v >= 1000 else f'{v:g}'))
    ax.set_xlabel('Lowest level [amol]', fontsize=fs.FONTSIZE)
    ax.set_ylim(-0.6, len(PEPTIDE_ORDER) - 0.4)
    ax.tick_params(labelsize=fs.FONTSIZE)
    _ms_legend(ax, line=False)
    fs.tidy(ax)
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=1.10, h_in=fs.AXES_H_IN)
    fs.save_panel(fig, OUT, f'supplementary_figure6{LETTER["lowest"]}'
                  f'_lowest_accepted_level', low.reset_index(), dpi=300)
    plt.close(fig)


def panel_cv(t):
    """d -- injection CV against amount, median line over peptide points."""
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    ax.axhline(20, color='#C0392B', lw=0.9, ls=':', zorder=1)
    lim = _limit_lines(ax)
    med = _points_and_median(ax, t, 'cv_pct')
    _log_amol_axis(ax)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_ylabel('CV across 4 injections [%]', fontsize=fs.FONTSIZE)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=1.45, h_in=fs.AXES_H_IN)
    print(med.pivot(index='amol', columns='ms_level').round(1).to_string())
    fs.save_panel(fig, OUT, f'supplementary_figure6{LETTER["cv"]}'
                  f'_cv_ms1_vs_ms2', t, dpi=300)
    plt.close(fig)


def panel_trace_r(t):
    """e -- MS2 summed-trace correlation per peptide and level."""
    rules = _rules()
    g = t[(t['ms_level'] == 'MS2') & (t['level'] < t['level'].max())].copy()
    acc, rej = g[g['accepted']], g[~g['accepted']]
    lo, hi = rej['trace_r'].max(), acc['trace_r'].min()
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    ax.axhspan(lo, hi, color='#EEEEEE', lw=0, zorder=0)
    ax.axhline(rules.MIN_TRACE_R, color='#555555', lw=1.0, ls=(0, (5, 3)),
               zorder=1)
    rng = np.random.default_rng(1)
    c = MS_COLOR['MS2']
    for sub, ok in ((acc, True), (rej, False)):
        x = sub['amol'].to_numpy() * 10 ** rng.uniform(-0.06, 0.06, len(sub))
        ax.scatter(x, sub['trace_r'], s=24, marker=LEVEL_MARKER['MS2'],
                   facecolor=mcolors.to_rgba(c, 0.7) if ok else 'white',
                   edgecolor='black' if ok else c,
                   linewidth=0.5 if ok else 1.0, zorder=3,
                   label='accepted' if ok else 'rejected')
    _log_amol_axis(ax)
    ax.set_xlim(1.2, 3300)
    ax.set_ylim(-1.0, 1.05)
    ax.yaxis.set_major_locator(FixedLocator([-1, -0.5, 0, 0.5, 1]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_ylabel('Trace correlation with 20 fmol', fontsize=fs.FONTSIZE)
    ax.tick_params(labelsize=fs.FONTSIZE)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2,
              frameon=False, fontsize=fs.LEGEND_FONTSIZE, handletextpad=0.2,
              columnspacing=0.6, borderpad=0.1)
    fs.tidy(ax)
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=1.15, h_in=fs.AXES_H_IN)
    print(f'  h  MS2 trace r: accepted >= {hi:.2f} (n={len(acc)}), '
          f'rejected <= {lo:.2f} (n={len(rej)}), threshold '
          f'{rules.MIN_TRACE_R:.2f}')
    fs.save_panel(fig, OUT, f'supplementary_figure6{LETTER["trace_r"]}'
                  f'_trace_r_acceptance',
                  g[['peptide', 'level', 'amol', 'trace_r', 'accepted']],
                  dpi=300)
    plt.close(fig)


def _e396_local():
    """E396's driver re-pointed at the paper's local raw copies."""
    if E396_DIR not in sys.path:
        sys.path.insert(0, E396_DIR)
    import analyze_6x5 as a
    return _rules().configure(a, use_saved=True)


def trace_table(peptide=None, run=None):
    """trace_table, read from the data package or measured and written there."""
    peptide, run = peptide or TRACE_PEPTIDE, run or TRACE_RUN
    path = cfg.data('H032_E396', f'trace_table_{peptide}_{run}.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    t = _measure_trace_table(peptide, run)
    t.to_csv(path, index=False)
    return t


def _measure_trace_table(peptide=None, run=None):
    """Every level's MS2 and MS1 XIC for TRACE_PEPTIDE in TRACE_RUN."""
    a = _e396_local()
    core.raw.register_sil_modifications()
    tg = a.method_table()
    peptide, run = peptide or TRACE_PEPTIDE, run or TRACE_RUN
    ms = a._load(run)
    ms2 = a._ms2_transitions(tg)
    ms1 = core.raw.precursor_isotope_transitions(tg, n_isotopes=3)
    sel = tg.index[tg['sequence'] == peptide]
    win = a._ms2_window(ms, tg.loc[sel, 'q1_mz'].to_numpy())
    rows = []
    for idx in sel:
        lvl = int(tg.loc[idx, 'level'])
        rt, y = core.raw.summed_fragment_xic(
            ms, ms2[ms2['target_idx'] == idx], rt_range=win,
            match='precursor', match_tol=a.Q1_TOL, recentre=False)
        rows += [{'level': lvl, 'ms_level': 'MS2', 'rt_min': r,
                  'intensity': v} for r, v in zip(rt, y)]
        x = core.raw.extract_xic(
            ms, ms1.loc[ms1['target_idx'] == idx, 'product_mz'].to_numpy(),
            ms_level=1, rt_range=win, recentre=False)
        rows += [{'level': lvl, 'ms_level': 'MS1', 'rt_min': r,
                  'intensity': v}
                 for r, v in zip(x['rt'], x['intensity'])]
    return pd.DataFrame(rows)


def panel_traces(peptide=None, run=None, out=None):
    """a -- the XICs themselves, MS2 over MS1, one column per level."""
    peptide, run = peptide or TRACE_PEPTIDE, run or TRACE_RUN
    tr = trace_table(peptide, run)
    lv2 = pd.read_csv(MS2_LEVELS)
    lv1 = pd.read_csv(MS1_LEVELS)
    meta = {}
    for name, lv in (('MS2', lv2), ('MS1', lv1)):
        s_ = lv[(lv['run'] == run) & (lv['peptide'] == peptide)]
        meta[name] = s_.set_index('level')
    levels = sorted(tr['level'].unique(), reverse=True)
    amol = {int(r.level): r.amol for r in lv2.itertuples()
            if r.peptide == peptide}
    fig, axes = plt.subplots(2, len(levels), figsize=(5.9, 3.3))
    src = []
    for j, lvl in enumerate(levels):
        for i, name in enumerate(('MS2', 'MS1')):
            ax = axes[i, j]
            m = meta[name].loc[lvl]
            anchor, fwhm = float(m['anchor_rt']), float(m['fwhm'])
            lo = anchor - TRACE_HALF_FWHM * fwhm
            hi = anchor + TRACE_HALF_FWHM * fwhm
            g = tr[(tr['level'] == lvl) & (tr['ms_level'] == name)
                   & (tr['rt_min'] >= lo) & (tr['rt_min'] <= hi)]
            scale = float(g['intensity'].max()) or 1.0
            y = g['intensity'] / scale
            ax.axvspan(anchor - 1.5 * fwhm, anchor + 1.5 * fwhm,
                       color='#EBEBEB', lw=0, zorder=0)
            ax.axvline(anchor, color='#9A9A9A', lw=0.7, ls=':', zorder=1)
            col = MS_COLOR[name]
            ax.plot(g['rt_min'], y, color=col, lw=1.1,
                    ls=LEVEL_STYLE[name], zorder=2)
            ax.scatter(g['rt_min'], y, s=5, color=col, zorder=3,
                       linewidths=0)
            ax.set_ylim(-0.05, 1.55)
            ax.set_yticks([0, 0.5, 1] if j == 0 else [])
            ax.set_xlim(lo, hi)
            ax.set_xticks([round(anchor, 2)])
            if i == 0:
                ax.set_xticklabels([])
                ax.text(0.5, 1.10, f'{amol[lvl]:,.0f} amol',
                        transform=ax.transAxes, ha='center', va='bottom',
                        fontsize=fs.FONTSIZE)
            ax.tick_params(labelsize=fs.FONTSIZE)
            for side in ('top', 'right') + (('left',) if j else ()):
                ax.spines[side].set_visible(False)
            if j == 0:
                ax.set_ylabel('Relative' + chr(10) + 'intensity',
                              fontsize=fs.FONTSIZE)
                ax.text(0.03, 0.97, name, transform=ax.transAxes,
                        ha='left', va='top', fontsize=fs.FONTSIZE,
                        color=col, fontweight='bold')
            top = float(meta[name].loc[max(levels), 'ions'])
            fold = (float(m['ions']) / top) / (amol[lvl] / amol[max(levels)])
            src.append(g.assign(scale=scale, anchor_rt=anchor, fwhm=fwhm,
                                trace_r=m['trace_r'],
                                measured_over_expected=fold))
    fig.supxlabel('Retention time [min]', fontsize=fs.FONTSIZE)
    fig.tight_layout(w_pad=0.6, h_pad=0.5)
    if out is not None:
        fig.savefig(os.path.join(out, f'traces_{peptide}_{run}.png'),
                    dpi=110, bbox_inches='tight')
    else:
        fs.save_panel(fig, OUT, f'supplementary_figure6{LETTER["traces"]}'
                      f'_xic_ms1_vs_ms2', pd.concat(src, ignore_index=True),
                      dpi=300)
    plt.close(fig)


def main(*letters):
    wanted = set(letters) or set(LETTER.values())
    fs.init()
    fs.start_build(script=__file__, inputs=[MS2_LEVELS, MS1_LEVELS],
                   out_dir=OUT)
    if LETTER['traces'] in wanted:
        print('panel a -- XIC traces ...')
        panel_traces()
    rules = _rules()
    t = pd.concat([per_level(MS2_LEVELS, 'MS2', rules),
                   per_level(MS1_LEVELS, 'MS1', rules)], ignore_index=True)
    print(t.pivot_table(index=['peptide', 'level'], columns='ms_level',
                        values=['relative', 'cv_pct', 'accepted'])
          .round(3).to_string())
    if LETTER['response'] in wanted:
        panel_response(t)
    if LETTER['lowest'] in wanted:
        panel_lowest(t)
    if LETTER['cv'] in wanted:
        panel_cv(t)
    if LETTER['trace_r'] in wanted:
        panel_trace_r(t)
    fs.report_stale(OUT, prefix='supplementary_figure')


if __name__ == '__main__':
    main(*sys.argv[1:])
