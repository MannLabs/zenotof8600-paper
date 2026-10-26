"""Supplementary figure 7 -- the second half of figure 6, in full."""

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
import pandas as pd
from scipy import stats

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = cfg.panels(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import figure_style as fs
sys.path.insert(0, os.path.join(ROOT, 'figure6'))

ZT_OCC = os.path.join(cfg.DATA_ROOT, 'H032_E397', 'ztscan_ps129_occupancy.csv')
LETTER = {'occupancy': 'a', 'light_site': 'b', 'cid': 'c', 'ead': 'd'}
LIGHT_SITE = cfg.data('H032_E397', 'light_site_ions.csv')
ZT_SITE = os.path.join(cfg.DATA_ROOT, 'H032_E397', 'ztscan_site_ions.csv')
LOW_AREA = 20.0
ORDER = ('PFF-', 'PFF+')
COLOR = {'PFF-': core.PALETTE_SINGLE[2], 'PFF+': core.PALETTE_SINGLE[0]}
METHODS = ('MRM-HR', 'ZT Scan')
TICK = {'MRM-HR': r'MRM$^{\mathrm{HR}}$'}
SITE_COLOR = {'MRM-HR': core.PALETTE_SINGLE[0], 'ZT Scan': core.PALETTE_SINGLE[1]}


def panel_spectra(letters):
    import spectra
    spectra.panel(letters, scale_to='all', out=OUT, y_tick_chars=7, h_in=1.12,
                  name_fmt='supplementary_figure7{letter}_{act}_spectrum_full')


def occupancy_table():
    import occupancy
    mrm = occupancy.table()[['run', 'condition', 'occupancy']]
    mrm = mrm.assign(method='MRM-HR', condition=mrm['condition'].astype(str))
    zt = pd.read_csv(ZT_OCC)[['run', 'group', 'occupancy']].rename(
        columns={'group': 'condition'}).assign(method='ZT Scan')
    return pd.concat([mrm, zt], ignore_index=True)


def panel_occupancy(t):
    """c -- two bars per method, every injection a point."""
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    rng = np.random.default_rng(0)
    x, xt, rows = 0.0, [], []
    for m in METHODS:
        xs = []
        for c in ORDER:
            v = t[(t.method == m) & (t.condition == c)]['occupancy'].to_numpy()
            ax.bar(x, v.mean(), width=0.8, color=COLOR[c], edgecolor='black',
                   linewidth=0.8, zorder=2)
            ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v,
                       s=fs.POINT_SIZE, color='black', alpha=0.75,
                       linewidth=0.3, edgecolor='white', zorder=5)
            rows.append({'method': m, 'condition': c, 'mean': v.mean(),
                         'sd': v.std(ddof=1), 'n': len(v),
                         'values': ';'.join(f'{a:.3f}' for a in v)})
            xs.append(x)
            x += 1.0
        a = t[(t.method == m) & (t.condition == 'PFF+')]['occupancy']
        b = t[(t.method == m) & (t.condition == 'PFF-')]['occupancy']
        print(f'    {m:8s} PFF- {b.mean():5.2f}% (sd {b.std():.2f})  '
              f'PFF+ {a.mean():5.2f}% (sd {a.std():.2f})  '
              f'{a.mean() / b.mean():.2f}x')
        rows[-1]['fold'] = a.mean() / b.mean()
        xt.append(np.mean(xs))
        x += 0.6
    ax.set_xticks(xt)
    ax.set_xticklabels([TICK.get(m, m) for m in METHODS], fontsize=fs.FONTSIZE)
    ax.set_ylabel('pS129 occupancy [%]', fontsize=fs.FONTSIZE)
    ax.set_ylim(0, None)
    h = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[c], edgecolor='black',
                       linewidth=0.8, label=c) for c in ORDER]
    ax.legend(handles=h, loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2,
              frameon=False, fontsize=fs.LEGEND_FONTSIZE, borderpad=0.1)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    fig.tight_layout()
    fs.set_bar_geometry_inches(fig, ax)
    fs.save_panel(fig, OUT, f'supplementary_figure7{LETTER["occupancy"]}'
                  f'_ps129_occupancy_by_method', pd.DataFrame(rows), dpi=300)
    plt.close(fig)


def site_ion_table(path=LIGHT_SITE, zt_path=ZT_SITE):
    """Light and heavy signal of every site ion, both methods, summed over the four PFF+ runs of each."""
    d = pd.read_csv(path)
    m = d.pivot_table(index='ion', columns='form', values='ions',
                      aggfunc='sum').reset_index()
    m['series'] = m['ion'].str[0]
    m['number'] = m['ion'].str[1:].astype(int)
    m = m[m['number'] >= 8]
    m = m.assign(method='MRM-HR', activation='EAD', charge=1, unit='ions',
                 runs='B1-B4_repeat', light_ions=m['light'],
                 heavy_ions=m['heavy'])
    z = pd.read_csv(zt_path)
    t = z.pivot_table(index='ion', columns='form', values='area',
                      aggfunc='sum').reset_index()
    t['series'] = t['ion'].str[0]
    t['number'] = t['ion'].str.extract(r'^[a-z](\d+)')[0].astype(int)
    t['charge'] = t['ion'].str.extract(r'\+(\d+)$')[0].astype(int)
    t = t.assign(method='ZT Scan', activation='CID', unit='area',
                 runs='B5-B8_repeat', light_ions=np.nan, heavy_ions=np.nan)
    s = pd.concat([m, t], ignore_index=True)
    s['l_over_h'] = s['light'] / s['heavy']
    s['low_count'] = ((s['method'] == 'ZT Scan')
                      & (s[['light', 'heavy']].min(axis=1) < LOW_AREA))
    return s[['method', 'activation', 'runs', 'ion', 'series', 'number',
              'charge', 'unit', 'light', 'heavy', 'light_ions', 'heavy_ions',
              'l_over_h', 'low_count']]


def panel_light_site(path=LIGHT_SITE, zt_path=ZT_SITE):
    """d -- light/heavy ratio of the site-determining ions of the ENDOGENOUS pS129 peptide, by acquisition method."""
    s = site_ion_table(path, zt_path)
    mrm = s[s['method'] == 'MRM-HR']
    med = float(np.median(mrm['l_over_h']))
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    off = {('MRM-HR', 'y', 1): -0.27, ('MRM-HR', 'z', 1): -0.09,
           ('ZT Scan', 'y', 1): 0.09, ('ZT Scan', 'y', 2): 0.27}
    mk = {('y', 1): 'o', ('z', 1): 's', ('y', 2): '^'}
    for _, r in s.iterrows():
        c = SITE_COLOR[r['method']]
        ax.scatter(r['number'] + off[(r['method'], r['series'], r['charge'])], r['l_over_h'],
                   marker=mk[(r['series'], r['charge'])], s=fs.POINT_SIZE * 1.6,
                   facecolor='white' if r['low_count'] else c,
                   edgecolor=c if r['low_count'] else 'black',
                   linewidth=1.2 if r['low_count'] else 0.5, zorder=3)
    ax.axhline(med, ls='--', color=SITE_COLOR['MRM-HR'], lw=1.0, zorder=1)
    ax.set_yscale('log')
    ax.set_ylim(0.02, 12)
    ax.set_xlim(7.45, 13.55)
    ax.set_xticks(range(8, 14))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xlabel('Site ion number (y/z)', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Light / heavy', fontsize=fs.FONTSIZE)
    from matplotlib.lines import Line2D
    kw = dict(ls='none', markersize=7, markeredgewidth=0.5,
              markeredgecolor='black')
    h = [Line2D([], [], marker='o', color=SITE_COLOR['MRM-HR'],
                label=TICK['MRM-HR'] + ' EAD', **kw),
         Line2D([], [], marker='o', color=SITE_COLOR['ZT Scan'],
                label='ZT Scan CID', **kw),
         Line2D([], [], marker='o', markerfacecolor='white',
                markeredgecolor='0.35', ls='none', markersize=7,
                markeredgewidth=1.2, label=f'area < {LOW_AREA:g}'),
         Line2D([], [], ls='--', color=SITE_COLOR['MRM-HR'], lw=1.0,
                label='median MRM$^{HR}$'),
         Line2D([], [], marker='s', color='0.6', label='z ion', **kw),
         Line2D([], [], marker='^', color='0.6', label='2+', **kw)]
    ax.legend(handles=h, loc='lower left', ncol=2, frameon=False,
              fontsize=fs.LEGEND_FONTSIZE, handletextpad=0.2,
              columnspacing=0.8, borderpad=0.2, labelspacing=0.3)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    fig.tight_layout()
    fs.save_panel(fig, OUT, f'supplementary_figure7{LETTER["light_site"]}'
                  f'_light_site_ions', s.assign(mrm_median_l_over_h=med),
                  dpi=300)
    plt.close(fig)
    for m, g in s.groupby('method'):
        lo = g[g['low_count']]
        print(f'    {m:8s} {len(g)} ions, L/H {g.l_over_h.min():.3f}-'
              f'{g.l_over_h.max():.3f} (median {g.l_over_h.median():.2f}); '
              f'low count: {", ".join(lo.ion) or "none"}')


def main(*letters):
    wanted = set(letters) or set(LETTER.values())
    fs.init()
    fs.start_build(script=__file__, inputs=[ZT_OCC, LIGHT_SITE, ZT_SITE], out_dir=OUT)
    spec = [LETTER[k] for k in ('cid', 'ead') if LETTER[k] in wanted]
    if spec:
        print('panels c, d -- fully scaled spectra ...')
        panel_spectra(spec)
    if LETTER['occupancy'] in wanted:
        print('panel a -- pS129 occupancy by method ...')
        panel_occupancy(occupancy_table())
    if LETTER['light_site'] in wanted:
        print('panel b -- endogenous site ions ...')
        panel_light_site()
    fs.report_stale(OUT, prefix='supplementary_figure')


if __name__ == '__main__':
    main(*sys.argv[1:])
