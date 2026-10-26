"""Supplementary figure 6 a-c -- Q1 resolution and transmission in Zeno MRMHR."""

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
from matplotlib.patches import Rectangle

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = cfg.panels(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import figure_style as fs

DATA = os.path.join(cfg.DATA_ROOT, 'H032_E276')
RAW = cfg.raw('H032_E276', 'raw')
CACHE = cfg.raw('ms_data_cache')
MEASUREMENTS = os.path.join(DATA, 'q1_resolution_measurements.csv')
TRANSITIONS = os.path.join(DATA, 'q1_resolution_transitions.csv')
TRACES = os.path.join(DATA, 'q1_resolution_traces.csv')
TARGETS = os.path.join(DATA, 'targeted_peptides_alphabase.csv')
PREFIX = '20260210_Zeno2_Eno0_12p95min_TiHe_SA_H032_E276_'

SETTINGS = [('high', 'A9', 0.5), ('unit', 'A10', 0.7), ('low', 'A11', 1.2),
            ('open', 'A12', 10.0)]
COLOR = {s: core.PALETTE_SINGLE[i] for i, (s, _, _) in enumerate(SETTINGS)}
LEVELS_AMOL = [200000, 20000, 2000, 200, 20]
Q1_TOL = 0.004
UNIQUE_Q1_TH = 6.0
UNIQUE_PPM = 30.0
MIN_TRACE_R = 0.80
MIN_AMOL = 20000
EXAMPLE = 'LASVSVSR'
LETTER = {'windows': 'a', 'xic': 'b', 'transmission': 'c'}


def load(vial):
    return core.raw.load_ms_data(
        os.path.join(RAW, f'{PREFIX}{vial}.wiff'), cache_dir=CACHE,
        reader_options={'keep_k_peaks_per_spec': 10 ** 7}, verbose=False)


def targets():
    from alphabase.peptide.precursor import calc_precursor_mz
    core.raw.register_sil_modifications()
    t = pd.read_csv(TARGETS)
    t['nAA'] = t['sequence'].str.len()
    calc_precursor_mz(t)
    t['level'] = t.groupby('sequence').cumcount()
    t['amol'] = t['level'].map(dict(enumerate(LEVELS_AMOL)))
    return t


def transitions(t):
    """y 1+ above the precursor, unique among co-transmissible targets."""
    f = core.raw.target_fragments(t, activation='cid', min_mz=150,
                                  max_mz=2000)
    y = f[(f['ion_type'] == 'y') & (f['fragment_charge'] == 1)
          & (f['product_mz'] > f['precursor_mz'])].copy()
    keep = []
    for i, row in y.iterrows():
        q1 = t.loc[row.target_idx, 'precursor_mz']
        near = t.index[(abs(t['precursor_mz'] - q1) <= UNIQUE_Q1_TH)
                       & (t.index != row.target_idx)]
        other = f[f['target_idx'].isin(near)]['product_mz'].to_numpy()
        ppm = np.abs(other - row.product_mz) / row.product_mz * 1e6
        keep.append(not (ppm < UNIQUE_PPM).any())
    y = y[keep]
    n = y.groupby('target_idx').size()
    print(f'    unique y ions per target: median {int(n.median())}, '
          f'min {int(n.min())} ({len(n)} targets)')
    return y


def fragment_traces(ms, q1, frags):
    """rt, {ion_number: intensity} -- one XIC per transition, same scans."""
    rt, out = None, {}
    for _, f in frags.iterrows():
        x = core.raw.targeted_xic(ms, float(q1), float(f.product_mz),
                                  tol=Q1_TOL)
        rt = x['rt'].to_numpy()
        out[int(f.ion_number)] = x['intensity'].to_numpy()
    return rt, out


def measure(t, y):
    """Per target: the summed trace (acceptance, b) and every transition."""
    rows, trows, traces = [], [], {}
    for setting, vial, _ in SETTINGS:
        ms = load(vial)
        for seq, g in t.groupby('sequence', sort=False):
            per = {i: fragment_traces(ms, t.loc[i, 'precursor_mz'],
                                      y[y.target_idx == i])
                   for i in g.index if (y.target_idx == i).any()}
            tr = {i: (rt, np.sum(list(d.values()), axis=0))
                  for i, (rt, d) in per.items()}
            anchor = g.index[0]
            rt_a, in_a = tr[anchor]
            pk = core.raw.pick_peak(rt_a, in_a, baseline='offpeak',
                                    interpolate_points=500, min_sn=3)
            lo = pk['apex_rt'] - 1.5 * pk['fwhm']
            hi = pk['apex_rt'] + 1.5 * pk['fwhm']
            win = (rt_a >= lo) & (rt_a <= hi)
            for i, (rt, inten) in tr.items():
                area = core.raw.integrate_peak(rt, inten, lo, hi,
                                               baseline='offpeak')['area']
                r = np.corrcoef(inten[win], in_a[win])[0, 1] \
                    if win.sum() > 2 else np.nan
                rows.append({'setting': setting, 'run': vial, 'sequence': seq,
                             'target_idx': i, 'amol': t.loc[i, 'amol'],
                             'area': area, 'trace_r': r,
                             'apex_rt': pk['apex_rt'], 'fwhm_min': pk['fwhm']})
                for n, yi in per[i][1].items():
                    a = core.raw.integrate_peak(rt, yi, lo, hi,
                                                baseline='offpeak')['area']
                    trows.append({'setting': setting, 'target_idx': i,
                                  'ion_number': n, 'area': a})
            if seq == EXAMPLE:
                traces[setting] = tr[anchor]
        print(f'    {setting:5s} {vial}: measured')
    return pd.DataFrame(rows), pd.DataFrame(trows), traces


def complement_distribution(t, i, n):
    """(mass offset in Da, abundance) of the neutral remainder of y_n."""
    from alphabase.peptide.precursor import (calc_precursor_isotope,
                                             calc_precursor_mz)
    seq = t.loc[i, 'sequence']
    k = len(seq) - n
    mods, sites = [], []
    if isinstance(t.loc[i, 'mods'], str) and t.loc[i, 'mods']:
        for mod, site in zip(t.loc[i, 'mods'].split(';'),
                             str(t.loc[i, 'mod_sites']).split(';')):
            if 0 < int(site) <= k:
                mods.append(mod)
                sites.append(site)
    row = pd.DataFrame({'sequence': [seq[:k]], 'mods': [';'.join(mods)],
                        'mod_sites': [';'.join(sites)], 'charge': [1],
                        'nAA': [k]})
    calc_precursor_mz(row)
    iso = calc_precursor_isotope(row)
    cols = [c for c in iso.columns if c.startswith('i_')]
    ab = iso[cols].to_numpy(float)[0]
    return np.arange(ab.size) - int(np.argmax(ab)), ab / ab.sum()


def envelope(t, i):
    """(offset from mono in m/z, relative abundance) of target i's envelope."""
    from alphabase.peptide.precursor import calc_precursor_isotope
    iso = calc_precursor_isotope(t.loc[[i]].copy())
    cols = [c for c in iso.columns if c.startswith('i_')]
    inten = iso[cols].to_numpy(float)[0]
    mono = int(np.argmax(inten))
    z = int(t.loc[i, 'charge'])
    return (np.arange(inten.size) - mono) * 1.0033548 / z, inten / inten.sum()


def transmission(m, tm, t):
    """Per-transition transmission of the isolated isotope, relative to open."""
    wide = m.pivot_table(index='target_idx', columns='setting',
                         values=['area', 'trace_r'])
    ok_cal = wide.index[(wide['trace_r'] >= MIN_TRACE_R).all(axis=1)
                        & (wide['area'] > 0).all(axis=1)]
    ok_cal = [i for i in ok_cal if t.loc[i, 'amol'] >= MIN_AMOL]
    per = tm[tm.target_idx.isin(ok_cal)].pivot_table(
        index=['target_idx', 'ion_number'], columns='setting', values='area')
    per = per[(per > 0).all(axis=1)]
    rows = []
    for (i, n), r in per.iterrows():
        off_da, c = complement_distribution(t, i, n)
        z = int(t.loc[i, 'charge'])
        passed = {s: float(c[np.abs(off_da * 1.0033548 / z) <= w / 2].sum())
                  for s, _, w in SETTINGS}
        for s, _, _ in SETTINGS:
            f = passed[s] / passed['open']
            raw = r[s] / r['open'] * 100
            rows.append({'sequence': t.loc[i, 'sequence'], 'target_idx': i,
                         'amol': t.loc[i, 'amol'], 'transition': f'y{n}',
                         'setting': s, 'relative_area_pct': raw,
                         'isolated_share_rel_open': f,
                         'transmission_pct': raw / f})
    long = pd.DataFrame(rows)
    print(f'    {len(ok_cal)} calibrants, {len(per)} transitions')
    for s, _, _ in SETTINGS:
        g = long[long.setting == s]
        print(f'      {s:5s} area/open {g.relative_area_pct.median():6.1f}%  '
              f'isolated share {100 * g.isolated_share_rel_open.median():5.1f}%'
              f'  transmission {g.transmission_pct.median():6.1f}% (IQR '
              f'{g.transmission_pct.quantile(.25):.0f}-'
              f'{g.transmission_pct.quantile(.75):.0f})')
    return long


def panel_windows(t):
    i = t.index[t['sequence'] == EXAMPLE][0]
    off, inten = envelope(t, i)
    mz = float(t.loc[i, 'precursor_mz']) + off
    keep = inten > 0.001 * inten.max()
    mz, inten = mz[keep], inten[keep] / inten.max()
    centre = float(mz[np.argmax(inten)])

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    rows = []
    for setting, _, w in reversed(SETTINGS):
        ax.add_patch(Rectangle((centre - w / 2, 0), w, 1.0,
                               facecolor=COLOR[setting], alpha=0.22,
                               edgecolor=COLOR[setting], lw=1.2, zorder=2,
                               label=f'{setting} ({w:g})'))
        rows.append({'setting': setting, 'width_mz': w,
                     'low_mz': centre - w / 2, 'high_mz': centre + w / 2})
    ax.vlines(mz, 0, inten, color='black', lw=2.0, zorder=3)
    rows += [{'isotope_mz': a, 'relative_abundance': b}
             for a, b in zip(mz, inten)]
    ax.set_xlim(centre - 2.5, centre + 2.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('m/z', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Relative abundance', fontsize=fs.FONTSIZE)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc='lower left', bbox_to_anchor=(0, 1.01),
              ncol=2, frameon=False, fontsize=fs.LEGEND_FONTSIZE,
              title='Q1 isolation width (m/z)',
              title_fontsize=fs.LEGEND_FONTSIZE, borderpad=0.1,
              handlelength=1.0, columnspacing=0.8)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=1.45, h_in=fs.AXES_H_IN)
    fs.save_panel(fig, OUT, f'supplementary_figure6{LETTER["windows"]}'
                  f'_q1_isolation_windows', pd.DataFrame(rows), dpi=300)
    plt.close(fig)


def panel_xic(traces):
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    rows = []
    apex = core.raw.pick_peak(*traces['open'], baseline='offpeak',
                              interpolate_points=500)['apex_rt']
    top = max(traces[s][1].max() for s, _, _ in SETTINGS)
    exp = int(np.floor(np.log10(top)))
    for setting, _, _ in SETTINGS:
        rt, inten = traces[setting]
        m = np.abs(rt - apex) <= 0.08
        ax.plot((rt[m] - apex) * 60, inten[m] / 10 ** exp,
                color=COLOR[setting], lw=1.6, label=setting)
        rows += [{'setting': setting, 'rt_min': a, 'intensity': b}
                 for a, b in zip(rt[m], inten[m])]
    ax.set_xlabel('Time from apex [s]', fontsize=fs.FONTSIZE)
    ax.set_ylabel(f'Fragment intensity [×10$^{{{exp}}}$]',
                  fontsize=fs.FONTSIZE)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2,
              frameon=False, fontsize=fs.LEGEND_FONTSIZE, borderpad=0.1,
              handlelength=1.0, columnspacing=0.8)
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=1.45, h_in=fs.AXES_H_IN)
    fs.save_panel(fig, OUT, f'supplementary_figure6{LETTER["xic"]}'
                  f'_q1_resolution_xic', pd.DataFrame(rows), dpi=300)
    plt.close(fig)


def panel_transmission(long):
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    rng = np.random.default_rng(0)
    rows = []
    for k, (setting, _, _) in enumerate(SETTINGS):
        v = long.loc[long.setting == setting, 'transmission_pct'].to_numpy()
        med = float(np.median(v))
        ax.boxplot([v], positions=[k], widths=0.6, showfliers=False,
                   patch_artist=True, zorder=2,
                   boxprops=dict(facecolor=COLOR[setting], edgecolor='black',
                                 linewidth=0.8),
                   medianprops=dict(color='black', linewidth=1.4),
                   whiskerprops=dict(color='black', linewidth=0.8),
                   capprops=dict(color='black', linewidth=0.8))
        ax.scatter(k + rng.uniform(-0.16, 0.16, v.size), v, s=14,
                   color='black', alpha=0.7, linewidth=0.3, edgecolor='white',
                   zorder=5)
        rows.append({'setting': setting, 'median_pct': med, 'n': v.size,
                     'q25_pct': float(np.percentile(v, 25)),
                     'q75_pct': float(np.percentile(v, 75))})
    ax.set_xticks(range(len(SETTINGS)))
    ax.set_xticklabels([s for s, _, _ in SETTINGS], fontsize=fs.FONTSIZE)
    ax.set_ylabel('Transmission relative to open [%]', fontsize=fs.FONTSIZE)
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=1.45, h_in=fs.AXES_H_IN)
    src = pd.concat([pd.DataFrame(rows).assign(kind='summary'),
                     long.assign(kind='calibrant')], ignore_index=True)
    fs.save_panel(fig, OUT, f'supplementary_figure6{LETTER["transmission"]}'
                  f'_q1_resolution_transmission', src, dpi=300)
    plt.close(fig)


def main(*letters):
    wanted = set(letters) or set(LETTER.values())
    fs.init()
    fs.start_build(script=__file__, inputs=[DATA], out_dir=OUT)
    t = targets()
    if LETTER['windows'] in wanted:
        print('panel a -- isolation windows ...')
        panel_windows(t)
    if wanted & {LETTER['xic'], LETTER['transmission']}:
        if all(os.path.exists(p) for p in (MEASUREMENTS, TRANSITIONS, TRACES)):
            m, tm = pd.read_csv(MEASUREMENTS), pd.read_csv(TRANSITIONS)
            tr = pd.read_csv(TRACES)
            traces = {k: (g['rt'].to_numpy(), g['intensity'].to_numpy())
                      for k, g in tr.groupby('setting', sort=False)}
        else:
            print('panels b, c -- measuring four resolution runs ...')
            m, tm, traces = measure(t, transitions(t))
            m.to_csv(MEASUREMENTS, index=False)
            tm.to_csv(TRANSITIONS, index=False)
            pd.concat([pd.DataFrame({'setting': k, 'rt': v[0],
                                     'intensity': v[1]})
                       for k, v in traces.items()]).to_csv(TRACES, index=False)
        if LETTER['xic'] in wanted:
            panel_xic(traces)
        if LETTER['transmission'] in wanted:
            panel_transmission(transmission(m, tm, t))


if __name__ == '__main__':
    main(*sys.argv[1:])
