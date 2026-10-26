"""Figure 6, panel a -- the Promega 6x5 standard as extracted ion chromatograms."""

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

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(cfg.DATA_ROOT, 'H032_E396')
OUT = cfg.panels(__file__)
E396_DIR = cfg.data('H032_E396')

RAW_CACHE = cfg.raw('ms_data_cache')

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

RUN = 'A1'

PEPTIDE_ORDER = ['LASVSVSR', 'LGFTDLFSK', 'LLSLGAGEFK', 'VVGGLVALR',
                 'YVYVADVAAK']

WINDOW_MIN = 0.22

AXES_W_IN = 1.75

TIC_FILL, TIC_INK = '#D8D8D8', '#7A7A7A'


def _e396():
    if E396_DIR not in sys.path:
        sys.path.insert(0, E396_DIR)
    import analyze_6x5 as a6
    import rules_6x5
    return rules_6x5.configure(a6, use_saved=True)


def tic(ms):
    """Total ion current, every scan, summed PER CYCLE."""
    sd = ms.spectrum_df
    inten = ms.peak_df['intensity'].to_numpy()
    per_scan = np.array([inten[a:b].sum() for a, b in
                         zip(sd['peak_start_idx'].to_numpy(),
                             sd['peak_stop_idx'].to_numpy())], float)
    o = np.argsort(sd['rt'].to_numpy())
    rt, per_scan, lvl = (sd['rt'].to_numpy()[o], per_scan[o],
                         sd['ms_level'].to_numpy()[o])
    rt1 = rt[lvl == 1]
    cycle = np.clip(np.searchsorted(rt1, rt, side='right') - 1, 0, len(rt1) - 1)
    return rt1, np.bincount(cycle, weights=per_scan, minlength=len(rt1))


def traces():
    """Per target: the summed-transition XIC, from the driver's own machinery."""
    a6 = _e396()
    core.raw.register_sil_modifications()
    tg = a6.method_table()
    frags = core.raw.target_fragments(tg, frag_types=a6.FRAG_TYPES,
                                      min_mz=150.0, max_mz=3000.0)
    ms = a6._load(RUN)
    picked = a6.choose_transitions(tg, frags)

    q1tab = core.precursor_q1_mz(tg, isotope='most_abundant')
    out = []
    for i, row in tg.iterrows():
        pep = row['sequence']
        if pep not in PEPTIDE_ORDER or pep not in picked:
            continue
        sub = frags[frags['target_idx'] == i]
        mzs = [float(m['product_mz'].iloc[0])
               for it, num, z in picked[pep]
               for m in [sub[(sub['ion_type'] == it)
                             & (sub['ion_number'] == num)
                             & (sub['fragment_charge'] == z)]]
               if len(m)]
        if not mzs:
            continue
        q1 = float(q1tab.loc[i, 'q1_mz'])
        rt = y = None
        for mz in mzs:
            r, v = a6._xic(ms, q1, float(mz))
            rt, y = (r, v) if y is None else (rt, y + v)
        out.append({'peptide': pep, 'level': int(row['level']),
                    'rt': rt, 'ions': y})
    return out


TRACES = cfg.data('H032_E396', 'standard_traces.csv')
TIC = cfg.data('H032_E396', 'standard_tic.csv')


def cached_traces():
    """traces(), read from the data package or measured and written there."""
    if os.path.exists(TRACES):
        t = pd.read_csv(TRACES)
        return [{'peptide': p, 'level': int(lv), 'rt': g['rt'].to_numpy(),
                 'ions': g['ions'].to_numpy()}
                for (p, lv), g in t.groupby(['peptide', 'level'], sort=False)]
    tr = traces()
    pd.concat([pd.DataFrame({'peptide': x['peptide'], 'level': x['level'],
                             'rt': x['rt'], 'ions': x['ions']}) for x in tr]
              ).to_csv(TRACES, index=False)
    return tr


def cached_tic():
    """tic() of the run, read from the data package or measured and written."""
    if os.path.exists(TIC):
        t = pd.read_csv(TIC)
        return t['rt'].to_numpy(), t['tic'].to_numpy()
    rt1, cur = tic(_e396()._load(RUN))
    pd.DataFrame({'rt': rt1, 'tic': cur}).to_csv(TIC, index=False)
    return rt1, cur


def main(letter='a'):
    fs.init()
    fs.start_build(script=__file__, inputs=[DATA], out_dir=OUT)
    print(f'extracting {RUN} ...', flush=True)
    tr = cached_traces()
    colors = {p: core.PALETTE_SINGLE[i % len(core.PALETTE_SINGLE)]
              for i, p in enumerate(PEPTIDE_ORDER)}

    apex = {}
    for t in tr:
        if t['level'] == 5:
            apex[t['peptide']] = float(t['rt'][int(np.argmax(t['ions']))])

    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 3.4))

    rt1, cur = cached_tic()
    bg = ax.twinx()
    bg.fill_between(rt1, 0, cur, color=TIC_FILL, lw=0, zorder=0)
    bg.set_ylim(0, float(np.nanmax(cur)) * 1.02)
    bg.set_ylabel('TIC', fontsize=fs.FONTSIZE, color=TIC_INK)
    bg.tick_params(axis='y', labelsize=fs.FONTSIZE, colors=TIC_INK)
    bg.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    bg.yaxis.get_offset_text().set(fontsize=fs.FONTSIZE, color=TIC_INK)
    bg.spines['right'].set(visible=True, color=TIC_INK,
                           linewidth=ax.spines['left'].get_linewidth())
    for side in ('top', 'left'):
        bg.spines[side].set_visible(False)
    ax.set_zorder(bg.get_zorder() + 1)
    ax.patch.set_visible(False)
    print(f'  TIC: {len(rt1):,} cycles, '
          f'{rt1.min():.2f}-{rt1.max():.2f} min, '
          f'{cur.sum():.3e} total, peak cycle {cur.max():.3e}')

    rows = [{'peptide': 'TIC', 'level': -1, 'run': RUN,
             'rt_min': float(a), 'ions': float(b)}
            for a, b in zip(rt1, cur)]
    for t in sorted(tr, key=lambda t: (apex.get(t['peptide'], 0), t['level'])):
        pep, c = t['peptide'], colors[t['peptide']]
        if pep not in apex:
            continue
        keep = np.abs(t['rt'] - apex[pep]) <= WINDOW_MIN
        rt, y = t['rt'][keep], t['ions'][keep]
        ax.plot(rt, y, color=c, lw=1.0, alpha=0.9, zorder=3)
        rows += [{'peptide': pep, 'level': t['level'], 'run': RUN,
                  'rt_min': float(a), 'ions': float(b)}
                 for a, b in zip(rt, y)]

    handles = [plt.Line2D([], [], color=colors[p], lw=1.5, label=p)
               for p in sorted(PEPTIDE_ORDER, key=lambda p: apex.get(p, 0))
               if p in apex]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='#D8D8D8',
                                 edgecolor='none', label='TIC'))
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.07),
              ncol=2, frameon=False, fontsize=fs.LEGEND_FONTSIZE,
              handlelength=1.1, handletextpad=0.4, borderpad=0.0,
              columnspacing=0.8, labelspacing=0.25)

    ax.set_xlabel('Retention time [min]', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Targeted peptide intensity', fontsize=fs.FONTSIZE)
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=fs.FONTSIZE)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(fs.FONTSIZE)
    fs.tidy(ax)

    for p in PEPTIDE_ORDER:
        if p in apex:
            n = sum(1 for t in tr if t['peptide'] == p)
            print(f'  {p:11s} apex {apex[p]:.2f} min, {n} levels drawn')

    print(f'  TIC peak / tallest transition: '
          f'{cur.max() / max(r["ions"] for r in rows if r["level"] > 0):.0f}x')

    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=AXES_W_IN, h_in=fs.AXES_H_IN)
    bg.set_position(ax.get_position())
    fs.save_panel(fig, OUT, f'figure6{letter}_6x5_chromatogram',
                  pd.DataFrame(rows), dpi=300)
    plt.close(fig)
    fs.report_stale(OUT, prefix='figure')


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['a']))
