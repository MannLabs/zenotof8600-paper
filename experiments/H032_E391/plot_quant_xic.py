"""Quant-transition chromatograms, heavy and light, phospho and unmodified."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import zenotof8600_analytics as core

from annotate_spectra import MAX_FRAGMENT_CHARGE
from make_final_method import all_species
from score_fragment_coverage import (FRAGMENT_TOL_PPM, FRAG_TYPES, PREFIX,
                                     RAW_DIR)
from score_quant_feasibility import QUANT_SET

OUT_DIR = pathlib.Path(cfg.cache('H032_E391', 'analysis'))
CACHE_DIR = pathlib.Path(cfg.raw('ms_data_cache'))

RUN = 'AB12'

APEX = {'pS129': 5.430, 'unmod': 5.337}

PLOT_HALF_MIN = 1.0

COMPARE_HALF_MIN = 0.12

COMPARE_RUNS = (('pure standard', 'A6'), ('HEK background', 'B10'))

XIC_TOL_PPM = 13.7

PROTON = 1.00727646


def draw_transitions(ax, ms, sp, frags, i, half_min, apex, label_legend,
                     quant_set=QUANT_SET):
    """Plot every quant transition of target `i`, un-normalised, plus the sum."""
    pal = core.PALETTE_SINGLE
    q1 = float(sp.loc[i, 'q1_mz'])
    sub = frags[(frags['target_idx'] == i)
                & (frags['fragment_charge'] <= MAX_FRAGMENT_CHARGE)]
    lo, hi = apex - half_min, apex + half_min
    total, rt_ref, rows = None, None, []
    wanted = []
    for ion_type, number in quant_set:
        t = sub[(sub['ion_type'] == ion_type)
                & (sub['ion_number'] == number)
                & (sub['fragment_charge'] == 1)]
        if not t.empty:
            wanted.append((f'{ion_type}{number}',
                           float(t['product_mz'].iloc[0])))
    if not wanted:
        return []
    cols, base = [], None
    for _, m in wanted:
        x = core.raw.targeted_xic(ms, q1, m, mz_tol=XIC_TOL_PPM)
        base = x[['rt']] if base is None else base
        cols.append(x['intensity'].to_numpy())
    xic = base.copy()
    for j, c in enumerate(cols):
        xic[f'c{j}'] = c
    keep = (xic['rt'] >= lo) & (xic['rt'] <= hi)
    for k, ((name, mz), col) in enumerate(zip(wanted, xic.columns[1:])):
        ion_type, number = name[0], name[1:]
        rt = xic.loc[keep, 'rt'].to_numpy()
        y = xic.loc[keep, col].to_numpy()
        if rt.size == 0:
            continue
        ax.plot(rt, y, lw=1.0, color=pal[k % len(pal)],
                label=name, alpha=0.9)
        total = y.copy() if total is None else total + y
        rt_ref = rt
        rows.append(pd.DataFrame({'ion': name, 'mz': round(mz, 4),
                                  'rt': rt, 'intensity': y}))
    if total is not None:
        ax.plot(rt_ref, total, lw=1.8, color='0.25', label='sum', zorder=5)
    ax.axvline(apex, ls='--', lw=1, color='0.5')
    ax.set_xlabel('RT [min]', fontsize=10)
    ax.set_ylabel('counts', fontsize=10)
    if label_legend:
        ax.legend(fontsize=9, frameon=False, ncol=2)
    return rows


def compare_to_standard():
    """The same transitions in the pure standard and in HEK, heavy channel."""
    core.raw.register_sil_modifications()
    sp = core.precursor_q1_mz(all_species(), isotope='most_abundant')
    frags = core.raw.target_fragments(sp, frag_types=FRAG_TYPES['ead'],
                                      min_mz=150.0, max_mz=3000.0)
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    rows = []
    for c, (label, run) in enumerate(COMPARE_RUNS):
        ms = core.raw.load_ms_data(
            str(RAW_DIR / f'{PREFIX}{run}.wiff'), cache_dir=str(CACHE_DIR),
            reader_options={'keep_k_peaks_per_spec': 10 ** 7})
        for r, base in enumerate(('pS129', 'unmod')):
            ax = axes[r, c]
            hit = sp.index[sp['compound_id'] == f'aSyn_h_{base}']
            if not len(hit):
                ax.axis('off')
                continue
            got = draw_transitions(ax, ms, sp, frags, hit[0],
                                   COMPARE_HALF_MIN, APEX[base],
                                   label_legend=(r == 0 and c == 0))
            for g in got:
                g.insert(0, 'run', run)
                g.insert(0, 'compound_id', f'aSyn_h_{base}')
                rows.append(g)
            n_scan = len(got[0]) if got else 0
            ax.set_title(
                f'{"phospho S129" if base == "pS129" else "unmodified"} | '
                f'{label} ({run})', fontsize=11, fontweight='bold')
            ax.annotate(f'{n_scan} scans in {2 * COMPARE_HALF_MIN * 60:.0f} s',
                        (0.98, 0.96), xycoords='axes fraction', ha='right',
                        va='top', fontsize=8, color='0.3')
    fig.suptitle('quant transitions, heavy channel | y is counts, not '
                 'normalised; each panel on its own scale', fontsize=11)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(OUT_DIR / f'quant_xic_standard_vs_hek.{ext}', dpi=300,
                    bbox_inches='tight')
    plt.close(fig)
    pd.concat(rows, ignore_index=True).to_csv(
        OUT_DIR / 'quant_xic_standard_vs_hek_sourcedata.csv', index=False)
    print(f'wrote {OUT_DIR / "quant_xic_standard_vs_hek.pdf"} '
          '(+ .png, _sourcedata.csv)')


def main():
    core.raw.register_sil_modifications()
    sp = core.precursor_q1_mz(all_species(), isotope='most_abundant')
    frags = core.raw.target_fragments(sp, frag_types=FRAG_TYPES['ead'],
                                      min_mz=150.0, max_mz=3000.0)
    ms = core.raw.load_ms_data(
        str(RAW_DIR / f'{PREFIX}{RUN}.wiff'), cache_dir=str(CACHE_DIR),
        reader_options={'keep_k_peaks_per_spec': 10 ** 7})

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    rows = []
    for r, base in enumerate(('pS129', 'unmod')):
        for c, form in enumerate(('h', 'l')):
            ax = axes[r, c]
            cid = f'aSyn_{form}_{base}'
            hit = sp.index[sp['compound_id'] == cid]
            if not len(hit):
                ax.axis('off')
                continue
            got = draw_transitions(ax, ms, sp, frags, hit[0], PLOT_HALF_MIN,
                                   APEX[base], label_legend=(r == 0 and c == 0))
            for g in got:
                g.insert(0, 'run', RUN)
                g.insert(0, 'channel', form)
                g.insert(0, 'compound_id', cid)
                rows.append(g)
            ax.set_title(
                f'{"phospho S129" if base == "pS129" else "unmodified"} | '
                f'{"heavy (spike)" if form == "h" else "light (endogenous)"}',
                fontsize=11, fontweight='bold')
    fig.suptitle(f'quant transitions, {RUN} (HEK background) | '
                 'y is counts, not normalised', fontsize=11)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(OUT_DIR / f'quant_xic_{RUN}.{ext}', dpi=300,
                    bbox_inches='tight')
    plt.close(fig)
    pd.concat(rows, ignore_index=True).to_csv(
        OUT_DIR / f'quant_xic_{RUN}_sourcedata.csv', index=False)
    print(f'wrote {OUT_DIR / f"quant_xic_{RUN}.pdf"} (+ .png, _sourcedata.csv)')


if __name__ == '__main__':
    main()
    compare_to_standard()
