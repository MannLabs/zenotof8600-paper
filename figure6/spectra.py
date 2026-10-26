"""Figure 6, panel f -- CID against EAD on the alpha-synuclein assay peptide."""

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
from matplotlib.ticker import FuncFormatter
import pandas as pd

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = cfg.panels(__file__)
E391_DIR = cfg.data('H032_E391')

RAW_CACHE = cfg.raw('ms_data_cache')
DATA = os.path.join(cfg.DATA_ROOT, 'H032_E391')

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

SPECIES = 'heavy pS129'

PANELS = (('CID', 'A1', 'cid', None),
          ('EAD 15 eV', 'A6', 'ead', None))

AXES_W_IN = fs.PANEL_W_IN * 1.70
AXES_H_IN = 0.90

Y_TICK_CHARS = 6

FIGURE_SPACE = ' '


_LADDERS = []


def _draw_ladder(ax, sequence, cover, highlight_sites=(), y=1.17,
                 x0=0.015, x1=0.985, fontsize=9.0):
    """`annotate_spectra.draw_ladder` with the bond symbols moved off the letters."""
    n = len(sequence)
    xs = np.linspace(x0, x1, n)
    for i, aa in enumerate(sequence):
        hl = (i + 1) in highlight_sites
        ax.text(xs[i], y, aa, transform=ax.transAxes, ha='center', va='center',
                fontsize=fontsize, family='monospace',
                fontweight='bold' if hl else 'normal',
                color='#D6403A' if hl else '#222222')
    _LADDERS.append((ax, xs, (x1 - x0) / (n - 1), y, dict(cover), fontsize))


def _place_bond_symbols(fig, colors):
    """Draw the recorded cleavage symbols, clear of the letters, in points."""
    tops = {}
    for ax, xs, step, y, cover, fontsize in _LADDERS:
        h_in = ax.get_position().height * fig.get_size_inches()[1]
        pt = 1.0 / 72.0 / h_in
        half = fontsize * 0.36
        gap, reach = (half + 1.6) * pt, (half + 6.0) * pt
        for bond, (cls, side) in cover.items():
            x = xs[bond - 1] + step / 2
            colour = colors[cls]
            up = -1 if side == 'c' else +1
            foot = +1 if side == 'c' else -1
            ax.plot([x, x], [y + up * gap, y + up * reach],
                    transform=ax.transAxes, color=colour, lw=1.0,
                    clip_on=False, zorder=6)
            ax.plot([x, x + foot * step * 0.32], [y + up * reach] * 2,
                    transform=ax.transAxes, color=colour, lw=1.0,
                    clip_on=False, zorder=6)
            if up > 0:
                tops[ax] = max(tops.get(ax, y), y + reach)
        tops.setdefault(ax, y + reach)
    return tops


SITE_DIAGNOSTIC_NUMBERS = (11, 12)


def _quant_ions():
    from occupancy import QUANT_SET
    return set(QUANT_SET)


def _classify(row, cterm):
    """`annotate_spectra.classify` with the paper's ion roles (see above)."""
    if row.ion_type in cterm:
        quant = _quant_ions()
        if row.fragment_charge == 1:
            if (row.ion_type, row.ion_number) in quant:
                return 'quant'
            cid = not any(str(t).startswith('z') for t in cterm)
            if cid and row.ion_type == 'y' and row.ion_number in {
                    n for _, n in quant}:
                return 'quant'
        if row.ion_number in SITE_DIAGNOSTIC_NUMBERS:
            return 'site-diagnostic'
    return 'other'


def _e391():
    """Import E391's modules, which resolve each other by bare module name."""
    if E391_DIR not in sys.path:
        sys.path.insert(0, E391_DIR)
    import annotate_cid_vs_ead as cve
    import annotate_spectra as asp
    from score_cid_sweep import summed_at
    from score_sweep import energy_labelled, summed_for
    import score_fragment_coverage as sfc
    sfc.RAW_DIR = sfc.pathlib.Path(cfg.raw('H032_E391', 'raw'))
    sfc.CACHE_DIR = sfc.pathlib.Path(RAW_CACHE)
    return cve, asp, summed_at, energy_labelled, summed_for, sfc


SUMMED = cfg.data('H032_E391', 'summed_spectra')


def summed(asp, sfc, runs, run, q1, frag_mz):
    """(centroided summed spectrum, number of spectra summed) for one run."""
    path = os.path.join(SUMMED, f'{run}_{SPECIES}.csv')
    if os.path.exists(path):
        t = pd.read_csv(path)
        return t[['mz', 'intensity']], int(t['n_spectra'].iloc[0])
    if run not in runs:
        runs[run] = sfc.load(run)
    spec, _, _apex, n = asp.summed_spectrum(runs[run], q1, frag_mz)
    if spec is not None and len(spec):
        os.makedirs(SUMMED, exist_ok=True)
        spec.assign(n_spectra=n).to_csv(path, index=False)
    return spec, n


def panel(letters, scale_to='assay', out=None, name_fmt=None, h_in=AXES_H_IN,
          y_tick_chars=Y_TICK_CHARS):
    """One FILE per activation condition, so each takes its own letter."""
    cve, asp, summed_at, energy_labelled, summed_for, sfc = _e391()
    core.raw.register_sil_modifications()
    asp.draw_ladder = _draw_ladder
    asp.classify = _classify

    tg = sfc.targets()
    q1tab = core.precursor_q1_mz(tg, isotope='most_abundant')
    i = tg.index[tg['species'] == SPECIES][0]
    q1 = float(q1tab.loc[i, 'q1_mz'])

    runs, built = {}, []
    for letter, (label, run, act, sel) in zip(letters, PANELS):
        _LADDERS.clear()
        frags = core.raw.target_fragments(tg, frag_types=sfc.FRAG_TYPES[act],
                                          min_mz=150.0, max_mz=3000.0)
        sub = frags[(frags['target_idx'] == i)
                    & (frags['fragment_charge'] <= asp.MAX_FRAGMENT_CHARGE)]
        with_prec = pd.concat(
            [sub, asp.precursor_species((q1 - 1.00727646) * 4)],
            ignore_index=True)

        spec, n_summed = summed(asp, sfc, runs, run, q1,
                                with_prec['product_mz'].astype(float))
        if spec is None or not len(spec):
            raise ValueError(f'{label}: nothing matched for {SPECIES} at '
                             f'Q1 {q1:.3f} in run {run}')

        fig, ax = plt.subplots(figsize=(AXES_W_IN + 1.2, h_in + 1.3))
        f, cov = asp.draw(
            ax, spec, with_prec, sfc.CTERM_SERIES_FOR[act],
            f'{label} | {n_summed} spectra summed', sfc.SEQUENCE,
            highlight_sites=(sfc.PHOSPHO_SITE,) if 'pS129' in SPECIES else (),
            scale_to=scale_to)
        f.insert(0, 'panel', label)
        f.insert(0, 'species', SPECIES)
        f.insert(0, 'sequence_coverage', round(cov, 4))
        ax.tick_params(labelsize=fs.FONTSIZE)
        ax.xaxis.label.set_fontsize(fs.FONTSIZE)
        ax.yaxis.label.set_fontsize(fs.FONTSIZE)
        lab = ax.get_ylabel()
        if lab:
            ax.set_ylabel(lab[:1].upper() + lab[1:], fontsize=fs.FONTSIZE)
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f'{v:.0f}'.rjust(y_tick_chars, FIGURE_SPACE)))
        lo, hi = ax.get_ylim()
        shown = [t for t in ax.get_yticks() if lo <= t <= hi]
        widest = max(len(f'{t:.0f}') for t in shown)
        if widest > y_tick_chars:
            raise ValueError(f'{label}: y tick needs {widest} characters, '
                             f'y_tick_chars is {y_tick_chars}')

        GREY = ('ions matched', 'sequence coverage', 'y scaled',
                'scaled independently')
        for t in list(ax.texts):
            if any(g in t.get_text() for g in GREY):
                t.remove()

        leg, handles, labels_ = ax.get_legend(), [], []
        if leg is not None:
            handles = list(leg.legend_handles)
            labels_ = [t.get_text() for t in leg.get_texts()]
            leg.remove()
        if letter == letters[0] and handles:
            leg = ax.legend(handles, labels_, loc='lower center',
                            bbox_to_anchor=(0.5, 1.16), ncol=len(labels_),
                            frameon=False, fontsize=fs.LEGEND_FONTSIZE,
                            handlelength=1.3, handletextpad=0.4,
                            columnspacing=1.2)
        else:
            leg = None

        fig.tight_layout()
        fs.set_axes_size_inches(fig, ax, w_in=AXES_W_IN, h_in=h_in)
        tops = _place_bond_symbols(fig, asp.CLASS_COLORS)
        if leg is not None:
            leg.set_bbox_to_anchor((0.5, tops[ax] + 0.05),
                                   transform=ax.transAxes)
        print(f'    {letter}  {label:12s} {n_summed:2d} scans  '
              f'{int((f["observed_intensity"] > 0).sum()):3d} ions  '
              f'coverage {cov:.0%}'
              + ('  (carries the class key)' if leg is not None else ''))
        built.append((letter, act, label, fig, ax, f))

    lo = min(a.get_xlim()[0] for *_, a, _ in built)
    hi = max(a.get_xlim()[1] for *_, a, _ in built)
    print(f'    shared m/z range {lo:.0f}-{hi:.0f} Th '
          + ', '.join(f'{lab} drew {a.get_xlim()[1]:.0f}'
                      for _l, _act, lab, _fig, a, _f in built))
    for _letter, act, _label, fig, ax, f in built:
        ax.set_xlim(lo, hi)
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        print(f'       left edge to y axis: '
              f'{ax.get_position().x0 * fig.get_size_inches()[0] - bb.x0:.3f} in')
        name = (name_fmt or 'figure6{letter}_{act}_spectrum').format(
            letter=_letter, act=act)
        fs.save_panel(fig, out or OUT, name, f, dpi=300)
        plt.close(fig)


def main(*letters):
    letters = letters or ('h', 'i')
    fs.init()
    fs.start_build(script=__file__, inputs=[DATA], out_dir=OUT)
    print(f'panels {", ".join(letters)} ... ({SPECIES})')
    panel(letters)
    fs.report_stale(OUT, prefix='figure')


if __name__ == '__main__':
    main(*sys.argv[1:])
