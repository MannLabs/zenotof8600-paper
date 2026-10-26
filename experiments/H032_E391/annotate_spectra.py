"""Annotated CID and EAD spectra for the alpha-synuclein pS129 assay, H032_E391."""

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

from score_fragment_coverage import (CACHE_DIR, CTERM_SERIES_FOR,
                                     FRAGMENT_TOL_PPM, FRAG_TYPES,
                                     MIN_COUNTS, PHOSPHO_SITE, PREFIX,
                                     RAW_DIR, SEQUENCE, activation_of,
                                     load, targets)

ANALYSIS_DIR = pathlib.Path(cfg.cache('H032_E391', 'analysis'))

OUT_SUBDIR = 'spectra'

SPECIES = ('heavy pS129', 'heavy pS129 +16', 'heavy pS129 +32',
           'heavy unmod', 'heavy unmod +16', 'heavy unmod +32')
RUNS = ('A1', 'A2')

QUANT_NUMBERS = (6, 8, 9, 10, 11)
DIAGNOSTIC_NUMBERS = (5, 12, 16)

CLASS_COLORS = {
    'quant': core.PALETTE_SINGLE[0],
    'site-diagnostic': core.PALETTE_SINGLE[1],
    'other': core.PALETTE_SINGLE[2],
    'precursor': core.PALETTE_SINGLE[3],
}
UNMATCHED_COLOR = '#BBBBBB'

ELECTRON_MASS = 0.000548579909

TITLE_PAD_PT = 64

Y_HEADROOM = 1.15
LABEL_INSIDE_FRAC = 0.80

SUM_BIN_PPM = 3.0

MAX_FRAGMENT_CHARGE = 2
LABEL_CLASSES = ('quant', 'site-diagnostic')

SERIES_LABEL = {'c_lossH': 'c•', 'z': 'z•',
                'z_addH': 'z′', 'z_lossH': 'z'}

MIN_LABEL_SEP_PT = 7.5
LABEL_LEADER_LW = 0.4

LABEL_HALO_LW = 2.0
LABEL_HALO_COLOR = 'white'

MAX_LABELS = 20

MIN_LABEL_Y_FRAC = 0.10


def classify(row, cterm):
    """quant / site-diagnostic / other, by the ion's role in the assay."""
    if row.ion_type in cterm:
        if row.ion_number in QUANT_NUMBERS:
            return 'quant'
        if row.ion_number in DIAGNOSTIC_NUMBERS:
            return 'site-diagnostic'
    return 'other'


def precursor_species(neutral, charge=4):
    """Intact precursor and its charge-reduced products (EAD)."""
    rows = []
    for z in range(charge, 0, -1):
        k = charge - z
        mz = (neutral + charge * 1.00727646 + k * ELECTRON_MASS) / z
        rows.append({'ion_type': 'M', 'ion_number': 0, 'fragment_charge': z,
                     'product_mz': mz})
    return pd.DataFrame(rows)


def target_spectra(ms, q1, ms_level=2):
    """MS2 scans belonging to ONE target - now `core.raw.target_scans`."""
    return core.raw.target_scans(ms, q1, ms_level=ms_level)


def summed_spectrum(ms, q1, frag_mz, min_counts=MIN_COUNTS, rt_bracket=None,
                    rt_width=None):
    """Sum MS2 over the precursor's own FWHM; returns (mz, intensity, rt_range)."""
    ms2 = target_spectra(ms, q1)
    xic = core.raw.extract_xic(ms2, list(frag_mz), mz_tol=FRAGMENT_TOL_PPM,
                               ms_level=2)
    if rt_bracket is not None:
        xic = xic[(xic['rt'] >= rt_bracket[0]) & (xic['rt'] <= rt_bracket[1])]
        if len(xic) < 3:
            return None
    pk = core.raw.pick_peak(xic['rt'].to_numpy(), xic['intensity'].to_numpy(),
                            baseline='offpeak', interpolate_points=500)
    if pk is None or not np.isfinite(pk.get('apex_rt', np.nan)):
        return None
    if rt_width is not None:
        rt_range = (float(pk['apex_rt']) - rt_width / 2,
                    float(pk['apex_rt']) + rt_width / 2)
    else:
        rt_range = (float(pk['left_rt']), float(pk['right_rt']))
    summed = core.raw.sum_spectra(ms2, rt_range=rt_range, ms_level=2,
                                  bin_tol=SUM_BIN_PPM)
    cen = core.raw.centroid_profile(summed['mz'].to_numpy(),
                                    summed['intensity'].to_numpy())
    spec = pd.DataFrame({'mz': cen[0], 'intensity': cen[1]})
    spec = spec[spec['intensity'] >= min_counts].reset_index(drop=True)
    sd = ms2.spectrum_df
    n_spectra = int(((sd['rt'] >= rt_range[0])
                     & (sd['rt'] <= rt_range[1])).sum())
    return spec, rt_range, float(pk['apex_rt']), n_spectra


LADDER_PRIORITY = ('site-diagnostic', 'quant', 'other', 'precursor')

NTERM_SERIES, CTERM_SERIES = ('a', 'b', 'c'), ('x', 'y', 'z')


def bond_coverage(hit, n_residues):
    """Backbone bonds covered by at least one matched fragment."""
    cover = {}
    for r in hit.itertuples():
        if r.ion_type in NTERM_SERIES:
            bond, side = int(r.ion_number), 'n'
        elif r.ion_type in CTERM_SERIES:
            bond, side = n_residues - int(r.ion_number), 'c'
        else:
            continue
        if not 1 <= bond <= n_residues - 1:
            continue
        prev = cover.get(bond)
        if prev is None or (LADDER_PRIORITY.index(r.ion_class)
                            < LADDER_PRIORITY.index(prev[0])):
            cover[bond] = (r.ion_class, side)
    return cover


def draw_ladder(ax, sequence, cover, highlight_sites=(), y=1.17,
                x0=0.015, x1=0.985, fontsize=9.0):
    """Sequence above the axes with a tick on every cleaved bond."""
    n = len(sequence)
    xs = np.linspace(x0, x1, n)
    step = (x1 - x0) / (n - 1)
    for i, aa in enumerate(sequence):
        hl = (i + 1) in highlight_sites
        ax.text(xs[i], y, aa, transform=ax.transAxes, ha='center', va='center',
                fontsize=fontsize, family='monospace',
                fontweight='bold' if hl else 'normal',
                color='#D6403A' if hl else '#222222')
    for bond, (cls, side) in cover.items():
        x = xs[bond - 1] + step / 2
        colour = CLASS_COLORS[cls]
        dy, foot = (0.030, +1) if side == 'c' else (-0.030, -1)
        ax.plot([x, x], [y - 0.022, y + 0.022], transform=ax.transAxes,
                color=colour, lw=1.0, clip_on=False, zorder=6)
        ax.plot([x, x + foot * step * 0.32], [y + dy, y + dy],
                transform=ax.transAxes, color=colour, lw=1.0, clip_on=False,
                zorder=6)


def _spread(xs, min_sep):
    """Push values apart to `min_sep` spacing, keeping each cluster centred."""
    out = list(xs)
    for i in range(1, len(out)):
        if out[i] - out[i - 1] < min_sep:
            out[i] = out[i - 1] + min_sep
    for i in range(len(out) - 2, -1, -1):
        if out[i + 1] - out[i] < min_sep:
            out[i] = out[i + 1] - min_sep
    shift = np.mean(np.asarray(xs) - np.asarray(out))
    return [v + shift for v in out]


def _place_labels(ax, rows, ymax):
    """Label every row, spread along x, each tied to its peak by a leader."""
    import matplotlib.patheffects as pe
    ax.figure.canvas.draw()
    to_pt = ax.figure.dpi / 72.0
    x_pt = [ax.transData.transform((m, 0))[0] / to_pt
            for m in rows['observed_mz']]
    placed_pt = _spread(x_pt, MIN_LABEL_SEP_PT)
    inv = ax.transData.inverted()
    halo = [pe.withStroke(linewidth=LABEL_HALO_LW, foreground=LABEL_HALO_COLOR)]

    for r, x_new_pt in zip(rows.itertuples(), placed_pt):
        x_new = inv.transform((x_new_pt * to_pt, 0))[0]
        peak_y = min(r.observed_intensity, ymax)
        inside = peak_y > LABEL_INSIDE_FRAC * ymax
        base = (peak_y - 0.03 * ymax if inside
                else max(peak_y + 0.03 * ymax, MIN_LABEL_Y_FRAC * ymax))
        colour = CLASS_COLORS[r.ion_class]
        ax.plot([r.observed_mz, x_new], [peak_y, base], color=colour,
                lw=LABEL_LEADER_LW, zorder=4, solid_capstyle='butt')
        ax.annotate(r.label, (x_new, base), xytext=(0, -1 if inside else 1),
                    textcoords='offset points', ha='center',
                    va='top' if inside else 'bottom', fontsize=7,
                    rotation=90, color=colour, zorder=6,
                    path_effects=halo)


def draw(ax, spec, frags, cterm, title, sequence, highlight_sites=(),
         scale_to='assay'):
    """One annotated panel; returns (source-data rows, sequence coverage)."""
    mz = spec['mz'].to_numpy()
    inten = spec['intensity'].to_numpy()

    obs_int, obs_mz = core.raw.match_peaks(mz, inten,
                                           frags['product_mz'].to_numpy(),
                                           tol=FRAGMENT_TOL_PPM, reduce='max')
    f = frags.copy()
    f['observed_mz'] = obs_mz
    f['observed_intensity'] = obs_int
    f['ion_class'] = [('precursor' if r.ion_type == 'M' else classify(r, cterm))
                      for r in f.itertuples()]
    f['label'] = [(f'[M+{r.fragment_charge}H]' if r.ion_type == 'M'
                   else f'{SERIES_LABEL.get(r.ion_type, r.ion_type)}'
                        f'{r.ion_number}')
                  + ('' if r.fragment_charge == 1 or r.ion_type == 'M'
                     else f'$^{{{r.fragment_charge}+}}$')
                  for r in f.itertuples()]
    hit = f[f['observed_intensity'] > 0]

    assay = hit[hit['ion_class'].isin(('quant', 'site-diagnostic'))]
    site = hit[hit['ion_class'] == 'site-diagnostic']
    if scale_to == 'all':
        top = float(inten.max())
    elif scale_to == 'site' and len(site):
        top = float(site['observed_intensity'].max())
    elif len(assay):
        top = float(assay['observed_intensity'].max())
    elif len(hit):
        top = float(hit['observed_intensity'].max())
    else:
        top = float(inten.max())
    matched_frac = hit['observed_intensity'].sum() / inten.sum() if inten.sum() else 0.0

    ax.vlines(mz, 0, inten, color=UNMATCHED_COLOR, lw=0.6, zorder=1)
    for cls, g in hit.groupby('ion_class'):
        ax.vlines(g['observed_mz'], 0, g['observed_intensity'],
                  color=CLASS_COLORS[cls], lw=1.4, zorder=3, label=cls)
    ymax = top * Y_HEADROOM
    ax.set_ylim(0, ymax)
    to_label = hit[hit['ion_class'].isin(LABEL_CLASSES)]
    if len(to_label) > MAX_LABELS:
        to_label = to_label.nlargest(MAX_LABELS, 'observed_intensity')
    to_label = to_label.sort_values('observed_mz').reset_index(drop=True)
    if len(to_label):
        _place_labels(ax, to_label, ymax)
    hi = float(np.max(mz[inten > 0])) if (inten > 0).any() else float(mz.max())
    ax.set_xlim(0, hi * 1.03)
    ax.set_xlabel('m/z', fontsize=10)
    ax.set_ylabel('intensity', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=TITLE_PAD_PT)
    cover = bond_coverage(hit, len(sequence))
    seq_cov = len(cover) / (len(sequence) - 1)
    draw_ladder(ax, sequence, cover, highlight_sites=highlight_sites)
    ax.annotate(f'{len(hit)}/{len(f)} ions matched · {matched_frac:.0%} of TIC'
                f'\nsequence coverage {seq_cov:.0%} '
                f'({len(cover)}/{len(sequence) - 1} bonds)'
                + ('\ny scaled to the tallest peak in the spectrum'
                   if scale_to == 'all'
                   else '\ny scaled to the tallest assay ion'),
                (0.99, 0.97), xycoords='axes fraction', ha='right', va='top',
                fontsize=8, color='0.3')
    ax.legend(fontsize=9, frameon=False, loc='lower center',
              bbox_to_anchor=(0.5, 1.02), ncol=4, handlelength=1.4,
              columnspacing=1.6)
    return f, seq_cov


def _parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--runs', default=','.join(RUNS),
                    help='comma-separated vials, CID first (default %(default)s)')
    ap.add_argument('--subdir', default=OUT_SUBDIR,
                    help='subfolder under analysis/ (default %(default)s)')
    return ap.parse_args(argv)


def main(runs=RUNS, subdir=OUT_SUBDIR):
    out_dir = ANALYSIS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    core.raw.register_sil_modifications()
    tg = targets()
    rows = []

    for species in SPECIES:
        i = tg.index[tg['species'] == species][0]
        fig, axes = plt.subplots(2, 1, figsize=(8, 11))
        panels = 0
        for ax, run in zip(axes, runs):
            ms = load(run)
            act, _ = activation_of(ms)
            frags = core.raw.target_fragments(
                tg, frag_types=FRAG_TYPES[act], min_mz=150.0, max_mz=3000.0)
            sub = frags[frags['target_idx'] == i]
            q1 = float(sub['precursor_mz'].iloc[0])
            sub = pd.concat([sub, precursor_species((q1 - 1.00727646) * 4)],
                            ignore_index=True)
            got = summed_spectrum(ms, q1, sub['product_mz'].astype(float))
            if got is None:
                ax.axis('off')
                continue
            spec, rt_range, apex, n_spectra = got
            title = (f'{species} | {act.upper()} | Q1 {q1:.3f} | '
                     f'{apex:.2f} min | {n_spectra} spectra summed')
            f, seq_cov = draw(ax, spec, sub, CTERM_SERIES_FOR[act], title, SEQUENCE,
                              highlight_sites=(PHOSPHO_SITE,)
                              if 'pS129' in species else ())
            f.insert(0, 'sequence_coverage', round(seq_cov, 4))
            f.insert(0, 'run', run)
            f.insert(1, 'activation', act)
            f.insert(2, 'species', species)
            rows.append(f)
            panels += 1
            print(f'{species:14s} {run} {act.upper():4s} apex {apex:.3f} min  '
                  f'{(f["observed_intensity"] > 0).sum()}/{len(f)} ions  '
                  f'sequence coverage {seq_cov:.0%}')
        if panels:
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.42, top=0.92)
            stem = species.replace(' ', '_').replace('+', 'ox')
            for ext in ('pdf', 'png'):
                fig.savefig(out_dir / f'annotated_{stem}.{ext}', dpi=300,
                            bbox_inches='tight')
            print(f'  wrote {out_dir / f"annotated_{stem}.pdf"} (+ .png)')
        plt.close(fig)

    src = pd.concat(rows, ignore_index=True)
    src.to_csv(out_dir / 'annotated_spectra_sourcedata.csv', index=False)
    print(f'\nwrote {out_dir / "annotated_spectra_sourcedata.csv"}')


if __name__ == '__main__':
    args = _parse_args()
    main(runs=tuple(args.runs.split(',')), subdir=args.subdir)
