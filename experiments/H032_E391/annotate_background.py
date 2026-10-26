"""The same peptide with and without a HEK background, EAD 15 eV."""

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

from annotate_spectra import (MAX_FRAGMENT_CHARGE, draw, precursor_species,
                              summed_spectrum)
from score_fragment_coverage import (CTERM_SERIES_FOR, FRAG_TYPES, PHOSPHO_SITE,
                                     PREFIX, RAW_DIR, SEQUENCE, targets)

OUT_DIR = pathlib.Path(cfg.cache('H032_E391', 'analysis', 'spectra_background'))
CACHE_DIR = pathlib.Path(cfg.raw('ms_data_cache'))

PANELS = (('standard only, 15 ms', 'A6'),
          ('HEK, scheduled, 15 ms', 'B1'),
          ('HEK, unscheduled, 15 ms', 'B4'),
          ('HEK, unscheduled, 5 ms', 'B7'))

SPECIES = ('heavy pS129', 'heavy unmod')

RT_BRACKET_MIN = 0.15
REFERENCE_RUN = 'A6'


def main(panels=PANELS, name='background', out_dir=OUT_DIR):
    """Draw one figure per species over `panels`, in both y scalings."""
    OUT_DIR = out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    core.raw.register_sil_modifications()
    tg = targets()
    q1tab = core.precursor_q1_mz(tg, isotope='most_abundant')
    frags = core.raw.target_fragments(tg, frag_types=FRAG_TYPES['ead'],
                                      min_mz=150.0, max_mz=3000.0)
    runs = {r: core.raw.load_ms_data(
        str(RAW_DIR / f'{PREFIX}{r}.wiff'), cache_dir=str(CACHE_DIR),
        reader_options={'keep_k_peaks_per_spec': 10 ** 7})
        for _, r in panels}

    rows = []
    for species in SPECIES:
        i = tg.index[tg['species'] == species][0]
        q1 = float(q1tab.loc[i, 'q1_mz'])
        sub = frags[(frags['target_idx'] == i)
                    & (frags['fragment_charge'] <= MAX_FRAGMENT_CHARGE)]
        prec_mz = precursor_species((q1 - 1.00727646) * 4)
        with_prec = pd.concat([sub, prec_mz], ignore_index=True)

        extracted = []
        bracket, width = None, None
        for label, run in panels:
            got = summed_spectrum(runs[run], q1,
                                  sub['product_mz'].astype(float),
                                  rt_bracket=bracket, rt_width=width)
            if got is not None and run == REFERENCE_RUN:
                bracket = (got[2] - RT_BRACKET_MIN, got[2] + RT_BRACKET_MIN)
                width = got[1][1] - got[1][0]
            extracted.append((label, run, got))

        stem = species.replace(' ', '_')
        for scale_to, suffix in (('assay', ''), ('all', '_scaled_all')):
            fig, axes = plt.subplots(len(extracted), 1,
                                     figsize=(8, 5.5 * len(extracted)))
            drawn = 0
            for ax, (label, run, got) in zip(axes, extracted):
                if got is None:
                    ax.axis('off')
                    continue
                spec, _, apex, n_window = got
                pv, _ = core.raw.match_peaks(
                    spec['mz'].to_numpy(), spec['intensity'].to_numpy(),
                    prec_mz['product_mz'].to_numpy(), tol=10.0, reduce='max')
                frac = pv.sum() / max(spec['intensity'].sum(), 1)
                title = (f'{species} | {label} ({run}) | Q1 {q1:.3f} | '
                         f'{apex:.2f} min | {n_window} scans | '
                         f'precursor {frac:.0%} of MS2 TIC')
                f, cov = draw(ax, spec, with_prec, CTERM_SERIES_FOR['ead'],
                              title, SEQUENCE,
                              highlight_sites=(PHOSPHO_SITE,)
                              if 'pS129' in species else (),
                              scale_to=scale_to)
                drawn += 1
                if scale_to == 'assay':
                    f.insert(0, 'run', run)
                    f.insert(0, 'panel', label)
                    f.insert(0, 'species', species)
                    rows.append(f)
                    print(f'{species:14s} {label:18s} {run}  '
                          f'{int((f["observed_intensity"] > 0).sum()):3d} ions  '
                          f'coverage {cov:.0%}  precursor {frac:.1%} of TIC  '
                          f'TIC {spec["intensity"].sum():.3g}  {n_window} scans')
            if drawn:
                fig.tight_layout()
                fig.subplots_adjust(hspace=0.45, top=0.94)
                for ext in ('pdf', 'png'):
                    fig.savefig(OUT_DIR / f'{name}_{stem}{suffix}.{ext}',
                                dpi=300, bbox_inches='tight')
                print(f'  wrote {OUT_DIR / f"{name}_{stem}{suffix}.pdf"}'
                      ' (+ .png)')
            plt.close(fig)

    pd.concat(rows, ignore_index=True).to_csv(
        OUT_DIR / f'{name}_spectra_sourcedata.csv', index=False)
    print(f'\nwrote {OUT_DIR / f"{name}_spectra_sourcedata.csv"}')


if __name__ == '__main__':
    main()
