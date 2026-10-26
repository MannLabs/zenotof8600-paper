"""CID at two collision energies against EAD 15 eV, same species, same peak."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

import zenotof8600_analytics as core

from annotate_spectra import draw, precursor_species, MAX_FRAGMENT_CHARGE
from score_cid_sweep import summed_at
from score_sweep import energy_labelled, summed_for
from score_fragment_coverage import (CTERM_SERIES_FOR, FRAG_TYPES, PHOSPHO_SITE,
                                     PREFIX, RAW_DIR, SEQUENCE, targets)

OUT_DIR = pathlib.Path(cfg.cache('H032_E391', 'analysis', 'spectra_cid_vs_ead'))
CACHE_DIR = pathlib.Path(cfg.raw('ms_data_cache'))

PANELS = (('CID -9 V', 'A10', 'cid', -9),
          ('CID 0 V (as run)', 'A10', 'cid', 0),
          ('EAD 15 eV', 'A9', 'ead', 15))

CE_OFFSETS_V = (-9, -6, -3, 0, +3)
SPECIES = ('heavy pS129', 'heavy unmod', 'heavy pS129 +16')


def load_raw(run):
    return core.raw.load_ms_data(
        str(RAW_DIR / f'{PREFIX}{run}.wiff'), cache_dir=str(CACHE_DIR),
        reader_options={'keep_k_peaks_per_spec': 10 ** 7})


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    core.raw.register_sil_modifications()
    tg = targets()
    q1tab = core.precursor_q1_mz(tg, isotope='most_abundant')
    runs = {'A10': load_raw('A10'), 'A9': energy_labelled(load_raw('A9'))}

    rows = []
    for species in SPECIES:
        i = tg.index[tg['species'] == species][0]
        q1 = float(q1tab.loc[i, 'q1_mz'])
        fig, axes = plt.subplots(len(PANELS), 1,
                                 figsize=(8, 5.5 * len(PANELS)))
        drawn = 0
        for ax, (label, run, act, sel) in zip(axes, PANELS):
            ms = runs[run]
            frags = core.raw.target_fragments(tg, frag_types=FRAG_TYPES[act],
                                              min_mz=150.0, max_mz=3000.0)
            sub = frags[(frags['target_idx'] == i)
                        & (frags['fragment_charge'] <= MAX_FRAGMENT_CHARGE)]
            with_prec = pd.concat(
                [sub, precursor_species((q1 - 1.00727646) * 4)],
                ignore_index=True)
            tgt = with_prec['product_mz'].to_numpy()
            if act == 'cid':
                s = ms.spectrum_df
                nces = sorted(s.loc[(s['ms_level'] == 2)
                                    & (abs(s['precursor_mz'] - q1) < 2e-3),
                                    'nce'].dropna().unique())
                nce = float(nces[CE_OFFSETS_V.index(sel)])
                got = summed_at(ms, q1, nce, tgt)
                got = None if got is None else (got[0], got[1])
                detail = f'{nce:.0f} V'
            else:
                g = summed_for(ms, q1, sel, tgt)
                got = None if g is None else (g[0], g[2])
                detail = f'{sel} eV'
            if got is None:
                ax.axis('off')
                continue
            spec, n_summed = got
            title = (f'{species} | {label} ({detail}) | Q1 {q1:.3f} | '
                     f'{n_summed} spectra summed')
            f, cov = draw(ax, spec, with_prec, CTERM_SERIES_FOR[act], title,
                          SEQUENCE,
                          highlight_sites=(PHOSPHO_SITE,)
                          if 'pS129' in species else ())
            f.insert(0, 'panel', label)
            f.insert(0, 'species', species)
            f.insert(0, 'sequence_coverage', round(cov, 4))
            rows.append(f)
            drawn += 1
            print(f'{species:18s} {label:18s} {n_summed:2d} scans  '
                  f'{int((f["observed_intensity"] > 0).sum()):3d} ions  '
                  f'coverage {cov:.0%}')
        if drawn:
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.45, top=0.94)
            stem = species.replace(' ', '_').replace('+', 'ox')
            for ext in ('pdf', 'png'):
                fig.savefig(OUT_DIR / f'cid_vs_ead_{stem}.{ext}', dpi=300,
                            bbox_inches='tight')
            print(f'  wrote {OUT_DIR / f"cid_vs_ead_{stem}.pdf"} (+ .png)')
        plt.close(fig)

    pd.concat(rows, ignore_index=True).to_csv(
        OUT_DIR / 'cid_vs_ead_sourcedata.csv', index=False)
    print(f'\nwrote {OUT_DIR / "cid_vs_ead_sourcedata.csv"}')


if __name__ == '__main__':
    main()
