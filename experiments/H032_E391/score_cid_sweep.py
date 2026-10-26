"""The CID collision-energy sweep, H032_E391 A10."""

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

from annotate_spectra import (MAX_FRAGMENT_CHARGE, MIN_COUNTS, SUM_BIN_PPM,
                              precursor_species)
from score_fragment_coverage import (FRAGMENT_TOL_PPM, FRAG_TYPES, PREFIX,
                                     RAW_DIR, targets)

OUT_DIR = pathlib.Path(cfg.cache('H032_E391', 'analysis'))
CACHE_DIR = pathlib.Path(cfg.raw('ms_data_cache'))
RUN = 'A10'

QUANT_SET = (('y', 6), ('y', 8), ('y', 9), ('y', 10), ('y', 11))
LOCALISE_PAIR = (('y', 11), ('y', 13))

CE_OFFSETS_V = (-9, -6, -3, 0, +3)

USABLE_COUNTS = 100


def summed_at(ms, q1, nce, target_mz):
    """Summed, centroided spectrum for one species at one collision energy."""
    sub = core.raw.filter_spectra(ms, ms_level=2, nce=nce)
    xic = core.raw.extract_xic(sub, list(target_mz), mz_tol=FRAGMENT_TOL_PPM,
                               ms_level=2, precursor_mz=q1)
    rt, inten = xic['rt'].to_numpy(), xic['intensity'].to_numpy()
    if rt.size < 5 or inten.max() <= 0:
        return None
    pk = core.raw.pick_peak(rt, inten, baseline='offpeak',
                            interpolate_points=500)
    if pk is None or not np.isfinite(pk.get('apex_rt', np.nan)):
        return None
    rt_range = (float(pk['left_rt']), float(pk['right_rt']))
    n_summed = int(((rt >= rt_range[0]) & (rt <= rt_range[1])).sum())
    summed = core.raw.sum_spectra(sub, rt_range=rt_range, ms_level=2,
                                  precursor_mz=q1, bin_tol=SUM_BIN_PPM)
    cen = core.raw.centroid_profile(summed['mz'].to_numpy(),
                                    summed['intensity'].to_numpy())
    spec = pd.DataFrame({'mz': cen[0], 'intensity': cen[1]})
    return spec[spec['intensity'] >= MIN_COUNTS].reset_index(drop=True), n_summed


def intensities(spec, frags, wanted):
    out = {}
    for ion_type, number in wanted:
        r = frags[(frags['ion_type'] == ion_type)
                  & (frags['ion_number'] == number)
                  & (frags['fragment_charge'] == 1)]
        if r.empty:
            out[f'{ion_type}{number}'] = 0.0
            continue
        v, _ = core.raw.match_peaks(spec['mz'].to_numpy(),
                                    spec['intensity'].to_numpy(),
                                    r['product_mz'].to_numpy(),
                                    tol=FRAGMENT_TOL_PPM, reduce='max')
        out[f'{ion_type}{number}'] = float(v[0])
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    core.raw.register_sil_modifications()
    tg = targets()
    q1tab = core.precursor_q1_mz(tg, isotope='most_abundant')
    ms = core.raw.load_ms_data(str(RAW_DIR / f'{PREFIX}{RUN}.wiff'),
                               cache_dir=str(CACHE_DIR),
                               reader_options={'keep_k_peaks_per_spec': 10 ** 7})
    s = ms.spectrum_df
    frags = core.raw.target_fragments(tg, frag_types=FRAG_TYPES['cid'],
                                      min_mz=150.0, max_mz=3000.0)

    rows = []
    for i, t in tg.iterrows():
        q1 = float(q1tab.loc[i, 'q1_mz'])
        sub = frags[(frags['target_idx'] == i)
                    & (frags['fragment_charge'] <= MAX_FRAGMENT_CHARGE)]
        allmz = pd.concat(
            [sub, precursor_species((q1 - 1.00727646) * 4)], ignore_index=True)
        nces = sorted(s.loc[(s['ms_level'] == 2)
                            & (np.isclose(s['precursor_mz'], q1, atol=2e-3)),
                            'nce'].dropna().unique())
        if len(nces) != len(CE_OFFSETS_V):
            print(f'{t["species"]}: {len(nces)} energies found {nces} - '
                  f'expected {len(CE_OFFSETS_V)}')
        for nce, off in zip(nces, CE_OFFSETS_V):
            got = summed_at(ms, q1, float(nce), allmz['product_mz'].to_numpy())
            if got is None:
                continue
            spec, n_summed = got
            q = intensities(spec, sub, QUANT_SET)
            l = intensities(spec, sub, LOCALISE_PAIR)
            qv = np.array(list(q.values()))
            rec = {'species': t['species'], 'ce_v': float(nce),
                   'offset_v': off, 'n_summed': n_summed,
                   'quant_sum': round(float(qv.sum())),
                   'quant_weakest': round(float(qv.min())),
                   'quant_detected': int((qv > 0).sum()),
                   'quant_usable': int((qv >= USABLE_COUNTS).sum()),
                   'y11': round(l['y11']), 'y13': round(l['y13']),
                   'localised': bool(l['y11'] > 0 and l['y13'] > 0)}
            rec.update({k: round(v) for k, v in q.items()})
            rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_csv(OUT_DIR / 'cid_sweep_scores_sourcedata.csv', index=False)

    names = [f'{a}{b}' for a, b in QUANT_SET]
    print(f'CID quant set: {", ".join(names)}   localisation: y11 + y13\n')
    for col, title in (
            ('quant_detected', f'quant transitions detected (of {len(QUANT_SET)})'),
            ('quant_usable', f'quant transitions >= {USABLE_COUNTS} counts'),
            ('quant_weakest', 'weakest quant transition'),
            ('quant_sum', 'summed quant intensity'),
            ('y13', 'y13 (localisation) intensity')):
        p = d.pivot_table(index='species', columns='offset_v', values=col)
        print(f'=== {title} ===')
        print(p.astype(int).to_string())
        print()

    print('=== pooled ===')
    agg = d.groupby('offset_v').agg(
        detected=('quant_detected', 'sum'), usable=('quant_usable', 'sum'),
        weakest_median=('quant_weakest', 'median'),
        quant_sum=('quant_sum', 'sum'),
        localised=('localised', 'sum'), y13=('y13', 'sum'),
        scans=('n_summed', 'mean'))
    print(agg.round(1).to_string())
    print(f'\nbest by usable       : {int(agg["usable"].idxmax()):+d} V')
    print(f'best by localisation : {int(agg["localised"].idxmax()):+d} V')

    _figure(d)
    print(f'\nwrote {OUT_DIR / "cid_sweep_scores_sourcedata.csv"}')


def _figure(d):
    palette = core.PALETTE_SINGLE
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    panels = [('quant_usable', f'quant transitions >= {USABLE_COUNTS} counts'),
              ('quant_weakest', 'weakest quant transition'),
              ('y13', 'y13 (localisation) intensity'),
              ('quant_sum', 'summed quant intensity')]
    for ax, (col, title) in zip(axes.ravel(), panels):
        for i, (sp, g) in enumerate(d.groupby('species')):
            g = g.sort_values('offset_v')
            ax.plot(g['offset_v'], g[col], 'o-', lw=1.6, ms=4,
                    color=palette[i % len(palette)], label=sp, alpha=0.9)
        ax.set_xlabel('collision energy offset (V)', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
    axes[0][0].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(OUT_DIR / f'cid_sweep_scores.{ext}', dpi=300,
                    bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT_DIR / "cid_sweep_scores.pdf"} (+ .png)')


if __name__ == '__main__':
    main()
