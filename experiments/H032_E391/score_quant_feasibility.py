"""Is the best HEK setting good enough to call a relative change?"""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg
import pathlib

import numpy as np
import pandas as pd

import zenotof8600_analytics as core

from annotate_spectra import (MAX_FRAGMENT_CHARGE, precursor_species,
                              summed_spectrum)
from annotate_devmode import PANELS
from annotate_background import REFERENCE_RUN, RT_BRACKET_MIN, SPECIES
from score_fragment_coverage import (FRAGMENT_TOL_PPM, FRAG_TYPES, PREFIX,
                                     RAW_DIR, targets)

OUT_DIR = pathlib.Path(cfg.cache('H032_E391', 'analysis'))
CACHE_DIR = pathlib.Path(cfg.raw('ms_data_cache'))

QUANT_SET = (('z', 6), ('z', 5), ('z', 7), ('y', 9), ('z', 11))
LOCALISE = (('y', 11), ('y', 13))

NEIGHBOURHOOD_TH = 5.0

COUNTS_PER_ION = 20.0

MIN_COUNTS_USABLE = 200
MIN_SB_USABLE = 3.0

FOLD_CHANGES = (1.2, 1.5, 2.0)
N_REPLICATES = 4


def cv_with_background(signal, background):
    """Counting-limited CV of a background-subtracted signal."""
    if signal <= 0:
        return np.nan
    s, b = signal / COUNTS_PER_ION, background / COUNTS_PER_ION
    return np.sqrt(s + 2 * b) / s


def main():
    core.raw.register_sil_modifications()
    tg = targets()
    q1tab = core.precursor_q1_mz(tg, isotope='most_abundant')
    frags = core.raw.target_fragments(tg, frag_types=FRAG_TYPES['ead'],
                                      min_mz=150.0, max_mz=3000.0)
    runs = {r: core.raw.load_ms_data(
        str(RAW_DIR / f'{PREFIX}{r}.wiff'), cache_dir=str(CACHE_DIR),
        reader_options={'keep_k_peaks_per_spec': 10 ** 7})
        for _, r in PANELS}

    rows = []
    for species in SPECIES:
        i = tg.index[tg['species'] == species][0]
        q1 = float(q1tab.loc[i, 'q1_mz'])
        sub = frags[(frags['target_idx'] == i)
                    & (frags['fragment_charge'] <= MAX_FRAGMENT_CHARGE)]
        prec = precursor_species((q1 - 1.00727646) * 4)
        allf = pd.concat([sub, prec], ignore_index=True)
        bracket, width = None, None
        for label, run in PANELS:
            got = summed_spectrum(runs[run], q1,
                                  sub['product_mz'].astype(float),
                                  rt_bracket=bracket, rt_width=width)
            if got is None:
                continue
            spec, rng, apex, n_scans = got
            if run == REFERENCE_RUN:
                bracket = (apex - RT_BRACKET_MIN, apex + RT_BRACKET_MIN)
                width = rng[1] - rng[0]
            mz, inten = spec['mz'].to_numpy(), spec['intensity'].to_numpy()
            assigned = np.zeros(len(mz), bool)
            for m in allf['product_mz']:
                assigned |= np.abs(mz - m) <= m * FRAGMENT_TOL_PPM * 1e-6
            un_mz, un_int = mz[~assigned], inten[~assigned]
            for kind, wanted in (('quant', QUANT_SET),
                                 ('localise', LOCALISE)):
                for ion_type, number in wanted:
                    r = sub[(sub['ion_type'] == ion_type)
                            & (sub['ion_number'] == number)
                            & (sub['fragment_charge'] == 1)]
                    if r.empty:
                        continue
                    th = float(r['product_mz'].iloc[0])
                    v, _ = core.raw.match_peaks(mz, inten, np.array([th]),
                                                tol=FRAGMENT_TOL_PPM,
                                                reduce='max')
                    s = float(v[0])
                    near = np.abs(un_mz - th) <= NEIGHBOURHOOD_TH
                    b = float(np.median(un_int[near])) if near.any() else 0.0
                    rows.append({
                        'species': species, 'panel': label, 'run': run,
                        'kind': kind, 'ion': f'{ion_type}{number}',
                        'mz': round(th, 4), 'n_scans': n_scans,
                        'counts': round(s), 'background': round(b),
                        'signal_over_background':
                            round(s / b, 2) if b else np.inf,
                        'ions': round(s / COUNTS_PER_ION, 1),
                        'cv_poisson': round(np.sqrt(COUNTS_PER_ION / s), 4)
                        if s > 0 else np.nan,
                        'cv_with_background': round(cv_with_background(s, b), 4),
                        'usable': bool(s >= MIN_COUNTS_USABLE
                                       and (b == 0 or s / b >= MIN_SB_USABLE)),
                    })
    d = pd.DataFrame(rows)
    d.to_csv(OUT_DIR / 'quant_feasibility_sourcedata.csv', index=False)

    pd.set_option('display.width', 220)
    for species in SPECIES:
        s = d[d['species'] == species]
        print(f'\n=== {species}: designated transitions ===')
        print(s.pivot_table(index=['kind', 'ion'], columns='run',
                            values=['counts', 'signal_over_background'],
                            sort=False).round(1).to_string())
        print('\n  usable transitions (>= %d counts and S/B >= %.0f):'
              % (MIN_COUNTS_USABLE, MIN_SB_USABLE))
        print('   ' + s.groupby('run')['usable'].sum().to_string()
              .replace('\n', '\n   '))

    print('\n=== combined precision of the quant set, per run ===')
    print('(transitions summed; the counting floor on that total, then with '
          'the background term)')
    for species in SPECIES:
        for run in [r for _, r in PANELS]:
            g = d[(d['species'] == species) & (d['run'] == run)
                  & (d['kind'] == 'quant')]
            if g.empty:
                continue
            S, B = g['counts'].sum(), g['background'].sum()
            cv = cv_with_background(S, B)
            mdfc = _min_fold_change(cv)
            print(f'  {species:14s} {run:4s}  S={S:7.0f}  B={B:6.0f}  '
                  f'S/B={S / B if B else np.inf:5.1f}  '
                  f'ions={S / COUNTS_PER_ION:6.0f}  '
                  f'CV_count={np.sqrt(COUNTS_PER_ION / S):6.2%}  CV_+bg={cv:6.2%}  '
                  f'-> smallest resolvable fold change at n={N_REPLICATES}: '
                  f'{mdfc:.2f}x (counting only)')
    print(f'\nwrote {OUT_DIR / "quant_feasibility_sourcedata.csv"}')


def _min_fold_change(cv, n=N_REPLICATES, z=2.8):
    """Fold change a two-sample comparison resolves at this CV."""
    return float(np.exp(z * cv * np.sqrt(2 / n)))


if __name__ == '__main__':
    main()
