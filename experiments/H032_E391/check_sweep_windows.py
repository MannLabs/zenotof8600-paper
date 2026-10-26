"""Did any scheduled window clip its peak?"""

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

from score_fragment_coverage import (CTERM_SERIES_FOR, FRAGMENT_TOL_PPM,
                                     FRAG_TYPES, PREFIX, RAW_DIR, targets)

OUT_DIR = pathlib.Path(cfg.cache('H032_E391', 'analysis'))
CACHE_DIR = pathlib.Path(cfg.raw('ms_data_cache'))
RUN = 'A9'

ENERGIES_EV = (0, 7, 12, 15, 17, 20)

CYCLE_GAP_S = 0.15

EDGE_WARN_FRAC = 0.05
MARGIN_WARN_BASE_WIDTHS = 2.0
BASE_WIDTH_PER_FWHM = 1.70
LOCALISE_NUMBER = 12


def split_by_energy(spectra):
    """Offset of each scan within its cycle -> which energy produced it."""
    rt = spectra['rt'].to_numpy()
    gaps = np.diff(rt) * 60
    offset, k = np.zeros(len(rt), int), 0
    for j in range(len(rt)):
        if j == 0:
            k = 0
        elif gaps[j - 1] > CYCLE_GAP_S:
            k = 0
        else:
            k += 1
        offset[j] = k
    return offset


def summed_xic(peaks_mz, peaks_int, spectra, target_mz):
    """Summed matched-fragment intensity per scan."""
    out = np.zeros(len(spectra))
    for n, r in enumerate(spectra.itertuples()):
        v, _ = core.raw.match_peaks(peaks_mz[r.peak_start_idx:r.peak_stop_idx],
                                    peaks_int[r.peak_start_idx:r.peak_stop_idx],
                                    target_mz, tol=FRAGMENT_TOL_PPM,
                                    reduce='max')
        out[n] = float(v.sum())
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
    pmz, pint = ms.peak_df['mz'].to_numpy(), ms.peak_df['intensity'].to_numpy()
    frags = core.raw.target_fragments(tg, frag_types=FRAG_TYPES['ead'],
                                      min_mz=150.0, max_mz=3000.0)

    rows = []
    for i, t in tg.iterrows():
        q1 = float(q1tab.loc[i, 'q1_mz'])
        sub = frags[(frags['target_idx'] == i) & (frags['fragment_charge'] <= 2)]
        tgt = sub['product_mz'].to_numpy()
        loc = sub[(sub['ion_number'] == LOCALISE_NUMBER)
                  & (sub['ion_type'].isin(CTERM_SERIES_FOR['ead']))]
        q = (s[(s['ms_level'] == 2)
               & (np.isclose(s['precursor_mz'], q1, atol=2e-3))]
             .sort_values('rt').reset_index(drop=True))
        if q.empty:
            print(f'{t["species"]}: no scans at Q1 {q1:.4f}')
            continue
        q['offset'] = split_by_energy(q)

        for off, ke in enumerate(ENERGIES_EV):
            sel = q[q['offset'] == off]
            if len(sel) < 5:
                continue
            rt = sel['rt'].to_numpy()
            inten = summed_xic(pmz, pint, sel, tgt)
            pk = core.raw.pick_peak(rt, inten, baseline='offpeak',
                                    interpolate_points=500)
            if pk is None or not np.isfinite(pk.get('apex_rt', np.nan)):
                continue
            apex, top = float(pk['apex_rt']), float(pk['apex_intensity'])
            fwhm_s = float(pk['fwhm']) * 60
            lo, hi = float(rt.min()), float(rt.max())
            base_w = fwhm_s * BASE_WIDTH_PER_FWHM
            edge_lo = float(np.mean(inten[:2])) / max(top, 1)
            edge_hi = float(np.mean(inten[-2:])) / max(top, 1)
            cadence = (hi - lo) * 60 / max(len(rt) - 1, 1)
            loc_int = summed_xic(pmz, pint, sel, loc['product_mz'].to_numpy()).sum()
            rows.append({
                'species': t['species'], 'ke_ev': ke, 'q1': round(q1, 4),
                'n_scans': len(rt), 'window_s': round((hi - lo) * 60, 1),
                'apex_rt': round(apex, 3), 'fwhm_s': round(fwhm_s, 2),
                'points_per_peak': round(base_w / cadence, 1),
                'edge_lo_frac': round(edge_lo, 4),
                'edge_hi_frac': round(edge_hi, 4),
                'margin_lo_bw': round((apex - lo) * 60 / base_w, 2),
                'margin_hi_bw': round((hi - apex) * 60 / base_w, 2),
                'summed_intensity': round(float(inten.sum())),
                'localise_intensity': round(float(loc_int)),
                'clipped': bool(edge_lo > EDGE_WARN_FRAC
                                or edge_hi > EDGE_WARN_FRAC),
                'tight': bool(min((apex - lo) * 60, (hi - apex) * 60) / base_w
                              < MARGIN_WARN_BASE_WIDTHS)})

    d = pd.DataFrame(rows)
    print(f'{RUN}: per species per electron energy\n')
    print(d[['species', 'ke_ev', 'n_scans', 'window_s', 'apex_rt', 'fwhm_s',
             'points_per_peak', 'edge_lo_frac', 'edge_hi_frac',
             'margin_lo_bw']].to_string(index=False))

    print()
    bad, tight = d[d['clipped']], d[d['tight'] & ~d['clipped']]
    if len(bad):
        print(f'CLIPPED - a window edge above {EDGE_WARN_FRAC:.0%} of apex:')
        for r in bad.itertuples():
            print(f'  {r.species} @ {r.ke_ev} eV: '
                  f'{max(r.edge_lo_frac, r.edge_hi_frac):.1%}')
    else:
        print(f'No peak clipped: every window edge is below '
              f'{EDGE_WARN_FRAC:.0%} of its apex.')
    if len(tight):
        print(f'TIGHT - under {MARGIN_WARN_BASE_WIDTHS:g} base widths of margin:')
        for r in tight.itertuples():
            print(f'  {r.species} @ {r.ke_ev} eV: {r.margin_lo_bw:.2f}')
    print(f'\nmedian FWHM {d.fwhm_s.median():.2f} s, '
          f'median points per peak {d.points_per_peak.median():.1f}')

    d.to_csv(OUT_DIR / 'sweep_window_check.csv', index=False)
    print(f'\nwrote {OUT_DIR / "sweep_window_check.csv"}')


if __name__ == '__main__':
    main()
