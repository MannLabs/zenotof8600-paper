"""Full scoring of the EAD energy sweep, H032_E391 A9."""

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

from annotate_spectra import (MIN_COUNTS, SUM_BIN_PPM, bond_coverage,
                              classify, precursor_species)
from check_sweep_windows import CYCLE_GAP_S, ENERGIES_EV, split_by_energy
from explain_cid_noise import internal_ions
from score_fragment_coverage import (CTERM_SERIES_FOR, FRAGMENT_TOL_PPM,
                                     FRAG_TYPES, PHOSPHO_SITE, PREFIX, RAW_DIR,
                                     SEQUENCE, targets)

OUT_DIR = pathlib.Path(cfg.cache('H032_E391', 'analysis'))
CACHE_DIR = pathlib.Path(cfg.raw('ms_data_cache'))
RUN = 'A9'
LOCALISE_NUMBER = 12


def energy_labelled(ms):
    """Tag every MS2 scan with the electron energy that produced it."""
    s = ms.spectrum_df
    s['energy_ev'] = np.nan
    for q1 in sorted(s.loc[s['ms_level'] == 2, 'precursor_mz'].unique()):
        sel = s[(s['ms_level'] == 2) & (s['precursor_mz'] == q1)].sort_values('rt')
        offs = split_by_energy(sel)
        s.loc[sel.index, 'energy_ev'] = [ENERGIES_EV[o] if o < len(ENERGIES_EV)
                                         else np.nan for o in offs]
    return ms


def summed_for(ms, q1, energy, target_mz):
    """Summed, centroided spectrum of one species at one energy."""
    sub = core.raw.filter_spectra(ms, ms_level=2,
                                  query=f'energy_ev == {energy}')
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
    spec = spec[spec['intensity'] >= MIN_COUNTS].reset_index(drop=True)
    return spec, float(pk['apex_rt']), n_summed, int(rt.size)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    core.raw.register_sil_modifications()
    tg = targets()
    q1tab = core.precursor_q1_mz(tg, isotope='most_abundant')
    ms = energy_labelled(core.raw.load_ms_data(
        str(RAW_DIR / f'{PREFIX}{RUN}.wiff'), cache_dir=str(CACHE_DIR),
        reader_options={'keep_k_peaks_per_spec': 10 ** 7}))
    frags = core.raw.target_fragments(tg, frag_types=FRAG_TYPES['ead'],
                                      min_mz=150.0, max_mz=3000.0)
    internals = np.array([m for _, m, _ in
                          internal_ions(SEQUENCE, PHOSPHO_SITE)])

    rows = []
    for i, t in tg.iterrows():
        q1 = float(q1tab.loc[i, 'q1_mz'])
        sub = frags[(frags['target_idx'] == i) & (frags['fragment_charge'] <= 2)]
        sub = pd.concat([sub, precursor_species((q1 - 1.00727646) * 4)],
                        ignore_index=True)
        tgt = sub['product_mz'].to_numpy()
        for ke in ENERGIES_EV:
            got = summed_for(ms, q1, ke, tgt)
            if got is None:
                continue
            spec, apex, n_summed, n_window = got
            mz, inten = spec['mz'].to_numpy(), spec['intensity'].to_numpy()
            obs, _ = core.raw.match_peaks(mz, inten, tgt,
                                          tol=FRAGMENT_TOL_PPM, reduce='max')
            f = sub.copy()
            f['obs'] = obs
            f['ion_class'] = [('precursor' if r.ion_type == 'M'
                               else classify(r, CTERM_SERIES_FOR['ead']))
                              for r in f.itertuples()]
            hit = f[f['obs'] > 0]
            quant = hit[hit['ion_class'] == 'quant']
            loc = hit[(hit['ion_number'] == LOCALISE_NUMBER)
                      & (hit['ion_type'].isin(CTERM_SERIES_FOR['ead']))]
            claimed = np.zeros(len(mz), bool)
            for m in tgt:
                claimed |= np.abs(mz - m) <= m * FRAGMENT_TOL_PPM * 1e-6
            internal_hit = np.zeros(len(mz), bool)
            for m in internals:
                internal_hit |= np.abs(mz - m) <= m * FRAGMENT_TOL_PPM * 1e-6
            internal_hit &= ~claimed
            total = max(float(inten.sum()), 1.0)
            rows.append({
                'species': t['species'], 'ke_ev': ke, 'apex_rt': round(apex, 3),
                'n_summed': n_summed, 'n_window': n_window, 'n_peaks': len(mz),
                'n_ions': int(len(hit)),
                'sequence_coverage': round(
                    len(bond_coverage(hit, len(SEQUENCE))) / (len(SEQUENCE) - 1), 4),
                'n_quant': int(len(quant)),
                'quant_intensity': round(float(quant['obs'].sum())),
                'n_localise': int(len(loc)),
                'localise_intensity': round(float(loc['obs'].sum())),
                'matched_tic_frac': round(float(hit['obs'].sum()) / total, 4),
                'matched_intensity': round(float(hit['obs'].sum())),
                'precursor_intensity': round(
                    float(hit.loc[hit['ion_class'] == 'precursor', 'obs'].sum())),
                'fragment_intensity': round(
                    float(hit.loc[hit['ion_class'] != 'precursor', 'obs'].sum())),
                'total_intensity': round(total),
                'internal_tic_frac': round(float(inten[internal_hit].sum()) / total, 4),
            })

    d = pd.DataFrame(rows)
    d.to_csv(OUT_DIR / 'sweep_scores_sourcedata.csv', index=False)

    for col, title in (('quant_intensity', 'QUANT-ion intensity'),
                       ('localise_intensity', 'LOCALISATION-ion intensity'),
                       ('n_localise', 'localisation ions detected'),
                       ('sequence_coverage', 'sequence coverage'),
                       ('internal_tic_frac', 'internal fragments, TIC share')):
        p = d.pivot_table(index='species', columns='ke_ev', values=col)
        print(f'\n=== {title} ===')
        print((p.round(3) if p.values.max() <= 1 else p.astype(int)).to_string())

    print('\n=== pooled over all six species ===')
    agg = d.groupby('ke_ev').agg(
        quant=('quant_intensity', 'sum'),
        localisation=('localise_intensity', 'sum'),
        n_localise=('n_localise', 'sum'),
        coverage=('sequence_coverage', 'mean'),
        internal=('internal_tic_frac', 'mean'),
        matched=('matched_tic_frac', 'mean'))
    print(agg.round(3).to_string())

    _figure(d, agg)
    print(f'\nwrote {OUT_DIR / "sweep_scores_sourcedata.csv"}')


def _figure(d, agg):
    palette = core.PALETTE_SINGLE
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    panels = [('quant_intensity', 'quant-ion intensity'),
              ('localise_intensity', 'localisation-ion intensity'),
              ('sequence_coverage', 'sequence coverage'),
              ('internal_tic_frac', 'internal fragments (TIC share)')]
    for ax, (col, title) in zip(axes.ravel(), panels):
        for i, (sp, g) in enumerate(d.groupby('species')):
            g = g.sort_values('ke_ev')
            ax.plot(g['ke_ev'], g[col], 'o-', lw=1.6, ms=4,
                    color=palette[i % len(palette)], label=sp, alpha=0.9)
        ax.set_xlabel('electron KE (eV)', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
    axes[0][0].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(OUT_DIR / f'sweep_scores.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT_DIR / "sweep_scores.pdf"} (+ .png)')


if __name__ == '__main__':
    main()
