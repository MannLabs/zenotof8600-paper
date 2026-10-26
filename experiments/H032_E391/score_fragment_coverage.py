"""Do CID and EAD cover the fragments the alpha-synuclein pS129 assay needs?"""

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

RAW_DIR = pathlib.Path(cfg.raw('H032_E391', 'raw'))
CACHE_DIR = pathlib.Path(cfg.raw('ms_data_cache'))
OUT_DIR = pathlib.Path(cfg.cache('H032_E391', 'analysis'))
PREFIX = '20260919_Zeno2_Eno0_6p4min_TiHe_SA_H032_E391_'

EAD_NCE, CID_NCE = 12.0, (50.0, 51.0, 53.0, 54.0)

SEQUENCE = 'NEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'
CHARGE = 4
PHOSPHO_SITE, LABEL_SITE = 27, 38

MET_SITES = (25, 14)
LABEL = 'Label:13C(3)15N(1)@A'

FRAGMENT_TOL_PPM = 10.0

FRAG_TYPES = {
    'cid': ['a_z1', 'a_z2', 'b_z1', 'b_z2', 'y_z1', 'y_z2'],
    'ead': core.raw.FRAGMENT_TYPES['ead_extended'] + ['z_lossH_z1',
                                                      'z_lossH_z2'],
}

CTERM_SERIES_FOR = {'cid': ('y',),
                    'ead': ('y', 'z', 'z_addH', 'z_lossH')}

MIN_COUNTS = 30
MAX_APEX_DELTA_MIN = 0.02
MIN_SN = 3.0

QUANT_IONS = (6, 8, 9, 10, 11)
LOCALISE_PAIR = (11, 12)

READER_OPTIONS = {'keep_k_peaks_per_spec': 10 ** 7}


def targets():
    rows = []
    for phospho in (False, True):
        for n_ox in (0, 1, 2):
            mods = [(s, 'Oxidation@M') for s in MET_SITES[:n_ox]]
            if phospho:
                mods.append((PHOSPHO_SITE, 'Phospho@S'))
            mods.append((LABEL_SITE, LABEL))
            mods.sort()
            rows.append({
                'species': ('heavy ' + ('pS129' if phospho else 'unmod')
                            + {0: '', 1: ' +16', 2: ' +32'}[n_ox]),
                'sequence': SEQUENCE,
                'mods': ';'.join(m for _, m in mods),
                'mod_sites': ';'.join(str(s) for s, _ in mods),
                'charge': CHARGE})
    return pd.DataFrame(rows)


def activation_of(ms):
    """Read CID / EAD off the file's NCE column."""
    nce = ms.spectrum_df.loc[ms.spectrum_df['ms_level'] == 2, 'nce']
    vals = sorted(nce.dropna().unique())
    if any(abs(v - EAD_NCE) < 0.5 for v in vals):
        return 'ead', vals
    if any(any(abs(v - c) < 0.5 for c in CID_NCE) for v in vals):
        return 'cid', vals
    raise SystemExit(f'cannot tell activation from NCE values {vals}')


def load(run):
    return core.raw.load_ms_data(str(RAW_DIR / f'{PREFIX}{run}.wiff'),
                                 cache_dir=str(CACHE_DIR),
                                 reader_options=READER_OPTIONS)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    core.raw.register_sil_modifications()
    tg = targets()

    rows = []
    for run in ('A1', 'A2'):
        ms = load(run)
        act, nce_vals = activation_of(ms)
        ms2 = core.raw.filter_spectra(ms, ms_level=2)
        frags = core.raw.target_fragments(
            tg, frag_types=FRAG_TYPES[act], min_mz=150.0, max_mz=3000.0)
        print(f'\n=== {run}: {act.upper()} (NCE {nce_vals}) ===')
        for i, t in tg.iterrows():
            sub = frags[frags['target_idx'] == i]
            q1 = float(sub['precursor_mz'].iloc[0])
            win = core.raw.filter_spectra(ms2, query=None)
            prec_xic = core.raw.extract_xic(
                win, list(sub['product_mz'].astype(float)),
                mz_tol=FRAGMENT_TOL_PPM, ms_level=2, precursor_mz=q1)
            pk = core.raw.pick_peak(prec_xic['rt'].to_numpy(),
                                    prec_xic['intensity'].to_numpy(),
                                    baseline='offpeak', interpolate_points=500)
            if pk is None or not np.isfinite(pk.get('apex_rt', np.nan)):
                print(f'  {t["species"]:18s} no peak')
                continue
            apex = float(pk['apex_rt'])
            for f in sub.itertuples():
                x = core.raw.extract_xic(win, [float(f.product_mz)],
                                         mz_tol=FRAGMENT_TOL_PPM, ms_level=2,
                                         precursor_mz=q1)
                fp = core.raw.pick_peak(x['rt'].to_numpy(),
                                        x['intensity'].to_numpy(),
                                        baseline='offpeak',
                                        interpolate_points=500, min_sn=MIN_SN)
                ok = (fp is not None and np.isfinite(fp.get('apex_rt', np.nan))
                      and abs(float(fp['apex_rt']) - apex) <= MAX_APEX_DELTA_MIN)
                rows.append({
                    'run': run, 'activation': act, 'species': t['species'],
                    'precursor_mz': round(q1, 4), 'precursor_apex_rt': round(apex, 3),
                    'ion_type': f.ion_type, 'ion_number': int(f.ion_number),
                    'fragment_charge': int(f.fragment_charge),
                    'product_mz': round(float(f.product_mz), 4),
                    'detected': bool(ok),
                    'apex_rt': None if fp is None else round(float(fp['apex_rt']), 3),
                    'apex_intensity': 0.0 if not ok else float(fp['apex_intensity']),
                    'sn': None if fp is None else round(float(fp['sn']), 1)})
            det = [r for r in rows if r['run'] == run
                   and r['species'] == t['species'] and r['detected']]
            print(f'  {t["species"]:18s} apex {apex:.3f} min, '
                  f'{len(det)}/{len(sub)} fragments co-eluting')

    src = pd.DataFrame(rows)
    src.to_csv(OUT_DIR / 'fragment_coverage_sourcedata.csv', index=False)
    _report(src)
    print(f'\nwrote {OUT_DIR / "fragment_coverage_sourcedata.csv"}')


def _report(src):
    cterm = {'cid': 'y', 'ead': 'z'}
    print('\n\n=== QUANT ions, singly charged, detected? ===')
    for act in ('cid', 'ead'):
        s = cterm[act]
        d = src[(src['activation'] == act) & (src['ion_type'] == s)
                & (src['fragment_charge'] == 1)]
        if d.empty:
            continue
        print(f'\n  {act.upper()}  ({s} ions)')
        hdr = '  '.join(f'{s}{n:<5d}' for n in QUANT_IONS)
        print(f'    {"species":18s} {hdr}')
        for sp, g in d.groupby('species', sort=False):
            cells = []
            for n in QUANT_IONS:
                r = g[g['ion_number'] == n]
                cells.append('  --   ' if r.empty or not r['detected'].iloc[0]
                             else f'{r["apex_intensity"].iloc[0]:7.1e}')
            print(f'    {sp:18s} ' + ' '.join(cells))

    print('\n=== LOCALISATION pair (n11 unmodified + n12 at +80) ===')
    lo, hi = LOCALISE_PAIR
    for act in ('cid', 'ead'):
        s = cterm[act]
        d = src[(src['activation'] == act) & (src['ion_type'] == s)
                & (src['ion_number'].isin(LOCALISE_PAIR))]
        if d.empty:
            continue
        print(f'\n  {act.upper()}')
        for sp, g in d.groupby('species', sort=False):
            if 'pS129' not in sp:
                continue
            bits = []
            for n in LOCALISE_PAIR:
                r = g[(g['ion_number'] == n)].sort_values('fragment_charge')
                for rr in r.itertuples():
                    mark = 'yes' if rr.detected else 'NO '
                    bits.append(f'{s}{n}^{rr.fragment_charge}+ {mark} '
                                f'({rr.apex_intensity:.1e})')
            print(f'    {sp:18s} ' + '  '.join(bits))


if __name__ == '__main__':
    main()
