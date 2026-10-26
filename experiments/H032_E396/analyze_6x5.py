"""Promega 6x5 reference mix on the 6.4 min gradient - H032_E396."""

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

HERE = pathlib.Path(cfg.data('H032_E396'))
RAW_DIR = pathlib.Path(cfg.raw('H032_E396', 'raw'))
CACHE_DIR = cfg.raw('ms_data_cache')
OUT_DIR = pathlib.Path(cfg.cache('H032_E396', 'analysis'))
PREFIX = '20260921_Zeno2_Eno0_6p4min_TiHe_SA_H032_E396_'
RUNS = ('A1', 'A2', 'A3', 'A4')

NOMINAL = {5: 20.0, 4: 2.0, 3: 0.2, 2: 0.02, 1: 0.002}

Q1_TOL = 0.004

FRAG_TYPES = ('b_z1', 'b_z2', 'y_z1', 'y_z2')

XIC_TOL_PPM = 'auto'

COUNTS_PER_ION = 20.0

REFERENCE_RUN = 'A1'
REFERENCE_LEVEL = 5
N_TRANSITIONS = 6

SELECT_LEVEL = 2

SELECT_MIN_R = 0.60
SELECT_MIN_R_BY_PEPTIDE = {'LLSLGAGEFK': 0.50, 'VVGGLVALR': 0.50}

SELECT_MIN_SN = 1.0
SELECT_MIN_SN_BY_PEPTIDE = {'LLSLGAGEFK': 0.50}

INTEG_HALF_MIN = 4.0 / 60.0
BASELINE_EXCLUDE_MIN = 6.0 / 60.0

MIN_TRACE_R = 0.80

MIN_TRACE_R_BY_PEPTIDE = {'YVYVADVAAK': 0.75}


def min_trace_r(peptide):
    """The acceptance bar for one peptide, or for a column of peptides."""
    if isinstance(peptide, str):
        return MIN_TRACE_R_BY_PEPTIDE.get(peptide, MIN_TRACE_R)
    return peptide.map(lambda p: MIN_TRACE_R_BY_PEPTIDE.get(p, MIN_TRACE_R))

COELUTION_TOL_MIN = 0.02
MIN_COELUTING = 3


def method_table():
    """Q1 and scheduled RT as the acquisition table set them."""
    cols = ['compound_id', 'group', 'q1_mz', 'tof_lo', 'tof_hi', 'accum',
            'dp', 'ce', 'ce_spread', 'rt_min', 'rt_tol_s']
    m = pd.read_csv(HERE / 'mrmhr_promega_6x5.txt', sep='\t', header=None,
                    names=cols)
    t = pd.read_csv(HERE / 'targets' / 'promega_6x5_targets.csv')
    return t.merge(m[['compound_id', 'q1_mz', 'rt_min', 'ce']],
                   on='compound_id')


def _load(run):
    return core.raw.load_ms_data(
        str(RAW_DIR / f'{PREFIX}{run}.wiff'), cache_dir=CACHE_DIR,
        reader_options={'keep_k_peaks_per_spec': 10 ** 7})


def assert_one_target_per_selection(ms, tg):
    """No scan selection may span two compounds - checked, not assumed."""
    m2 = ms.spectrum_df[ms.spectrum_df['ms_level'] == 2]
    bad = {}
    for r in tg.itertuples():
        hit = m2.loc[(m2['precursor_mz'] - r.q1_mz).abs() <= Q1_TOL,
                     'precursor_mz'].unique()
        if len(hit) != 1:
            bad[r.compound_id] = [round(float(h), 5) for h in hit]
    if bad:
        raise SystemExit(
            f'Q1_TOL={Q1_TOL} does not isolate these targets: {bad}\n'
            'Tighten it below half the closest spacing.')


def _xic(ms, q1, mz):
    x = core.raw.targeted_xic(ms, q1, mz, tol=Q1_TOL, mz_tol=XIC_TOL_PPM)
    return x['rt'].to_numpy(), x['intensity'].to_numpy() / COUNTS_PER_ION


def _integrate(rt, y, apex):
    """Ions over the peak, and the 1-sigma an empty channel would show."""
    off = np.abs(rt - apex) > BASELINE_EXCLUDE_MIN
    base = float(np.mean(y[off])) if off.sum() >= 5 else 0.0
    sd = float(np.std(y[off])) if off.sum() >= 5 else np.nan
    peak = np.abs(rt - apex) <= INTEG_HALF_MIN
    n = float(peak.sum())
    return float(np.sum(y[peak] - base)), sd * np.sqrt(n), n


def choose_transitions(tg, frags):
    """Per peptide: the brightest ions that are also CLEAN at level 2."""
    ms = _load(REFERENCE_RUN)
    picked = {}
    for pep, grp in tg.groupby('sequence'):
        ref = grp[grp['level'] == REFERENCE_LEVEL].iloc[0]
        lo = grp[grp['level'] == SELECT_LEVEL].iloc[0]
        rsub = frags[frags['target_idx'] == ref.name]
        lsub = frags[frags['target_idx'] == lo.name]

        tot, rt = None, None
        for r in rsub.itertuples():
            t, y = _xic(ms, float(ref['q1_mz']), float(r.product_mz))
            tot = y if tot is None else tot + y
            rt = t
        apex = float(rt[int(np.argmax(tot))])

        cand = []
        for r in rsub.itertuples():
            key = (r.ion_type, int(r.ion_number), int(r.fragment_charge))
            lr = lsub[(lsub['ion_type'] == key[0])
                      & (lsub['ion_number'] == key[1])
                      & (lsub['fragment_charge'] == key[2])]
            if lr.empty:
                continue
            rt5, y5 = _xic(ms, float(ref['q1_mz']), float(r.product_mz))
            rt2, y2 = _xic(ms, float(lo['q1_mz']),
                           float(lr['product_mz'].iloc[0]))
            m = np.abs(rt2 - apex) <= INTEG_HALF_MIN
            if m.sum() <= 3 or np.std(y2[m]) == 0:
                continue
            tmpl = np.interp(rt2[m], rt5, y5)
            if np.std(tmpl) == 0:
                continue
            corr = float(np.corrcoef(y2[m], tmpl)[0, 1])
            ions, sig, _n = _integrate(rt2, y2, apex)
            cand.append({'key': key, 'r': corr, 'bright': float(y5.sum()),
                         'ions': ions,
                         'sn': ions / (3 * sig) if sig else np.nan})
        c = pd.DataFrame(cand)
        bar = SELECT_MIN_R_BY_PEPTIDE.get(pep, SELECT_MIN_R)
        snbar = SELECT_MIN_SN_BY_PEPTIDE.get(pep, SELECT_MIN_SN)
        clean = c[(c['r'] >= bar) & (c['ions'] > 0) & (c['sn'] >= snbar)]
        if len(clean):
            sel = clean.nlargest(N_TRANSITIONS, 'bright')
            how = f'{len(clean)} clean at level {SELECT_LEVEL}, r>={bar:g}'
        else:
            sel = c.nlargest(N_TRANSITIONS, 'bright')
            how = (f'FALLBACK: brightest, nothing clean at level '
                   f'{SELECT_LEVEL}')
        picked[pep] = [tuple(k) for k in sel['key']]
        print(f'  {pep:12s} ' + ', '.join(
            f'{t}{n}{"++" if z == 2 else "+"}' for t, n, z in picked[pep])
            + f'   [{how}]')
    return picked


def accepted_levels(pep, g):
    """{level: bool} -- which levels of one peptide enter its curve."""
    return g.groupby('level')['trace_r'].mean() >= min_trace_r(pep)


def fit_table(d):
    """One calibration row per peptide, from a per-level table like main's."""
    fit_rows = []
    for pep, g in d.groupby('peptide'):
        ok = accepted_levels(pep, g)
        used = g[g['level'].isin(set(ok[ok].index)) & (g['ions'] > 0)]
        if used['level'].nunique() < 3:
            fit_rows.append({'peptide': pep,
                             'n_levels': used['level'].nunique()})
            continue
        fit = core.raw.calibration_fit(
            used['amol'].to_numpy(), used['ions'].to_numpy(),
            decoy_area=used['decoy_area'].to_numpy())
        lv = fit['levels']
        fit_rows.append({
            'peptide': pep, 'n_levels': fit['n_levels'],
            'levels_amol': ','.join(f'{c:g}'
                                    for c in sorted(lv['concentration'])),
            'slope': fit['slope'], 'r2': fit['r_squared'],
            'LOD_amol': fit['lod'], 'LOQ_amol': fit['loq'],
            'LOD_decoy_amol': fit['lod_decoy'],
            'accuracy_min_%': lv['accuracy_pct'].min(),
            'accuracy_max_%': lv['accuracy_pct'].max()})
    f = pd.DataFrame(fit_rows)
    return f


def main(ms_level=2):
    """Measure, fit and write the 6x5 tables."""
    core.raw.register_sil_modifications()
    tg = method_table()
    if ms_level == 1:
        selected = core.raw.precursor_isotope_transitions(tg, n_isotopes=3)
        print(f'MS1: precursor isotopes M0-M2 of {len(tg)} targets')
    else:
        selected = _ms2_transitions(tg)

    runs = {r: _load(r) for r in RUNS}
    if ms_level == 2:
        d = core.raw.quantify_targets(
            runs, selected, reference_by='sequence',
            fragment_filter=None, match='precursor', match_tol=Q1_TOL,
            ms_level=2, verbose=False)
    else:
        d = _quantify_ms1(runs, selected, tg)
    _report_and_write(d, tg)


def _ms2_window(ms, q1_values):
    """(rt_lo, rt_hi): where this run actually acquired MS2 for these Q1s."""
    sd = ms.spectrum_df
    m2 = sd[sd['ms_level'] == 2]
    pm = m2['precursor_mz'].to_numpy()
    hit = np.zeros(len(m2), bool)
    for q in q1_values:
        hit |= np.abs(pm - q) <= Q1_TOL
    rt = m2['rt'].to_numpy()[hit]
    return float(rt.min()), float(rt.max())


def _quantify_ms1(runs, selected, tg):
    """MS1 survey, one peptide at a time, inside that peptide's MS2 window."""
    parts = []
    for pep, sub in selected.groupby('sequence'):
        q1s = tg.loc[tg['sequence'] == pep, 'q1_mz'].to_numpy()
        for run, ms in runs.items():
            win = _ms2_window(ms, q1s)
            parts.append(core.raw.quantify_targets(
                {run: ms}, sub, rt_range=win, reference_by='sequence',
                reference_selector={'level': REFERENCE_LEVEL},
                ms_level=1, verbose=False))
    return pd.concat(parts, ignore_index=True)


def _ms2_transitions(tg):
    """The fixed fragment transitions, chosen once on the reference run."""
    frags = core.raw.target_fragments(tg, frag_types=FRAG_TYPES,
                                      min_mz=150.0, max_mz=3000.0)
    print('transitions fixed on '
          f'{REFERENCE_RUN}, level {REFERENCE_LEVEL}:')
    assert_one_target_per_selection(_load(REFERENCE_RUN), tg)
    picked = choose_transitions(tg, frags)
    sel_idx = []
    for pep, ions in picked.items():
        for idx in tg.index[tg['sequence'] == pep]:
            sub = frags[frags['target_idx'] == idx]
            for it, num, z in ions:
                m = sub[(sub['ion_type'] == it) & (sub['ion_number'] == num)
                        & (sub['fragment_charge'] == z)]
                sel_idx.extend(m.index)
    return frags.loc[sorted(sel_idx)]


def _report_and_write(d, tg):
    d = d.merge(tg[['compound_id', 'level']].assign(target_idx=tg.index),
                on='target_idx')
    d = d.rename(columns={'sequence': 'peptide', 'area': 'ions'})
    d['amol'] = d['level'].map(NOMINAL) * 1000.0
    d['over_limit'] = d['signal_to_decoy'] > 3.0
    d['measured'] = (d['over_limit']
                     & (d['trace_r'] >= min_trace_r(d['peptide'])))

    OUT_DIR.mkdir(exist_ok=True)
    d.to_csv(OUT_DIR / '6x5_levels_sourcedata.csv', index=False)
    pd.set_option('display.width', 240)

    print('\n=== peak area per level, mean of the four injections ===')
    m = d.pivot_table(index='peptide', columns='level', values='ions',
                      aggfunc='mean')
    print(m.round(0).to_string())

    print('\n=== over 3x the measured decoy background, runs out of 4 ===')
    print(d.pivot_table(index='peptide', columns='level', values='over_limit',
                        aggfunc='sum').astype(int).to_string())

    print(f'\n=== MEASURED per run: over the 3 sd limit AND trace r >= '
          f'{MIN_TRACE_R} vs level {REFERENCE_LEVEL}, of {len(RUNS)} runs '
          f'({MIN_TRACE_R_BY_PEPTIDE or "no"} overrides) ===')
    print(f'    (a level enters the CURVE on its MEAN r, below - one dip in '
          f'one of {len(RUNS)}\n     injections is scatter on a weak '
          f'channel, not evidence of background)')
    print(d.pivot_table(index='peptide', columns='level', values='measured',
                        aggfunc='sum').astype(int).to_string())
    print('  trace correlation with level 5 (mean over runs):')
    print(d.pivot_table(index='peptide', columns='level',
                        values='trace_r').round(3).to_string())
    print('  apex offset from the anchor, seconds (mean over runs):')
    print(d.pivot_table(index='peptide', columns='level',
                        values='rt_delta_s').round(2).to_string())

    print('\n=== CV across the four injections, % ===')
    cv = (d.groupby(['peptide', 'level'])['ions']
           .agg(lambda x: 100 * x.std(ddof=1) / x.mean()).unstack('level'))
    print(cv.round(1).to_string())

    print('\n=== step between consecutive levels (fold) ===')
    step = m.div(m.shift(axis=1).values)
    print(step.round(2).to_string())

    print('\n=== calibration, from core.raw.calibration_fit ===')
    f = fit_table(d)
    print(f.to_string(index=False, float_format=lambda v: f'{v:,.4g}'))
    f.to_csv(OUT_DIR / '6x5_calibration.csv', index=False)
    print('  LOD/LOQ are 3.3 and 10 x the intercept SE over the slope; '
          'LOD_decoy is\n  the level whose fitted area equals 3x the '
          'measured decoy background.')

    _plot(d)
    print(f'\nwrote {OUT_DIR / "6x5_levels_sourcedata.csv"}')


def _plot(d):
    core.init_plotting()
    peptides = sorted(d['peptide'].unique())
    colour = dict(zip(peptides, core.PALETTE_SINGLE))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    src = []
    n_runs = d['run'].nunique()
    for pep in peptides:
        g = d[d['peptide'] == pep]
        m = g.groupby('level')['ions'].mean()
        lim = g.groupby('level')['decoy_area'].mean()
        ok = accepted_levels(pep, g)
        cv = g.groupby('level')['ions'].agg(
            lambda x: 100 * x.std(ddof=1) / x.mean())
        axes[0].plot(m.index[ok], m[ok].to_numpy(), 'o-', lw=1.6, ms=5,
                     color=colour[pep], label=pep)
        axes[0].plot(m.index[~ok], m[~ok].clip(lower=1).to_numpy(), 'o',
                     ms=5, mfc='none', color=colour[pep])
        axes[0].plot(lim.index, lim.to_numpy(), ':', lw=1.0,
                     color=colour[pep])
        axes[1].plot(cv.index[ok], cv[ok].to_numpy(), 'o-', lw=1.6, ms=5,
                     color=colour[pep], label=pep)
        src += [{'peptide': pep, 'level': k, 'mean_ions': m[k],
                 'decoy_area': lim[k], 'cv_percent': cv[k],
                 'measured': bool(ok[k])} for k in m.index]

    axes[0].set_yscale('log')
    axes[0].set_xlabel('isotopologue level', fontsize=10)
    axes[0].set_ylabel('peak area (mean of 4 injections)', fontsize=10)
    axes[0].set_title('response curve', fontsize=11, fontweight='bold')
    axes[0].set_xticks([1, 2, 3, 4, 5])
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].axhline(20, color='0.6', lw=0.8, ls='--')
    axes[1].set_xlabel('isotopologue level', fontsize=10)
    axes[1].set_ylabel('CV across injections [%]', fontsize=10)
    axes[1].set_title('injection precision', fontsize=11, fontweight='bold')
    axes[1].set_xticks([1, 2, 3, 4, 5])
    for ax in axes:
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    fig.suptitle('Promega 6x5 mix, 6.4 min gradient | dotted = measured '
                 'decoy background', fontsize=11, fontweight='bold')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(OUT_DIR / f'6x5_response.{ext}', dpi=300,
                    bbox_inches='tight')
    pd.DataFrame(src).to_csv(OUT_DIR / '6x5_response_sourcedata.csv',
                             index=False)
    plt.close(fig)
    print(f'  wrote {OUT_DIR / "6x5_response.pdf"}')


if __name__ == '__main__':
    main()
