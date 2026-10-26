"""Calibration curve for one peptide of the 6x5 standard."""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import zenotof8600_analytics as core

import analyze_6x5 as a

DEFAULT_PEPTIDE = 'LGFTDLFSK'

AMOL_PER_FMOL = 1000.0
UNIT = 'amol'

SUB = 'analysis'


LOD_K, LOQ_K = 3.3, 10.0

def main(peptide=DEFAULT_PEPTIDE):
    sub = SUB
    src = a.HERE / sub / '6x5_levels_sourcedata.csv'
    d = pd.read_csv(src)
    d = d[(d['peptide'] == peptide) & (d['kind'] != 'diagnostic')] \
        if 'kind' in d.columns else d[d['peptide'] == peptide]
    if not len(d):
        raise SystemExit(f'{peptide} not in {src}')

    d['amount'] = d['level'].map(a.NOMINAL) * AMOL_PER_FMOL
    per = d.groupby('level').agg(
        amount=('amount', 'first'), ions=('ions', 'mean'),
        sd=('ions', 'std'), lim=('decoy_area', 'mean'),
        r=('trace_r', 'mean'), n=('ions', 'size')).reset_index()
    bar = a.min_trace_r(peptide)
    used = per[per['r'] >= bar].copy()
    if len(used) < 3:
        raise SystemExit(f'{peptide}: only {len(used)} usable levels')

    fit_pts = d[d['level'].isin(used['level']) & (d['ions'] > 0)]
    cal = core.raw.calibration_fit(fit_pts['amount'].to_numpy(),
                                   fit_pts['ions'].to_numpy(),
                                   decoy_area=fit_pts['decoy_area'].to_numpy()
                                   if 'decoy_area' in fit_pts else None)
    lod, loq = cal['lod'], cal['loq']
    lin_slope, lin_intercept = cal['slope'], cal['intercept']
    sigma_blank, lod_decoy = cal['sigma_blank'], cal['lod_decoy']
    r2 = cal['r_squared']

    def to_amount(ions):
        return (ions - lin_intercept) / lin_slope

    per['back_amol'] = np.nan
    pos = per['ions'] > 0
    per.loc[pos, 'back_amol'] = to_amount(per.loc[pos, 'ions'])
    per['accuracy_%'] = per['back_amol'] / per['amount'] * 100
    per['cv_%'] = per['sd'] / per['ions'] * 100
    per['used'] = per['r'] >= bar
    lowest_validated = float(used['amount'].min())

    pd.set_option('display.width', 200)
    print(f'=== {peptide}, n={int(per["n"].max())} injections ===')
    print(per[['level', 'amount', 'ions', 'cv_%', 'back_amol', 'accuracy_%',
               'r', 'used']].to_string(
        index=False, float_format=lambda v: f'{v:,.3g}'))
    print(f'\n  fit over {len(used)} levels, 1/x^2 weighted, on LINEAR values:')
    print(f'    area = {lin_slope:.4g} x {UNIT} + {lin_intercept:.4g}'
          f',   r2 = {r2:.4f}')
    print(f'  from that curve (ICH, over {len(fit_pts)} injections):')
    print(f'    LOD {lod:.3g} {UNIT} ({LOD_K:g} sigma/S),  '
          f'LOQ {loq:.3g} {UNIT} ({LOQ_K:g} sigma/S)')
    print(f'    sigma(blank) {sigma_blank:.3g} (intercept SE), '
          f'S {lin_slope:.3g} area/{UNIT}')
    if lod_decoy == lod_decoy:
        print(f'    LOD from the MEASURED decoy background: '
              f'{lod_decoy:.3g} {UNIT}')
    print(f'  lowest level that actually validated: {lowest_validated:g} '
          f'{UNIT}')
    print('    <- the working limit.  The two diverge where INTERFERENCE, '
          'not noise, stops the curve.')
    acc = per.loc[per['used'], 'accuracy_%']
    print(f'  back-calculated accuracy over the used levels: '
          f'{acc.min():.0f}-{acc.max():.0f}% of nominal')

    core.init_plotting()
    out_dir = a.HERE / sub / peptide
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    col = core.PALETTE_SINGLE[0]

    grid = np.logspace(np.log10(per['amount'].min() / 2),
                       np.log10(per['amount'].max() * 2), 100)
    pred = lin_slope * grid + lin_intercept
    vis = pred > 0
    ax.plot(grid[vis], pred[vis], '-', color='0.35', lw=1.2, zorder=1)
    ok = per[per['used']]
    bad = per[~per['used']]
    for _, r in ok.iterrows():
        pts = d.loc[d['level'] == r['level'], 'ions']
        ax.scatter([r['amount']] * len(pts), pts, s=18, color='black',
                   alpha=0.75, linewidth=0.3, edgecolor='white', zorder=4)
    ax.scatter(ok['amount'], ok['ions'], s=55, color=col, edgecolor='black',
               linewidth=0.8, zorder=3, label='used in the fit')
    if len(bad):
        ax.scatter(bad['amount'], bad['ions'].clip(lower=1), s=55,
                   facecolor='none', edgecolor=col, linewidth=1.0, zorder=3,
                   label='rejected (not the peptide)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(min(per['amount'].min(), lod) / 3, per['amount'].max() * 3)
    lo = max(per['ions'][per['ions'] > 0].min() / 3, 1.0)
    ax.set_ylim(lo, per['ions'].max() * 3)
    ax.axvline(lod, color='0.6', lw=0.8, ls=':')
    ax.axvline(loq, color='0.6', lw=0.8, ls='--')
    ax.annotate(f'LOQ {loq:.2g} {UNIT}', xy=(loq, lo), xytext=(3, 3),
                textcoords='offset points', fontsize=8, rotation=90,
                va='bottom', ha='left', color='0.35')
    ax.annotate(f'LOD {lod:.2g} {UNIT}', xy=(lod, lo), xytext=(3, 3),
                textcoords='offset points', fontsize=8, rotation=90,
                va='bottom', ha='left', color='0.35')
    ax.set_xlabel(f'on column [{UNIT}]', fontsize=10)
    n_inj = int(per['n'].max())
    ax.set_ylabel(f'peak area (mean of {n_inj} injections)', fontsize=10)
    ax.set_title(f'{peptide}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    lo, hi = used['amount'].min(), used['amount'].max()
    txt = (f'$r^2$ {r2:.4f}  (1/$x^2$ weighted)\n'
           f'{len(used)} levels, {lo:g}-{hi:g} {UNIT}')
    ax.text(0.97, 0.03, txt, transform=ax.transAxes, fontsize=8,
            ha='right', va='bottom')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    name = f'6x5_calibration_{peptide}'
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)

    srcd = d[['run', 'level', 'amount', 'ions', 'decoy_area',
              'trace_r']].copy()
    srcd = srcd.rename(columns={'amount': 'amol'})
    srcd.insert(0, 'peptide', peptide)
    srcd.to_csv(out_dir / f'{name}_sourcedata.csv', index=False)
    print(f'\nwrote {out_dir / (name + ".pdf")}')


def run_all():
    """Every peptide, each into its own folder."""
    sub = SUB
    d = pd.read_csv(a.HERE / sub / '6x5_levels_sourcedata.csv')
    done, skipped = [], []
    for pep in sorted(d['peptide'].unique()):
        try:
            main(pep)
            done.append(pep)
        except SystemExit as exc:
            skipped.append(f'{pep} ({exc})')
        print()
    print(f'=== {len(done)} curves written: ' + ', '.join(done))
    if skipped:
        print('    skipped: ' + '; '.join(skipped))


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args or args[0] == 'all':
        run_all()
    else:
        main(args[0])
