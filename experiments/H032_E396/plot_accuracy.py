"""Accuracy and precision of the 6x5 curve, all peptides."""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import zenotof8600_analytics as core

import analyze_6x5 as a
from plot_calibration_curve import AMOL_PER_FMOL, SUB, UNIT

ACCEPTANCE_PCT = 20.0


def main():
    sub = SUB
    d = pd.read_csv(a.HERE / sub / '6x5_levels_sourcedata.csv')
    d['amount'] = d['level'].map(a.NOMINAL) * AMOL_PER_FMOL

    rows = []
    for pep, g in d.groupby('peptide'):
        keep = a.accepted_levels(pep, g)
        used_levels = set(keep[keep].index)
        fit_pts = g[g['level'].isin(used_levels) & (g['ions'] > 0)]
        if fit_pts['level'].nunique() < 3:
            continue
        cal = core.raw.calibration_fit(
            fit_pts['amount'].to_numpy(), fit_pts['ions'].to_numpy(),
            decoy_area=(fit_pts['decoy_area'].to_numpy()
                        if 'decoy_area' in fit_pts else None))
        slope, intercept = cal['slope'], cal['intercept']
        for r in g.itertuples():
            back = ((r.ions - intercept) / slope
                    if r.ions > 0 else np.nan)
            rows.append({'peptide': pep, 'run': r.run, 'level': r.level,
                         'amol': r.amount, 'ions': r.ions,
                         'back_amol': back,
                         'accuracy_pct': back / r.amount * 100
                                         if back == back
                                         else np.nan,
                         'used': r.level in used_levels})
    s = pd.DataFrame(rows)
    out_dir = a.HERE / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    s.to_csv(out_dir / '6x5_accuracy_sourcedata.csv', index=False)

    pd.set_option('display.width', 200)
    ok = s[s['used']]
    print('=== back-calculated accuracy, % of nominal ===')
    print(ok.pivot_table(index='peptide', columns='amol',
                         values='accuracy_pct').round(1).to_string())
    print(f'\n  all {len(ok)} measurements within '
          f'{ok["accuracy_pct"].min():.0f}-{ok["accuracy_pct"].max():.0f}% '
          f'of nominal; '
          f'{(ok["accuracy_pct"].sub(100).abs() <= ACCEPTANCE_PCT).mean()*100:.0f}% '
          f'inside +-{ACCEPTANCE_PCT:g}%')
    print(f'\n=== precision, CV across the {d['run'].nunique()} injections ===')
    cv = (ok.groupby(['peptide', 'amol'])['ions']
            .agg(lambda x: 100 * x.std(ddof=1) / x.mean()).unstack('amol'))
    print(cv.round(1).to_string())

    core.init_plotting()
    peptides = sorted(s['peptide'].unique())
    colour = dict(zip(peptides, core.PALETTE_SINGLE))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for y in (100 - ACCEPTANCE_PCT, 100 + ACCEPTANCE_PCT):
        axes[0].axhline(y, color='0.6', lw=0.8, ls='--', zorder=1)
    axes[0].axhline(100, color='0.4', lw=0.8, zorder=1)
    n_rejected = int(s.loc[~s['used'] & s['accuracy_pct'].notna(),
                           ['peptide', 'level']].drop_duplicates().shape[0])
    for pep in peptides:
        g = s[(s['peptide'] == pep) & s['used']]
        m = g.groupby('amol')['accuracy_pct'].mean()
        axes[0].plot(m.index, m.to_numpy(), 'o-', lw=1.5, ms=5,
                     color=colour[pep], label=pep, zorder=4)
        cvp = (g.groupby('amol')['ions']
                .agg(lambda x: 100 * x.std(ddof=1) / x.mean()))
        axes[1].plot(cvp.index, cvp.to_numpy(), 'o-', lw=1.5, ms=5,
                     color=colour[pep], label=pep)

    axes[0].set_xscale('log')
    used_acc = s.loc[s['used'], 'accuracy_pct'].dropna()
    lo = min(100 - ACCEPTANCE_PCT - 5, used_acc.min() - 5)
    hi = max(100 + ACCEPTANCE_PCT + 5, used_acc.max() + 5)
    axes[0].set_ylim(lo, hi)
    axes[0].set_ylabel('back-calculated / nominal [%]', fontsize=10)
    axes[0].set_title('accuracy', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=8, frameon=False, loc='upper left')
    axes[1].set_xscale('log')
    axes[1].axhline(20, color='0.6', lw=0.8, ls='--')
    axes[1].set_ylabel('CV across injections [%]', fontsize=10)
    axes[1].set_title('precision', fontsize=11, fontweight='bold')
    for ax in axes:
        ax.set_xlabel(f'on column [{UNIT}]', fontsize=10)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    fig.suptitle(f'Promega 6x5 | dashed = +-{ACCEPTANCE_PCT:g}% accuracy / {ACCEPTANCE_PCT:g}% CV | '
                 f'levels that are the peptide only '
                 f'({n_rejected} levels rejected, not drawn)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'6x5_accuracy.{ext}', dpi=300,
                    bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {out_dir / "6x5_accuracy.pdf"}')


if __name__ == '__main__':
    main()
