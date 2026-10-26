"""Is the Q1 mass axis drifting, or is the per-slice fit just noisy?"""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, _os.path.join(cfg.REPO_ROOT, 'workflows'))
import zenotof8600_analytics as core
from alphabase.peptide.fragment import (
    create_fragment_mz_dataframe, get_charged_frag_types)

from q1_calibration_limit import complement_dist, NEUTRON

IN_DIR = _os.path.join(cfg.cache('H032_E376', 'pseudo_ms2'), 'q1_patch_dumps')
OUT_DIR = cfg.cache('H032_E376', 'q1_calibration')
MZ_TOL = 0.02
CAL_BINS = 12


def ions_with_identity(seqs, charges):
    """b/y m/z per precursor, each tagged with its complement's mean heavy atom count - the predicted Q1 offset in units of 1.0034/z."""
    prec = pd.DataFrame({'sequence': list(seqs), 'mods': '', 'mod_sites': '',
                         'charge': list(charges)})
    prec['nAA'] = prec.sequence.str.len()
    fdf = create_fragment_mz_dataframe(prec,
                                       get_charged_frag_types(['b', 'y'], 1))
    out = []
    for (a, b), seq, z, n_aa in zip(zip(prec.frag_start_idx,
                                        prec.frag_stop_idx),
                                    prec.sequence, prec.charge, prec.nAA):
        blk = fdf.iloc[int(a):int(b)]
        mz, k = [], []
        for r in range(len(blk)):
            for col, comp, water in (('b_z1', str(seq)[r + 1:], True),
                                     ('y_z1', str(seq)[:r + 1], False)):
                v = float(blk[col].iloc[r])
                if not (150 <= v <= 2000) or not comp:
                    continue
                d = complement_dist(comp, water)
                if d is None:
                    continue
                d = np.asarray(d)
                mz.append(v)
                k.append(float((d * np.arange(len(d))).sum())
                         * NEUTRON / float(z))
        o = np.argsort(mz)
        out.append((np.asarray(mz)[o], np.asarray(k)[o]))
    return out


def affine(mz, err, bins=CAL_BINS):
    """a + b x m/z on BIN MEDIANS - tails would drag a least-squares fit."""
    ok = np.isfinite(mz) & np.isfinite(err)
    if ok.sum() < 40:
        return np.nan, np.nan
    b = core.binned_stats(mz[ok], err[ok], bins=bins)
    x, y = b['x'].to_numpy(float), b['median'].to_numpy(float)
    g = np.isfinite(x) & np.isfinite(y)
    if g.sum() < 3:
        return np.nan, np.nan
    s, i = np.polyfit(x[g], y[g], 1)
    return float(i), float(s)


def load(tag):
    rows = []
    for p in sorted(glob.glob(os.path.join(
            IN_DIR, f'patchdump_{tag}_rt*_peaks.parquet'))):
        stem = p[:-len('_peaks.parquet')]
        rt = float(os.path.basename(stem).split('_rt')[1])
        tab = pd.read_parquet(p)
        ids = pd.read_csv(stem + '_precursors.csv')
        ions = ions_with_identity(ids.sequence.to_numpy(),
                                  ids.z.astype(int).to_numpy())
        rec = []
        for p_, sub in tab.groupby('owner'):
            by = sub[sub.is_by]
            if len(by) < 3 or int(p_) >= len(ions):
                continue
            fm, fk = ions[int(p_)]
            if not len(fm):
                continue
            mm = by.mz.to_numpy()
            j = np.searchsorted(fm, mm)
            j = np.clip(j, 0, len(fm) - 1)
            hit = np.abs(fm[j] - mm) <= MZ_TOL
            j2 = np.clip(j - 1, 0, len(fm) - 1)
            hit2 = np.abs(fm[j2] - mm) <= MZ_TOL
            use = np.where(hit, j, np.where(hit2, j2, -1))
            ok = use >= 0
            if ok.sum() < 3:
                continue
            rec.append(dict(
                rt_slice=rt, owner=int(p_),
                prec_mz=float(sub.prec_mz.iloc[0]),
                seq_len=len(str(ids.sequence.iloc[int(p_)])),
                z=int(ids.z.iloc[int(p_)]),
                dev=float(np.median(by.dev.to_numpy()[ok])),
                iso_off=float(np.mean(fk[use[ok]])),
                n_by=int(ok.sum())))
        rows.append(pd.DataFrame(rec))
        print(f'  RT {rt}: {len(rec)} precursors', flush=True)
    if not rows:
        sys.exit(f'no dumps for {tag} in {IN_DIR}')
    return pd.concat(rows, ignore_index=True)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'A5'
    os.makedirs(OUT_DIR, exist_ok=True)
    core.init_plotting()
    d = load(tag)
    print(f'{len(d):,} precursors over {d.rt_slice.nunique()} slices')

    per = []
    for rt, s in d.groupby('rt_slice'):
        i_, sl = affine(s.prec_mz.to_numpy(), s.dev.to_numpy())
        per.append(dict(rt_slice=rt, intercept=i_, slope=sl,
                        corr500=i_ + sl * 500 if np.isfinite(i_) else np.nan,
                        n=len(s), seq_len=s.seq_len.mean(),
                        iso_off=s.iso_off.mean(), prec_mz=s.prec_mz.mean()))
    per = pd.DataFrame(per)
    gi, gs = affine(d.prec_mz.to_numpy(), d.dev.to_numpy())
    d['res_global'] = d.dev - (gi + gs * d.prec_mz)
    per['res_global'] = per.rt_slice.map(
        d.groupby('rt_slice').res_global.median())
    print(f'\nglobal axis: {gi:+.4f} Th {gs:+.3e} x m/z  '
          f'(corr@500 {gi + gs * 500:+.3f} Th)')
    print(f'  {"RT":>5s} {"n":>4s} {"corr@500":>9s} {"resid":>8s} '
          f'{"len":>5s} {"iso_off":>8s}')
    for _, r in per.iterrows():
        print(f'  {r.rt_slice:5.1f} {int(r.n):4d} {r.corr500:+9.3f} '
              f'{r.res_global:+8.3f} {r.seq_len:5.1f} {r.iso_off:8.3f}')

    sd_per = float(np.nanstd(per.corr500))
    sd_res = float(np.nanstd(per.res_global))
    print(f'\n  SD of corr@500 across slices : {sd_per:.3f} Th')
    print(f'  SD of the median residual    : {sd_res:.3f} Th   <- what a '
          f'GLOBAL axis would leave behind')

    within = []
    for rt, s in d.groupby('rt_slice'):
        if len(s) < 40:
            continue
        x = s.iso_off - s.iso_off.mean()
        y = s.res_global - s.res_global.median()
        b = core.binned_stats(x.to_numpy(float), y.to_numpy(float), bins=7)
        bx, by = b['x'].to_numpy(float), b['median'].to_numpy(float)
        g = np.isfinite(bx) & np.isfinite(by)
        if g.sum() >= 3:
            within.append(np.polyfit(bx[g], by[g], 1)[0])
    print(f'  within-slice slope of residual on isotope offset: '
          f'{np.mean(within):+.2f} (1.0 = the isotope model explains it), '
          f'n = {len(within)} slices')
    r_rt = float(np.corrcoef(per.rt_slice, per.corr500)[0, 1])
    r_iso = float(np.corrcoef(per.iso_off, per.corr500)[0, 1])
    r_len = float(np.corrcoef(per.seq_len, per.corr500)[0, 1])
    print(f'  corr@500 vs RT r = {r_rt:+.2f} · vs mean isotope offset '
          f'r = {r_iso:+.2f} · vs mean peptide length r = {r_len:+.2f}')

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    hue = core.PALETTE_SINGLE[0]
    a = ax[0]
    a.plot(per.rt_slice, per.corr500, color=hue, lw=2.0, marker='o', ms=6,
           markeredgecolor='white', markeredgewidth=0.8,
           label='per-slice fit')
    a.axhline(gi + gs * 500, color='black', lw=1.4, ls='--',
              label='one global axis')
    a.set_xlabel('retention time [min]', fontsize=10)
    a.set_ylabel('predicted Q1 correction at m/z 500 [Th]', fontsize=10)
    a.set_title(f'Does the axis move down the gradient?\n'
                f'r = {r_rt:+.2f}', fontsize=11, fontweight='bold')
    a.legend(fontsize=8.5, frameon=False)

    a = ax[1]
    a.plot(per.rt_slice, per.res_global, color=hue, lw=2.0, marker='o', ms=6,
           markeredgecolor='white', markeredgewidth=0.8)
    a.axhline(0, color='black', lw=1.0, ls=':')
    a.axhspan(-0.28, 0.28, color='gray', alpha=0.12)
    a.text(per.rt_slice.min(), 0.28, ' Q1 tolerance ±0.28 Th', fontsize=8,
           va='bottom')
    a.set_xlabel('retention time [min]', fontsize=10)
    a.set_ylabel('median residual under the global axis [Th]', fontsize=10)
    a.set_title(f'What one global axis would leave\n'
                f'SD {sd_res:.3f} Th', fontsize=11, fontweight='bold')

    a = ax[2]
    a.scatter(per.iso_off, per.corr500, s=60, color=hue,
              edgecolor='white', linewidth=0.8, zorder=5)
    for _, r in per.iterrows():
        a.annotate(f'{r.rt_slice:.0f}', (r.iso_off, r.corr500), fontsize=7.5,
                   xytext=(4, 3), textcoords='offset points')
    a.set_xlabel('mean predicted isotope offset of the slice [Th]',
                 fontsize=10)
    a.set_ylabel('predicted Q1 correction at m/z 500 [Th]', fontsize=10)
    a.set_title(f'Is it composition, not the instrument?\n'
                f'r = {r_iso:+.2f}  (labels = RT)', fontsize=11,
                fontweight='bold')

    for a in ax.ravel():
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    fig.suptitle(f'Stability of the Q1 mass axis across the gradient · {tag}',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    b_ = os.path.join(OUT_DIR, f'q1_axis_stability_{tag}')
    fig.savefig(b_ + '.pdf', bbox_inches='tight')
    fig.savefig(b_ + '.png', dpi=300, bbox_inches='tight')
    per.to_csv(b_ + '_sourcedata.csv', index=False)
    d.to_csv(b_ + '_precursors_sourcedata.csv', index=False)

    X = np.column_stack([np.ones(len(d)), d.prec_mz.to_numpy(float),
                         d.rt_slice.to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, d.dev.to_numpy(float), rcond=None)
    res = d.dev.to_numpy(float) - X @ beta

    mz_v, rt_v = d.prec_mz.to_numpy(float), d.rt_slice.to_numpy(float)
    dev_v = d.dev.to_numpy(float)
    slope_rt = float(beta[2])
    flat = dev_v - slope_rt * rt_v
    n_knot = 20
    edges = np.quantile(mz_v, np.linspace(0, 1, n_knot + 1))
    w = np.clip(np.digitize(mz_v, edges[1:-1]), 0, n_knot - 1)
    knots, vals = [], []
    for k in range(n_knot):
        m = w == k
        if m.sum() >= 20:
            knots.append(float(np.median(mz_v[m])))
            vals.append(float(np.median(flat[m])))
    corr = np.interp(mz_v, knots, vals) + slope_rt * rt_v
    res_free = dev_v - corr

    import json
    cal = dict(kind='table', mz_knots=knots, mz_corr=vals,
               slope_rt=slope_rt,
               affine=dict(intercept=float(beta[0]), slope_mz=float(beta[1]),
                           slope_rt=slope_rt),
               n_precursors=int(len(d)),
               n_slices=int(d.rt_slice.nunique()),
               rt_range=[float(d.rt_slice.min()), float(d.rt_slice.max())],
               residual_sd_th=float(np.std(res_free)),
               affine_residual_sd_th=float(np.std(res)),
               per_slice_residual_sd_th=float(
                   pd.Series(res_free).groupby(d.rt_slice).median().std()),
               note='correction_Th = interp(prec_mz, mz_knots, mz_corr) '
                    '+ slope_rt*rt_min')
    print(f'  shape-free residual SD {np.std(res_free):.4f} Th vs affine '
          f'{np.std(res):.4f} Th ({n_knot} m/z knots)')
    path = os.path.join(OUT_DIR, f'q1_axis_{tag}.json')
    with open(path, 'w') as fh:
        json.dump(cal, fh, indent=2)
    print(f'\ncalibration: {beta[0]:+.4f} {beta[1]:+.4e} x m/z '
          f'{beta[2]:+.4e} x RT   (per-slice residual SD '
          f'{cal["per_slice_residual_sd_th"]:.4f} Th)')
    print(f'wrote {b_}.pdf / .png and {os.path.basename(path)}')


if __name__ == '__main__':
    main()
