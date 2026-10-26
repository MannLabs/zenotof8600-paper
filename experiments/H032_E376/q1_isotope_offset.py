"""The precursor-isotope offset of a fragment's Q1 centroid, and what it costs."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, _os.path.join(cfg.REPO_ROOT, 'workflows'))
sys.path.insert(0, _os.path.join(cfg.REPO_ROOT, 'experiments', 'H032_E376'))
from ztscan_common import PROTON, iso_pattern

HERE = _os.path.join(cfg.REPO_ROOT, 'experiments', 'H032_E376')
NEUTRON = 1.00335
N_ISO = 6
H2O = 18.010565

AA = {
    'G': 57.02146, 'A': 71.03711, 'S': 87.03203, 'P': 97.05276,
    'V': 99.06841, 'T': 101.04768, 'C': 103.00919, 'L': 113.08406,
    'I': 113.08406, 'N': 114.04293, 'D': 115.02694, 'Q': 128.05858,
    'K': 128.09496, 'E': 129.04259, 'M': 131.04049, 'H': 137.05891,
    'F': 147.06841, 'R': 156.10111, 'Y': 163.06333, 'W': 186.07931,
}


def residue_mass(seq):
    return sum(AA.get(c, 0.0) for c in seq)


def complement_mass(seq, kind, num):
    """Neutral mass of the fragment on the OTHER side of the same cleavage."""
    L = len(seq)
    n = int(num)
    if n <= 0 or n >= L:
        return np.nan
    if kind == 'b':
        return residue_mass(seq[n:]) + H2O
    return residue_mass(seq[:L - n])


def offset_from_complement(comp_mass, z_prec):
    w = np.asarray(iso_pattern(float(comp_mass), N_ISO), float)
    if w.sum() <= 0:
        return 0.0
    k = np.arange(len(w))
    return float((k * w).sum() / w.sum() * NEUTRON / int(z_prec))


def offset_from_mz(frag_mz, frag_z, prec_mz, prec_z):
    """The same offset, from m/z alone - and the form to prefer."""
    pm = float(prec_mz) * int(prec_z) - int(prec_z) * PROTON
    fm = float(frag_mz) * int(frag_z) - int(frag_z) * PROTON
    cm = pm - fm
    if not np.isfinite(cm) or cm <= 0:
        return 0.0
    return offset_from_complement(cm, prec_z)


def add_offset(d):
    """Add an ``iso_offset`` column to a survey fragment table."""
    cm = np.array([complement_mass(s, k, n) for s, k, n
                   in zip(d.sequence.astype(str), d.frag_kind.astype(str),
                          d.frag_num.to_numpy())])
    key = list(zip(np.round(np.nan_to_num(cm), 0).astype(int),
                   d.z.astype(int)))
    cache, out = {}, np.zeros(len(d))
    for i, (m, z) in enumerate(key):
        if (m, z) not in cache:
            if m <= 0:
                cache[(m, z)] = 0.0
            else:
                w = np.asarray(iso_pattern(float(m), N_ISO), float)
                kk = np.arange(len(w))
                cache[(m, z)] = (float((kk * w).sum() / w.sum() * NEUTRON / z)
                                 if w.sum() > 0 else 0.0)
        out[i] = cache[(m, z)]
    d = d.copy()
    d['complement_mass'] = cm
    d['iso_offset'] = out
    return d


def sig(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.nan
    return float((np.percentile(x, 75) - np.percentile(x, 25)) / 1.349)


CONDITIONS = [('A5', 1.0), ('A9', 2.8), ('B1', 4.3), ('B5', 5.7), ('B9', 7.2)]
STEM = 'capped_q1'
MIN_FRAG = 4


def main():
    print('Q1 centroid error against known precursors, before and after '
          'removing the\npredicted precursor-isotope offset. sigma_IQR in Th, '
          'per-condition offset removed.\n')
    print(f'{"":>16s} {"precursors":>11s} {"within-peptide":>15s} '
          f'{"across-precursor":>17s}')
    print(f'{"condition":>16s} {"":>11s} {"raw":>7s} {"corr":>7s} '
          f'{"raw":>8s} {"corr":>8s}')
    for tag, nom in CONDITIONS:
        p = os.path.join(HERE, f'{STEM}_{tag}_fragments.csv')
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        if 'rt_slice' not in d.columns:
            d['rt_slice'] = 0.0
        d = add_offset(d)
        d['cen_corr'] = d.q1_centroid - d.iso_offset
        d['_fin'] = np.isfinite(d.q1_centroid)
        w = d[d._fin].groupby(['rt_slice', 'precursor'])
        within_raw = w.q1_centroid.apply(sig).median()
        within_cor = w.cen_corr.apply(sig).median()
        g = d.groupby(['rt_slice', 'precursor']).agg(
            theo=('prec_theo_mz', 'first'), n=('_fin', 'sum'),
            cen=('q1_centroid', 'median'),
            cen_corr=('cen_corr', 'median')).reset_index()
        g = g[(g.n >= MIN_FRAG) & np.isfinite(g.cen)]
        e_raw = g.cen - g.theo
        e_cor = g.cen_corr - g.theo
        print(f'{tag + f" ({nom:g} Th)":>16s} {len(g):11,d} '
              f'{within_raw:7.3f} {within_cor:7.3f} '
              f'{sig(e_raw - e_raw.median()):8.3f} '
              f'{sig(e_cor - e_cor.median()):8.3f}')
    print('\n  "within-peptide" is the median over precursors of the scatter '
          'of their own\n  fragments\' centroids - the quantity the '
          'correction is aimed at. "across-\n  precursor" is the scatter of '
          'the per-precursor medians, which is what decides\n  whether two '
          'co-isolated peptides land apart.')
    by_mass()


def by_mass(tag='A9'):
    """Where the correction should and should not pay, by precursor mass."""
    p = os.path.join(HERE, f'{STEM}_{tag}_fragments.csv')
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    if 'rt_slice' not in d.columns:
        d['rt_slice'] = 0.0
    d = add_offset(d)
    d['cen_corr'] = d.q1_centroid - d.iso_offset
    d['_fin'] = np.isfinite(d.q1_centroid)
    g = d.groupby(['rt_slice', 'precursor']).agg(
        theo=('prec_theo_mz', 'first'), zc=('z', 'first'), n=('_fin', 'sum'),
        cen=('q1_centroid', 'median'),
        cen_corr=('cen_corr', 'median')).reset_index()
    g = g[(g.n >= MIN_FRAG) & np.isfinite(g.cen)].copy()
    g['mass'] = g.theo * g.zc - g.zc * 1.00728
    g['bin'] = pd.qcut(g.mass, 4, labels=False)
    print(f'\n\n{tag}: the same correction by PRECURSOR MASS quartile\n')
    print(f'{"mass range (Da)":>18s} {"n":>6s} {"raw":>8s} {"corr":>8s} '
          f'{"change":>8s}')
    for b, h in g.groupby('bin'):
        e_raw = h.cen - h.theo
        e_cor = h.cen_corr - h.theo
        a = sig(e_raw - e_raw.median())
        c = sig(e_cor - e_cor.median())
        print(f'{f"{h.mass.min():.0f}-{h.mass.max():.0f}":>18s} '
              f'{len(h):6,d} {a:8.3f} {c:8.3f} {(c - a) / a:+8.1%}')
    print('\n  The prediction is that the gain grows with precursor mass. '
          'A flat or\n  reversed pattern would mean the correction is '
          'removing something else.')


if __name__ == '__main__':
    main()
