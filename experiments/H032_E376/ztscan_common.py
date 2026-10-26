"""Primitives shared by the targeted ZT Scan Q1 analyses."""

from __future__ import annotations

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
from functools import lru_cache

import h5py
import hdf5plugin
import numpy as np
import pandas as pd

from alphabase.constants.atom import parse_formula
from alphabase.constants.isotope import formula_dist
from alphabase.peptide.fragment import (
    create_fragment_mz_dataframe, get_charged_frag_types)


OUT_DIR = cfg.cache('H032_E376', 'pseudo_ms2')


FRAG_MZ_RANGE = (150.0, 2000.0)


def by_ions(sequences, charges, with_charge=False):
    """b/y m/z per sequence: 1+ always, 2+ as well for 2+ and higher precursors."""
    prec = pd.DataFrame({'sequence': list(sequences), 'mods': '',
                         'mod_sites': '', 'charge': list(charges)})
    prec['nAA'] = prec.sequence.str.len()
    fdf = create_fragment_mz_dataframe(prec,
                                       get_charged_frag_types(['b', 'y'], 2))
    out, sites, zs = [], [], []
    for (a, b), z, n_aa in zip(zip(prec.frag_start_idx, prec.frag_stop_idx),
                               prec.charge, prec.nAA):
        cols = ['b_z1', 'y_z1'] + (['b_z2', 'y_z2'] if int(z) >= 2 else [])
        blk = fdf.iloc[int(a):int(b)][cols].to_numpy()
        site = np.repeat(np.arange(1, len(blk) + 1), blk.shape[1])
        zcol = np.tile([int(c.split('_z')[1]) for c in cols], len(blk))
        v = blk.ravel()
        keep = (v >= FRAG_MZ_RANGE[0]) & (v <= FRAG_MZ_RANGE[1])
        v, site, zcol = v[keep], site[keep], zcol[keep]
        o = np.argsort(v)
        v, site, zcol = v[o], site[o], zcol[o]
        uniq = np.concatenate(([True], np.diff(v) > 1e-6))
        out.append(v[uniq])
        sites.append(site[uniq])
        zs.append(zcol[uniq])
    return (out, sites, zs) if with_charge else (out, sites)


def load_axis(tag):
    """The measured Q1 calibration, if one has been fitted for this file."""
    import json
    p = os.path.join(os.path.dirname(OUT_DIR), 'q1_calibration',
                     f'q1_axis_{tag}.json')
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        c = json.load(fh)
    if c.get('kind') == 'table':
        extra = []
        if c.get('n_precursors'):
            extra.append(f'{c["n_precursors"]:,} precursors')
        if c.get('n_slices'):
            extra.append(f'{c["n_slices"]} slices')
        if c.get('residual_sd_th') is not None:
            extra.append(f'residual SD {c["residual_sd_th"]:.4f} Th')
        if c.get('source'):
            extra.append(str(c['source']))
        print(f'  Q1 axis from {os.path.basename(p)}: shape-free table over '
              f'{len(c["mz_knots"])} m/z knots {c["slope_rt"]:+.3e} x RT'
              + (f' ({", ".join(extra)})' if extra else ''), flush=True)
        return c
    print(f'  Q1 axis from {os.path.basename(p)}: {c["intercept"]:+.4f} '
          f'{c["slope_mz"]:+.3e} x m/z {c["slope_rt"]:+.3e} x RT',
          flush=True)
    return (c['intercept'], c['slope_mz'], c['slope_rt'])


def q1_correction(axis, mz, rt):
    """The Q1 mass-axis correction, for either form of calibration."""
    mz = np.asarray(mz, float)
    rt = np.asarray(rt, float)
    if isinstance(axis, dict):
        return (np.interp(mz, axis['mz_knots'], axis['mz_corr'])
                + axis['slope_rt'] * rt)
    a, b = axis[0], axis[1]
    c = axis[2] if len(axis) > 2 else 0.0
    return a + b * mz + c * rt


def load_ms1(hdf_path, rt_lo, rt_hi):
    """Every MS1 (TOFMS) scan of the slice, read once and held (~25 MB)."""
    with h5py.File(hdf_path, 'r') as h:
        sg = h['ms_data']['spectrum_df']
        ml, rt = sg['ms_level'][:], sg['rt'][:]
        st = sg['peak_start_idx'][:].astype(np.int64)
        sp = sg['peak_stop_idx'][:].astype(np.int64)
        k = np.where((ml == 1) & (rt >= rt_lo - 0.1) & (rt <= rt_hi + 0.1))[0]
        mz_ds = h['ms_data']['peak_df']['mz']
        in_ds = h['ms_data']['peak_df']['Intensity']
        scans = []
        for i in k:
            a, b = int(st[i]), int(sp[i])
            m, v = mz_ds[a:b], in_ds[a:b]
            o = np.argsort(m, kind='stable')
            scans.append((m[o].astype(float), v[o].astype(float)))
    return scans, rt[k]


PROTON = 1.00727646


N_ISO_FIT = 4


AVERAGINE = dict(C=4.9384, H=7.7583, N=1.3577, O=1.4773, S=0.0417)
AVERAGINE_MASS = 111.1254
ISO_MASS_BIN = 25.0


@lru_cache(maxsize=8192)
def iso_pattern_cached(mass_bin, n):
    """Expected PEPTIDE isotope envelope at this mass, from averagine."""
    nres = max(float(mass_bin), AVERAGINE_MASS) / AVERAGINE_MASS
    comp = {e: int(round(v * nres)) for e, v in AVERAGINE.items()}
    f = ''.join(f'{e}({v})' for e, v in comp.items() if v > 0)
    d = np.asarray(formula_dist(parse_formula(f))[0], float)[:n]
    s = d.sum()
    return tuple(d / s) if s > 0 else tuple(d)


def iso_pattern(mass, n=N_ISO_FIT):
    return np.asarray(iso_pattern_cached(
        round(max(float(mass), 1.0) / ISO_MASS_BIN) * ISO_MASS_BIN, n))
