"""H032_E376 runs, search outputs and the Q1 patches the targeted Q1 analyses read."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os

import numpy as np
import pandas as pd

ROOT = cfg.data('H032_E376')
RUN = '20260905_Zeno2_Eno0_32p8min_TiHe_SA_H032_E376_'
HDF = cfg.raw('ztscan_full', RUN + 'A9.hdf')
FEAT = os.path.join(ROOT, '40SPD_2p8Th_1p48s', 'lfq.dia.features.csv')
TAG = 'A9'

RT_TOL = 0.06
FRAG_TOL = 0.02

N_PATCHES = 8
Q1_WIDTH = 20.0
N_CYCLES = 11
MIN_PRECURSORS = 4

RT_GRID = np.arange(12.0, 30.1, 1.0)
Q1_GRID = np.arange(400.0, 900.1, Q1_WIDTH)
MIN_RT_SEP = 2.0
MIN_Q1_SEP = 40.0


def identified():
    """Unmodified identified precursors of run TAG, most abundant first."""
    f = pd.read_csv(FEAT, low_memory=False)
    num = lambda c: pd.to_numeric(f[c], errors='coerce')
    d = pd.DataFrame({'seq': f['Peptide'].astype(str), 'z': num('z'),
                      'mz': num(f'{RUN}{TAG} m/z'),
                      'rt': num(f'{RUN}{TAG} RT mean'),
                      'area': num(f'{RUN}{TAG} Normalized Area')}).dropna()
    d = d[d.seq.str.fullmatch(r'[A-Z]+').fillna(False) & (d.area > 0)]
    return d.sort_values('area', ascending=False).drop_duplicates(['seq', 'z'])


def choose_patches(d):
    """Densest RT x Q1 cells, thinned so the set spans the space."""
    cand = []
    for rt in RT_GRID:
        for q0 in Q1_GRID:
            sel = d[(np.abs(d.rt - rt) <= RT_TOL) & (d.mz >= q0)
                    & (d.mz <= q0 + Q1_WIDTH)]
            if len(sel) >= MIN_PRECURSORS:
                cand.append((len(sel), float(rt), float(q0)))
    cand.sort(reverse=True)
    out = []
    for n, rt, q0 in cand:
        if any(abs(rt - r) < MIN_RT_SEP and abs(q0 - q) < MIN_Q1_SEP
               for _, r, q in out):
            continue
        out.append((n, rt, q0))
        if len(out) == N_PATCHES:
            break
    return out
