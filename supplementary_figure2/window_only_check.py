"""Check S2c's "Q1 window only" reference against real precursor m/z."""

import os
import sys

import numpy as np
import pandas as pd

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'figure2'))
import figure2 as f2

N_PHASES = 50


def window_only(mz, w, n_phases=N_PHASES):
    out = []
    for ph in np.arange(n_phases) / n_phases * w:
        centre = np.floor((mz - ph) / w) * w + ph + w / 2
        out.append(core.sigma_iqr(mz - centre))
    return np.asarray(out)


def main():
    rows = []
    for tag, w in f2.Q1_WHOLEFILE_TAGS.items():
        mz = f2.wholefile_placements(tag)['mz'].to_numpy()
        s = window_only(mz, w)
        rows.append({'width_th': w, 'n': len(mz),
                     'analytical': w / 2 / 1.349,
                     'empirical_median': float(np.median(s)),
                     'empirical_min': float(s.min()),
                     'empirical_max': float(s.max())})
    t = pd.DataFrame(rows)
    t['ratio'] = t['empirical_median'] / t['analytical']
    print(t.round(4).to_string(index=False))
    return t


if __name__ == '__main__':
    main()
