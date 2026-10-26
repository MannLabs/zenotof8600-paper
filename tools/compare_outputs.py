"""Compare the source data of rebuilt panels with a reference panel set."""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zt8600_config as cfg

RTOL = 1e-9


def same(a, b):
    """'' when equal, otherwise a short description of the first difference."""
    if list(a.columns) != list(b.columns):
        return f'columns differ: {list(a.columns)} vs {list(b.columns)}'
    if len(a) != len(b):
        return f'{len(a)} vs {len(b)} rows'
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            bad = ~(np.isclose(xv, yv, rtol=RTOL, atol=0) | (np.isnan(xv) & np.isnan(yv)))
            if bad.any():
                i = int(np.flatnonzero(bad)[0])
                return f'{c}: row {i} {xv[i]!r} vs {yv[i]!r} ({int(bad.sum())} rows)'
        elif not x.astype(str).equals(y.astype(str)):
            i = int(np.flatnonzero(x.astype(str).to_numpy() != y.astype(str).to_numpy())[0])
            return f'{c}: row {i} {x.iloc[i]!r} vs {y.iloc[i]!r}'
    return ''


def main(reference, figures):
    figures = figures or sorted(
        d for d in os.listdir(reference)
        if os.path.isdir(os.path.join(reference, d, 'panels'))
        and d.startswith(('figure', 'supplementary_figure')))
    n_ok = n_bad = n_missing = 0
    for fig in figures:
        for ref in sorted(glob.glob(os.path.join(reference, fig, 'panels',
                                                 '*_sourcedata.csv'))):
            name = os.path.basename(ref)
            if name.lower().startswith('deleteme'):
                continue
            new = os.path.join(cfg.OUTPUT_ROOT, fig, 'panels', name)
            if not os.path.exists(new):
                n_missing += 1
                print(f'  missing  {fig}/{name}')
                continue
            diff = same(pd.read_csv(new), pd.read_csv(ref))
            if diff:
                n_bad += 1
                print(f'  DIFFERS  {fig}/{name}: {diff}')
            else:
                n_ok += 1
    print(f'{n_ok} identical, {n_bad} different, {n_missing} not rebuilt')
    return n_bad == 0


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1], sys.argv[2:]) else 1)
