"""Choosing WHICH precursors to measure, without biasing the answer."""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 0
MZ_BINS = 5
ABUNDANCE_BINS = 5


def stratified_sample(df, n, *, mz_col='mz', intensity_col='intensity',
                      mz_bins=MZ_BINS, abundance_bins=ABUNDANCE_BINS,
                      top_fraction=1.0, seed=RNG_SEED):
    """Even coverage of the m/z x abundance plane.

    Parameters
      df:             one row per candidate.
      n:              how many rows to return. All of them if ``df`` is
                      smaller.
      mz_col,
      intensity_col:  column names; the intensity axis is binned on log10.
      top_fraction:   restrict to the most abundant fraction before
                      stratifying; 1.0 keeps the full dynamic range.
      seed:           random seed.
    """
    d = df.dropna(subset=[mz_col, intensity_col])
    d = d[d[intensity_col] > 0]
    if d.empty:
        return d
    if top_fraction and top_fraction < 1.0:
        cut = d[intensity_col].quantile(1 - top_fraction)
        d = d[d[intensity_col] >= cut]
        if d.empty:
            return d
    d = d.reset_index(drop=True)
    if len(d) <= n:
        return d.sort_values(mz_col).reset_index(drop=True)

    mzb = pd.qcut(d[mz_col], min(mz_bins, d[mz_col].nunique()),
                  labels=False, duplicates='drop')
    ab = pd.qcut(np.log10(d[intensity_col]),
                 min(abundance_bins, d[intensity_col].nunique()),
                 labels=False, duplicates='drop')
    cells = list(d.groupby([mzb, ab]).indices.values())
    rng = np.random.default_rng(seed)
    quota = max(1, n // max(len(cells), 1))
    chosen, short = [], 0
    for idx in cells:
        take = min(quota, len(idx))
        chosen.append(rng.choice(idx, take, replace=False))
        short += quota - take
    chosen = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    if short > 0:
        rest = np.setdiff1d(np.arange(len(d)), chosen)
        if len(rest):
            chosen = np.concatenate(
                [chosen, rng.choice(rest, min(short, len(rest)),
                                    replace=False)])
    return d.iloc[np.sort(chosen)].sort_values(mz_col).reset_index(drop=True)
